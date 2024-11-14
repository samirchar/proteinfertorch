from proteinfertorch.proteinfer import ProteInfer
from proteinfertorch.data import ProteinDataset, create_multiple_loaders
from proteinfertorch.utils import read_json, read_yaml, generate_vocabularies, to_device
from proteinfertorch.config import get_logger, ACTIVATION_MAP
from proteinfertorch.utils import save_evaluation_results
import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
import re
from collections import defaultdict
from torcheval.metrics import MultilabelAUPRC, BinaryAUPRC, BinaryF1Score, MultilabelBinnedAUPRC, BinaryBinnedAUPRC, Mean
from torcheval.metrics.toolkit import sync_and_compute_collection, reset_metrics



logger = get_logger()

# Argument parser setup
parser = argparse.ArgumentParser(description="Train and/or Test the ProteInfer model.")

parser.add_argument('--config-dir',
                    type=str,
                    default="config",
                    help="Path to the configuration directory (default: config)")
initial_args, _ = parser.parse_known_args()

config = read_yaml(
    os.path.join(initial_args.config_dir, "config.yaml")
)

parser.add_argument(
    "--data-path",
    type=str,
    required=True,
    help="Path to the data fasta file."
)

parser.add_argument(
    "--vocabulary-path",
    type=str,
    required=True,
    help="Path to the vocabulary file"
)

parser.add_argument(
    "--weights-path",
    type=str,
    required=True,
    help="Path to the weights file"
)

parser.add_argument(
    "--output-dir",
    type=str,
    default=config["default_dirs"]["outputs"],
    help="Path to the output directory"
)

parser.add_argument(
    "--name",
    type=str,
    default="ProteInfer", #TODO: add uuid
    help="Name of the W&B run. If not provided, a name will be generated.",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=config["inference"]["threshold"]
)

parser.add_argument(
    "--deduplicate",
    action="store_true",
    default=False,
    help="Deduplicate sequences in the dataset",
)

parser.add_argument(
    "--max-sequence-length",
    type=int,
    default=config["data"]["max_sequence_length"],
    help="Maximum sequence length to consider",
)

parser.add_argument(
    "--unpin-memory",
    action="store_true",
    default=False,
    help="Whether to unpin memory for dataloaders",
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=config["training"]["num_workers"],
    help="Number of workers for dataloaders",
)

parser.add_argument(
    "--map-bins",
    type=int,
    default=config["inference"]["map_bins"],
    help="Number of bins for MAP calculation",
)
parser.add_argument(
    "--save-prediction-results",
    action="store_true",
    default=False,
    help="Save predictions and ground truth dataframe for validation and/or test",
)

args = parser.parse_args()

test_dataset = ProteinDataset(
        data_path = args.data_path,
        vocabulary_path = args.vocabulary_path,
        deduplicate = args.deduplicate,
        max_sequence_length = args.max_sequence_length,
        logger=None
        )

dataset_specs = [
                    {"dataset": test_dataset,"name":"test","shuffle": False,"drop_last": False,"batch_size": config["inference"]["batch_size"]}
                    ]
# dataset_specs = [
#                     {"dataset": ProteinDataset(),"type": "train","name":"train","shuffle": True,"drop_last": True,"batch_size": 32},
#                     {"dataset": ProteinDataset(),"type": "validation","name":"validation","shuffle": False,"drop_last": False,"batch_size": 32},
#                     {"dataset": ProteinDataset(),"type": "test","name":"test","shuffle": False,"drop_last": False,"batch_size": 32}
#                     ]

# Define data loaders
loaders = create_multiple_loaders(
    dataset_specs = dataset_specs,
    num_workers = args.num_workers,
    pin_memory = not args.unpin_memory,
    world_size = 1,
    rank = 0
)

#Extract the first two characters of the model weights path as the architecture type: go or ec
architecture_type = args.weights_path.split("/")[-1][:2]
architecture_config = read_yaml(
    os.path.join(initial_args.config_dir, config['architecture_configs'][architecture_type])
)

num_labels = architecture_config["architecture"]["output_dim"]
model = ProteInfer.from_pretrained(
    weights_path=args.weights_path,
    num_labels=num_labels,
    input_channels=architecture_config["architecture"]["input_dim"],
    output_channels=architecture_config["architecture"]["output_embedding_dim"],
    kernel_size=architecture_config["architecture"]["kernel_size"],
    activation=ACTIVATION_MAP[architecture_config["architecture"]["activation"]],
    dilation_base=architecture_config["architecture"]["dilation_base"],
    num_resnet_blocks=architecture_config["architecture"]["num_resnet_blocks"],
    bottleneck_factor=architecture_config["architecture"]["bottleneck_factor"],
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model = model.eval()

# label_normalizer = read_json(paths["PARENTHOOD_LIB_PATH"])


for loader_name, loader in loaders.items():
    logger.info(f"##{loader_name}##")
    test_results = defaultdict(list)
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    metrics = {
        "map_micro": BinaryAUPRC(device="cpu") if args.map_bins is None else BinaryBinnedAUPRC(device="cpu", threshold=args.map_bins),
        "map_macro": MultilabelAUPRC(device="cpu", num_labels=num_labels) if args.map_bins is None else MultilabelBinnedAUPRC(device="cpu", num_labels=num_labels, threshold=args.map_bins),
        "f1_micro": BinaryF1Score(device=device, threshold=args.threshold),
        "avg_loss":  Mean(device=device)
    }

    with torch.no_grad(), torch.amp.autocast(enabled=True,device_type=device.type):
        for batch_idx, batch in tqdm(enumerate(loader[0]), total=len(loader[0])):
            # Unpack the validation or testing batch
            (
                sequence_onehots,
                sequence_lengths,
                sequence_ids,
                label_multihots
            ) = (
                batch["sequence_onehots"],
                batch["sequence_lengths"],
                batch["sequence_ids"],
                batch["label_multihots"]
            )
            
            sequence_onehots, sequence_lengths, label_multihots = to_device(
                device, sequence_onehots, sequence_lengths, label_multihots
            )

            logits = model(sequence_onehots, sequence_lengths)

            probabilities = torch.sigmoid(logits)

            metrics["map_micro"].update(probabilities.cpu().flatten(), label_multihots.cpu().flatten())
            metrics["map_macro"].update(probabilities.cpu(), label_multihots.cpu())
            metrics["f1_micro"].update(probabilities.flatten(), label_multihots.flatten())
            metrics["avg_loss"].update(bce_loss(logits, label_multihots.float()))

            if args.save_prediction_results:
                test_results["sequence_ids"].append(sequence_ids)
                test_results["logits"].append(logits.cpu())
                test_results["labels"].append(label_multihots.cpu())

        final_metrics = sync_and_compute_collection(metrics)
        #Convert items in final_metrics to scalars if they are tensors and log metrics
        final_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in final_metrics.items()}
        logger.info(final_metrics)
        

        if args.save_prediction_results:
            for key in test_results.keys():
                if key == "sequence_ids":
                    test_results[key] = np.array(
                        [j for i in test_results["sequence_ids"] for j in i]
                    )
                else:
                    test_results[key] = torch.cat(test_results[key]).numpy()
            logger.info("saving predictions...")
            save_evaluation_results(
                results=test_results,
                label_vocabulary=loader[0].dataset.label_vocabulary,
                run_name=args.name,
                output_dir=args.output_dir,
                data_split_name=loader_name,
                logger=logger,
            )
            logger.info("Done saving predictions...")

torch.cuda.empty_cache()
