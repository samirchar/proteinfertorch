from proteinfertorch.proteinfer import ProteInfer
from proteinfertorch.data import ProteinDataset, create_multiple_loaders
from proteinfertorch.utils import read_json, read_yaml, to_device
from proteinfertorch.config import get_logger, ACTIVATION_MAP
from proteinfertorch.utils import save_evaluation_results, probability_normalizer
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

# Arguments that must be parsed first
parser_first = argparse.ArgumentParser(add_help=False)

parser_first.add_argument('--config-dir',
                    type=str,
                    default="config",
                    help="Path to the configuration directory (default: config)")


parser_first.add_argument(
    "--weights-path",
    type=str,
    required=True,
    help="Path to the weights file"
)

initial_args, _ = parser_first.parse_known_args()

config = read_yaml(
    os.path.join(initial_args.config_dir, "config.yaml")
)

task = initial_args.weights_path.split("/")[-1][:2]
task_defaults = config[f'{task}_defaults']

# Argument parser setup. The rest of the args are loaded after the initial args. All args are then updated with the initial args.
parser = argparse.ArgumentParser(description="Inference with ProteInfer model.",parents=[parser_first])

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
    "--output-dir",
    type=str,
    default=config["paths"]["outputs"],
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
    default=task_defaults["threshold"],
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

parser.add_argument(
    "--parenthood-path",
    type=str,
    default=config["paths"]["parenthood_2019"],
    help="""Path to the parenthood file. 
            Must align with annotations release used in data path.
            Can be None to skip normalization""",
)


# load args
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



num_labels = task_defaults["output_dim"]
model = ProteInfer.from_tf_pretrained(
    weights_path=args.weights_path,
    num_labels=num_labels,
    input_channels=task_defaults["input_dim"],
    output_channels=task_defaults["output_embedding_dim"],
    kernel_size=task_defaults["kernel_size"],
    activation=ACTIVATION_MAP[task_defaults["activation"]],
    dilation_base=task_defaults["dilation_base"],
    num_resnet_blocks=task_defaults["num_resnet_blocks"],
    bottleneck_factor=task_defaults["bottleneck_factor"],
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model = model.eval()

label_normalizer = read_json(args.parenthood_path) if args.parenthood_path is not None else None

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

    prob_norm = probability_normalizer(
                        label_vocab=loader[0].dataset.label_vocabulary,
                        applicable_label_dict = label_normalizer,
                        )
    
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

            if args.parenthood_path is not None:
                probabilities =  torch.tensor(
                    prob_norm(probabilities.cpu().numpy()),
                    device=device,
                )

            metrics["map_micro"].update(probabilities.cpu().flatten(), label_multihots.cpu().flatten())
            metrics["map_macro"].update(probabilities.cpu(), label_multihots.cpu())
            metrics["f1_micro"].update(probabilities.flatten(), label_multihots.flatten())
            metrics["avg_loss"].update(bce_loss(logits, label_multihots.float()))

            if args.save_prediction_results:
                test_results["sequence_ids"].append(sequence_ids)
                test_results["logits"].append(logits.cpu())
                test_results["labels"].append(label_multihots.cpu())
                test_results["probabilities"].append(probabilities.cpu())

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
