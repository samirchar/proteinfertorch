from proteinfertorch.data import ProteinDataset, create_multiple_loaders
from proteinfertorch.proteinfer import ProteInfer
from proteinfertorch.utils import read_json, read_yaml, generate_vocabularies, to_device
from proteinfertorch.config import get_logger, ACTIVATION_MAP
import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
import re
from collections import defaultdict
from torcheval.metrics import MultilabelAUPRC, BinaryAUPRC
from torch.cuda.amp import autocast

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
    help="Path to the data file"
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
    "--name",
    type=str,
    default="ProteInfer",
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
    "--save-prediction-results",
    action="store_true",
    default=False,
    help="Save predictions and ground truth dataframe for validation and/or test",
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

args = parser.parse_args()

test_dataset = ProteinDataset(
        data_path = args.data_path,
        vocabulary_path = args.vocabulary_path,
        deduplicate = args.deduplicate,
        max_sequence_length = args.max_sequence_length,
        logger=None
        )

dataset_specs = [
                    {"dataset": test_dataset,"name":"test","shuffle": False,"drop_last": False,"batch_size": config["inference"]["test_batch_size"]}
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
    world_size = world_size,
    rank = rank
)

#Extract the first two characters of the model weights path as the architecture type: go or ec
architecture_type = args.weights_path.split("/")[-1][:2]
architecture = read_yaml(config['architecture_config'][architecture_type])

model = ProteInfer.from_pretrained(
    weights_path=args.weights_path,
    num_labels=architecture["output_dim"],
    input_channels=architecture["input_dim"],
    output_channels=architecture["output_embedding_dim"],
    kernel_size=architecture["kernel_size"],
    activation=ACTIVATION_MAP[architecture["activation"]],
    dilation_base=architecture["dilation_base"],
    num_resnet_blocks=architecture["num_resnet_blocks"],
    bottleneck_factor=architecture["bottleneck_factor"],
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model = model.eval()

# label_normalizer = read_json(paths["PARENTHOOD_LIB_PATH"])

PROTEINFER_VOCABULARY = generate_vocabularies(
    file_path=config["paths"][full_data_path]
)["label_vocab"]

for loader_name, loader in loaders.items():
    print(loader_name, len(loader[0].dataset.label_vocabulary))

    represented_labels = [
        label in loader[0].dataset.label_vocabulary for label in PROTEINFER_VOCABULARY
    ]

    test_metrics = eval_metrics.get_metric_collection_with_regex(
        pattern="f1_m.*",
        threshold=args.threshold,
        num_labels=label_sample_sizes["test"]
        if (params["IN_BATCH_SAMPLING"] or params["GRID_SAMPLER"]) is False
        else None,
    )

    bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
    focal_loss = FocalLoss(
        gamma=config["params"]["FOCAL_LOSS_GAMMA"],
        alpha=config["params"]["FOCAL_LOSS_ALPHA"],
    )
    total_bce_loss = 0
    total_focal_loss = 0
    test_results = defaultdict(list)

    mAP_micro = BinaryAUPRC(device="cpu")
    mAP_macro = MultilabelAUPRC(device="cpu", num_labels=label_sample_sizes["test"])

    with torch.no_grad(), autocast(enabled=True):
        for batch_idx, batch in tqdm(enumerate(loader[0]), total=len(loader[0])):
            # Unpack the validation or testing batch
            (
                sequence_onehots,
                sequence_lengths,
                sequence_ids,
                label_multihots,
                label_embeddings,
            ) = (
                batch["sequence_onehots"],
                batch["sequence_lengths"],
                batch["sequence_ids"],
                batch["label_multihots"],
                batch["label_embeddings"],
            )
            sequence_onehots, sequence_lengths, label_multihots = to_device(
                device, sequence_onehots, sequence_lengths, label_multihots
            )

            logits = model(sequence_onehots, sequence_lengths)
            if args.only_represented_labels:
                logits = logits[:, represented_labels]

            probabilities = torch.sigmoid(logits)

            if not args.only_inference:
                test_metrics(probabilities, label_multihots)

                if loader_name in ["validation", "test"]:
                    mAP_micro.update(
                        probabilities.cpu().flatten(), label_multihots.cpu().flatten()
                    )
                    mAP_macro.update(probabilities.cpu(), label_multihots.cpu())

                total_bce_loss += bce_loss(logits, label_multihots.float())
                total_focal_loss += focal_loss(logits, label_multihots.float())

            if args.save_prediction_results:
                test_results["sequence_ids"].append(sequence_ids)
                test_results["logits"].append(logits.cpu())
                test_results["labels"].append(label_multihots.cpu())

        if not args.only_inference:
            test_metrics = test_metrics.compute()
            test_metrics.update({"bce_loss": total_bce_loss / len(loader[0])})
            test_metrics.update({"focal_loss": total_focal_loss / len(loader[0])})

            if loader_name in ["validation", "test"]:
                test_metrics.update(
                    {"map_micro": mAP_micro.compute(), "map_macro": mAP_macro.compute()}
                )

        print("\n\n", "=" * 20)
        print(f"##{loader_name}##")
        print(test_metrics)
        print("=" * 20, "\n\n")

        if args.save_prediction_results:
            for key in test_results.keys():
                if key == "sequence_ids":
                    test_results[key] = np.array(
                        [j for i in test_results["sequence_ids"] for j in i]
                    )
                else:
                    test_results[key] = torch.cat(test_results[key]).numpy()
            print("saving resuts...")
            save_evaluation_results(
                results=test_results,
                label_vocabulary=loader[0].dataset.label_vocabulary,
                run_name=f"{task}_{args.name}" + (str(args.model_weights_id)
                if args.model_weights_id is not None
                else ""),
                output_dir=config["paths"]["RESULTS_DIR"],
                data_split_name=loader_name,
                save_as_h5=True,
            )
            print("Done saving resuts...")
torch.cuda.empty_cache()
