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
from huggingface_hub import login
from dotenv import load_dotenv


"""
example usage: 
- python bin/get_embeddings.py --data-path data/random_split/test_GO.fasta --weights-dir samirchar/proteinfertorch-go-random-13731645

"""

# load_dotenv()
# login(token=os.getenv("HF_TOKEN"))
logger = get_logger()

# Arguments that must be parsed first
parser_first = argparse.ArgumentParser(add_help=False)

parser_first.add_argument('--config-dir',
                    type=str,
                    default="config",
                    help="Path to the configuration directory (default: config)")


initial_args, _ = parser_first.parse_known_args()

config = read_yaml(
    os.path.join(initial_args.config_dir, "config.yaml")
)


# Argument parser setup. The rest of the args are loaded after the initial args. All args are then updated with the initial args.
parser = argparse.ArgumentParser(description="Inference with ProteInfer model.",parents=[parser_first])

parser.add_argument(
    "--data-path",
    type=str,
    required=True,
    help="Path to the data fasta file."
)

parser.add_argument(
    "--weights-dir",
    type=str,
    required=True,
    help="Directory to the model weights either on huggingface or local"
)

parser.add_argument(
    "--output-dir",
    type=str,
    default=os.path.join(config["paths"]["outputs"],"embeddings"),
    help="Path to the output directory"
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=config["inference"]["batch_size"],
    help="Batch size for inference",
)

parser.add_argument(
    "--num-embedding-partitions",
    type=int,
    default=10,
    help="Number of embedding partitions",
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
    default=config["inference"]["num_workers"],
    help="Number of workers for dataloaders",
)

# load args
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load datasets
test_dataset = ProteinDataset(
        data_path = args.data_path,
        logger=None
        )

dataset_specs = [{"dataset": test_dataset,"name":"test","shuffle": False,"drop_last": False,"batch_size": args.batch_size}]

# Define data loaders
loaders = create_multiple_loaders(
    dataset_specs = dataset_specs,
    num_workers = args.num_workers,
    pin_memory = not args.unpin_memory,
    world_size = 1,
    rank = 0
)

model = ProteInfer.from_pretrained(
    pretrained_model_name_or_path=args.weights_dir,
).to(device).eval()

for loader_name, loader in loaders.items():
    logger.info(f"## Extracting embeddings ##")
    
    embeddings = []
    partition_idxs = sorted(np.arange(len(loader)) % args.num_embedding_partitions) # Assign each batch to a partition

    with torch.no_grad(), torch.amp.autocast(enabled=True,device_type=device.type): 
        for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
            partition_idx = partition_idxs[batch_idx]
            next_partition_idx = float("inf") if batch_idx >= len(loader) - 1 else partition_idxs[batch_idx+1]

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

            batch_embeddings = model.get_embeddings(sequence_onehots, sequence_lengths)
            
            embeddings.append(batch_embeddings.cpu())

            # If in new partition or last partition, save the embeddings and reset the embeddings list
            if next_partition_idx > partition_idx:
                embeddings = torch.cat(embeddings, dim=0)
                torch.save(embeddings, os.path.join(args.output_dir, f"partition_{partition_idx}.pt"))
                logger.info(f"Saved {len(embeddings)} embeddings for partition {partition_idx}")
                embeddings = []
                