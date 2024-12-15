from proteinfertorch import CONFIG_FILE
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
- python bin/inference.py --data-path data/random_split/test_GO.fasta --vocabulary-path data/random_split/full_GO.fasta --weights-dir samirchar/proteinfertorch-go-random-13731645 --map-bins 50

"""

load_dotenv()
login(token=os.getenv("HF_TOKEN"))
logger = get_logger()

# Arguments that must be parsed first
parser_first = argparse.ArgumentParser(add_help=False)

parser_first.add_argument('--config-path',
                    type=str,
                    default=CONFIG_FILE,
                    required=False,
                    help="Path to the configuration yaml path (default: config/config.yaml)")

initial_args, _ = parser_first.parse_known_args()

config = read_yaml(initial_args.config_path)


# Argument parser setup. The rest of the args are loaded after the initial args. All args are then updated with the initial args.
parser = argparse.ArgumentParser(description="Inference with ProteInfer model.",parents=[parser_first])

parser.add_argument(
    "--data-path",
    type=str,
    required=True,
    help="Path to the data fasta file."
)

parser.add_argument(
    "--fasta-separator",
    type=str,
    default=config["data"]["fasta_separator"],
    help="The separator of the header (A.K.A. description or labels) in the FASTA file."
)

parser.add_argument(
    "--vocabulary-path",
    type=str,
    required=True,
    help="Path to the vocabulary file"
)  #TODO: instead of inferring vocab from fasta everytime, should create static vocab json

parser.add_argument(
    "--weights-dir",
    type=str,
    required=True,
    help="Directory to the model weights either on huggingface or local"
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
    default=0.88,
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
    "--batch-size",
    type=int,
    default=config["inference"]["batch_size"],
    help="Batch size for inference",
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

# variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
label_normalizer = read_json(args.parenthood_path) if args.parenthood_path is not None else None


#Load datasets
test_dataset = ProteinDataset(
        data_path = args.data_path,
        vocabulary_path = args.vocabulary_path,
        deduplicate = args.deduplicate,
        max_sequence_length = args.max_sequence_length,
        fasta_separator = args.fasta_separator,
        logger=None
        )

if test_dataset.dataset_has_labels:
    logger.info(f"Dataset includes ground truth labels, will use them for evaluation")
else:
    logger.info(f"Dataset does not include ground truth labels, will only output predictions")

num_labels = len(test_dataset.label_vocabulary)

dataset_specs = [
                    {"dataset": test_dataset,"name":"test","shuffle": False,"drop_last": False,"batch_size": args.batch_size}
                    ]

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

assert model.output_layer.out_features == num_labels, "Number of labels in the model does not match the number of labels in the dataset"


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
                        label_vocab=loader.dataset.label_vocabulary,
                        applicable_label_dict = label_normalizer,
                        )
    
    with torch.no_grad(), torch.amp.autocast(enabled=True,device_type=device.type): 
        for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
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

            if test_dataset.dataset_has_labels:
                metrics["map_micro"].update(probabilities.cpu().flatten(), label_multihots.cpu().flatten())
                metrics["map_macro"].update(probabilities.cpu(), label_multihots.cpu())
                metrics["f1_micro"].update(probabilities.flatten(), label_multihots.flatten())
                metrics["avg_loss"].update(bce_loss(logits, label_multihots.float()))

            if args.save_prediction_results:
                test_results["sequence_ids"].append(sequence_ids)
                test_results["logits"].append(logits.cpu())
                test_results["probabilities"].append(probabilities.cpu())

                if test_dataset.dataset_has_labels:
                    test_results["labels"].append(label_multihots.cpu())

        if test_dataset.dataset_has_labels:
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
                label_vocabulary=loader.dataset.label_vocabulary,
                run_name=args.name,
                output_dir=args.output_dir,
                data_split_name=loader_name,
                logger=logger,
            )
            logger.info("Done saving predictions...")

torch.cuda.empty_cache()
