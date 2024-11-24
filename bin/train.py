from proteinfertorch.proteinfer import ProteInfer
from proteinfertorch.data import ProteinDataset, create_multiple_loaders
from proteinfertorch.utils import read_json, read_yaml, to_device
from proteinfertorch.config import get_logger, ACTIVATION_MAP
from proteinfertorch.utils import save_evaluation_results, probability_normalizer, seed_everything
import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
import datetime
import wandb
import re
from collections import defaultdict
from torcheval.metrics import MultilabelAUPRC, BinaryAUPRC, BinaryF1Score, MultilabelBinnedAUPRC, BinaryBinnedAUPRC, Mean
from torcheval.metrics.toolkit import sync_and_compute_collection, reset_metrics
from huggingface_hub import login
from dotenv import load_dotenv
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist



"""
example usage: 
- python bin/train.py 

"""

def main():

    # Arguments that must be parsed first
    parser_first = argparse.ArgumentParser(add_help=False)

    parser_first.add_argument('--config-dir',
                        type=str,
                        default="config",
                        help="Path to the configuration directory (default: config)")


    parser_first.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task for the model. Either go or ec"
    )

    initial_args, _ = parser_first.parse_known_args()

    config = read_yaml(
        os.path.join(initial_args.config_dir, "config.yaml")
    )


    # Argument parser setup. The rest of the args are loaded after the initial args. All args are then updated with the initial args.
    parser = argparse.ArgumentParser(description="Train ProteInfer model.",parents=[parser_first])

    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to the training data fasta file."
    )

    parser.add_argument(
        "--val-data-path",
        type=str,
        required=True,
        help="Path to the validation data fasta file."
    )

    parser.add_argument(
        "--test-data-path",
        type=str,
        required=False,
        help="Path to the test data fasta file."
    )
    parser.add_argument(
        "--vocabulary-path",
        type=str,
        required=False,
        help="Path to the vocabulary file"
    ) #TODO: instead of inferring vocab from fasta everytime, should create static vocab json

    parser.add_argument(
        "--model-dir",
        type=str,
        required=False,
        default=None,
        help="Directory to the model either on huggingface or local. If not provided, a new model will be initialized."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=config["paths"]["outputs"],
        help="Path to the output directory"
    )

    parser.add_argument(
        "--parenthood-path",
        type=str,
        default=config["paths"]["parenthood_2019"],
        help="""Path to the parenthood file. 
                Must align with annotations release used in data path.
                Can be None to skip normalization""",
        required=False
    )

    parser.add_argument(
        "--name",
        type=str,
        default="proteinfertorch", 
        help="Name of the W&B run and other generated files.",
        required=False
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
        "--train-batch-size",
        type=int,
        default=config["train"]["train_batch_size"],
        help="Batch size for training data",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=config["train"]["test_batch_size"],
        help="Batch size for val and test data",
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
        "--save-val-test-metrics",
        action="store_true",
        default=False,
        help="Append val/test metrics to json",
    )

    parser.add_argument(
        "-n",
        "--nodes",
        default=1,
        type=int,
        metavar="N",
        help="Number of nodes (default: 1)",
    )

    parser.add_argument(
        "-g", "--gpus", default=1, type=int, help="Number of gpus per node (default: 1)"
    )

    parser.add_argument(
        "-nr", "--nr", default=0, type=int, help="Ranking within the nodes"
    )

    parser.add_argument(
        "--amlt",
        action="store_true",
        default=False,
        help="Run job on Amulet. Default is False.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Use Weights & Biases for logging. Default is False.",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default=config["wandb"]["project"],
        help="Wandb project name",
    )

    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=config["wandb"]["entity"],
        help="Wandb entity name",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=config["training"]["seed"],
        help="Seed for reproducibility",
    )

    # load args
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    if args.amlt:
        # os.environ['MASTER_ADDR'] = os.environ['MASTER_IP']
        args.nr = int(os.environ["NODE_RANK"])
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "8889"

    mp.spawn(train, nprocs=args.gpus, args=(args,))


def train(gpu,args):
    # initiate logger
    logger = get_logger()

    # Calculate GPU rank (based on node rank and GPU rank within the node) and initialize process group
    args.rank = args.nr * args.gpus + gpu
    
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=args.world_size, rank=rank
    )

    print(
        f"{'=' * 50}\n"
        f"Initializing GPU {gpu}/{args.gpus-1} on node {args.nr};\n"
        f"    or, gpu {rank+1}/{args.world_size} for all nodes.\n"
        f"{'=' * 50}"
    )
    
    # Check if master process
    is_master = rank == 0

    # Set the GPU device, if using
    torch.cuda.set_device(rank)
    device = torch.device("cuda:" + str(rank) if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    seed_everything(args.seed)
    

    # Set the timezone for logging
    os.environ["TZ"] = "US/Pacific"
    time.tzset()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S %Z").strip()

    # Initialize W&B, if using
    if is_master and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.name}_{timestamp}",
            config=vars(args),
            sync_tensorboard=False,
            entity=args.wandb_entity,
        )

    # label normalizer
    label_normalizer = read_json(args.parenthood_path) if args.parenthood_path is not None else None

    #Load datasets
    test_dataset = ProteinDataset(
            data_path = args.test_data_path,
            vocabulary_path = args.vocabulary_path,
            deduplicate = args.deduplicate,
            max_sequence_length = args.max_sequence_length,
            logger=None
            )

    val_dataset = ProteinDataset(
            data_path = args.val_data_path,
            vocabulary_path = args.vocabulary_path,
            deduplicate = args.deduplicate,
            max_sequence_length = args.max_sequence_length,
            logger=None
            )

    train_dataset = ProteinDataset(
            data_path = args.data_path,
            vocabulary_path = args.vocabulary_path,
            deduplicate = args.deduplicate,
            max_sequence_length = args.max_sequence_length,
            logger=None
            )

    data_loader_specs = [
                        {"dataset": test_dataset,"name":"test","shuffle": False,"drop_last": False,"batch_size": args.test_batch_size},
                        {"dataset": val_dataset,"name":"validation","shuffle": False,"drop_last": False,"batch_size": args.test_batch_size},
                        {"dataset": train_dataset,"name":"train","shuffle": True,"drop_last": True,"batch_size": args.train_batch_size},
                        ]

    # Define data loaders
    loaders = create_multiple_loaders(
        dataset_specs = data_loader_specs,
        num_workers = args.num_workers,
        pin_memory = not args.unpin_memory,
        world_size = args.world_size,
        rank = rank
    )


    model = ProteInfer.from_pretrained(
        pretrained_model_name_or_path=args.model_dir,
    ).to(device).eval()

    num_labels = model.output_layer.out_features

    # Wrap the model in DDP for distributed computing and sync batchnorm if needed.
    # TODO: This may be more flexible to allow for layer norm.
    if args.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model.to(rank), device_ids=[rank], find_unused_parameters=True)

    #Define metrics
    test_metrics = {
        "map_micro": BinaryAUPRC(device="cpu") if args.map_bins is None else BinaryBinnedAUPRC(device="cpu", threshold=args.map_bins),
        "map_macro": MultilabelAUPRC(device="cpu", num_labels=num_labels) if args.map_bins is None else MultilabelBinnedAUPRC(device="cpu", num_labels=num_labels, threshold=args.map_bins),
        "f1_micro": BinaryF1Score(device=device, threshold=args.threshold),
        "avg_loss":  Mean(device=device)
    }

    train_metrics = {test_metrics["avg_loss"],test_metrics["f1_micro"]} #Only need loss and f1 for training because mAP are compute intensive

    for loader_name, loader in loaders.items():
        
        #Reset metrics
        reset_metrics(test_metrics)
        reset_metrics(train_metrics)

        logger.info(f"##{loader_name}##")

        if label_normalizer is not None:
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

            final_metrics = sync_and_compute_collection(metrics)
            #Convert items in final_metrics to scalars if they are tensors and log metrics
            final_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in final_metrics.items()}
            logger.info(final_metrics)
            


    ####### CLEANUP #######
    logger.info(f"\n{'='*100}\nTraining  COMPLETE\n{'='*100}\n")
    # W&B, MLFlow amd optional metric results saving
    if is_master:
        # Optionally save val/test results in json
        if args.save_val_test_metrics:
            metrics_results.append(run_metrics)
            write_json(metrics_results, args.save_val_test_metrics_file)
        # Log test metrics
        if args.test_paths_names:
            if args.use_wandb:
                wandb.log(all_test_metrics)

        # Log val metrics
        if args.validation_path_name:
            if args.use_wandb:
                wandb.log(validation_metrics)

        # Close metric loggers
        if args.use_wandb:
            wandb.finish()

    # Loggers
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
    # Torch
    torch.cuda.empty_cache()
    dist.destroy_process_group()