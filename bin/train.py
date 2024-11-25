from proteinfertorch.proteinfer import ProteInfer
from proteinfertorch.data import ProteinDataset, create_multiple_loaders
from proteinfertorch.utils import read_json, read_yaml, to_device, save_checkpoint
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
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_




"""
example usage: 

From HF weights pretrained:
- python bin/train.py --train-data-path data/random_split/train_GO.fasta --validation-data-path data/random_split/dev_GO.fasta --test-data-path data/random_split/test_GO.fasta --vocabulary-path data/random_split/full_GO.fasta --model-dir <username>/proteinfertorch-go-random-13731645 --task go --map-bins 50 --use-wandb

From random weights with possibly custom architecture: #TODO: modify code to allow for custom architecture
- python bin/train.py --train-data-path data/random_split/train_GO.fasta --validation-data-path data/random_split/dev_GO.fasta --test-data-path data/random_split/test_GO.fasta --vocabulary-path data/random_split/full_GO.fasta --model-dir <username>/proteinfertorch-go-random-13731645 --task go --map-bins 50 --use-wandb

"""

def main():

    # Arguments that must be parsed first
    parser_first = argparse.ArgumentParser(add_help=False)

    parser_first.add_argument('--config-dir',
                        type=str,
                        default="config",
                        required=False,
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
        "--validation-data-path",
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
        "--lr",
        type=float,
        default=config["train"]["lr"],
        help="Learning rate for the optimizer"
    )

    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default=config["train"]["lr_scheduler"],
        help="Learning rate scheduler for the optimizer"
    )

    parser.add_argument(
        "--lr-decay",
        type=float,
        default=config["train"]["lr_decay"],
        help="Learning rate decay for the optimizer"
    )

    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=config["train"]["lr_decay_steps"],
        help="Number of steps to decay the learning rate"
    )

    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=config["train"]["lr_warmup_steps"],
        help="Number of steps to warm up the learning rate"
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=config["train"]["weight_decay"],
        help="Weight decay for the optimizer"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=config["train"]["epochs"],
        help="Number of epochs to train the model"
    )

    parser.add_argument(
        "--epochs-per-validation",
        type=int,
        default=config["train"]["epochs_per_validation"],
        help="Number of epochs between each validation"
    )

    parser.add_argument(
        "--always-save-checkpoint",
        action="store_true",
        default=False,
        help="Save the model checkpoint after every validation"
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=config["train"]["gradient_accumulation_steps"], 
        help="Number of gradient accumulation steps"
    )

    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=config["train"]["gradient_clip"],
        help="Gradient clipping value"
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
        default=config["train"]["num_workers"],
        help="Number of workers for dataloaders",
    )

    parser.add_argument(
        "--map-bins",
        type=int,
        default=config["inference"]["map_bins"],
        help="Number of bins for MAP calculation",
    )

    parser.add_argument(
        "--save-eval-metrics",
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
        default=config["train"]["seed"],
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


def evaluate(loader, model, metrics, loss_fn, device, prob_norm=None):
    reset_metrics(metrics.values())
    with torch.no_grad(), torch.amp.autocast(enabled=True,device_type=device.type): 
        for _, batch in tqdm(enumerate(loader), total=len(loader)):
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

            if prob_norm is not None:
                probabilities =  torch.tensor(
                    prob_norm(probabilities.cpu().numpy()),
                    device=device,
                )

            metrics["map_micro"].update(probabilities.cpu().flatten(), label_multihots.cpu().flatten())
            metrics["map_macro"].update(probabilities.cpu(), label_multihots.cpu())
            metrics["f1_micro"].update(probabilities.flatten(), label_multihots.flatten())
            metrics["avg_loss"].update(loss_fn(logits, label_multihots.float()))

        #Compute metrics
        metrics = sync_and_compute_collection(metrics)
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

        return metrics
            

def train(gpu,args):
    # initiate logger
    logger = get_logger()
    load_dotenv()

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

    # Set the seed for reproducibility
    seed_everything(seed=args.seed,device=device)
    

    # Set the timezone for logging
    os.environ["TZ"] = "US/Pacific"
    time.tzset()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S %Z").strip()
    args.name = f"{args.name}_{timestamp}"

    # Initialize W&B, if using
    if is_master and args.use_wandb:
        wandb.login(key=os.environ["WANDB_API_KEY"],relogin=True)
        wandb.init(
            project=args.wandb_project,
            name=args.name,
            config=vars(args),
            sync_tensorboard=False,
            entity= args.wandb_entity
            
        )

    #Load datasets
    test_dataset = ProteinDataset(
            data_path = args.test_data_path,
            vocabulary_path = args.vocabulary_path,
            deduplicate = args.deduplicate,
            max_sequence_length = args.max_sequence_length,
            logger=None
            )

    validation_dataset = ProteinDataset(
            data_path = args.validation_data_path,
            vocabulary_path = args.vocabulary_path,
            deduplicate = args.deduplicate,
            max_sequence_length = args.max_sequence_length,
            logger=None
            )

    train_dataset = ProteinDataset(
            data_path = args.train_data_path,
            vocabulary_path = args.vocabulary_path,
            deduplicate = args.deduplicate,
            max_sequence_length = args.max_sequence_length,
            logger=None
            )

    data_loader_specs = [
                        {"dataset": test_dataset,"name":"test","shuffle": False,"drop_last": False,"batch_size": args.test_batch_size},
                        {"dataset": validation_dataset,"name":"validation","shuffle": False,"drop_last": False,"batch_size": args.test_batch_size},
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

    # label normalizer
    prob_norm = probability_normalizer(
                        label_vocab=loaders["train"].dataset.label_vocabulary,
                        applicable_label_dict = read_json(args.parenthood_path)
                        )  if args.parenthood_path is not None else None

    # Load model
    model = ProteInfer.from_pretrained(
        pretrained_model_name_or_path=args.model_dir,
    ).to(device).eval()

    num_labels = model.output_layer.out_features

    # Wrap the model in DDP for distributed computing and sync batchnorm if needed.
    # TODO: This may be more flexible to allow for layer norm.
    if args.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model.to(rank), device_ids=[rank])

    #Define metrics
    eval_metrics = {
        "map_micro": BinaryAUPRC(device="cpu") if args.map_bins is None else BinaryBinnedAUPRC(device="cpu", threshold=args.map_bins),
        "map_macro": MultilabelAUPRC(device="cpu", num_labels=num_labels) if args.map_bins is None else MultilabelBinnedAUPRC(device="cpu", num_labels=num_labels, threshold=args.map_bins),
        "f1_micro": BinaryF1Score(device=device, threshold=args.threshold),
        "avg_loss":  Mean(device=device)
    }

    reset_metrics(eval_metrics.values())

    train_metrics = { k:v for k,v in eval_metrics.items() if k in ["avg_loss","f1_micro"]} #Only need loss and f1 for training because mAP are compute intensive

    #Loss function, optimizer, grad scaler, and label normalizer
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()
    
    ## TRAINING LOOP ##
    best_validation_loss = float("inf")
    training_step = 0 #Keep track of the number of training steps

    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch + 1} / {args.epochs}")
        reset_metrics(train_metrics.values()) #Reset metrics for the epoch

        #Set the epoch for the sampler to shuffle the data
        if hasattr(loaders["train"].sampler, "set_epoch"):
            loaders["train"].sampler.set_epoch(epoch) 

        for batch_idx, batch in tqdm(enumerate(loaders['train']), total=len(loaders['train'])):
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
            with torch.amp.autocast(enabled=True,device_type=device.type): 
                # Forward pass
                logits = model(sequence_onehots, sequence_lengths)

                # Compute loss, normalized by the number of gradient accumulation step  
                loss = loss_fn(logits, label_multihots.float()) / args.gradient_accumulation_steps
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()

            # Gradient accumulation every gradient_accumulation_steps
            if (training_step % args.gradient_accumulation_steps == 0) or ( 
                batch_idx + 1 == len(loaders)
            ):
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)

                # Apply gradient clipping
                if args.gradient_clip is not None:
                    clip_grad_norm_(
                        model.module.parameters(), max_norm=args.gradient_clip
                    )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_metrics["f1_micro"].update(logits.detach().flatten(), label_multihots.detach().flatten())
            train_metrics["avg_loss"].update(loss_fn(logits.detach(), label_multihots.detach().float()))

        
        #Compute train and validation epoch metrics
        train_metrics = sync_and_compute_collection(train_metrics)
        train_metrics_float = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in train_metrics.items()}
        # Log metrics
        logger.info(f"Finished epoch {epoch}")
        logger.info("Train metrics:\n{}".format(train_metrics_float))
        if args.use_wandb and is_master:
            wandb.log(train_metrics_float, step=epoch)

        # Validation every args.epochs_per_validation
        if (epoch + 1) % args.epochs_per_validation == 0:
            validation_metrics = evaluate(loader=loaders["validation"],
                                    model=model,
                                    metrics=eval_metrics,
                                    loss_fn=loss_fn,
                                    device=device,
                                    prob_norm=prob_norm
                                    )
            validation_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in validation_metrics.items()}

            # Log metrics
            logger.info("Validation metrics:\n{}".format(validation_metrics))
            if args.use_wandb and is_master:
                wandb.log(validation_metrics, step=epoch)

            # Save model checkpoint
            if is_master & (args.always_save_checkpoint or validation_metrics["avg_loss"] < best_validation_loss):
                if validation_metrics["avg_loss"] < best_validation_loss:
                    checkpoint_path = os.path.join(args.output_dir,"checkpoints", f"{args.name}_best_checkpoint_{timestamp}.pt")
                elif args.always_save_checkpoint:
                    checkpoint_path = os.path.join(args.output_dir,"checkpoints", f"{args.name}_checkpoint_epoch_{epoch}_{timestamp}.pt")

                save_checkpoint(model = model.module,
                                optimizer = optimizer,
                                epoch = epoch,
                                validation_metrics = validation_metrics,
                                train_metrics = train_metrics_float,
                                model_path = checkpoint_path
                                )
    
    ####### CLEANUP #######
    logger.info(f"\n{'='*100}\nTraining  COMPLETE\n{'='*100}\n")
    # W&B, MLFlow amd optional metric results saving
    if is_master:
        # Optionally save val/test results in json
        # if args.save_eval_metrics:
        #     metrics_results.append(run_metrics)
        #     write_json(metrics_results, args.save_eval_metrics_file)

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

if __name__ == "__main__":
    main()