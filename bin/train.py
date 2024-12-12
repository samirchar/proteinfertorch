from proteinfertorch.proteinfer import ProteInfer
from proteinfertorch.data import ProteinDataset, create_multiple_loaders
from proteinfertorch.utils import read_json, read_yaml, to_device, save_checkpoint
from proteinfertorch.config import get_logger, ACTIVATION_MAP
from proteinfertorch.utils import save_evaluation_results, probability_normalizer, seed_everything, get_model
import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
import datetime
import wandb
import csv
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
from torch.optim.lr_scheduler import LinearLR, SequentialLR, LambdaLR
from proteinfertorch.schedulers import ExponentialDecay



"""
example usage: 

From HF weights pretrained:
- python bin/train.py --train-data-path data/random_split/train_GO.fasta --validation-data-path data/random_split/dev_GO.fasta --test-data-path data/random_split/test_GO.fasta --vocabulary-path data/random_split/full_GO.fasta --weights-dir <username>/proteinfertorch-go-random-13731645 --map-bins 50 --use-wandb

From random weights with possibly custom architecture: #TODO: modify code to allow for custom architecture
- python bin/train.py --train-data-path data/random_split/train_GO.fasta --validation-data-path data/random_split/dev_GO.fasta --test-data-path data/random_split/test_GO.fasta --vocabulary-path data/random_split/full_GO.fasta --map-bins 50 --use-wandb

"""

def main():

    # Arguments that must be parsed first
    parser_first = argparse.ArgumentParser(add_help=False)

    parser_first.add_argument('--config-dir',
                        type=str,
                        default="config",
                        required=False,
                        help="Path to the configuration directory (default: config)")



    initial_args, _ = parser_first.parse_known_args()

    config = read_yaml(
        os.path.join(initial_args.config_dir, "config.yaml")
    )
    

    # Argument parser setup. The rest of the args are loaded after the initial args. All args are then updated with the initial args.
    parser = argparse.ArgumentParser(description="Train ProteInfer model.",parents=[parser_first])

    # Data paths arguments
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
        "--fasta-separator",
        type=str,
        default=config["data"]["fasta_separator"],
        help="The separator of the header (A.K.A. description or labels) in the FASTA file."
    )

    parser.add_argument(
        "--vocabulary-path",
        type=str,
        required=False,
        default=None,
        help="Path to the vocabulary file"
    ) #TODO: instead of inferring vocab from fasta everytime, should create static vocab json

    parser.add_argument(
        "--weights-dir",
        type=str,
        required=False,
        default=None,
        help="Directory to the model weights either on huggingface or local. If not provided, a new model will be initialized."
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
        default= None, # config["paths"]["parenthood_2019"],
        help="""Path to the parenthood file. 
                Must align with annotations release used in data path.
                Can be None to skip normalization""",
        required=False
    )

    # Model architecture arguments
    parser.add_argument(
        "--input-dim",
        type=int,
        default=config["base_architecture"]["input_dim"],
        help="Input dimension for the model. Tyically 20 for amino acids."
    )

    parser.add_argument(
        "--output-embedding-dim",
        type=int,
        default=config["base_architecture"]["output_embedding_dim"],
        help="Output embedding dimension for the model."
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=config["base_architecture"]["kernel_size"],
        help="Kernel size for the model."
    )

    parser.add_argument(
        "--activation",
        type=str,
        default=config["base_architecture"]["activation"],
        help="Activation function for the model."
    )

    parser.add_argument(
        "--dilation-base",
        type=int,
        default=config["base_architecture"]["dilation_base"],
        help="Dilation base for the model."
    )

    parser.add_argument(
        "--num-resnet-blocks",
        type=int,
        default=config["base_architecture"]["num_resnet_blocks"],
        help="Number of resnet blocks for the model."
    )

    parser.add_argument(
        "--bottleneck-factor",
        type=int,
        default=config["base_architecture"]["bottleneck_factor"],
        help="Bottleneck factor for the model within resnet blocks."
    )

    # Training arguments

    parser.add_argument(
        "--seed",
        type=int,
        default=config["train"]["seed"],
        help="Seed for reproducibility",
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
        "--lr-scheduler-staircase",
        action="store_true",
        default=False,
        help="Whether to use staircase decay for the learning rate"
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
        "--no-checkpoints",
        action="store_true",
        default=False,
        help="Do not save any model checkpoints"
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
        "--amlt",
        action="store_true",
        default=False,
        help="Run job on Amulet. Default is False.",
    )

    # Distributed training arguments
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

    # W&B arguments
    parser.add_argument(
        "--name",
        type=str,
        default="proteinfertorch", 
        help="Name of the W&B run and other generated files.",
        required=False
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


    # load args
    args = parser.parse_args()

    

    args.world_size = args.gpus * args.nodes
    if args.world_size > 1:
        if args.amlt:
            # os.environ['MASTER_ADDR'] = os.environ['MASTER_IP']
            args.nr = int(os.environ["NODE_RANK"])
        else:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "8889"

        mp.spawn(train, nprocs=args.gpus, args=(args,))
    else:
        train(0, args)
        


def evaluate(loader, model, metric_collection, loss_fn, device, prob_norm=None):
    reset_metrics(metric_collection.values())
    model.eval()
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

            #Create CPU Versions
            # label_multihots = label_multihots.cpu()
            # probabilities = probabilities.cpu()

            #Create flattened versions
            label_multihots_flat = label_multihots.flatten()
            probabilities_flat = probabilities.flatten()

            metric_collection["map_micro"].update(probabilities_flat, label_multihots_flat)
            metric_collection["map_macro"].update(probabilities, label_multihots)
            metric_collection["f1_micro"].update(probabilities_flat, label_multihots_flat)
            metric_collection["avg_loss"].update(loss_fn(logits, label_multihots.float()))

        #Compute metrics
        metrics = sync_and_compute_collection(metric_collection)
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

    model.train()
    reset_metrics(metric_collection.values())
    return metrics
            

def train(gpu,args):
    # torch.cuda.memory._record_memory_history()

    # initiate logger
    logger = get_logger()
    load_dotenv()



    # Calculate GPU rank (based on node rank and GPU rank within the node) and initialize process group
    args.rank = args.nr * args.gpus + gpu
    
    rank = args.nr * args.gpus + gpu

    if args.world_size > 1:
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

    if is_master:
        logger.info(f"Using device: {device}")
        # Log the arguments
        logger.info(f"Arguments: {args}")

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
            fasta_separator = args.fasta_separator,
            logger=None
            )

    validation_dataset = ProteinDataset(
            data_path = args.validation_data_path,
            vocabulary_path = args.vocabulary_path,
            deduplicate = args.deduplicate,
            max_sequence_length = args.max_sequence_length,
            fasta_separator = args.fasta_separator,
            logger=None
            )

    train_dataset = ProteinDataset(
            data_path = args.train_data_path,
            vocabulary_path = args.vocabulary_path,
            deduplicate = args.deduplicate,
            max_sequence_length = args.max_sequence_length,
            fasta_separator = args.fasta_separator,
            logger=None
            )

    # Assert dataset has labels
    assert train_dataset.has_labels & test_dataset.has_labels & validation_dataset.has_labels, "All datasets must have labels for training"

    num_labels = len(train_dataset.label_vocabulary)

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
    if args.weights_dir is not None:
        if is_master:
            logger.info(f"Loading model from pre-trained weights in {args.weights_dir}")

        model = ProteInfer.from_pretrained(
            pretrained_model_name_or_path=args.weights_dir,
        ).to(device)

        assert model.output_layer.out_features == num_labels, "Number of labels in the model does not match the number of labels in the dataset"
        
    else:
        if is_master:
            logger.info(f"Initializing model from random weights")
        model = ProteInfer(
            num_labels=num_labels,
            input_channels=args.input_dim,
            output_channels=args.output_embedding_dim,
            kernel_size=args.kernel_size,
            activation=ACTIVATION_MAP[args.activation],
            dilation_base=args.dilation_base,
            num_resnet_blocks=args.num_resnet_blocks,
            bottleneck_factor=args.bottleneck_factor,
        ).to(device)


    # Wrap the model in DDP for distributed computing and sync batchnorm if needed.
    # TODO: This may be more flexible to allow for layer norm.
    if args.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu])

    #Define metrics
    metric_collection_eval = {
        "map_micro": BinaryAUPRC(device="cpu") if args.map_bins is None else BinaryBinnedAUPRC(device="cpu", threshold=args.map_bins),
        "map_macro": MultilabelAUPRC(device="cpu", num_labels=num_labels) if args.map_bins is None else MultilabelBinnedAUPRC(device="cpu", num_labels=num_labels, threshold=args.map_bins),
        "f1_micro": BinaryF1Score(device=device, threshold=args.threshold),
        "avg_loss":  Mean(device=device)
    }

    metric_collection_train = { k:v for k,v in metric_collection_eval.items() if k in ["avg_loss","f1_micro"]} #Only need loss and f1 for training because mAP are compute intensive

    #Loss function, optimizer, grad scaler, and label normalizer
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.AdamW(get_model(model).parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    #LR Schedule
    lr_warmup_scheule = LinearLR(optimizer=optimizer,
                                start_factor=1/args.lr_warmup_steps,
                                end_factor=1,
                                total_iters=args.lr_warmup_steps
                                )

    lr_decay_schedule = LambdaLR(optimizer = optimizer,
                                    lr_lambda=ExponentialDecay(decay_steps=args.lr_decay_steps, decay_rate=args.lr_decay, staircase=args.lr_scheduler_staircase)
                                    )

    lr_scheduler = SequentialLR(optimizer=optimizer,
                                schedulers = [lr_warmup_scheule, lr_decay_schedule],
                                milestones=[args.lr_warmup_steps])
    
    ## TRAINING LOOP ##
    best_validation_loss = float("inf")
    model.train() #Set the model to training mode

    # Watch the model with W&B
    if args.use_wandb:
        wandb.watch(model, log = 'gradients')

    for epoch in range(args.epochs):
        if is_master:
            logger.info(f"Starting epoch {epoch + 1} / {args.epochs}")
        reset_metrics(metric_collection_train.values()) #Reset metrics for the epoch

        #Set the epoch for the sampler to shuffle the data
        if hasattr(loaders["train"].sampler, "set_epoch"):
            loaders["train"].sampler.set_epoch(epoch) 

        for batch_idx, batch in tqdm(enumerate(loaders['train']), total=len(loaders['train'])):
            
            # Calculate training step considering gradient accumulation
            global_batch_idx = (epoch * len(loaders['train']) + batch_idx)
            global_step =  global_batch_idx // args.gradient_accumulation_steps
            

            # Unpack the validation or testing batch
            (
                sequence_onehots,
                sequence_lengths,
                _,
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

            max_seq_length,min_seq_length = sequence_lengths.max().item(),sequence_lengths.min().item() #TODO: remove this

            if is_master and args.use_wandb:
                wandb.log(
                    {"max_seq_length": max_seq_length,
                    "min_seq_length": min_seq_length,
                    "max_to_min_seq_length_ratio": max_seq_length/min_seq_length},
                    step=global_step
                )
                
            with torch.amp.autocast(enabled=True,device_type=device.type): 
                # Forward pass
                logits = model(sequence_onehots, sequence_lengths)
                

                # Compute loss, normalized by the number of gradient accumulation step  
                loss = loss_fn(logits, label_multihots.float())
                scaled_loss = loss / args.gradient_accumulation_steps
            
            
            # Backward pass with mixed precision
            scaler.scale(scaled_loss).backward()
            

            # Gradient accumulation every gradient_accumulation_steps or at the end of the epoch
            if (batch_idx % args.gradient_accumulation_steps == 0) or ( 
                batch_idx + 1 == len(loaders["train"])
            ):
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)

                # Apply gradient clipping
                if args.gradient_clip is not None:
                    clip_grad_norm_(
                        get_model(model).parameters(), max_norm=args.gradient_clip
                    )
                #optimization step
                scaler.step(optimizer)
                scaler.update()

                # Zero the gradients
                optimizer.zero_grad()

                # Update learning rate
                lr_scheduler.step()

                

                if is_master and args.use_wandb:
                    wandb.log({"per_batch_learning_rate": optimizer.param_groups[0]["lr"]}, step=global_step)
                    wandb.log({"per_batch_loss": loss.item()}, step=global_step) #TODO: this is not the average loss, but the loss for the last batch. Good enough for debugging.

            metric_collection_train["f1_micro"].update(logits.detach().flatten(), label_multihots.detach().flatten())
            metric_collection_train["avg_loss"].update(loss.detach())
            
        
        
            
        #Compute train and validation epoch metrics
        train_metrics = sync_and_compute_collection(metric_collection_train)
        train_metrics = {f"train_{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in train_metrics.items()}
        
        # Log metrics
        if is_master:
            logger.info(f"Finished epoch {epoch}")
            logger.info("Train metrics:\n{}".format(train_metrics))
            if args.use_wandb:
                wandb.log({**train_metrics,
                           **{"epoch":epoch, 
                              "learning_rate": optimizer.param_groups[0]["lr"]}}, step=global_step)

        # Validation every args.epochs_per_validation
        if (epoch + 1) % args.epochs_per_validation == 0:
            validation_metrics = evaluate(loader=loaders["validation"],
                                    model=model,
                                    metric_collection=metric_collection_eval,
                                    loss_fn=loss_fn,
                                    device=device,
                                    prob_norm=prob_norm
                                    )
            validation_metrics = {f"validation_{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in validation_metrics.items()}
            

            # Log metrics
            if is_master:
                logger.info("Validation metrics:\n{}".format(validation_metrics))
                if args.use_wandb:
                    wandb.log(validation_metrics, step=global_step)

            # Save model checkpoint
            if (is_master 
                & (args.always_save_checkpoint or validation_metrics["validation_avg_loss"] < best_validation_loss)
                & (not args.no_checkpoints)
                ):
                if validation_metrics["validation_avg_loss"] < best_validation_loss:
                    checkpoint_path = os.path.join(args.output_dir,"checkpoints", f"{args.name}_best_checkpoint_{timestamp}.pt")
                elif args.always_save_checkpoint:
                    checkpoint_path = os.path.join(args.output_dir,"checkpoints", f"{args.name}_checkpoint_epoch_{epoch}_{timestamp}.pt")

                save_checkpoint(model = get_model(model),
                                optimizer = optimizer,
                                epoch = epoch,
                                validation_metrics = validation_metrics,
                                train_metrics = train_metrics,
                                model_path = checkpoint_path
                                )
                best_validation_loss = validation_metrics["validation_avg_loss"]
    
    ####### CLEANUP #######

    # W&B, MLFlow amd optional metric results saving
    if is_master:
        logger.info(f"\n{'='*100}\nTraining  COMPLETE\n{'='*100}\n")
        # Optionally save val/test results in json
        # if args.save_metric_collection_eval:
        #     metrics_results.append(run_metrics)
        #     write_json(metrics_results, args.save_metric_collection_eval_file)

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
    if args.world_size > 1:
        dist.destroy_process_group()

    # torch.cuda.memory._dump_snapshot('memory_snapshot_v2.pickle')
if __name__ == "__main__":
    main()