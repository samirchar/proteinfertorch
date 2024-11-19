from proteinfertorch.proteinfer import ProteInfer
from proteinfertorch.utils import read_yaml
from proteinfertorch.config import ACTIVATION_MAP
import torch
import os
import argparse

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

parser = argparse.ArgumentParser(description="Inference with ProteInfer model.",parents=[parser_first])
parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="The directory containing all tf pkl weights",
    )

parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The directory to save the pytorch weights",
    )

args = parser.parse_args()

#Process all weights in tf_weights
model_weights = os.listdir(args.input_dir)

for model_weight in model_weights:
    if "go" in model_weight and "random" in model_weight: #TODO: remove this. This should work for all tasks and data splits
        task,data_split,model_id = model_weight.split("_")

        task_defaults = config[f'{task}_defaults']
        num_labels = task_defaults["output_dim"]

        model = ProteInfer.from_tf_pretrained(
            weights_path=os.path.join(args.input_dir,model_weight),
            num_labels=num_labels,
            input_channels=task_defaults["input_dim"],
            output_channels=task_defaults["output_embedding_dim"],
            kernel_size=task_defaults["kernel_size"],
            activation=ACTIVATION_MAP[task_defaults["activation"]],
            dilation_base=task_defaults["dilation_base"],
            num_resnet_blocks=task_defaults["num_resnet_blocks"],
            bottleneck_factor=task_defaults["bottleneck_factor"],
        )
        model.save_pretrained(os.path.join(args.output_dir,f"{task}_{data_split}_{model_id}"))