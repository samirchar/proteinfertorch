from proteinfertorch.proteinfer import ProteInfer
from proteinfertorch.utils import read_yaml, HF_MODEL_CARD_TEMPLATE
from proteinfertorch.config import ACTIVATION_MAP
import re
import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import login, ModelCard
'''

This script converts the weights of a ProteInfer model from TensorFlow to PyTorch, and optionally pushes the models to the Huggingface Hub.

Example usage: python bin/convert_tf_weights_to_pytorch.py --input-dir data/model_weights/tf_weights --output-dir data/model_weights/ --push-to-hub --hf-username <username>

'''
load_dotenv()
login(token=os.getenv("HF_TOKEN"))
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
        help="Directory to save pytorch weights."
    )

parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the models to the huggingface hub"
    )

parser.add_argument(
    "--hf-username",
    type=str,
    default=None,
    help="Huggingface username"
)

parser.add_argument(
    "--regex-pattern",
    type=str,
    default="(go|ec)-(random|clustered)-[0-9]+",
    help="Regex pattern to filter models"
)

args = parser.parse_args()

assert not (args.push_to_hub ^ (args.hf_username is not None)), "Please provide both hf_username and push_to_hub or neither"


### Upload Models to Huggingface Hub ###
#Process all weights in tf_weights
model_weights = os.listdir(args.input_dir)
base_architecture = config['base_architecture']
p = re.compile(f"{args.regex_pattern}")

for model_weight in model_weights:
    if p.match(model_weight):
        task,data_split,model_id= model_weight.split("-")
        model_id = model_id.split(".")[0]

        task_defaults = config[f'{task}_{data_split}_defaults']
        num_labels = task_defaults["output_dim"]

        model = ProteInfer.from_tf_pretrained(
            weights_path=os.path.join(args.input_dir,model_weight),
            num_labels=num_labels,
            input_channels=base_architecture["input_dim"],
            output_channels=base_architecture["output_embedding_dim"],
            kernel_size=base_architecture["kernel_size"],
            activation=ACTIVATION_MAP[base_architecture["activation"]],
            dilation_base=base_architecture["dilation_base"],
            num_resnet_blocks=base_architecture["num_resnet_blocks"],
            bottleneck_factor=base_architecture["bottleneck_factor"],
        )
        model_name = f"{task}-{data_split}-{model_id}"
        model.save_pretrained(os.path.join(args.output_dir,f"{model_name}")) 
        
        if args.push_to_hub:
            card = ModelCard(
                HF_MODEL_CARD_TEMPLATE.format(
                    model_name=model_name.upper(),
                    task=task.upper(),
                    data_split=data_split.upper(),
                    model_id=model_id
                )
            )
            model.push_to_hub(
                repo_id = f"{args.hf_username}/proteinfertorch-{model_name}",
                private=True
            )
            card.push_to_hub(
                repo_id = f"{args.hf_username}/proteinfertorch-{model_name}"
            )


    