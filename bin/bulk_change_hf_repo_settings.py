from proteinfertorch.proteinfer import ProteInfer
from proteinfertorch.utils import read_yaml
from proteinfertorch.config import ACTIVATION_MAP
import torch
import os
import re
import argparse
from dotenv import load_dotenv
from huggingface_hub import login, list_models,update_repo_visibility
'''

This script converts the weights of a ProteInfer model from TensorFlow to PyTorch, and optionally pushes the models to the Huggingface Hub.

Example usage: python bin/convert_tf_weights_to_pytorch.py --input-dir data/model_weights/tf_weights --output-dir data/model_weights/ --push-to-hub --hf-username <username>

'''
load_dotenv()
login(token=os.getenv("HF_TOKEN"))
# Arguments that must be parsed first

parser = argparse.ArgumentParser(description="Bulk updates for models in the Huggingface Hub.")

parser.add_argument(
    "--hf-username",
    type=str,
    default=None,
    help="Huggingface username"
)

parser.add_argument(
    "--regex-pattern",
    type=str,
    default="proteinfertorch-(go|ec)-(random|clustered)-[0-9]+",
    help="Regex pattern to filter models"
)

parser.add_argument(
    "--visibility",
    type=str,
    default=None,
    required=False,
    help="Visibility of the models"
)

args = parser.parse_args()
p = re.compile(f"{args.hf_username}/{args.regex_pattern}")
matching_model_ids = [model.id for model in list_models(author=args.hf_username) if p.match(model.id)]

for model_id in matching_model_ids:
    if args.visibility is not None:
        update_repo_visibility(model_id, private=args.visibility == "private")
        print(f"Updated visibility of {model_id} to {args.visibility}")
    