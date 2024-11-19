import os
import subprocess
import argparse
from pathlib import Path

# Define the list of IDs
def run_command(command):
    """Utility function to run a shell command"""
    result = subprocess.run(command, shell=True, check=True, text=True)
    return result

def download_and_extract_model(task:str,data_split:str,id:int,output_dir:str):
    """Download, extract, and export the model if it doesn't exist"""
    output_dir = Path(output_dir)
    model_file_name = f"{task}_{data_split}_{id}"
    model_file = output_dir / Path(str(model_file_name)+".pkl")
    if not os.path.exists(model_file):
        noxtype = "noxpnd" if (task == "ec" and data_split == "random") else "noxpd2"
        tar_file = f"{noxtype}_cnn_swissprot_{task}_{data_split}_swiss-cnn_for_swissprot_{task}_{data_split}-{id}.tar.gz"
        download_url = f"https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/models/zipped_models/{tar_file}"
        
        # Download the model
        run_command(f"wget {download_url}")
        
        # Extract the tar file
        run_command(f"tar -xvzf {tar_file}")
        
        # Move extracted model to cached models directory
        model_id_dir = tar_file.replace(".tar.gz", "")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        run_command(f"mv -f {model_id_dir} {output_dir}")
        
        # Run the export script
        run_command(f"conda run -n proteinfer python bin/export_proteinfer.py --model-path {output_dir / model_id_dir} --output-dir {output_dir} --model-name {model_file_name}")

        #Clean up. Orginal weights are not needed.
        
        run_command(f"rm -rf {output_dir / model_id_dir}")
    
    else:
        print(f"{id} weights already exist")

def main(task:str, data_split:str, ids:list, output_dir:str):

    # Loop over the IDs and handle the models
    for id in ids:
        download_and_extract_model(task = task, data_split = data_split, id = id, output_dir = output_dir)

if __name__ == "__main__":


    """
    Sample usage: python bin/download_proteinfer_weights.py --task ec --data-split clustered --ids 13704042 13704073 13704099 --output-dir data/model_weights/
    """
    parser = argparse.ArgumentParser(description="Run protein inference on multiple models")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task for the model. Either go or ec"
    )

    parser.add_argument(
        "--data-split",
        type=str,
        required=True,
        help="Data split for the model. Either random or clustered"
    )
    
    parser.add_argument(
        "--ids",
        nargs="+",
        required=True,
        help="List of proteinfer model ids",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The directory to store the new ProteInfer weights",
    )


    args = parser.parse_args()
    main(task = args.task, ids = args.ids, output_dir=args.output_dir, data_split=args.data_split)
