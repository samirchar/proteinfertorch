# ProteInferTorch

## Description

Unofficial PyTorch version of ProteInfer (https://github.com/google-research/proteinfer), originally implemented in TensorFlow 1.X. 

ProteInfer is a model for protein function prediction that is trained to predict the functional properties of protein sequences using Deep Learning. Authors provide pre-trained models for two tasks: Gene Ontology (GO) and Enzyme Commission (EC) number prediction, as well as two data splits two data splits: random and clustered. Additionally, for every task and data split combination, authors trained multiple models using different random seeds. 

This repo contains PyTorch code to run inference, train, or extract embeddings for four ProteInfer models - one for each task/data split combination. All model weights are hosted in [Hugging Face 🤗](https://huggingface.co/samirchar).

The table below summarizes ProteInferTorch's performance on the original ProteInfer test sets using the Pytorch converted weights:

| Data Split | Task  | ProteInfer  | ProteInferTorch | Weights 🤗 |
|------------|-------|-------------|-----------------|------------|
| random     | GO    | 0.885       |      0.886      |  [Link](https://huggingface.co/samirchar/proteinfertorch-go-random-13731645)       |
| clustered  | GO    | Not Reported|      0.784      |  [Link](https://huggingface.co/samirchar/proteinfertorch-go-clustered-13703731)       |
| random     | EC    | 0.977       |      0.979      |  [Link](https://huggingface.co/samirchar/proteinfertorch-ec-random-13685140)       |
| clustered  | EC    | 0.914       |      0.914      |  [Link](https://huggingface.co/samirchar/proteinfertorch-ec-clustered-13704042)       |


TODO: ProteInferTorch's performance when training from scratch (i.e., random weights)

## Table Of Contents
<!-- toc -->

- [Installation](#installation)
- [Config](#config)
- [Data](#data)
- [Inference](#inference)
- [Extract Embeddings](#extract-embeddings)
- [Train](#train)
- [Citation](#citation)
- [Additional scripts](#additional-scripts)
  * [Create datasets](#create-datasets)
  * [Extract TF weights](#extract-tf-weights)

<!-- tocstop -->

## Installation
```
git clone https://github.com/samirchar/proteinfertorch
cd proteinfertorch
conda env create -f environment.yml
conda activate proteinfertorch
pip install -e ./  # make sure ./ is the dir including setup.py
```

## Config
All default hyperparameters and default arugments for the scripts are stored in `config/config.yaml`. 

## Data

All tge data to train and run inference with ProteInferTorch is available in the data.zip file (XXGB) hosted in Zenodo using the following command *from the ProteInferTorch root folder*

```
sudo apt-get install unzip
curl -O https://zenodo.org/records/13897920/files/data.zip?download=1
unzip data.zip
```

The data folder has the following structure:
* **data/**
    * **random_split/**: contains the train, dev, test fasta files for all tasks using the random split method
    * **clustered_split/**: contains the train, dev, test fasta files for all tasks using the clustered split method
    * **parenthood/**: holds a JSON with the EC and GO graphs, used by ProteInfer to normalize output probabilities.

## Input data format
This package uses the standard [FASTA](https://en.wikipedia.org/wiki/FASTA_format) format, which is a standard in bioinformatics. All scripts have an optional arugment called --fasta-separator that defaults to " " and represents how elements of header (i.e., sequence id and labels, if any) are separated.

## Inference
To run inference simply run and evaluate model performance run:

```
python bin/inference.py --data-path data/random_split/test_GO.fasta --vocabulary-path data/random_split/vocabularies/full_GO.json --weights-dir samirchar/proteinfertorch-go-random-13731645
```

To save the prediction logits, probabilities and labels, add the flag --save-prediction-results


## Extract Embeddings
Users can extract and save ProteInferTorch embeddings using the get_embeddings.py script. The embeddings will be stored in one or more .pt files inside of --output-dir, depending on the number of --num-embedding-partitions specified.

```
python bin/get_embeddings.py --data-path data/random_split/test_GO.fasta --weights-dir samirchar/proteinfertorch-go-random-13731645
```
By default, --num-embedding-partitions=10.

## Train
The model can be trained from scratch or from pretrained weights depending on the value of the --weights-dir argument.

To train from scratch run:
```
python bin/train.py --train-data-path data/random_split/train_GO.fasta --validation-data-path data/random_split/dev_GO.fasta --test-data-path data/random_split/test_GO.fasta --vocabulary-path data/random_split/vocabularies/full_GO.json
```

To start from pretrained weights:
```
python bin/train.py --train-data-path data/random_split/train_GO.fasta --validation-data-path data/random_split/dev_GO.fasta --test-data-path data/random_split/test_GO.fasta --vocabulary-path data/random_split/full_GO.fasta --weights-dir samirchar/proteinfertorch-go-random-13731645 
```

## Citation

If you use this model in your work, I would greatly appreciate it if you could cite it as follows:

```bibtex
@misc{yourname2024pytorchmodel,
  title={ProteInferTorch: a PyTorch implementation of ProteInfer},
  version={v1.0.0},
  author={Samir Char},
  year={2024},
  month={12},
  day={08},
  doi={10.5281/zenodo.1234567},
  url={https://github.com/samirchar/proteinfertorch}
}
```

## Additional scripts
This section describes additional scripts available in the bin folder

### Create datasets

The following code create train, dev and test FASTA files for both tasks and data splits from the original datasets in tfrecord format.

```
conda env create -f proteinfer_conda_requirements.yml
conda activate proteinfer
python bin/make_proteinfer_dataset.py --data-input-dir data/clustered_split/tfrecords/ --data-output-dir data/clustered_split/ --vocab-output-dir data/clustered_split/vocabularies/ --annotation-types GO
python bin/make_proteinfer_dataset.py --data-input-dir data/clustered_split/tfrecords/ --data-output-dir data/clustered_split/ --vocab-output-dir data/clustered_split/vocabularies/ --annotation-types EC
python bin/make_proteinfer_dataset.py --data-input-dir data/random_split/tfrecords/ --data-output-dir data/random_split/ --vocab-output-dir data/random_split/vocabularies/ --annotation-types GO
python bin/make_proteinfer_dataset.py --data-input-dir data/random_split/tfrecords/ --data-output-dir data/random_split/ --vocab-output-dir data/random_split/vocabularies/ --annotation-types EC
conda activate proteinfertorch
```

### Extract TF weights

Use the following code to download the original tensorflow weights for the two tasks and data splits, and convert them pkl format:

```
python bin/download_proteinfer_weights.py --task go --data-split clustered --ids 13703731 --output-dir data/model_weights/tf_weights/
python bin/download_proteinfer_weights.py --task go --data-split random --ids 13731645 --output-dir data/model_weights/tf_weights/
python bin/download_proteinfer_weights.py --task ec --data-split clustered --ids 13704042 --output-dir data/model_weights/tf_weights/
python bin/download_proteinfer_weights.py --task ec --data-split random --ids 13685140 --output-dir data/model_weights/tf_weights/
```


