import numpy as np
import collections
import torch
import pickle
from Bio import SeqIO
from Bio.Seq import Seq
import json
import yaml
import os
import pandas as pd
import random
import transformers
from proteinfertorch.config import get_logger
from Bio.SeqRecord import SeqRecord



EC_CLUSTERED_ENSEMBLE_ELEMENT_EXPERIMENT_IDS= ['13704042',
              '13704073',
              '13704099',
              '13704118',
              '13705302',
              '13705319',
              '13705355',
              '13705673',
              '13705807',
              '13706269',
              '13706312',
              '13706344',
              '13707448',
              '13707675',
              '13707943',
              '13708185',
              '13708683',
              '13708895',
              '13709160',
              '13709369',
              '13709407',
              '13709643',
              '13710003',
              '13711429',
              '13729732',
              '13730025',
              '13730058',
              '13730450',
              '13730829',
              '13730925',
              '13731238',
              '13731619',
              '13731744',
              '13731990']
EC_RANDOM_ENSEMBLE_ELEMENT_EXPERIMENT_IDS=['13685140']

# EC_RANDOM_ENSEMBLE_ELEMENT_EXPERIMENT_IDS = ['13703966',
#               '13704083',
#               '13704104',
#               '13704130',
#               '13705280',
#               '13705675',
#               '13705786',
#               '13705802',
#               '13705819',
#               '13705839',
#               '13706239',
#               '13706986',
#               '13707020',
#               '13707589',
#               '13707925',
#               '13708369',
#               '13708672',
#               '13708706',
#               '13708740',
#               '13708951',
#               '13709242',
#               '13709584',
#               '13709983',
#               '13710037',
#               '13711670',
#               '13729344',
#               '13730041',
#               '13730097',
#               '13730679',
#               '13730876',
#               '13730909',
#               '13731218',
#               '13731588',
#               '13731728',
#               '13731976']
GO_CLUSTERED_ENSEMBLE_ELEMENT_EXPERIMENT_IDS = ['13703731',
              '13704047',
              '13704079',
              '13704124',
              '13705320',
              '13705684',
              '13705720',
              '13705737',
              '13705762',
              '13706275',
              '13707011',
              '13707046',
              '13707576',
              '13707695',
              '13707754',
              '13708196',
              '13708690',
              '13708901',
              '13709163',
              '13709187',
              '13709225',
              '13709647',
              '13710390',
              '13710438',
              '13729994',
              '13730036',
              '13730664',
              '13730802',
              '13730888',
              '13731470',
              '13731575',
              '13731970',
              '13732043',
              '13732068']
GO_RANDOM_ENSEMBLE_ELEMENT_EXPERIMENT_IDS = ['13703706',
              '13703742',
              '13703997',
              '13704131',
              '13705631',
              '13705668',
              '13705677',
              '13705689',
              '13705708',
              '13705728',
              '13706170',
              '13706215',
              '13707414',
              '13707438',
              '13707732',
              '13708169',
              '13708676',
              '13708925',
              '13708995',
              '13709052',
              '13709428',
              '13709589',
              '13710370',
              '13710418',
              '13711677',
              '13729352',
              '13730011',
              '13730387',
              '13730746',
              '13730766',
              '13730958',
              '13731179',
              '13731598',
              '13731645',
              '13732022']

def read_yaml(data_path: str):
    with open(data_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def read_json(data_path: str):
    with open(data_path, "r") as file:
        data = json.load(file)
    return data


def write_json(data, data_path: str):
    with open(data_path, "w") as file:
        json.dump(data, file)


def read_pickle(file_path: str):
    with open(file_path, "rb") as p:
        item = pickle.load(p)
    return item

def read_fasta(data_path: str,
               sep: str =" ",
               ignore_labels = False
               ):
    """
    Reads a FASTA file and returns a list of tuples containing sequences, ids, and labels.
    """

    sequences_with_ids_and_labels = []
    for record in SeqIO.parse(data_path, "fasta"):
        sequence = str(record.seq)
        sequence_id = record.id


        # always return dummy labels unless we are not ignoring the labels and the labels are present
        labels = []
        has_labels = False
        
        # labels[0] contains the sequence ID, and the rest of the labels are GO terms.
        temp = record.description.split(sep)[1:] 
        has_labels = len(temp) > 0

        if has_labels and not ignore_labels:
            labels = temp

        # Return a tuple of sequence, sequence_id, and labels
        sequences_with_ids_and_labels.append((sequence, sequence_id, labels))

    return sequences_with_ids_and_labels, has_labels


def save_to_pickle(item, file_path: str):
    with open(file_path, "wb") as p:
        pickle.dump(item, p)


def save_to_fasta(sequence_id_labels_tuples, output_file):
    """
    Save a list of tuples in the form (sequence, [labels]) to a FASTA file.

    :param sequence_label_tuples: List of tuples containing sequences and labels
    :param output_file: Path to the output FASTA file
    """
    records = []
    for _, (
        sequence,
        id,
        labels,
    ) in enumerate(sequence_id_labels_tuples):
        # Create a description from labels, joined by space
        description = " ".join(labels)

        record = SeqRecord(Seq(sequence), id=id, description=description)
        records.append(record)

    # Write the SeqRecord objects to a FASTA file
    with open(output_file, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")
        print("Saved FASTA file to " + output_file)

def get_vocab_mappings(vocabulary):
    assert len(vocabulary) == len(set(vocabulary)), "items in vocabulary must be unique"
    term2int = {term: idx for idx, term in enumerate(vocabulary)}
    int2term = {idx: term for term, idx in term2int.items()}
    return term2int, int2term

def generate_vocabularies(data: list = None) -> dict:
    """
    Generate vocabularies based on the parsed fasta file using read_fasta.
    path must be .fasta file
    """

    vocabs = {
        "amino_acid_vocab": set(),
        "label_vocab": set(),
        "sequence_id_vocab": set(),
    }

    for sequence, sequence_id, labels in data:
        vocabs["sequence_id_vocab"].add(sequence_id)
        vocabs["label_vocab"].update(labels)
        vocabs["amino_acid_vocab"].update(list(sequence))

    for vocab_type in vocabs.keys():
        vocabs[vocab_type] = sorted(list(vocabs[vocab_type]))

    return vocabs

def transfer_tf_weights_to_torch(torch_model: torch.nn.Module, tf_weights_path: str):
    # Load tensorflow variables. Remove global step variable and add it as num_batches variable for each batchnorm
    tf_weights = read_pickle(tf_weights_path)
    # total training steps from the paper. Used for batch norm running statistics.
    num_batches = tf_weights["inferrer/global_step:0"]
    tf_weights.pop("inferrer/global_step:0")
    temp = {}
    for tf_name, tf_param in tf_weights.items():
        temp[tf_name] = tf_param
        if ("batch_normalization" in tf_name) & ("moving_variance" in tf_name):
            num_batches_name = "/".join(
                tf_name.split("/")[:-1] + ["num_batches_tracked:0"]
            )
            temp[num_batches_name] = np.array(num_batches)
    tf_weights = temp

    # Get pytorch model variables
    state_dict = torch_model.state_dict()
    state_dict_list = [(k, v) for k, v in state_dict.items()]

    with torch.no_grad():
        for (name, param), (tf_name, tf_param) in zip(
            state_dict_list, tf_weights.items()
        ):
            if tf_param.ndim >= 2:
                tf_param = np.transpose(
                    tf_param, tuple(sorted(range(tf_param.ndim), reverse=True))
                )

            assert (
                tf_param.shape == param.detach().numpy().shape
            ), f"{name} and {tf_name} don't have the same shape"
            state_dict[name] = torch.from_numpy(tf_param)

    torch_model.load_state_dict(state_dict)


def reverse_map(applicable_label_dict, label_vocab=None):
    """Flip parenthood dict to map parents to children.

    Args:
      applicable_label_dict: e.g. output of get_applicable_label_dict.
      label_vocab: e.g. output of inference_lib.vocab_from_model_base_path

    Returns:
      collections.defaultdict of k, v where:
      k: originally the values in applicable_label_dict
      v: originally the keys in applicable_label_dict.
      The defaultdict returns an empty frozenset for keys that are not found.
      This behavior is desirable for lifted clan label normalizers, where
      keys may not imply themselves.
    """
    # This is technically the entire transitive closure, so it is safe for DAGs
    # (e.g. GO labels).

    children = collections.defaultdict(set)
    for child, parents in applicable_label_dict.items():
        # Avoid adding children which don't appear in the vocab.
        if label_vocab is None or child in label_vocab:
            for parent in parents:
                children[parent].add(child)
    children = {k: frozenset(v) for k, v in children.items()}
    return collections.defaultdict(frozenset, children.items())


#TODO: this function could be reimplemented in pytorch
def normalize_confidences(predictions, label_vocab, applicable_label_dict):
    """Set confidences of parent labels to the max of their children.

    Args:
      predictions: [num_sequences, num_labels] ndarray.
      label_vocab: list of vocab strings in an order that corresponds to
        `predictions`.
      applicable_label_dict: Mapping from labels to their parents (including
        indirect parents).

    Returns:
      A numpy array [num_sequences, num_labels] with confidences where:
      if label_vocab[k] in applicable_label_dict[label_vocab[j]],
      then arr[i, j] >= arr[i, k] for all i.
    """
    vocab_indices = {v: i for i, v in enumerate(label_vocab)}
    children = reverse_map(applicable_label_dict, set(vocab_indices.keys()))

    # Only vectorize this along the sequences dimension as the number of children
    # varies between labels.
    label_confidences = []
    for label in label_vocab:
        child_indices = np.array([vocab_indices[child] for child in children[label]])
        if child_indices.size > 1:
            confidences = np.max(predictions[:, child_indices], axis=1)
            label_confidences.append(confidences)
        else:
            label_confidences.append(predictions[:, vocab_indices[label]])

    return np.stack(label_confidences, axis=1)

class probability_normalizer:
    def __init__(self, applicable_label_dict, label_vocab):
        self.label_vocab = label_vocab
        self.vocab_indices = {v: i for i, v in enumerate(label_vocab)}
        self.children = reverse_map(applicable_label_dict, set(self.vocab_indices.keys()))

    def __call__(self, predictions):
        # Only vectorize this along the sequences dimension as the number of children
        # varies between labels.
        label_confidences = []
        for label in self.label_vocab:
            child_indices = np.array([self.vocab_indices[child] for child in self.children[label]])
            if child_indices.size > 1:
                confidences = np.max(predictions[:, child_indices], axis=1)
                label_confidences.append(confidences)
            else:
                label_confidences.append(predictions[:, self.vocab_indices[label]])
        return np.stack(label_confidences, axis=1)



def to_device(device, *args):
    return [
        item.to(device) if isinstance(item, torch.Tensor) else None for item in args
    ]

def save_evaluation_results(
    results,
    label_vocabulary,
    run_name,
    output_dir,
    data_split_name,
    logger = None
):
    
    logger = logger if logger is not None else get_logger()

    # Do not need to check if is_master, since this function is only called by the master node
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logits_df_cols = label_vocabulary

    #Saving the labels_df
    if 'labels' in results:
        labels_df = pd.DataFrame(
            results["labels"], columns=label_vocabulary, index=results["sequence_ids"]
        )
        labels_df_output_path = os.path.join(
            output_dir, f"{data_split_name}_labels_{run_name}.h5"
        )
        logger.info(f"saving results to {labels_df_output_path}")
        labels_df.to_hdf(labels_df_output_path, key="labels_df", mode="w")

    #saving the logits df
    if 'logits' in results:
        logits_df = pd.DataFrame(
            results["logits"], columns=logits_df_cols, index=results["sequence_ids"]
        )
        logits_df_output_path = os.path.join(
            output_dir, f"{data_split_name}_logits_{run_name}.h5"
        )
        logger.info(f"saving results to {logits_df_output_path}")
        logits_df.to_hdf(logits_df_output_path, key="logits_df", mode="w")


    #Saving the probabilities df
    if 'probabilities' in results:
        probabilities_df = pd.DataFrame(
            results["probabilities"], columns=logits_df_cols, index=results["sequence_ids"]
        )
        probabilities_df_output_path = os.path.join(
            output_dir, f"{data_split_name}_probabilities_{run_name}.h5"
        )
        logger.info(f"saving results to {probabilities_df_output_path}")
        probabilities_df.to_hdf(probabilities_df_output_path, key="probabilities_df", mode="w")


def seed_everything(seed: int, device: str):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)


def save_checkpoint(model, optimizer, epoch, train_metrics, validation_metrics, model_path):
    """
    Save model and optimizer states as a checkpoint.

    Args:
    - model (torch.nn.Module): The model whose state we want to save.
    - optimizer (torch.optim.Optimizer): The optimizer whose state we want to save.
    - epoch (int): The current training epoch.
    - model_path (str): The path where the checkpoint will be saved.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": validation_metrics,
        "train_metrics": train_metrics
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    torch.save(checkpoint, model_path)


# function that returns model.module if model is a DistributedDataParallel object
# otherwise returns model
def get_model(model):
    return model.module if hasattr(model, "module") else model


def load_emeddings(dir:str, num_partitions:int = -1):
    partitions = os.listdir(dir)
    total_partitions = len(partitions)
    num_partitions = total_partitions if num_partitions == -1 else num_partitions
    embeddings = []
    for partition_idx in range(num_partitions):
        embeddings.append(torch.load(f"{dir}/{partitions[partition_idx]}"))

    return torch.cat(embeddings, dim=0)



HF_MODEL_CARD_TEMPLATE = '''
# Model Card for ProteInferTorch trained on {task} task with {data_split} data split

Unofficial PyTorch version of ProteInfer (https://github.com/google-research/proteinfer), originally implemented in TensorFlow 1.X. 

ProteInfer is a model for protein function prediction that is trained to predict the functional properties of protein sequences using Deep Learning.
Authors provide pre-trained models for two tasks: Gene Ontology (GO) and Enzyme Commission (EC) number prediction, as well as two data splits two data splits: random and clustered.
Additionally, for every task and data split combination, authors trained multiple models using different random seeds.

This model is trained on the **{task}** task with the **{data_split}** data split, and corresponds to the model with ID **{model_id}** in the original ProteInfer repository.

## Model Details

### Model Description

For all the details about the model, please refer to the original ProteInfer paper: https://elifesciences.org/articles/80942.

- **Developed by:** Samir Char, adapted from the original TensorFlow 1.X implementation by Google Research
- **Model type:** Dilated Convolutional Neural Network
- **License:** Apache

### Model Sources

- **Repository:** https://github.com/samirchar/proteinfertorch

## Uses

### Direct Use

This model is intended for research use. It can be used for protein function prediction tasks, such as Gene Ontology (GO) and Enzyme Commission (EC) number prediction, or
as a feature extractor for protein sequences.

### Downstream Use

This model can be fine-tuned for any task that can benefit from function-aware protein embeddings.

## Bias, Risks, and Limitations

- This model is intended for use on protein sequences. It is not meant for other biological sequences, such as DNA sequences.

## How to Get Started with the Model

```
git clone https://github.com/samirchar/proteinfertorch
cd proteinfertorch
conda env create -f environment.yml
conda activate proteinfertorch
pip install -e ./  # make sure ./ is the dir including setup.py
```

For detailed instructions on package usage, please refer to the README in model repo

## Evaluation

### Results

TODO: Add table comparing the performance of this model with the original TensorFlow 1.X implementation.


## Technical Specifications 

### Compute Infrastructure

8xV100 GPU cluster


## Citation

**BibTeX:**
If you use this model in your work, I would greatly appreciate it if you could cite it as follows:

```bibtex
@misc{{yourname2024pytorchmodel,
  title={{ProteInferTorch: a PyTorch implementation of ProteInfer}},
  version={{v1.0.0}},
  author={{Samir Char}},
  year={{2024}},
  month={{12}},
  day={{08}},
  doi={{10.5281/zenodo.14514368}},
  url={{https://github.com/samirchar/proteinfertorch}}
}}
```


## Model Card Authors

Samir Char

## Model Card Contact

Samir Char
'''
