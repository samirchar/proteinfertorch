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
from proteinfertorch.config import get_logger
from Bio.SeqRecord import SeqRecord

EC_RANDOM_ENSEMBLE_ELEMENT_EXPERIMENT_IDS = [
    '13703966', '13704083', '13704104', '13704130', '13705280', '13705675',
    '13705786', '13705802', '13705819', '13705839', '13706239', '13706986',
    '13707020', '13707589', '13707925', '13708369', '13708672', '13708706',
    '13708740', '13708951', '13709242', '13709584', '13709983', '13710037',
    '13711670', '13729344', '13730041', '13730097', '13730679', '13730876',
    '13730909', '13731218', '13731588', '13731728', '13731976',
]

GO_RANDOM_ENSEMBLE_ELEMENT_EXPERIMENT_IDS = [
    '13703706', '13703742', '13703997', '13704131', '13705631', '13705668',
    '13705677', '13705689', '13705708', '13705728', '13706170', '13706215',
    '13707414', '13707438', '13707732', '13708169', '13708676', '13708925',
    '13708995', '13709052', '13709428', '13709589', '13710370', '13710418',
    '13711677', '13729352', '13730011', '13730387', '13730746', '13730766',
    '13730958', '13731179', '13731598', '13731645', '13732022',
]

EC_CLUSTERED_ENSEMBLE_ELEMENT_EXPERIMENT_IDS = [
    
]

GO_CLUSTERED_ENSEMBLE_ELEMENT_EXPERIMENT_IDS = [

]


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


def read_fasta(data_path: str, sep=" "):
    """
    Reads a FASTA file and returns a list of tuples containing sequences, ids, and labels.
    """
    sequences_with_ids_and_labels = []

    for record in SeqIO.parse(data_path, "fasta"):
        sequence = str(record.seq)
        components = record.description.split(sep)
        # labels[0] contains the sequence ID, and the rest of the labels are GO terms.
        sequence_id = components[0]
        labels = components[1:]

        # Return a tuple of sequence, sequence_id, and labels
        sequences_with_ids_and_labels.append((sequence, sequence_id, labels))
    return sequences_with_ids_and_labels



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


def generate_vocabularies(file_path: str = None, data: list = None) -> dict:
    """
    Generate vocabularies based on the provided data path.
    path must be .fasta file
    """
    if not ((file_path is None) ^ (data is None)):
        raise ValueError("Only one of file_path OR data must be passed, not both.")
    vocabs = {
        "amino_acid_vocab": set(),
        "label_vocab": set(),
        "sequence_id_vocab": set(),
    }
    if file_path is not None:
        if isinstance(file_path, str):
            data = read_fasta(file_path)
        else:
            raise TypeError(
                "File not supported, vocabularies can only be generated from .fasta files."
            )

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
    labels_df = pd.DataFrame(
        results["labels"], columns=label_vocabulary, index=results["sequence_ids"]
    )
    labels_df_output_path = os.path.join(
        output_dir, f"{data_split_name}_labels_{run_name}.h5"
    )
    logger.info(f"saving results to {labels_df_output_path}")
    labels_df.to_hdf(labels_df_output_path, key="labels_df", mode="w")

    #saving the logits df
    logits_df = pd.DataFrame(
        results["logits"], columns=logits_df_cols, index=results["sequence_ids"]
    )
    logits_df_output_path = os.path.join(
        output_dir, f"{data_split_name}_logits_{run_name}.h5"
    )
    logger.info(f"saving results to {logits_df_output_path}")
    logits_df.to_hdf(logits_df_output_path, key="logits_df", mode="w")


    #Saving the probabilities df
    probabilities_df = pd.DataFrame(
        results["probabilities"], columns=logits_df_cols, index=results["sequence_ids"]
    )
    probabilities_df_output_path = os.path.join(
        output_dir, f"{data_split_name}_probabilities_{run_name}.h5"
    )
    logger.info(f"saving results to {probabilities_df_output_path}")
    probabilities_df.to_hdf(probabilities_df_output_path, key="probabilities_df", mode="w")