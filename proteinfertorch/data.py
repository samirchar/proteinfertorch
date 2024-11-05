import torch
import logging
import random
from collections import defaultdict
from joblib import Parallel, delayed, cpu_count
from functools import partial
from collections import Counter
from typing import List
import pandas as pd
import numpy as np
import blosum as bl
from torch.utils.data import Dataset, DataLoader
from protnote.data.collators import collate_variable_sequence_length
from protnote.utils.data import read_fasta, get_vocab_mappings
from protnote.utils.data import generate_vocabularies
from itertools import product
import numpy as np
from torch.utils.data import RandomSampler
from typing import Optional
import math
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import math
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    """
    Dataset class for protein sequences with GO annotations.
    """

    def __init__(
        self,
        logger=None
        ):
        """
        data_paths (dict): Dictionary containing paths to the data and vocabularies.
            data_path (str): Path to the FASTA file containing the protein sequences and corresponding GO annotations
            dataset_type (str): One of 'train', 'validation', or 'test'
            go_descriptions_path (str): Path to the pickled file containing the GO term descriptions mapped to GO term IDs
        deduplicate (bool): Whether to remove duplicate sequences (default: False)
        """
        self.logger = logger

        # Subset the data if subset_fraction is provided
        subset_fraction = config["params"][
            f"{self.dataset_type.upper()}_SUBSET_FRACTION"
        ]
        if subset_fraction < 1.0:
            logging.info(
                f"Subsetting {subset_fraction*100}% of the {self.dataset_type} set..."
            )
            self.data = self.data[: int(subset_fraction * len(self.data))]

        extract_vocabularies_from = config["params"]["EXTRACT_VOCABULARIES_FROM"]
        vocabulary_path = (
            config["paths"][extract_vocabularies_from]
            if extract_vocabularies_from is not None
            else self.data_path
        )
        self._preprocess_data(
            deduplicate=config["params"]["DEDUPLICATE"],
            max_sequence_length=config["params"]["MAX_SEQUENCE_LENGTH"],
            vocabulary_path=vocabulary_path,
        )

    def _preprocess_data(self, deduplicate, max_sequence_length, vocabulary_path):

        vocabularies = generate_vocabularies(data=self.data)

        self.amino_acid_vocabulary = vocabularies["amino_acid_vocab"]
        self.label_vocabulary = vocabularies["label_vocab"]
        self.sequence_id_vocabulary = vocabularies["sequence_id_vocab"]

        # Save mask of represented vocab
        self.represented_vocabulary_mask = [
            label in self.label_frequency for label in self.label_vocabulary
        ]

        self._process_vocab()

    # Helper functions for processing and loading vocabularies
    def _process_vocab(self):
        self._process_amino_acid_vocab()
        self._process_label_vocab()
        self._process_sequence_id_vocab()

    def _process_amino_acid_vocab(self):
        self.aminoacid2int, self.int2aminoacid = get_vocab_mappings(
            self.amino_acid_vocabulary
        )

    def _process_label_vocab(self):
        self.label2int, self.int2label = get_vocab_mappings(self.label_vocabulary)

    def _process_sequence_id_vocab(self):
        self.sequence_id2int, self.int2sequence_id = get_vocab_mappings(
            self.sequence_id_vocabulary
        )

    def __len__(self) -> int:
        return len(self.data)

    def process_example(
        self,
        sequence: str,
        sequence_id_alphanumeric: str,
        labels: list[str],
    ) -> dict:
        # One-hot encode the labels for use in the loss function (not a model input, so should not be impacted by augmentation)
        labels_ints = torch.tensor(
            [self.label2int[label] for label in labels], dtype=torch.long
        )

        # Convert the sequence and labels to integers for one-hot encoding (impacted by augmentation)
        amino_acid_ints = torch.tensor(
            [self.aminoacid2int[aa] for aa in sequence], dtype=torch.long
        )

        # Get the length of the sequence
        sequence_length = torch.tensor(len(amino_acid_ints))

        # Get multi-hot encoding of sequence and labels
        sequence_onehots = torch.nn.functional.one_hot(
            amino_acid_ints, num_classes=len(self.amino_acid_vocabulary)
        ).permute(1, 0)
        label_multihots = torch.nn.functional.one_hot(
            labels_ints, num_classes=len(self.label_vocabulary)
        ).sum(dim=0)

        # Return a dict containing the processed example
        # NOTE: In the collator, we will use the label token counts for only the first sequence in the batch
        return {
            "sequence_onehots": sequence_onehots,
            "sequence_id": sequence_id_alphanumeric,
            "sequence_length": sequence_length,
            "label_multihots": label_multihots
        }

    def __getitem__(self, idx) -> tuple:
        sequence, sequence_id, labels = self.data[idx]  
        return self.process_example(sequence, sequence_id, labels)

def create_multiple_loaders(
    datasets: list,
    num_workers: int = 2,
    pin_memory: bool = True,
    world_size: int = 1,
    rank: int = 0
) -> List[DataLoader]:
    loaders = defaultdict(list)
    for dataset_specs in datasets.items():
        batch_size_for_type = params[f"{dataset_type.upper()}_BATCH_SIZE"]

        for dataset in dataset_list:
            drop_last = True

            if dataset_type == "train":
                sequence_sampler = observation_sampler_factory(
                    dataset=dataset,
                    world_size=world_size,
                    rank=rank,
                    shuffle=True,
                )
            else:
                # Distributed Sampler without shuffling for validation or test
                sequence_sampler = observation_sampler_factory(
                    dataset=dataset,
                    world_size=world_size,
                    rank=rank,
                    shuffle=False,
                )
                drop_last = False

            loader = DataLoader(
                dataset,
                batch_size=batch_size_for_type,
                shuffle=False,
                collate_fn=partial(
                    collate_variable_sequence_length,
                    world_size=world_size,
                    rank=rank,
                ),
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                sampler=sequence_sampler
            )
            loaders[dataset_type].append(loader)

    return loaders


def observation_sampler_factory(
    shuffle: bool,
    dataset: Dataset = None,
    world_size: int = 1,
    rank: int = 0
    ):
    if world_size == 1:
        sampler = RandomSampler()
    else:
        assert dataset is not None, "DistributeSampler requires dataset"
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
    return sampler
