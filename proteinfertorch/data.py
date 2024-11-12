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
from proteinfertorch.collators import collate_variable_sequence_length
from proteinfertorch.utils import read_fasta, get_vocab_mappings, generate_vocabularies
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
    Dataset class for protein sequences with multilabel annotations.
    """

    def __init__(
        self,
        data_path: str,
        vocabulary_path: Optional[str] = None,
        deduplicate:bool = False,
        max_sequence_length:int = float("inf"),
        logger=None
        ):
        """

        """
        self.logger = logger
        self.data_path = data_path
        self.vocabulary_path = vocabulary_path
        self.deduplicate = deduplicate
        self.max_sequence_length = max_sequence_length
        

        self.vocabulary_path = (
            self.vocabulary_path
            if self.vocabulary_path is not None
            else self.data_path
        )
        self._preprocess_data(
            vocabulary_path=vocabulary_path,
        )

    def _preprocess_data(self, vocabulary_path: str):

        vocabularies = generate_vocabularies(data=vocabulary_path)

        self.amino_acid_vocabulary = vocabularies["amino_acid_vocab"]
        self.label_vocabulary = vocabularies["label_vocab"]
        self.sequence_id_vocabulary = vocabularies["sequence_id_vocab"]
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
    dataset_specs: list,
    num_workers: int = 2,
    pin_memory: bool = True,
    world_size: int = 1,
    rank: int = 0
) -> List[DataLoader]:
    loaders = defaultdict(list)
    for dataset_spec in dataset_specs.items():

        sequence_sampler = observation_sampler_factory(
            dataset=dataset_spec['dataset'],
            world_size=world_size,
            rank=rank,
            shuffle=dataset_spec['shuffle']
        )

        loader = DataLoader(
            dataset_spec['dataset'],
            batch_size=dataset_spec['batch_size'],
            shuffle=False,
            collate_fn=partial(
                collate_variable_sequence_length,
                world_size=world_size,
                rank=rank,
            ),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=dataset_spec['drop_last'],
            sampler=sequence_sampler
        )
        loaders[dataset_spec["name"]].append(loader)

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
