import torch
from typing import List, Tuple


def collate_variable_sequence_length(
    batch: List[Tuple],
    return_label_multihots=True,
    world_size=1,
    rank=0,
):
    """
    Collates a batch of data with variable sequence lengths. Pads sequences to the maximum length within the batch to handle the variable
    lengths.



    """

    # Determine the maximum sequence length in the batch
    max_length = max(item["sequence_length"] for item in batch)

    # Initialize lists to store the processed values
    processed_sequence_onehots = []
    processed_sequence_ids = []
    processed_sequence_lengths = []
    processed_label_multihots = []
    processed_label_token_counts = []
    processed_label_embeddings = None

    sampled_label_indices = None
    num_labels = batch[0]["label_multihots"].shape[0]

    # Loop through the batch
    for row in batch:
        # Get the sequence onehots, sequence length, sequence id, and label multihots
        sequence_onehots = row["sequence_onehots"]
        sequence_id = row["sequence_id"]
        sequence_length = row["sequence_length"]
        label_multihots = row["label_multihots"]

        # Set padding
        padding_length = max_length - sequence_length

        # Get the sequence dimension (e.g., 20 for amino acids)
        sequence_dim = sequence_onehots.shape[0]

        # Pad the sequence to the max_length and append to the processed_sequences list
        processed_sequence_onehots.append(
            torch.cat(
                (sequence_onehots, torch.zeros((sequence_dim, padding_length))), dim=1
            )
        )

        # Use the sampled labels for each element in the batch.
        if sampled_label_indices is not None:
            label_multihots = label_multihots[sampled_label_indices]

        # Append the other values to the processed lists
        processed_sequence_ids.append(sequence_id)
        processed_sequence_lengths.append(sequence_length)
        processed_label_multihots.append(label_multihots)

    processed_batch = {
        "sequence_onehots": torch.stack(processed_sequence_onehots),
        "sequence_ids": processed_sequence_ids,
        "sequence_lengths": torch.stack(processed_sequence_lengths),
        "label_embeddings": processed_label_embeddings,
        "label_token_counts": label_token_counts,
    }

    if return_label_multihots:
        processed_batch["label_multihots"] = torch.stack(processed_label_multihots)

    return processed_batch
