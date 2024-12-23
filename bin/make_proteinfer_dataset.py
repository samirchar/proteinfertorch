import os
import logging
from torchdata.datapipes.iter import FileLister, FileOpener
import argparse
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import json
from tqdm import tqdm

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

def process_sequence_tfrecord(record: dict, annotation_types: list):
    sequence = record["sequence"][0].decode()
    id = record["id"][0].decode()

    labels = set()

    # Some rows have no lavel column
    if "label" not in record:
        return None

    # Add all labels from desired annotation types
    for l in record["label"]:
        label = l.decode()
        label_type = label.split(":")[0]

        if label_type in annotation_types:
            labels.add(label)

    # Sequence with no annotation from selected types
    if not labels:
        return None

    return id, (sequence, list(labels))


def process_tfrecords(
    data_input_dir: str,
    data_output_dir: str,
    vocab_output_dir: str,
    annotation_types: list,
    pattern: str,
    pattern_name: str,
):
    # Load all tfrecords from desired data split
    datapipe1 = FileLister(str(data_input_dir), pattern)
    datapipe2 = FileOpener(datapipe1, mode="b")
    tfrecord_loader_dp = datapipe2.load_from_tfrecord()

    # Create paths
    data_output_path = os.path.join(data_output_dir , f"{pattern_name}_{'_'.join(annotation_types)}.fasta")
    vocab_output_path = os.path.join(vocab_output_dir, f"{pattern_name}_{'_'.join(annotation_types)}.json")

    records = []
    records_raw = []
    # Iterate over records, process and write to a fasta file
    for _, record in tqdm(enumerate(tfrecord_loader_dp)):
        processed_sequence = process_sequence_tfrecord(record, annotation_types)

        # Skipping sequence with no labels from desired annotations
        if processed_sequence is None:
            continue

        id, (sequence, labels) = processed_sequence

        description = " ".join(labels)
        record = SeqRecord(Seq(sequence), id=f"{id}", description=description)
        records.append(record)
        records_raw.append((sequence, id, labels))

    with open(
        data_output_path,
        "w",
    ) as output_handle:
        SeqIO.write(records, output_handle, "fasta")

    if pattern_name == "full":
        vocabulary = generate_vocabularies(data = records_raw)

        os.makedirs(vocab_output_dir, exist_ok=True)
        with open(vocab_output_path, "w") as file:
            json.dump(vocabulary, file)

if __name__ == "__main__":
    """
    Example usage: python bin/make_proteinfer_dataset.py --data-input-dir data/clustered_split/ --annotation-types GO
    """
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s: %(message)s", level=logging.NOTSET
    )
    parser = argparse.ArgumentParser()

    
    parser.add_argument(
        "--data-input-dir",
        required=True,
        help="Path to the directory containing the tfrecords. For proteinfer = random_split or clustered_split"
    )

    parser.add_argument(
        "--data-output-dir",
        required=True,
        help="Path to the directory to save the processed data"
    )

    parser.add_argument(
        "--vocab-output-dir",
        required=True,
        help="Path to the directory to save the vocabulary"
    )

    parser.add_argument(
        "--annotation-types",
        nargs="+",
        required=True
    )
    
    args = parser.parse_args()

    dirname = os.path.dirname(__file__)

    patterns = {
        "train": "train*.tfrecord",
        "dev": "dev*.tfrecord",
        "test": "test*.tfrecord",
        "full": "*.tfrecord",
    }

    for pattern_name, pattern in patterns.items():
        logging.info(f"Processing {pattern_name}")
        process_tfrecords(
            data_input_dir=args.data_input_dir,
            data_output_dir=args.data_output_dir,
            vocab_output_dir=args.vocab_output_dir,
            annotation_types=args.annotation_types,
            pattern=pattern,
            pattern_name=pattern_name,
        )
