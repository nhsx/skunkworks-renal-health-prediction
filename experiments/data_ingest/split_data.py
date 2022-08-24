# Script to load a jsonl record dataset and then split the dataset out into
# randomly selected groups of train/test/validate.
#
# Input path:
#    Directory to ingest and add split dataset to.
#
# Output files (Saved to input path directory):
#    <date timestamp>_split_log.txt (log of console outputs for tracking)
#    data_splits.json (dictionary of each set and the master key used)
#    ingest_records_output_lines_train.jsonl
#    ingest_records_output_lines_test.jsonl
#    ingest_records_output_lines_validate.jsonl
#    ingest_records_output_lines_calib.jsonl
#
# Data splits: `train`(~80%), `valid` (~5%), `calib` (5%), and `test` (~10%).
import sys
import logging
from pathlib import Path
import time
import random

from aki_predictions.file_operations import load_jsonl, append_jsonl, save_dictionary_json
from aki_predictions.data_utils import split_data


def main(args):  # noqa: C901
    """Normalisation prototype."""
    # Load artifacts directory
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    artifacts_dir = Path(args[0])
    # debug_dir = artifacts_dir / "debug"

    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(artifacts_dir, f"{timestamp}_normalisation_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)

    # Load artifacts
    root_logger.info(f"Artifacts directory: {artifacts_dir}")

    # load data
    root_logger.info("Loading data file...")
    data = load_jsonl(artifacts_dir / "ingest_records_output_lines_normalised.jsonl")

    # Extract list of keys
    root_logger.info(f"Total records: {len(data)}")

    # Select split of keys
    # Set Seed
    random.seed(0)

    # Randomise data
    root_logger.info("Shuffling data...")
    shuffled_data = random.sample(data, len(data))

    root_logger.info("Dividing data...")
    train, test, val, calib = split_data(shuffled_data, splits=[0.8, 0.1, 0.05, 0.05])

    root_logger.info(f"Train: {len(train)}, Test: {len(test)}, Validate: {len(val)}, Calib: {len(calib)}")

    # Save out listing of the key split
    train_k = [record["master_key"] for record in train]
    test_k = [record["master_key"] for record in test]
    val_k = [record["master_key"] for record in val]
    calib_k = [record["master_key"] for record in calib]

    root_logger.info("Saving index of keys...")
    splits = {"train": train_k, "test": test_k, "validate": val_k, "calib": calib_k}
    save_dictionary_json(artifacts_dir / "data_splits.json", splits)

    # Save out the split data as new jsonl
    root_logger.info("Saving data...")
    for record in train:
        append_jsonl(artifacts_dir / "ingest_records_output_lines_train.jsonl", record)
    for record in test:
        append_jsonl(artifacts_dir / "ingest_records_output_lines_test.jsonl", record)
    for record in val:
        append_jsonl(artifacts_dir / "ingest_records_output_lines_validate.jsonl", record)
    for record in calib:
        append_jsonl(artifacts_dir / "ingest_records_output_lines_calib.jsonl", record)


if __name__ == "__main__":
    main(sys.argv[1:])
