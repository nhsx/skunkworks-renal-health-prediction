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
from aki_predictions.data_processing import CAP_CENTILES


def main(data_dir, splits=None):  # noqa: C901
    """Splitting data for training - prototype."""
    root_logger = logging.getLogger(__name__)

    artifacts_dir = Path(data_dir)
    # debug_dir = artifacts_dir / "debug"

    if CAP_CENTILES:
        capped_string = ""
    else:
        capped_string = "_uncapped"

    # Load artifacts
    root_logger.info(f"Artifacts directory: {artifacts_dir}")

    # load data
    root_logger.info("Loading data file...")
    data = load_jsonl(artifacts_dir / f"ingest_records_output_lines_normalised{capped_string}.jsonl")

    # Extract list of keys
    root_logger.info(f"Total records: {len(data)}")

    # Select split of keys
    # Set Seed
    random.seed(0)

    # Randomise data
    root_logger.info("Shuffling data...")
    shuffled_data = random.sample(data, len(data))

    root_logger.info("Dividing data...")

    if splits is None:
        splits = [0.8, 0.1, 0.05, 0.05]

    train, test, val, calib = split_data(shuffled_data, splits=splits)

    root_logger.info(f"Train: {len(train)}, Test: {len(test)}, Validate: {len(val)}, Calib: {len(calib)}")

    # Save out listing of the key split
    train_k = [record["master_key"] for record in train]
    test_k = [record["master_key"] for record in test]
    val_k = [record["master_key"] for record in val]
    calib_k = [record["master_key"] for record in calib]

    root_logger.info("Saving index of keys...")
    splits = {"train": train_k, "test": test_k, "validate": val_k, "calib": calib_k}
    save_dictionary_json(artifacts_dir / f"data_splits{capped_string}.json", splits)

    # Save out the split data as new jsonl
    root_logger.info("Saving data...")
    for record in train:
        append_jsonl(artifacts_dir / f"ingest_records_output_lines_train{capped_string}.jsonl", record)
    for record in test:
        append_jsonl(artifacts_dir / f"ingest_records_output_lines_test{capped_string}.jsonl", record)
    for record in val:
        append_jsonl(artifacts_dir / f"ingest_records_output_lines_validate{capped_string}.jsonl", record)
    for record in calib:
        append_jsonl(artifacts_dir / f"ingest_records_output_lines_calib{capped_string}.jsonl", record)


if __name__ == "__main__":
    data_dir = sys.argv[1]
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")

    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(data_dir, f"{timestamp}_data_split_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)

    main(data_dir)
