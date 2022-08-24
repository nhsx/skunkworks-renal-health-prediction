"""
Script to convert JSONL with many entries into individual files with one for each entry.
Designed for supporting inference tasks.
"""
import sys
import time
import logging
from pathlib import Path

from aki_predictions.file_operations import load_jsonl, append_jsonl


def main(output_dir, data_file):
    """Main splitting method."""
    # Load data file
    root_logger = logging.getLogger(__name__)

    artifacts_dir = Path(output_dir)

    # Load artifacts
    root_logger.info(f"Artifacts directory: {artifacts_dir}")

    # load data
    root_logger.info("Loading data file...")
    data = load_jsonl(Path(data_file))

    # Save out each record as its own single line jsonl
    for record in data:
        output_path = Path(output_dir) / f"ingest_records_output_lines_{record['record_number']}.jsonl"
        if output_path.exists():
            root_logger.error(
                "Cannot append to existing files. Please specify a different directory"
                " or remove files before re-running."
            )
            break
        else:
            append_jsonl(output_path, record)


if __name__ == "__main__":
    output_dir = sys.argv[1]
    data = sys.argv[2]
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")

    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(output_dir, f"{timestamp}_inference_plotting_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)

    main(output_dir, data)
