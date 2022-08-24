# Script to load a jsonl record dataset and then work through data using a
# mapping definition to normalise all of the numerical values, with reference
# to a provided set of statistics.
#
# Input path:
#    Directory to ingest and add normalised dataset to.
#
# Output files (Saved to input path directory):
#    <date timestamp>_normalisation_log.txt (log of console outputs for tracking)
#    ingest_records_output_lines_normalised.jsonl (normalised data output)
#    debug/ingest_records_output_single_line_<n>_normalised.json
#        (output for first n records - default of 10, in structured pretty json)
import sys
import logging
from pathlib import Path
import time

from tqdm import tqdm

from aki_predictions.file_operations import load_json, load_jsonl, append_jsonl, save_dictionary_json
from aki_predictions.data_processing import CAP_CENTILES


def main(data_dir):  # noqa: C901
    """Main normalisation prototype."""
    root_logger = logging.getLogger(__name__)

    artifacts_dir = Path(data_dir)
    debug_dir = artifacts_dir / "debug"

    if CAP_CENTILES:
        capped_string = ""
    else:
        capped_string = "_uncapped"

    # Load artifacts
    root_logger.info(f"Artifacts directory: {artifacts_dir}")

    # load data
    root_logger.info("Loading data file...")
    data = load_jsonl(artifacts_dir / "ingest_records_output_lines.jsonl")

    # load numerical feature mapping
    root_logger.info("Loading numerical feature mapping...")
    numerical_feature_mapping = load_json(artifacts_dir / "numerical_feature_mapping.json")
    numerical_mapping_strings = [str(val) for val in numerical_feature_mapping.values()]

    # load feature statistics
    root_logger.info("Loading numerical feature statistics...")
    feature_statistics = load_json(artifacts_dir / "feature_statistics.json")

    # Work through each record
    root_logger.info("Processing records...")
    for i, record in enumerate(tqdm(data)):
        updated_record = record.copy()
        # Work through each entry for each record and normalise numerical feature values using statistics
        updated_events = []
        for event in record["episodes"][0]["events"]:
            updated_event = event.copy()

            # Check for numerical feature changes and normalise value
            new_entries = []
            for entry in event["entries"]:
                new_entry = entry.copy()

                # Check for numerical feature changes
                if entry["feature_idx"] in numerical_mapping_strings:
                    # Feature is a numerical feature
                    # find numerical feature statistics using feature index
                    if new_entry["feature_value"] != "":
                        feature_key = list(numerical_feature_mapping.keys())[
                            list(numerical_feature_mapping.values()).index(int(entry["feature_idx"]))
                        ]
                        stats = feature_statistics[feature_key]

                        # Process feature value
                        current_value = float(new_entry["feature_value"])
                        # Cap values within 1st and 99th centile
                        if CAP_CENTILES:
                            if current_value < stats["centile_1"]:
                                current_value = stats["centile_1"]
                            if current_value > stats["centile_99"]:
                                current_value = stats["centile_99"]
                            # Normalise values between centile_1 and 1
                            # Zero values cause issues with the dense/sparse mapping within the data loading
                            if stats["centile_99"] - stats["centile_1"]:
                                new_value = 0.0
                            else:
                                new_value = (current_value) / (stats["centile_99"] - stats["centile_1"])
                        else:
                            if stats["max"] == stats["min"]:
                                new_value = 0.0
                            else:
                                new_value = (current_value) / (stats["max"] - stats["min"])

                        # Update feature value
                        new_entry["feature_value"] = str(new_value)

                new_entries.append(new_entry)

            updated_event["entries"] = new_entries

            updated_events.append(updated_event)

        updated_record["episodes"][0]["events"] = updated_events

        # Write out record to new normalised dataset
        output_normalised_location = artifacts_dir / f"ingest_records_output_lines_normalised{capped_string}.jsonl"
        append_jsonl(output_normalised_location, updated_record)

        # Export subset of structured records
        if i < 10:
            output_temp_location = debug_dir / f"ingest_records_output_single_line_{i}_normalised{capped_string}.json"
            save_dictionary_json(output_temp_location, updated_record)


if __name__ == "__main__":
    data_dir = sys.argv[1]

    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(data_dir, f"{timestamp}_normalisation_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)

    main(data_dir)
