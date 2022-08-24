# Script to assess outputs from the occlusion analysis pipeline.
import sys
import time
import logging
from pathlib import Path

from aki_predictions.file_operations import load_json, load_jsonl, save_dictionary_json
from aki_predictions.ehr_prediction_modeling.utils.occlusion_utils import normalise_cross_entropies


def main(output_dir, feature_mapping_path, occlusion_output_path, metadata_mapping_path=None):
    """Analyse occlusion output data."""
    root_logger = logging.getLogger(__name__)

    artifacts_dir = Path(output_dir)

    # Load artifacts
    root_logger.info(f"Artifacts directory: {artifacts_dir}")

    # load mapping
    root_logger.info("Loading mapping file...")
    mapping = load_json(Path(feature_mapping_path))

    metadata_mapping = {}
    if metadata_mapping_path:
        root_logger.info("Loading metadata file...")
        metadata_mapping = load_json(Path(metadata_mapping_path))

    root_logger.info("Loading occlusion output file...")
    data = load_jsonl(Path(occlusion_output_path))

    # Convert jsonl to json list
    data = {key: value for entry in data for (key, value) in entry.items()}

    normalised_output = normalise_cross_entropies(data)

    # Invert feature mapping
    inverted_mapping = {str(v): k for k, v in mapping.items()}

    # Invert metadata mapping
    inverted_metadata_mapping = {str(v): k for k, v in metadata_mapping.items()}

    # mapped_output = {inverted_mapping[index]: value for index, value in normalised_output.items()}

    mapped_output = {}
    # Loop through mappings and detect sequence or metadata and apply name mapping
    for index, value in normalised_output.items():
        index_type = index.split("_")[0]
        index_value = index.split("_")[1]
        if index_type == "sequence":
            mapped_name = inverted_mapping[str(index_value)]
        elif index_type == "context":
            mapped_name = inverted_metadata_mapping[str(index_value)]
        elif index_type == "unoccluded":
            mapped_name = "unoccluded"
        mapped_output[f"{index_type}_{mapped_name}"] = value

    # Sort by cross entropy value
    sorted_output = dict(sorted(mapped_output.items(), key=lambda item: item[1], reverse=True))
    logging_n = 100
    root_logger.info(f"Top {logging_n} features are...")
    for i, (key, value) in enumerate(sorted_output.items()):
        if i < logging_n:
            root_logger.info(f"{key}: {value}")

    # Save normalised and mapped output
    output_location = artifacts_dir / "occlusion_analysis_output_mapped.json"
    save_dictionary_json(output_location, sorted_output)


if __name__ == "__main__":
    output_dir = sys.argv[1]
    feature_mapping_path = sys.argv[2]
    occlusion_output_path = sys.argv[3]
    metadata_mapping_path = sys.argv[4]
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")

    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(output_dir, f"{timestamp}_occlusion_analysis_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)

    main(output_dir, feature_mapping_path, occlusion_output_path, metadata_mapping_path)
