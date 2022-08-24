# Script to load a jsonl record dataset and then work through data using a
# mapping definition to find various metrics and report back.
#
# Input path:
#    Directory to ingest and add output metrics to.
#
# Output files (Saved to input path directory):
#    <date timestamp>_metrics_log.txt (log of console outputs for tracking)
#    metrics_statistics.json
#    metrics_statistics_splits.json
import sys
import logging
from pathlib import Path
import time
from collections import Counter

from tqdm import tqdm
import numpy as np
import pandas as pd

from aki_predictions.file_operations import load_json, load_jsonl, save_dictionary_json
from aki_predictions.data_processing import CAP_CENTILES


def find_split(master_key, splits):
    """Find data split for key.

    Args:
        master_key (string): Patient spell key
        splits (dictionary of lists): dictionary of lists of keys corresponding to data splits.

    Returns:
        (string): name of split that corresponds to provided master key.
    """
    for key, value in splits.items():
        if master_key in value:
            return key
    return None


def main(output_dir, data_dir):  # noqa: C901
    """Main metric extraction prototype."""
    root_logger = logging.getLogger(__name__)

    artifacts_dir = Path(output_dir)
    # debug_dir = artifacts_dir / "debug"

    if CAP_CENTILES:
        capped_string = ""
    else:
        capped_string = "_uncapped"

    # Load artifacts
    root_logger.info(f"Artifacts directory: {artifacts_dir}")

    # load data
    root_logger.info("Loading data file...")
    data = load_jsonl(Path(data_dir) / f"ingest_records_output_lines_normalised{capped_string}.jsonl")

    # Load data splits
    root_logger.info("Load data splits")
    splits = load_json(Path(data_dir) / f"data_splits{capped_string}.json")

    # Load metadata mapping
    root_logger.info("Load metadata mapping")
    metadata_mapping = load_json(Path(data_dir) / "metadata_mapping.json")

    # Invert metadata mapping
    inverted_metadata_mapping = {str(v): k for k, v in metadata_mapping.items()}

    total_stays = 0
    for split, keys in splits.items():
        total_stays += len(keys)
        print(f"{split}: {len(keys)}")

    print(f"total stays: {total_stays}")

    for split, keys in splits.items():
        print(f"{split}: percentage: {len(keys) / total_stays}")

    # Set up metrics to capture
    metrics_data_template = {
        "stay_length_days": None,
        "stay_feature_changes": None,
        "stay_outcome_mortality": None,
        "stay_outcome_itu": None,
        "stay_outcome_dialysis": None,
        "stay_adverse_outcome": None,
        "year_of_birth": None,
        "age_in_2022": None,
    }

    if "ethnic_origin" in metadata_mapping.keys():
        metrics_data_template["ethnic_origin"] = None
    if "sex" in metadata_mapping.keys():
        metrics_data_template["sex"] = None
    if "method_of_admission" in metadata_mapping.keys():
        metrics_data_template["method_of_admission"] = None

    metrics_data = []

    # Load data splits
    # Calculate above metrics for each split

    # Work through each record
    root_logger.info("Analyzing records...")
    root_logger.info(f"Size: {len(data)}")
    for _, record in enumerate(tqdm(data)):
        record_metrics = metrics_data_template.copy()
        record_metrics["record_number"] = record["record_number"]
        record_metrics["master_key"] = record["master_key"]
        record_metrics["split"] = find_split(record["master_key"], splits)

        # Stay length
        events = record["episodes"][0]["events"]
        if events:
            start_day = events[0]["patient_age"]
            end_day = events[-1]["patient_age"]

            record_metrics["stay_length_days"] = end_day - start_day
        else:
            record_metrics["stay_length_days"] = 0

        # Feature changes per spell
        feature_changes = 0
        for event in record["episodes"][0]["events"]:
            feature_changes = +len(event["entries"])
        record_metrics["stay_feature_changes"] = feature_changes

        # Stay adverse outcome
        adverse_outcome = False
        for event in record["episodes"][0]["events"]:
            if event["labels"]["adverse_outcome_in_spell"] == "1":
                adverse_outcome = True
        record_metrics["stay_adverse_outcome"] = adverse_outcome

        # Stay outcome mortality
        adverse_outcome_mortality = False
        for event in record["episodes"][0]["events"]:
            if event["labels"]["adverse_outcome_mortality_within_6h"] == "1":
                adverse_outcome_mortality = True
        record_metrics["stay_outcome_mortality"] = adverse_outcome_mortality

        # Stay outcome itu
        adverse_outcome_itu = False
        for event in record["episodes"][0]["events"]:
            if event["labels"]["adverse_outcome_itu_within_6h"] == "1":
                adverse_outcome_itu = True
        record_metrics["stay_outcome_itu"] = adverse_outcome_itu

        # Stay outcome dialysis
        adverse_outcome_dialysis = False
        for event in record["episodes"][0]["events"]:
            if event["labels"]["adverse_outcome_dialysis_within_6h"] == "1":
                adverse_outcome_dialysis = True
        record_metrics["stay_outcome_dialysis"] = adverse_outcome_dialysis

        # Year of birth
        record_metrics["year_of_birth"] = int(record[str(metadata_mapping["year_of_birth"])])

        record_metrics["age_in_2022"] = 2022 - int(record[str(metadata_mapping["year_of_birth"])])

        present_keys = []
        # Map metadata categoricals
        for key, _ in record.items():
            if key not in ["master_key", "record_number", "episodes", str(metadata_mapping["year_of_birth"])]:
                present_keys.append(key)

        for key in present_keys:
            if "ethnic_origin" in inverted_metadata_mapping[key]:
                record_metrics["ethnic_origin"] = str(inverted_metadata_mapping[key])
            if "sex" in inverted_metadata_mapping[key]:
                record_metrics["sex"] = str(inverted_metadata_mapping[key])
            if "method_of_admission" in inverted_metadata_mapping[key]:
                record_metrics["method_of_admission"] = str(inverted_metadata_mapping[key])

        # Log adverse outcomes
        if record_metrics["stay_adverse_outcome"] is True:
            if record_metrics["stay_outcome_mortality"]:
                root_logger.info(
                    f"Record: {record_metrics['record_number']}, split: {record_metrics['split']}, outcome: mortality"
                )
            if record_metrics["stay_outcome_itu"]:
                root_logger.info(
                    f"Record: {record_metrics['record_number']}, split: {record_metrics['split']}, outcome: itu"
                )
            if record_metrics["stay_outcome_dialysis"]:
                root_logger.info(
                    f"Record: {record_metrics['record_number']}, split: {record_metrics['split']}, outcome: dialysis"
                )

        metrics_data.append(record_metrics)

    # Calculate each metric
    output_metrics = {}
    for metric in metrics_data_template.keys():
        output_metrics[metric] = [record[metric] for record in metrics_data]

    metric_statistics = {}
    for metric, metric_info in output_metrics.items():
        root_logger.info(f"Metric: {metric}")
        if len(metric_info):
            if isinstance(metric_info[0], bool):
                # Evaluate boolean statistics
                metric_statistics[metric] = {
                    "count": sum(metric_info),
                    "percentage": (sum(metric_info) / len(data)) * 100,
                }

                root_logger.info(f"Metric: {metric}")

                for key, value in metric_statistics[metric].items():
                    root_logger.info(f"Stat: {key}, Value: {value}")
            elif isinstance(metric_info[0], str):
                # Evaluate categorical entry
                metric_statistics[metric] = {
                    "unique_values": [str(i) for i in list(set(metric_info))],
                    "unique_count": str(len(list(set(metric_info)))),
                    "frequency": {str(v): k for v, k in dict(Counter(metric_info)).items()},
                }

                root_logger.info(f"Metric: {metric}")

                for key, value in metric_statistics[metric].items():
                    root_logger.info(f"Stat: {key}, Value: {value}")
            else:
                # Evaluate numerical statistics
                mean = np.mean(metric_info)
                std = np.std(metric_info)
                maximum = np.max(metric_info)
                minimum = np.min(metric_info)
                centile_1 = np.percentile(metric_info, 1)
                centile_99 = np.percentile(metric_info, 99)

                metric_statistics[metric] = {
                    "mean": float(mean),
                    "std": float(std),
                    "max": float(maximum),
                    "min": float(minimum),
                    "centile_1": float(centile_1),
                    "centile_99": float(centile_99),
                }

                root_logger.info(f"Metric: {metric}")

                for key, value in metric_statistics[metric].items():
                    if key != "values":
                        root_logger.info(f"Stat: {key}, Value: {value}")

    # Store dataframe
    pd.DataFrame(metric_statistics).fillna(0).T.to_csv(artifacts_dir / "metrics_statistics_total.csv")

    # Save feature statistics
    output_location = artifacts_dir / f"metric_statistics{capped_string}.json"
    save_dictionary_json(output_location, metric_statistics)

    # Get metrics for different splits
    split_metrics = {key: {metric: [] for metric in metrics_data_template.keys()} for key in splits.keys()}
    for metric in metrics_data_template.keys():
        for record in metrics_data:
            split_metrics[record["split"]][metric].append(record[metric])

    split_statistics = {}
    for split in splits.keys():
        split_statistics[split] = {}
        for metric, metric_info in output_metrics.items():
            root_logger.info(f"Metric: {metric}")
            if isinstance(metric_info[0], bool):

                root_logger.info(f"Metric: {metric}")

                if split_metrics[split][metric]:
                    split_statistics[split][metric] = {
                        "count": sum(split_metrics[split][metric]),
                        "percentage": (sum(split_metrics[split][metric]) / len(split_metrics[split][metric])) * 100,
                    }
                    for key, value in split_statistics[split][metric].items():
                        root_logger.info(f"Split: {split}, Stat: {key}, Value: {value}")
            elif isinstance(metric_info[0], str):
                # Evaluate categorical entry
                if split_metrics[split][metric]:
                    split_statistics[split][metric] = {
                        "unique_values": [str(i) for i in list(set(metric_info))],
                        "unique_count": str(len(list(set(metric_info)))),
                        "frequency": {str(v): k for v, k in dict(Counter(metric_info)).items()},
                    }

                    root_logger.info(f"Metric: {metric}")

                    for key, value in split_statistics[split][metric].items():
                        root_logger.info(f"Stat: {key}, Value: {value}")
            else:
                # Evaluate numerical statistics
                if len(split_metrics[split][metric]):
                    mean = np.mean(split_metrics[split][metric])
                    std = np.std(split_metrics[split][metric])
                    maximum = np.max(split_metrics[split][metric])
                    minimum = np.min(split_metrics[split][metric])
                    centile_1 = np.percentile(split_metrics[split][metric], 1)
                    centile_99 = np.percentile(split_metrics[split][metric], 99)

                    split_statistics[split][metric] = {
                        "mean": float(mean),
                        "std": float(std),
                        "max": float(maximum),
                        "min": float(minimum),
                        "centile_1": float(centile_1),
                        "centile_99": float(centile_99),
                    }
                else:
                    split_statistics[split][metric] = {
                        "mean": "None",
                        "std": "None",
                        "max": "None",
                        "min": "None",
                        "centile_1": "None",
                        "centile_99": "None",
                    }

                root_logger.info(f"Metric: {metric}")

                for key, value in split_statistics[split][metric].items():
                    if key != "values":
                        root_logger.info(f"Split: {split}, Stat: {key}, Value: {value}")

    # Store raw dataframe
    split_stats_dataframe = pd.DataFrame(split_statistics)
    split_stats_dataframe.to_csv(artifacts_dir / "metrics_split_statistics_raw.csv")
    # Generate dataframe with split statistics metrics as columns
    df = split_stats_dataframe.T
    split_metrics = []
    for metric_name in df:
        col = df[metric_name].apply(pd.Series)
        for name in col.columns.values:
            new_name = str(name) + "_" + str(metric_name)
            col = col.rename(columns={name: new_name})
        split_metrics.append(col)
    split_metrics_full = pd.concat(split_metrics, axis=1)
    split_metrics_full.to_csv(artifacts_dir / "metrics_split_statistics_table.csv")

    # Save feature statistics
    output_location = artifacts_dir / f"metric_statistics_splits{capped_string}.json"
    save_dictionary_json(output_location, split_statistics)


if __name__ == "__main__":
    output_dir = sys.argv[1]
    data_dir = sys.argv[2]
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")

    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(output_dir, f"{timestamp}_metrics_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)

    main(output_dir, data_dir)
