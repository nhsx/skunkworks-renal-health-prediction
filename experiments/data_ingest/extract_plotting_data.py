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
import time
import logging
import os
import argparse
from pathlib import Path

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aki_predictions.file_operations import load_json, load_jsonl, save_dictionary_json


def parse_args():
    desc = "Extract plots and plotting data from raw data"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--plot_dir", type=str, default=None, help="Directory name of directory to save plots to")

    parser.add_argument(
        "--raw_data", type=str, default="ingest_records_output_lines.jsonl", help="Directory and file name of raw data"
    )

    parser.add_argument("--plotting_data", type=str, default=None, help="Existing plotting data file (pre-processed)")

    parser.add_argument("--wcap", type=bool, default=False, help="Cap data")

    parser.add_argument(
        "--feature_map",
        type=str,
        default="numerical_feature_mapping.json",
        help="Directory and file name of feature map",
    )

    return parser.parse_args()


def generate_distplot(x, feature, cap, dir):
    if len(x) <= 1:
        return False
    # creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.15, 0.85)})

    # Set up plots
    box_width = 0.5
    sns.boxplot(x, ax=ax_box, width=box_width, meanline=True, whis=[1, 99])
    ax_box.spines.right.set_visible(False)
    ax_box.spines.left.set_visible(False)
    ax_box.spines.top.set_visible(False)
    ax_box.spines.bottom.set_visible(False)
    ax_box.tick_params(bottom=False, left=False)

    # Display Mean
    mean = np.around(np.mean(x), 3)
    ax_box.text(
        mean, 0.7, "mean: " + str(mean), horizontalalignment="center", size="x-small", color="black", weight="semibold"
    )

    # Display 1st and 99th percentile
    Q99 = np.around(np.percentile(x, 99), 3)
    Q1 = np.around(np.percentile(x, 1), 3)
    ax_box.text(
        Q99, -0.6, "99th: " + str(Q99), horizontalalignment="center", size="x-small", color="black", weight="semibold"
    )
    ax_box.text(
        Q1, -0.6, "1st: " + str(Q1), horizontalalignment="center", size="x-small", color="black", weight="semibold"
    )

    # Display min/max
    min = np.around(np.min(x))
    max = np.around(np.max(x))
    ax_box.text(
        min, -0.3, "min: " + str(min), horizontalalignment="center", size="x-small", color="black", weight="semibold"
    )
    ax_box.text(
        max, -0.3, "max: " + str(max), horizontalalignment="center", size="x-small", color="black", weight="semibold"
    )

    if cap:
        ax_hist.text(
            0.8,
            1.01,
            "Capped at 1st and 99th Percentile",
            horizontalalignment="center",
            size="x-small",
            color="black",
            weight="semibold",
            transform=ax_hist.transAxes,
        )
        for i, sample in enumerate(x):
            current_value = sample
            if current_value < Q1:
                current_value = Q1
            if current_value > Q99:
                current_value = Q99
            x[i] = current_value

    ax_hist.text(
        0.1,
        1.4,
        feature,
        horizontalalignment="center",
        size="large",
        color="black",
        weight="semibold",
        transform=ax_hist.transAxes,
    )
    sns.histplot(data=x, ax=ax_hist)

    if dir is None:
        if os.path.exists("plots") is False:
            os.mkdir("plots")
        f.savefig(Path("./plots/") / (str(feature) + ".png"), bbox_inches="tight")
    else:
        f.savefig(Path(dir) / (str(feature).replace("/", " ") + ".png"), bbox_inches="tight")

    plt.clf()


def extract_raw_data(data_file, feature_map_file, plot_out_dir, existing_plot_file, cap=False):
    """Extract raw data from data file using feature map, generate plots for each feature."""
    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    timestamp = time.strftime("%Y-%m-%d-%H%M%S")

    file_handler = logging.FileHandler(
        "{0}/{1}.log".format(Path(feature_map_file).parent, f"{timestamp}_plotting_log.txt")
    )
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    root_logger.setLevel(logging.DEBUG)

    root_logger.info("Loading data file...")
    data = load_jsonl(data_file)

    # load numerical feature mapping
    root_logger.info("Loading feature mapping...")
    feature_mapping = load_json(feature_map_file)

    if existing_plot_file is None:
        feat_data = {}
        for k, v in feature_mapping.items():
            feat_data[v] = {"data": [], "feature": k}

        root_logger.info("Extracting raw numerical feature values...")
        for _, record in enumerate(tqdm(data)):
            for event in record["episodes"][0]["events"]:
                for entry in event["entries"]:
                    # Check for numerical feature changes
                    if int(entry["feature_idx"]) in feature_mapping.values():
                        if entry["feature_value"] != "":
                            feat_data[int(float(entry["feature_idx"]))]["data"].append(float(entry["feature_value"]))
        for k, v in feat_data.items():
            x = feat_data[k]["data"]
            feature = feat_data[k]["feature"]

            root_logger.info(f"Total entries for {feature}: {len(x)}")

            generate_distplot(x, feature, cap, plot_out_dir)

        save_dictionary_json(Path("raw_data_plotting.json"), feat_data)
    else:
        feat_data = load_json(existing_plot_file)
        for k, v in feat_data.items():
            x = feat_data[k]["data"]
            feature = feat_data[k]["feature"]

            root_logger.info(f"Total entries for {feature}: {len(x)}")

            generate_distplot(x, feature, cap, plot_out_dir)


# Function that might be useful for data analysis:
def extract_raw_data_summaries(data_file, feature_map_file):
    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)

    root_logger.info("Loading data file...")
    data = load_jsonl(data_file)

    # load numerical feature mapping
    root_logger.info("Loading feature mapping...")
    feature_mapping = load_json(feature_map_file)
    # Collect summary information of all data by record

    feat_summary_data = {}
    for k, v in feature_mapping.items():
        feat_summary_data[v] = {"mean": [], "min": [], "max": [], "std": [], "feature": k}

    root_logger.info("Processing records...")
    for _, record in enumerate(tqdm(data)):
        record_summary = {}
        for event in record["episodes"][0]["events"]:
            for entry in event["entries"]:
                # Check for numerical feature changes
                if int(entry["feature_idx"]) in feature_mapping.values():
                    if entry["feature_value"] != "":
                        if entry["feature_idx"] not in record_summary:
                            record_summary[entry["feature_idx"]] = []
                        record_summary[entry["feature_idx"]].append(float(entry["feature_value"]))

        for k, v in record_summary.items():
            feature_data = record_summary[k]
            feat_summary_data[int(k)]["mean"].append(np.mean(feature_data))
            feat_summary_data[int(k)]["min"].append(np.min(feature_data))
            feat_summary_data[int(k)]["max"].append(np.max(feature_data))
            feat_summary_data[int(k)]["std"].append(np.std(feature_data))

    save_dictionary_json("summary_data_plotting.jsonl", feat_summary_data)


def main():
    arg = parse_args()

    extract_raw_data(arg.raw_data, arg.feature_map, arg.plot_dir, arg.plotting_data, arg.wcap)


if __name__ == "__main__":
    main()
