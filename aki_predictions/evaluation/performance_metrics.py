# Script to load a json metrics file and generate relevant performance graphs.
#
# Input path:
#    Metrics filepath
#
# Output files (Saved to directory of input file):
#    <date timestamp>_performance_log.txt (log of console outputs for tracking)
import sys
import logging
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from aki_predictions.file_operations import load_json


def plot_performance(outcome, time_bins, metrics_data, artifacts_dir):
    """Plot performance graph for specific outcome."""
    performance = {
        "true_positives": [],
        "false_positives": [],
        "true_negatives": [],
        "false_negatives": [],
        "pr": [],
        "roc": [],
    }
    # Handle non-number value (inf) from divide by zeros while appending.
    for bin in time_bins:
        key = f"adverse_outcome_{outcome}_within_{bin}h"
        performance["true_positives"].append(
            metrics_data[key]["true_positives"] if not isinstance(metrics_data[key]["true_positives"], str) else 0.0
        )
        performance["false_positives"].append(
            metrics_data[key]["false_positives"] if not isinstance(metrics_data[key]["false_positives"], str) else 0.0
        )
        performance["true_negatives"].append(
            metrics_data[key]["true_negatives"] if not isinstance(metrics_data[key]["true_negatives"], str) else 0.0
        )
        performance["false_negatives"].append(
            metrics_data[key]["false_negatives"] if not isinstance(metrics_data[key]["false_negatives"], str) else 0.0
        )

        performance["pr"].append(metrics_data[key]["pr"])
        performance["roc"].append(metrics_data[key]["roc"])

    plot_interval_performance(outcome, time_bins, performance, artifacts_dir)

    plot_pr_curve(outcome, time_bins, performance, artifacts_dir)

    plot_roc_curve(outcome, time_bins, performance, artifacts_dir)


def plot_pr_curve(outcome, time_bins, performance, artifacts_dir):
    """Plot PR curve for current performance."""
    fig = plt.figure()

    ax1 = fig.add_subplot()
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title(f"PR curve for each time interval for outcome {outcome}")

    legend = []
    for i, _ in enumerate(time_bins):
        xvals = [val[0] if not isinstance(val[0], str) else 0.0 for val in performance["pr"][i]]
        yvals = [val[1] if not isinstance(val[1], str) else 0.0 for val in performance["pr"][i]]
        (l,) = ax1.plot(xvals, yvals)
        legend.append(l)

    ax1.legend(legend, [str(bin) for bin in time_bins])
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.savefig(artifacts_dir / f"{outcome}-pr.png", bbox_inches="tight", dpi=200)

    plt.close(fig)


def plot_roc_curve(outcome, time_bins, performance, artifacts_dir):
    """Plot ROC curve for current performance."""
    fig = plt.figure()

    ax1 = fig.add_subplot()
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC curve for each time interval for outcome {outcome}")

    legend = []
    for i, _ in enumerate(time_bins):
        xvals = [val[0] if not isinstance(val[0], str) else 0.0 for val in performance["roc"][i]]
        yvals = [val[1] if not isinstance(val[1], str) else 0.0 for val in performance["roc"][i]]
        (l,) = ax1.plot(xvals, yvals)
        legend.append(l)

    ax1.legend(legend, [str(bin) for bin in time_bins])
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.savefig(artifacts_dir / f"{outcome}-roc.png", bbox_inches="tight", dpi=200)

    plt.close(fig)


def plot_interval_performance(outcome, time_bins, performance, artifacts_dir):
    """Plot performance over time intervals."""
    fig = plt.figure()

    ax1 = fig.add_subplot()
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Plot of aggregated predictions for each time interval for outcome {outcome}")
    ax1.xaxis.set_major_locator(MultipleLocator(6))
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax1.xaxis.set_minor_locator(MultipleLocator(6))
    (l1,) = ax1.plot(time_bins, performance["true_positives"], linestyle="-", marker="o", color="b", markersize=3)
    (l2,) = ax1.plot(time_bins, performance["false_positives"], linestyle="--", marker="o", color="g", markersize=3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Count (True Negatives Only)")
    (l3,) = ax2.plot(time_bins, performance["true_negatives"], linestyle="-.", marker="o", color="r", markersize=3)

    (l4,) = ax1.plot(time_bins, performance["false_negatives"], linestyle=":", marker="o", color="y", markersize=3)

    ax1.legend(
        [l1, l2, l3, l4], ["True Positives", "False Positives", "True Negatives", "False Negatives"], loc="center left"
    )
    ax1.grid(axis="x")
    ax1.grid(axis="y")

    plt.savefig(artifacts_dir / f"{outcome}-performance.png", bbox_inches="tight", dpi=200)

    plt.close(fig)


def main(output_dir, metrics_file, root_logger=None):  # noqa: C901
    """Performance metrics plotting prototype."""
    if root_logger is None:
        root_logger = logging.getLogger()

    # Load artifacts
    root_logger.info(f"Artifacts directory: {output_dir}")

    # load data
    root_logger.info("Loading metrics file...")
    metrics = load_json(Path(metrics_file))

    metric_keys = ["mortality", "dialysis", "itu"]
    time_bins = [6, 12, 18, 24, 30, 36, 42, 48]

    root_logger.info(f"Outcomes: {metric_keys}")
    root_logger.info(f"Time intervals: {time_bins}")
    for key in metric_keys:
        root_logger.info(f"Processing key: {key}")
        plot_performance(key, time_bins, metrics, Path(output_dir))


if __name__ == "__main__":
    output_dir = sys.argv[1]
    metrics_file = sys.argv[2]

    timestamp = time.strftime("%Y-%m-%d-%H%M%S")

    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(output_dir, f"{timestamp}_performance_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)

    main(output_dir, metrics_file, root_logger)
