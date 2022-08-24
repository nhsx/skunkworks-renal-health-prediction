# Script to load a set of training output files and determine the most
# appropriate againist a performance metric criteria.
#
# Input path(s):
#     <performance metric files to compare>
#     --output/-o <output directory to save any artifacts to (plots/logs etc)>
#
# Example usage: python aki_predictions/evaluation/evaluate_models.py
#                   -o checkpoints/ttl\=120d/train/
#                   -m checkpoints/ttl\=120d/train/metrics-*.json
import sys
import logging
from pathlib import Path
import time
import argparse

from aki_predictions.file_operations import load_json, save_dictionary_json


def parse_args():
    """Parse command line arguments for this script."""
    desc = "Extract performance data and evaluate performance of each provided metrics file."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "-o", "--output_dir", type=str, default=None, help="Directory name of directory to save artifacts to."
    )

    parser.add_argument("-m", "--metrics", nargs="+", help="List of metric file arguments.")
    parser.add_argument("-a", "--adverse_outcome", type=str, default="mortality", help="Adverse outcome to compare.")

    return parser.parse_args()


def evaluate_metrics_data(metrics_data, metric_keys, time_bins, root_logger, adverse_outcome):
    """Evaluate and compare metrics data.

    Args:
        metrics_data (dict): dictionary of metrics files.
        metric_keys (list): list of outcomes.
        time_bins (list): list of time bins.
        root_logger (logging.logger): Logging instance.

    Returns:
        (dict): dictionary of evaluation metrics by metric file key.
    """
    outcome_keys = []
    for outcome in metric_keys:
        for bin in time_bins:
            outcome_keys.append(f"adverse_outcome_{outcome}_within_{bin}h")

    scores = {}
    for key, value in metrics_data.items():
        root_logger.info(f"Applying performance criteria to {key}")
        score_dict = {}
        for outcome_key in outcome_keys:
            # Calculate score metric
            ppv = value[outcome_key]["ppv"]
            sensitivity = 0.0
            if (value[outcome_key]["true_positives"] + value[outcome_key]["false_negatives"]) != 0:
                sensitivity = value[outcome_key]["true_positives"] / (
                    value[outcome_key]["true_positives"] + value[outcome_key]["false_negatives"]
                )

            specificity = 0.0
            if (value[outcome_key]["true_negatives"] + value[outcome_key]["false_positives"]) != 0:
                specificity = value[outcome_key]["true_negatives"] / (
                    value[outcome_key]["true_negatives"] + value[outcome_key]["false_positives"]
                )

            score_dict[outcome_key] = {}
            score_dict[outcome_key]["ppv"] = ppv
            score_dict[outcome_key]["npv"] = value[outcome_key]["npv"]
            score_dict[outcome_key]["sensitivity"] = sensitivity
            score_dict[outcome_key]["specificity"] = specificity
            score_dict[outcome_key]["true_positives"] = value[outcome_key]["true_positives"]
            score_dict[outcome_key]["false_positives"] = value[outcome_key]["false_positives"]
            score_dict[outcome_key]["true_negatives"] = value[outcome_key]["true_negatives"]
            score_dict[outcome_key]["false_negatives"] = value[outcome_key]["false_negatives"]

            # Take geometric mean of ppv and sensitivity.
            score_dict[outcome_key]["score"] = (ppv * sensitivity) ** (1 / 2)

            # Uncomment below lines to log score outputs for other time bins and outcomes.
            # root_logger.info(f"Calculated score of {score_dict[outcome_key]['score']} for {key}, {outcome_key}.")
            # root_logger.info(f"Relating to ppv of {score_dict[outcome_key]['ppv']} and
            #    sensitivity {score_dict[outcome_key]['sensitivity']}.")

        # Specify single outcome/time interval to optimise for.
        scores[key] = score_dict[f"adverse_outcome_{adverse_outcome}_within_24h"]
        root_logger.info(f"Calculated score of {scores[key]['score']} for default performance metric.")
        root_logger.info(f"Calculated ppv of {scores[key]['ppv']}.")
        root_logger.info(f"Calculated npv of {scores[key]['npv']}.")
        root_logger.info(f"Calculated sensitivity of {scores[key]['sensitivity']}.")
        root_logger.info(f"Calculated specificity of {scores[key]['specificity']}.")

        root_logger.info(f"Calculated tp of {scores[key]['true_positives']}.")
        root_logger.info(f"Calculated fp of {scores[key]['false_positives']}.")
        root_logger.info(f"Calculated tn of {scores[key]['true_negatives']}.")
        root_logger.info(f"Calculated fn of {scores[key]['false_negatives']}.")

    # Determine best from the scores
    highest_score = 0.0
    best_model = None

    # Specify metric to score against.
    scoring_field = "score"

    for model, score in scores.items():
        if score[scoring_field] > highest_score:
            highest_score = score[scoring_field]
            best_model = model

    if best_model is None:
        root_logger.warning(f"Unable to select best model for field {scoring_field}.")
    else:
        root_logger.info(f"Highest scoring model: {best_model} with score {highest_score}.")
        root_logger.info(f"Highest scoring model: {best_model} with ppv of {scores[best_model]['ppv']}.")
        root_logger.info(
            f"Highest scoring model: {best_model} with sensitivity of {scores[best_model]['sensitivity']}."
        )

    return scores


def _evaluate_performance(output_dir, metrics_files, adverse_outcome, root_logger=None):  # noqa: C901
    """Performance metrics plotting prototype."""
    if root_logger is None:
        root_logger = logging.getLogger()

    artifacts_dir = output_dir

    # Load artifacts
    root_logger.info(f"Artifacts directory: {artifacts_dir}")

    root_logger.info(f"Against outcome: {adverse_outcome}")

    # load data
    root_logger.info("Loading metrics files...")
    metrics_file_paths = metrics_files
    metrics_data = {}
    for metrics_file in metrics_file_paths:
        metrics_data[str(Path(metrics_file))] = load_json(Path(metrics_file))

    metric_keys = ["mortality", "dialysis", "itu"]
    time_bins = [6, 12, 18, 24, 30, 36, 42, 48]

    root_logger.info(f"Outcomes: {metric_keys}")
    root_logger.info(f"Time intervals: {time_bins}")

    # Evaluate metrics data
    score_outputs = evaluate_metrics_data(metrics_data, metric_keys, time_bins, root_logger, adverse_outcome)

    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    output_temp_location = artifacts_dir / f"{timestamp}_performance_comparison_scoring.json"
    save_dictionary_json(output_temp_location, score_outputs)


def main(output_dir, metrics, adverse_outcome=None):
    """Main method for performance evaluation."""
    if adverse_outcome is None:
        adverse_outcome = "mortality"
    _evaluate_performance(Path(output_dir), metrics, adverse_outcome)


if __name__ == "__main__":
    args = parse_args()

    # Load artifacts directory
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    artifacts_dir = Path(args.output_dir)

    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(artifacts_dir, f"{timestamp}_performance_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)

    main(args.output_dir, args.metrics, args.adverse_outcome)
