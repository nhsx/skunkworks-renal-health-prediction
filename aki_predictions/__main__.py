# CLI Tool to utilise the aki_predictions processing and model training pipeline.
import sys
import argparse
import logging
import time
from pathlib import Path

from aki_predictions.data_processing.ingest_data import main as ingest_main
from aki_predictions.data_processing.normalise_data import main as normalise_main
from aki_predictions.data_processing.split_data import main as split_main
from aki_predictions.data_processing.extract_metrics import main as survey_main
from aki_predictions.training.multiple_adverse_outcomes_training import main as training_main
from aki_predictions.training.multiple_adverse_outcomes_w_context_training import main as training_with_context_main
from aki_predictions.training.multiple_adverse_outcomes_training_hyper_p_scan import main as hyperp_main
from aki_predictions.training.multiple_adverse_outcomes_training_recurrent_block_scan import main as recurrent_cell_main
from aki_predictions.evaluation.performance_metrics import main as performance_metrics_main
from aki_predictions.training.multiple_adverse_outcomes_occlusion_analysis import main as occlusion_main
from aki_predictions.training.multiple_adverse_outcomes_w_context_occlusion_analysis import (
    main as occlusion_context_main,
)
from aki_predictions.evaluation.occlusion_analysis import main as occlusion_reporting_main
from aki_predictions.training.multiple_adverse_outcomes_threshold_sweep import main as threshold_sweep_main
from aki_predictions.training.multiple_adverse_outcomes_w_context_threshold_sweep import (
    main as threshold_sweep_context_main,
)
from aki_predictions.evaluation.evaluate_models import main as performance_comparison_main
from aki_predictions.inference.multiple_adverse_outcomes_inference import main as inference_main
from aki_predictions.inference.multiple_adverse_outcomes_inference import main as inference_context_main
from aki_predictions import __version__


def parse_arguments(args):
    """Parse top level command line arguments.

    Args:
        args (list): list of command line arguments

    Returns:
        (argparse.Namespace): Namespace object containing argument inputs
    """
    desc = f"CLI Tool for C308 Predictive Renal Health Proof of Concept v{__version__}"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("-o", "--output_dir", type=str, default=".", help="Output directory for logging.")

    subparsers = parser.add_subparsers(help="pipeline commands", dest="command")

    # Add ingest subparser
    ingest_subparser = subparsers.add_parser("ingest")
    ingest_subparser.add_argument(
        "-o", "--output_dir", type=str, default=".", help="Output directory for logging and artifacts."
    )
    ingest_subparser.add_argument("-c", "--config", type=str, default="", help="Data ingest config file location.")

    # Add survey subparser
    survey_subparser = subparsers.add_parser("survey")
    survey_subparser.add_argument(
        "-o", "--output_dir", type=str, default=".", help="Output directory for logging and artifacts."
    )
    survey_subparser.add_argument("-d", "--data", type=str, default="", help="Data location (ingest output).")

    # Add training subparser
    training_subparser = subparsers.add_parser("training")
    training_subparsers = training_subparser.add_subparsers(help="training commands", dest="training_command")

    main_training = training_subparsers.add_parser("default")
    main_training.add_argument(
        "-o", "--output_dir", type=str, default=".", help="Output directory for logging and artifacts."
    )
    main_training.add_argument("-d", "--data", type=str, default="", help="Data location (training data location).")
    main_training.add_argument("-s", "--steps", type=int, default=None, help="Training steps.")
    main_training.add_argument(
        "-c", "--context", type=bool, default=False, help="Include context features in training (True, False)."
    )
    main_training.add_argument(
        "--checkpoint_every", type=int, default=None, help="Save checkpoint every 'x' training steps."
    )
    main_training.add_argument(
        "--eval_every", type=int, default=None, help="Evaluate model performance every 'x' training steps."
    )
    main_training.add_argument(
        "--summary_every", type=int, default=None, help="Save summary events every 'x' training steps."
    )

    hyper_p_training = training_subparsers.add_parser("hyperp")
    hyper_p_training.add_argument(
        "-o", "--output_dir", type=str, default=".", help="Output directory for logging and artifacts."
    )
    hyper_p_training.add_argument("-d", "--data", type=str, default="", help="Data location (training data location).")
    hyper_p_training.add_argument(
        "-s", "--steps", type=int, default=None, help="Training steps for each hyper parameter value step."
    )

    recurrent_cell_training = training_subparsers.add_parser("recurrent_cell")
    recurrent_cell_training.add_argument(
        "-o", "--output_dir", type=str, default=".", help="Output directory for logging and artifacts."
    )
    recurrent_cell_training.add_argument(
        "-d", "--data", type=str, default="", help="Data location (training data location)."
    )
    recurrent_cell_training.add_argument(
        "-s", "--steps", type=int, default=None, help="Training steps for each recurrent cell option."
    )

    # Add evaluation subparser
    evaluation_subparser = subparsers.add_parser("evaluation")
    evaluation_subparsers = evaluation_subparser.add_subparsers(help="evaluation commands", dest="evaluation_command")

    performance_metrics_subparser = evaluation_subparsers.add_parser("performance_evaluation")
    performance_metrics_subparser.add_argument(
        "-o", "--output_dir", type=str, default=".", help="Output directory for logging and artifacts."
    )
    performance_metrics_subparser.add_argument("-m", "--metrics", type=str, default="", help="Metrics file.")

    occlusion_analysis_processing = evaluation_subparsers.add_parser("occlusion_processing")
    occlusion_analysis_processing.add_argument(
        "-o", "--output_dir", type=str, default=".", help="Output directory for logging and artifacts."
    )
    occlusion_analysis_processing.add_argument(
        "-d", "--data", type=str, default="", help="Data location (training data location)."
    )
    occlusion_analysis_processing.add_argument(
        "-s",
        "--steps",
        type=int,
        default=None,
        help="Training steps for occlusion processing (must equal original training steps.).",
    )
    occlusion_analysis_processing.add_argument(
        "-c",
        "--context",
        type=bool,
        default=False,
        help="Include context features in occlusion analysis (True, False).",
    )

    occlusion_analysis_reporting = evaluation_subparsers.add_parser("occlusion_reporting")
    occlusion_analysis_reporting.add_argument(
        "-o", "--output_dir", type=str, default=".", help="Output directory for logging and artifacts."
    )
    occlusion_analysis_reporting.add_argument("-f", "--feature", type=str, default="", help="Feature mapping file.")
    occlusion_analysis_reporting.add_argument("-d", "--metadata", type=str, default="", help="Metadata mapping file.")
    occlusion_analysis_reporting.add_argument(
        "-m", "--metrics", type=str, default="", help="Occlusion analysis metrics file."
    )
    occlusion_analysis_reporting.add_argument(
        "-c",
        "--context",
        type=bool,
        default=False,
        help="Include context features in occlusion analysis (True, False).",
    )

    # Threshold sweep
    threshold_sweep = evaluation_subparsers.add_parser("threshold_sweep")
    threshold_sweep.add_argument(
        "-o", "--output_dir", type=str, default=".", help="Output directory for logging and artifacts."
    )
    threshold_sweep.add_argument("-d", "--data", type=str, default="", help="Data location (training data location).")
    threshold_sweep.add_argument(
        "-s",
        "--steps",
        type=int,
        default=None,
        help="Training steps for threshold sweep (must equal original training steps.).",
    )
    threshold_sweep.add_argument(
        "-c",
        "--context",
        type=bool,
        default=False,
        help="Include context features in threshold sweep. Must match original training. (True, False).",
    )

    model_comparison = evaluation_subparsers.add_parser("performance_comparison")
    model_comparison.add_argument(
        "-o", "--output_dir", type=str, default=".", help="Output directory for logging and artifacts."
    )
    model_comparison.add_argument(
        "-a", "--adverse_outcome", type=str, default="mortality", help="Type of adverse outcome to compare."
    )
    model_comparison.add_argument("-m", "--metrics", nargs="+", help="List of metric file arguments.")

    # Add inference subparser
    inference_subparser = subparsers.add_parser("inference")

    inference_subparser.add_argument(
        "-o", "--output_dir", type=str, default=".", help="Output directory for logging and artifacts."
    )
    inference_subparser.add_argument(
        "-m",
        "--model",
        type=str,
        default="",
        help="Model directory (containing relevant model checkpoint to use for inference",
    )
    inference_subparser.add_argument(
        "-d",
        "--data",
        type=str,
        default="",
        help="Data location (path to directory which should also contain mapping files required).",
    )
    inference_subparser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=None,
        help="Step count (match model training step count).",
    )
    inference_subparser.add_argument(
        "-c",
        "--context",
        type=bool,
        default=False,
        help="Include context features in inference. Must match original training. (True, False).",
    )

    return parser.parse_args(args)


def setup_logging(output_dir):
    """Setup logging for tool."""
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")

    # Configure output directory and create if missing
    artifacts_dir = Path(output_dir)
    if artifacts_dir.is_dir() is False:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    log_path = artifacts_dir / f"{timestamp}_aki_predictions_tool_log.txt"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)


def run_ingest(output_dir, config_path):
    """Trigger data ingest."""
    return ingest_main(output_dir, config_path)


def run_normalise(data_dir):
    """Trigger data normalisation."""
    return normalise_main(data_dir)


def run_split(data_dir):
    """Trigger data splitting."""
    return split_main(data_dir)


def run_training(output_dir, data_dir, steps, checkpoint_every, eval_every, summary_every):
    """Trigger default training."""
    return training_main(output_dir, data_dir, steps, checkpoint_every, eval_every, summary_every)


def run_training_with_context(output_dir, data_dir, steps, checkpoint_every, eval_every, summary_every):
    """Trigger default training."""
    return training_with_context_main(output_dir, data_dir, steps, checkpoint_every, eval_every, summary_every)


def run_hyper_parameter_scan(output_dir, data_dir, steps):
    """Trigger default training."""
    return hyperp_main(output_dir, data_dir, steps)


def run_recurrent_cell_scan(output_dir, data_dir, steps):
    """Trigger default training."""
    return recurrent_cell_main(output_dir, data_dir, steps)


def run_survey(output_dir, data_dir):
    """Trigger data survey."""
    return survey_main(output_dir, data_dir)


def run_performance_analysis(output_dir, metrics_file):
    """Trigger performance analysis plot generation."""
    return performance_metrics_main(output_dir, metrics_file)


def run_occlusion_processing(output_dir, data_dir, steps):
    """Trigger occlusion processing."""
    return occlusion_main(output_dir, data_dir, steps)


def run_occlusion_processing_context(output_dir, data_dir, steps):
    """Trigger occlusion processing with context."""
    return occlusion_context_main(output_dir, data_dir, steps)


def run_occlusion_reporting(output_dir, feature_file, occlusion_file, metadata_file):
    """Trigger occlusion reporting."""
    return occlusion_reporting_main(output_dir, feature_file, occlusion_file, metadata_file)


def run_threshold_sweep(output_dir, data_dir, steps):
    """Trigger threshold sweep."""
    return threshold_sweep_main(output_dir, data_dir, steps)


def run_threshold_sweep_context(output_dir, data_dir, steps):
    """Trigger threshold sweep with context."""
    return threshold_sweep_context_main(output_dir, data_dir, steps)


def run_performance_comparison(output_dir, metrics_files, adverse_outcome):
    """Trigger performance analysis plot generation."""
    return performance_comparison_main(output_dir, metrics_files, adverse_outcome)


def run_inference(model_dir, data_path, steps):
    """Trigger inference processing."""
    return inference_main(model_dir, data_path, steps)


def run_inference_context(model_dir, data_path, steps):
    """Trigger inference processing."""
    return inference_context_main(model_dir, data_path, steps)


def main(args=sys.argv):  # noqa: C901
    """Main method."""
    options = parse_arguments(args[1:])

    setup_logging(options.output_dir)

    logger = logging.getLogger(__name__)

    if options.command == "ingest":
        # Run data ingest
        logger.info("Starting data ingest...")
        ingested_data_directory = run_ingest(options.output_dir, options.config)
        logger.info(f"Completed ingest into ingested data directory: {ingested_data_directory}")

        logger.info("Starting data normalisation...")
        run_normalise(ingested_data_directory)
        logger.info(f"Completed normalisation into data directory: {ingested_data_directory}")

        logger.info("Starting data splitting...")
        run_split(ingested_data_directory)
        logger.info(f"Completed training split into data directory: {ingested_data_directory}")
    elif options.command == "survey":
        # Run metric extraction on a given dataset.
        logger.info("Starting data survey...")
        run_survey(options.output_dir, options.data)
    elif options.command == "training":
        # Run training experiment
        if options.training_command == "default":
            # Run default training experiment
            if options.context:
                logger.info("Starting training with context features...")
                run_training_with_context(
                    options.output_dir,
                    options.data,
                    options.steps,
                    options.checkpoint_every,
                    options.eval_every,
                    options.summary_every,
                )
            else:
                logger.info("Starting training...")
                run_training(
                    options.output_dir,
                    options.data,
                    options.steps,
                    options.checkpoint_every,
                    options.eval_every,
                    options.summary_every,
                )
        if options.training_command == "hyperp":
            # Run learning rate sweep
            logger.info("Starting hyper parameter training sweep...")
            run_hyper_parameter_scan(options.output_dir, options.data, options.steps)
        if options.training_command == "recurrent_cell":
            # Run recurrent block test
            logger.info("Starting recurrent cell training sweep...")
            run_recurrent_cell_scan(options.output_dir, options.data, options.steps)
    elif options.command == "evaluation":
        # Run model evaluation
        if options.evaluation_command == "performance_evaluation":
            logger.info("Starting performance plotting...")
            run_performance_analysis(options.output_dir, options.metrics)
        # Occlusion analysis
        if options.evaluation_command == "occlusion_processing":
            if options.context:
                logger.info("Starting occlusion analysis with context features included...")
                feature_file, metadata_file, metrics_file = run_occlusion_processing_context(
                    options.output_dir, options.data, options.steps
                )
                run_occlusion_reporting(options.output_dir, feature_file, metrics_file, metadata_file)
            else:
                logger.info("Starting occlusion analysis...")
                feature_file, metrics_file = run_occlusion_processing(options.output_dir, options.data, options.steps)
                run_occlusion_reporting(options.output_dir, feature_file, metrics_file, None)
        if options.evaluation_command == "occlusion_reporting":
            if options.context:
                logger.info("Starting occlusion analysis reporting with context...")
                run_occlusion_reporting(options.output_dir, options.feature, options.metrics, options.metadata)
            else:
                logger.info("Starting occlusion analysis reporting...")
                run_occlusion_reporting(options.output_dir, options.feature, options.metrics, None)
        # Threshold sweep
        if options.evaluation_command == "threshold_sweep":
            if options.context:
                logger.info("Starting threshold sweep with context features included...")
                run_threshold_sweep_context(options.output_dir, options.data, options.steps)
            else:
                logger.info("Starting threshold sweep...")
                run_threshold_sweep(options.output_dir, options.data, options.steps)
        # £valuate models
        if options.evaluation_command == "performance_comparison":
            logger.info("Starting model performance comparison...")
            run_performance_comparison(options.output_dir, options.metrics, options.adverse_outcome)
    elif options.command == "inference":
        if options.context:
            logger.info("Starting inference with context...")
            run_inference_context(options.model, options.data, options.steps)
        else:
            logger.info("Starting inference...")
            run_inference(options.model, options.data, options.steps)


if __name__ == "__main__":
    main(sys.argv)
