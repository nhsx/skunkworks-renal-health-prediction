import sys
import logging
import time
from pathlib import Path

from aki_predictions.ehr_prediction_modeling import config as experiment_config
from aki_predictions.ehr_prediction_modeling import types
from aki_predictions.training.multiple_adverse_outcomes_training import run
from aki_predictions.data_processing import CAP_CENTILES


def _get_config(checkpoint_dir=""):
    data_path = str(Path(__file__).resolve().parents[2] / "data" / "data_ingest_index_full_2022-07-11-100305")
    print(data_path)

    if CAP_CENTILES:
        capped_string = ""
    else:
        capped_string = "_uncapped"

    data_locs_dict = {
        "records_dirpath": data_path,
        "train_filename": f"ingest_records_output_lines_train{capped_string}.jsonl",
        "valid_filename": f"ingest_records_output_lines_validate{capped_string}.jsonl",
        "test_filename": f"ingest_records_output_lines_test{capped_string}.jsonl",
        "calib_filename": f"ingest_records_output_lines_calib{capped_string}.jsonl",
        "category_mapping": "category_mapping.json",
        "feature_mapping": "feature_mapping.json",
        "numerical_feature_mapping": "numerical_feature_mapping.json",
        "sequence_giveaways": "sequence_giveaways.json",
    }
    shared_config_kwargs = {
        "tasks": (types.TaskNames.ITU_OUTCOME, types.TaskNames.DIALYSIS_OUTCOME, types.TaskNames.MORTALITY_OUTCOME)
    }
    config = experiment_config.get_config(
        data_locs_dict=data_locs_dict,
        num_steps=200001,  # 2 for testing
        eval_num_batches=None,  # to allow exiting via out of range error
        checkpoint_every_steps=1000,  # 1000 for full dataset, 1 for testing
        summary_every_steps=1000,  # 1000 for full dataset, 1 for testing
        eval_every_steps=2000,  # 1000 for full dataset, 1 for testing
        shared_config_kwargs=shared_config_kwargs,
        using_curriculum=True,
        shuffle=False,
        curriculum_starting_epoch=2,
        checkpoint_dir=checkpoint_dir,
    )
    return config


def _get_curriculum_eval_config():
    eval_config = _get_config()
    eval_config.using_curriculum = False
    eval_config.shuffle = False


def main(args):
    """Run training"""
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    artifacts_dir = Path(args[0])
    if artifacts_dir.is_dir() is False:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(artifacts_dir, f"{timestamp}_w_curriculum_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)

    config = _get_config(checkpoint_dir=artifacts_dir)
    # avoid curriculum accidentally getting turned off in config, as this is a curriculum learning experiment
    assert config.using_curriculum is True
    eval_config = _get_curriculum_eval_config()
    run(config, eval_config)


if __name__ == "__main__":
    main(sys.argv[1:])
