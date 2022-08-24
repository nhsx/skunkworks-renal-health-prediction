import sys
import logging
import time
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

from aki_predictions.ehr_prediction_modeling import config as experiment_config
from aki_predictions.ehr_prediction_modeling import types
from aki_predictions.training.multiple_adverse_outcomes_training import run
from aki_predictions.data_processing import CAP_CENTILES


def _get_config(data_dir, checkpoint_dir="", root_logger=None, steps=None, **kwargs):
    if root_logger is None:
        root_logger = logging.getLogger()
    root_logger.info(data_dir)

    if steps is None:
        # Default training run length
        steps = 60000

    if CAP_CENTILES:
        capped_string = ""
    else:
        capped_string = "_uncapped"

    data_locs_dict = {
        "records_dirpath": data_dir,
        "train_filename": f"ingest_records_output_lines_train{capped_string}.jsonl",
        "valid_filename": f"ingest_records_output_lines_validate{capped_string}.jsonl",
        "test_filename": f"ingest_records_output_lines_test{capped_string}.jsonl",
        "calib_filename": f"ingest_records_output_lines_calib{capped_string}.jsonl",
        "category_mapping": "category_mapping.json",
        "feature_mapping": "feature_mapping.json",
        "numerical_feature_mapping": "numerical_feature_mapping.json",
    }
    shared_config_kwargs = {
        "tasks": (types.TaskNames.ITU_OUTCOME, types.TaskNames.DIALYSIS_OUTCOME, types.TaskNames.MORTALITY_OUTCOME)
    }
    config = experiment_config.get_config(
        data_locs_dict=data_locs_dict,
        num_steps=steps,  # 2 for testing
        eval_num_batches=None,  # to allow exiting via out of range error
        checkpoint_every_steps=2000,  # 1000 for full dataset, 1 for testing
        summary_every_steps=1000,  # 1000 for full dataset, 1 for testing
        eval_every_steps=2000,  # 1000 for full dataset, 1 for testing
        shared_config_kwargs=shared_config_kwargs,
        using_curriculum=False,
        shuffle=True,
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )
    return config


def _get_eval_config(data_dir, checkpoint_dir="", root_logger=None, steps=None, **kwargs):
    if root_logger is None:
        root_logger = logging.getLogger()
    eval_config = _get_config(
        data_dir=data_dir, checkpoint_dir=checkpoint_dir, root_logger=root_logger, steps=steps, **kwargs
    )
    eval_config.using_curriculum = False
    eval_config.shuffle = False


def main(output_dir, data_dir, steps):
    """Run the hyperparameter scan"""
    root_logger = logging.getLogger(__name__)

    hyperparameter_scan = {"rnn_cell": [types.RNNCellType.SRU, types.RNNCellType.UGRNN, types.RNNCellType.LSTM]}
    hyperp_arrays = [np.array(list(val)) for val in hyperparameter_scan.values()]
    hyperp_meshgrids = np.meshgrid(*hyperp_arrays)
    flattened_hyperp_arrays = [np.ravel(arr) for arr in hyperp_meshgrids]
    for i in range(len(flattened_hyperp_arrays[0])):
        tf.keras.backend.clear_session()
        for key_i, key in enumerate(hyperparameter_scan.keys()):
            curr_checkpoint_dir = f"{output_dir}/hyperparameter_scan_{key}_{flattened_hyperp_arrays[key_i][i]}"

            hyperp_kwargs = {
                key: flattened_hyperp_arrays[key_i][i] for key_i, key in enumerate(hyperparameter_scan.keys())
            }

            config = _get_config(
                data_dir=data_dir,
                checkpoint_dir=curr_checkpoint_dir,
                root_logger=root_logger,
                steps=steps,
                **hyperp_kwargs,
            )
            eval_config = _get_eval_config(
                data_dir=data_dir,
                checkpoint_dir=curr_checkpoint_dir,
                root_logger=root_logger,
                steps=steps,
                **hyperp_kwargs,
            )
            run(config, eval_config, root_logger)


if __name__ == "__main__":
    output_dir = sys.argv[1]
    data_dir = sys.argv[2]
    steps = int(sys.argv[3])

    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parents[2] / "data" / "data_ingest_index_full_2022-07-11-100305")

    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    artifacts_dir = Path(output_dir)
    if artifacts_dir.is_dir() is False:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(artifacts_dir, f"{timestamp}_recurrent_block_scan_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)

    main(artifacts_dir, data_dir, steps)
