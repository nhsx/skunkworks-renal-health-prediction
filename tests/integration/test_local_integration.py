# These tests rely on local data - these may be relevant when working with new datasets.
from pathlib import Path

import pytest
import tensorflow.compat.v1 as tf
import numpy as np

from aki_predictions.ehr_prediction_modeling.data import tf_dataset
from aki_predictions.ehr_prediction_modeling import config as experiment_config
from aki_predictions.training.multiple_adverse_outcomes_training import _get_config, run
from aki_predictions.ehr_prediction_modeling import types


class TestMultipleAdverseOutcomesTraining:
    config_instance = _get_config(
        data_dir=str(Path("data") / "data_ingest_index_full_2022-07-11-100305"), expect_giveaways=False
    )
    local_data_check = pytest.mark.skipif(
        Path(config_instance.data.records_dirpath).exists() is False,
        reason="Test requires local data.",
    )

    @local_data_check
    def test_single_iteration_w_multiple_adverse_outcomes_completes_full_dataset(self, tmp_path):
        tf.keras.backend.clear_session()
        data_path = str(Path("data") / "data_ingest_index_full_2022-07-11-100305")
        data_locs_dict = {
            "records_dirpath": data_path,
            "train_filename": "ingest_records_output_lines_train_uncapped.jsonl",
            "valid_filename": "ingest_records_output_lines_validate_uncapped.jsonl",
            "test_filename": "ingest_records_output_lines_test_uncapped.jsonl",
            "calib_filename": "ingest_records_output_lines_calib_uncapped.jsonl",
            "category_mapping": "category_mapping.json",
            "feature_mapping": "feature_mapping.json",
            "numerical_feature_mapping": "numerical_feature_mapping.json",
        }
        shared_config_kwargs = {
            "tasks": (types.TaskNames.ITU_OUTCOME, types.TaskNames.DIALYSIS_OUTCOME, types.TaskNames.MORTALITY_OUTCOME)
        }
        # for updating ndim_dict values - debug to return of _get_mappings in data loader, then
        # ndim_dict = {'pres_s': max([int(el) for el in list(presence_map.values())]) + 1,
        # 'num_s': max([int(el) for el in list(numerical_map.values())]) + 1,
        # 'count_s': max([int(el) for el in list(feature_category_map.values())]) + 1}
        config = experiment_config.get_config(
            data_locs_dict=data_locs_dict,
            num_steps=2,
            eval_num_batches=2,
            shared_config_kwargs=shared_config_kwargs,
        )
        config.checkpoint.checkpoint_dir = str(tmp_path)

        run(config, config)


class TestOcclusionInBatchGenerator:
    config_instance = _get_config(
        data_dir=str(Path("data") / "data_ingest_index_full_2022-07-11-100305"), expect_giveaways=False
    )
    local_data_check = pytest.mark.skipif(
        Path(config_instance.data.records_dirpath).exists() is False,
        reason="Test requires local data.",
    )

    @local_data_check
    def test_time_steps_from_occluded_batch_can_be_matched_to_unoccluded(self):
        batch_gen = tf_dataset.JsonlBatchGenerator(
            _get_config(data_dir=str(Path("data") / "data_ingest_index_full_2022-07-11-100305")), False, None, "valid"
        )
        with tf.Session() as sess:
            batch = batch_gen.batch
            sess.run(batch_gen.iterator.initializer)
            fetches = {"batch": batch}
            fetches_np = sess.run(fetches)
            occlusion_field = list(batch_gen.presence_map.keys())[0]
            batch_gen.set_occluded_field(occlusion_field, occlusion_type="sequence")
            sess.run(batch_gen.iterator.initializer)
            occluded_fetches_np = sess.run(fetches)
            expected_unoccluded_val = float(batch_gen.presence_map[occlusion_field])

            #  make sure the field selected for occlusion has actually affected the sequences before checking
            #  time sequences
            assert expected_unoccluded_val in fetches_np["batch"].sequences["indexes_presence"].values.tolist()
            assert (
                expected_unoccluded_val
                not in occluded_fetches_np["batch"].sequences["indexes_presence"].values.tolist()
            )

            unoccluded_times = fetches_np["batch"].sequences["timestamp"]
            occluded_times = occluded_fetches_np["batch"].sequences["timestamp"]
            np.testing.assert_equal(unoccluded_times, occluded_times)
