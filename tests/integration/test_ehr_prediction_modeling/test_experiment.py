from pathlib import Path

import tensorflow.compat.v1 as tf

from aki_predictions.ehr_prediction_modeling import experiment
import aki_predictions.ehr_prediction_modeling.config as test_config
import aki_predictions.ehr_prediction_modeling.data


class TestExperiment:
    def test_single_iteration_completes(self, tmp_path):
        tf.keras.backend.clear_session()
        config = test_config.get_config(eval_num_batches=1)
        config.model.num_steps = 2
        config.checkpoint.checkpoint_dir = str(tmp_path)

        experiment.run(config)

    def test_jsondmbatchgenerator_can_be_used(self, tmp_path, monkeypatch):
        tf.keras.backend.clear_session()
        monkeypatch.setattr(
            experiment.tf_dataset,
            "BatchGenerator",
            aki_predictions.ehr_prediction_modeling.data.tf_dataset.JsonDMBatchGenerator,
        )
        config = test_config.get_config()
        config.eval_num_batches = 1
        for key in config.encoder.ndim_dict.keys():
            # increment ndims to deal with updating maps to never map to 0
            config.encoder.ndim_dict[key] = config.encoder.ndim_dict[key] + 1
        config.model.num_steps = 2
        config.checkpoint.checkpoint_dir = str(tmp_path)
        config.data.train_filename = "fake_patients_short.json"
        config.data.valid_filename = "eval_fake_patients.json"
        config.data.records_dirpath = str(Path(config.data.records_dirpath).parent)

        experiment.run(config)

    def test_jsonlbatchgenerator_can_be_used(self, tmp_path, monkeypatch):
        tf.keras.backend.clear_session()
        monkeypatch.setattr(
            experiment.tf_dataset,
            "BatchGenerator",
            aki_predictions.ehr_prediction_modeling.data.tf_dataset.JsonlBatchGenerator,
        )
        data_path = Path("tests") / "fixtures" / "test_data_ingest_output"
        data_locs_dict = {
            "records_dirpath": str(data_path),
            "train_filename": "ingest_records_output_lines_train_uncapped.jsonl",
            "valid_filename": "ingest_records_output_lines_validate_uncapped.jsonl",
            "test_filename": "ingest_records_output_lines_test_uncapped.jsonl",
            "calib_filename": "ingest_records_output_lines_calib_uncapped.jsonl",
            "category_mapping": "category_mapping.json",
            "feature_mapping": "feature_mapping.json",
            "numerical_feature_mapping": "numerical_feature_mapping.json",
        }
        config = test_config.get_config(data_locs_dict=data_locs_dict)
        config.eval_num_batches = None
        # if the number of presence, numeric, or category features change as generated in
        # JsonDMBatchGenerator._get_mappings, values in the below ndim_dict will need updating.
        # They need to be the maximum value in each case (< rather than <=)
        # config.encoder.ndim_dict = {"pres_s": 2, "num_s": 2, "count_s": 2}
        config.model.num_steps = 2
        config.checkpoint.checkpoint_dir = str(tmp_path)

        experiment.run(config)
