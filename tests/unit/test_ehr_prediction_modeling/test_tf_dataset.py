from pathlib import Path

import pytest

from aki_predictions.ehr_prediction_modeling import config
from aki_predictions.ehr_prediction_modeling.data import tf_dataset


class TestJsonlBatchGenerator:
    @pytest.fixture()
    def batch_generator(self):
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
        configuration = config.get_config(data_locs_dict=data_locs_dict)

        batch_generator = tf_dataset.JsonlBatchGenerator(
            config=configuration, is_training=True, task_coordinator=None, data_split_name="train"
        )
        return batch_generator

    def test_get_mappings_from_config(self, batch_generator):
        data_path = Path("tests") / "fixtures" / "test_data_ingest_output"
        (
            category_mapping_path,
            feature_mapping_path,
            numerical_feature_mapping_path,
            metadata_mapping_path,
            missing_metadata_mapping_path,
        ) = batch_generator._get_mapping_locations()

        assert category_mapping_path == data_path / "category_mapping.json"
        assert feature_mapping_path == data_path / "feature_mapping.json"
        assert numerical_feature_mapping_path == data_path / "numerical_feature_mapping.json"
