from aki_predictions.ehr_prediction_modeling import config
from aki_predictions.ehr_prediction_modeling import configdict


class TestGetConfig:
    def test_returns_config_dict(self):
        assert isinstance(config.get_config(), configdict.ConfigDict)


class TestGetDataConfig:
    def test_returns_config_dict(self):
        shared_config = config.shared_config(**{})
        assert isinstance(config.get_data_config(shared_config), configdict.ConfigDict)

    def test_passes_variables_from_data_locs_dict(self):
        shared_config = config.shared_config(**{})
        data_path = "data_path"
        data_locs_dict = {
            "records_dirpath": data_path,
            "train_filename": "training_data_filename",
            "valid_filename": "validation_data_filename",
            "test_filename": "test_data_filename",
            "calib_filename": "calib_data_filename",
            "category_mapping": "category_mapping_filename",
            "feature_mapping": "feature_mapping_filename",
            "numerical_feature_mapping": "numerical_feature_mapping_filename",
        }
        data_config = config.get_data_config(shared_config, data_locs_dict=data_locs_dict)
        assert data_config.records_dirpath == "data_path"
        assert data_config.train_filename == "training_data_filename"
        assert data_config.valid_filename == "validation_data_filename"
        assert data_config.test_filename == "test_data_filename"
        assert data_config.calib_filename == "calib_data_filename"
        assert data_config.category_mapping == "category_mapping_filename"
        assert data_config.feature_mapping == "feature_mapping_filename"
        assert data_config.numerical_feature_mapping == "numerical_feature_mapping_filename"
