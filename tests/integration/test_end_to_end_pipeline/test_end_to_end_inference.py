from pathlib import Path
import os

import tensorflow.compat.v1 as tf
import pytest
import numpy as np

import aki_predictions.ehr_prediction_modeling.config as test_config
from aki_predictions.ehr_prediction_modeling import types
from aki_predictions.training.multiple_adverse_outcomes_training import run as training_run
from aki_predictions.training.multiple_adverse_outcomes_training import get_checkpoint_dir
from aki_predictions.inference.multiple_adverse_outcomes_inference import run as inference_run
from aki_predictions.file_operations import load_json


class TestEndToEndInference:
    # Setup training run to generate 'empty' model file
    @pytest.fixture()
    def multiple_adverse_outcomes_w_context_config(self, tmpdir, data_locs_dict, shared_config_kwargs):
        def method(checkpoint_dir):
            if checkpoint_dir is None:
                checkpoint_dir = str(tmpdir)

            data_locs_dict["metadata_mapping"] = "metadata_mapping.json"
            data_locs_dict["missing_metadata_mapping"] = "missing_metadata_mapping.json"
            context_ndim_dict = {
                "diagnosis": 3,
                "ethnic_origin": 5,
                "method_of_admission": 7,
                "sex": 9,
                "year_of_birth": 1,
            }
            context_nact_dict = {
                "diagnosis": 2,
                "ethnic_origin": 1,
                "method_of_admission": 1,
                "sex": 1,
                "year_of_birth": 1,
            }
            context_features = list(context_ndim_dict.keys())
            var_len_context_features = ["diagnosis"]
            fixed_len_context_features = [
                cont_feat for cont_feat in context_features if cont_feat not in var_len_context_features
            ]
            shared_config_kwargs["context_features"] = context_features
            shared_config_kwargs["fixed_len_context_features"] = fixed_len_context_features
            shared_config_kwargs["var_len_context_features"] = var_len_context_features
            shared_config_kwargs["identity_lookup_features"] = [
                types.FeatureTypes.CATEGORY_COUNTS_SEQ,
                types.FeatureTypes.ETHNIC_ORIGIN,
                types.FeatureTypes.METHOD_OF_ADMISSION,
                types.FeatureTypes.SEX,
            ]
            shared_config_kwargs["encoder_layer_sizes"] = (
                [400, 400],
                [400, 400],
                [],
                [400, 400],
                [context_ndim_dict["ethnic_origin"], 400],
                [context_ndim_dict["method_of_admission"], 400],
                [context_ndim_dict["sex"], 400],
                [400, 400],
            )
            nact_dict = {
                types.FeatureTypes.PRESENCE_SEQ: 10,
                types.FeatureTypes.NUMERIC_SEQ: 3,
                types.FeatureTypes.CATEGORY_COUNTS_SEQ: 3,
                **context_nact_dict,
            }
            config = test_config.get_config(
                ndim_dict={"pres_s": 2, "num_s": 2, "count_s": 2, **context_ndim_dict},
                nact_dict=nact_dict,
                data_locs_dict=data_locs_dict,
                checkpoint_every_steps=1,
                num_steps=2,
                eval_num_batches=1,
                shared_config_kwargs=shared_config_kwargs,
                shuffle=False,
                checkpoint_dir=checkpoint_dir,
            )
            return config

        return method

    # Run inference on training output using test data
    @pytest.fixture()
    def multiple_adverse_outcomes_w_context_inference_config(self, data_locs_dict, shared_config_kwargs):
        def method(checkpoint_dir):
            data_locs_dict["train_filename"] = "ingest_records_output_lines_inference.jsonl"
            data_locs_dict["valid_filename"] = "ingest_records_output_lines_inference.jsonl"
            data_locs_dict["test_filename"] = "ingest_records_output_lines_inference.jsonl"
            data_locs_dict["calib_filename"] = "ingest_records_output_lines_inference.jsonl"

            data_locs_dict["metadata_mapping"] = "metadata_mapping.json"
            data_locs_dict["missing_metadata_mapping"] = "missing_metadata_mapping.json"
            context_ndim_dict = {
                "diagnosis": 3,
                "ethnic_origin": 5,
                "method_of_admission": 7,
                "sex": 9,
                "year_of_birth": 1,
            }
            context_nact_dict = {
                "diagnosis": 2,
                "ethnic_origin": 1,
                "method_of_admission": 1,
                "sex": 1,
                "year_of_birth": 1,
            }
            context_features = list(context_ndim_dict.keys())
            var_len_context_features = ["diagnosis"]
            fixed_len_context_features = [
                cont_feat for cont_feat in context_features if cont_feat not in var_len_context_features
            ]
            shared_config_kwargs["context_features"] = context_features
            shared_config_kwargs["fixed_len_context_features"] = fixed_len_context_features
            shared_config_kwargs["var_len_context_features"] = var_len_context_features
            shared_config_kwargs["identity_lookup_features"] = [
                types.FeatureTypes.CATEGORY_COUNTS_SEQ,
                types.FeatureTypes.ETHNIC_ORIGIN,
                types.FeatureTypes.METHOD_OF_ADMISSION,
                types.FeatureTypes.SEX,
            ]
            shared_config_kwargs["encoder_layer_sizes"] = (
                [400, 400],
                [400, 400],
                [],
                [400, 400],
                [context_ndim_dict["ethnic_origin"], 400],
                [context_ndim_dict["method_of_admission"], 400],
                [context_ndim_dict["sex"], 400],
                [400, 400],
            )
            nact_dict = {
                types.FeatureTypes.PRESENCE_SEQ: 10,
                types.FeatureTypes.NUMERIC_SEQ: 3,
                types.FeatureTypes.CATEGORY_COUNTS_SEQ: 3,
                **context_nact_dict,
            }
            config = test_config.get_config(
                ndim_dict={"pres_s": 2, "num_s": 2, "count_s": 2, **context_ndim_dict},
                nact_dict=nact_dict,
                data_locs_dict=data_locs_dict,
                num_steps=2,
                checkpoint_every_steps=1,
                eval_num_batches=1,
                shared_config_kwargs=shared_config_kwargs,
                shuffle=False,
                run_inference=True,
                checkpoint_dir=checkpoint_dir,
            )
            return config

        return method

    def test_end_to_end_inference(
        self, multiple_adverse_outcomes_w_context_config, multiple_adverse_outcomes_w_context_inference_config
    ):
        training_config = multiple_adverse_outcomes_w_context_config(None)
        # Uncomment below instead of above for local debugging.
        # training_config = multiple_adverse_outcomes_w_context_config(Path(os.getcwd()) / "none")

        tf.keras.backend.clear_session()
        training_run(training_config)

        output_dir = get_checkpoint_dir(training_config.checkpoint, "train")

        # Look in output dir for model
        print(output_dir)
        print(os.listdir(output_dir))
        assert Path.exists(Path(output_dir) / "model.ckpt-2.data-00000-of-00001")
        assert Path.exists(Path(output_dir) / "model.ckpt-2.index")
        assert Path.exists(Path(output_dir) / "model.ckpt-2.meta")

        inference_config = multiple_adverse_outcomes_w_context_inference_config(
            training_config.checkpoint.checkpoint_dir
        )
        print(inference_config.checkpoint.checkpoint_dir)

        tf.keras.backend.clear_session()
        inference_run(inference_config)

        inference_output_dir = get_checkpoint_dir(inference_config.checkpoint, "train")

        print(os.listdir(inference_output_dir))

        assert Path.exists(Path(inference_output_dir) / "inference_predictions.json")

        output_data = load_json(Path(inference_output_dir) / "inference_predictions.json")

        # Check keys present in output
        for key in ["MortalityOutcome", "ITUOutcome", "DialysisOutcome"]:
            assert key in output_data.keys()

            for interval in [6, 12, 18, 24, 30, 36, 42, 48]:
                assert str(interval) in output_data[key].keys()

                assert "values" in output_data[key][str(interval)].keys()
                assert "timestamps" in output_data[key][str(interval)].keys()
                assert len(output_data[key][str(interval)]["values"]) == len(
                    output_data[key][str(interval)]["timestamps"]
                )

                np.testing.assert_array_equal(output_data[key][str(interval)]["timestamps"], [0, 21600, 86400])
