from pathlib import Path
import os

import pytest
import numpy as np
import tensorflow.compat.v1 as tf

from aki_predictions.__main__ import main
from aki_predictions.training import multiple_adverse_outcomes_training
from aki_predictions.ehr_prediction_modeling import config as test_config
from aki_predictions.ehr_prediction_modeling import types
from aki_predictions.file_operations import save_dictionary_json


@pytest.fixture()
def data_locs_dict_test():
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
        "metadata_mapping": "metadata_mapping.json",
        "missing_metadata_mapping": "missing_metadata_mapping.json",
    }
    return data_locs_dict


@pytest.fixture()
def multiple_adverse_outcomes_training_test(tmp_path, data_locs_dict_test, shared_config_kwargs):
    data_path = Path("tests") / "fixtures" / "test_data_ingest_output"
    data_locs_dict_test["records_dirpath"] = str(data_path)
    data_locs_dict_test["sequence_giveaways"] = "sequence_giveaways.json"

    context_nact_dict = {"diagnosis": 2, "ethnic_origin": 1, "method_of_admission": 1, "sex": 1, "year_of_birth": 1}
    nact_dict = {
        types.FeatureTypes.PRESENCE_SEQ: 10,
        types.FeatureTypes.NUMERIC_SEQ: 3,
        types.FeatureTypes.CATEGORY_COUNTS_SEQ: 3,
        **context_nact_dict,
    }
    context_features = list(context_nact_dict.keys())
    var_len_context_features = ["diagnosis"]
    fixed_len_context_features = [
        cont_feat for cont_feat in context_features if cont_feat not in var_len_context_features
    ]
    identity_lookup_features = [
        types.FeatureTypes.CATEGORY_COUNTS_SEQ,
        types.FeatureTypes.ETHNIC_ORIGIN,
        types.FeatureTypes.METHOD_OF_ADMISSION,
        types.FeatureTypes.SEX,
    ]
    shared_config_kwargs = {
        "tasks": (types.TaskNames.ITU_OUTCOME, types.TaskNames.DIALYSIS_OUTCOME, types.TaskNames.MORTALITY_OUTCOME),
        "context_features": context_features,
        "fixed_len_context_features": fixed_len_context_features,
        "var_len_context_features": var_len_context_features,
        "identity_lookup_features": identity_lookup_features,
    }
    config = test_config.get_config(
        nact_dict=nact_dict,
        data_locs_dict=data_locs_dict_test,
        num_steps=2,  # 2 for testing
        eval_num_batches=None,  # None for full dataset (divided by batch size), 2 for testing
        checkpoint_every_steps=1,  # 2000 for full dataset, 1 for testing
        summary_every_steps=1,  # 1000 for full dataset, 1 for testing
        eval_every_steps=1,  # 2000 for full dataset, 1 for testing
        shared_config_kwargs=shared_config_kwargs,
        using_curriculum=False,
        shuffle=True,
        checkpoint_dir=str(tmp_path),
        threshold_range=np.concatenate((np.arange(0.001, 0.01, 0.001), np.arange(0.01, 1, 0.01)), axis=0),
    )
    return config


@pytest.fixture()
def multiple_adverse_outcomes_training_test_eval(tmp_path, data_locs_dict_test, shared_config_kwargs):
    data_path = Path("tests") / "fixtures" / "test_data_ingest_output"
    data_locs_dict_test["records_dirpath"] = str(data_path)
    data_locs_dict_test["sequence_giveaways"] = "sequence_giveaways.json"

    context_nact_dict = {"diagnosis": 2, "ethnic_origin": 1, "method_of_admission": 1, "sex": 1, "year_of_birth": 1}
    nact_dict = {
        types.FeatureTypes.PRESENCE_SEQ: 10,
        types.FeatureTypes.NUMERIC_SEQ: 3,
        types.FeatureTypes.CATEGORY_COUNTS_SEQ: 3,
        **context_nact_dict,
    }
    context_features = list(context_nact_dict.keys())
    var_len_context_features = ["diagnosis"]
    fixed_len_context_features = [
        cont_feat for cont_feat in context_features if cont_feat not in var_len_context_features
    ]
    identity_lookup_features = [
        types.FeatureTypes.CATEGORY_COUNTS_SEQ,
        types.FeatureTypes.ETHNIC_ORIGIN,
        types.FeatureTypes.METHOD_OF_ADMISSION,
        types.FeatureTypes.SEX,
    ]
    shared_config_kwargs = {
        "tasks": (types.TaskNames.ITU_OUTCOME, types.TaskNames.DIALYSIS_OUTCOME, types.TaskNames.MORTALITY_OUTCOME),
        "context_features": context_features,
        "fixed_len_context_features": fixed_len_context_features,
        "var_len_context_features": var_len_context_features,
        "identity_lookup_features": identity_lookup_features,
    }
    config = test_config.get_config(
        nact_dict=nact_dict,
        data_locs_dict=data_locs_dict_test,
        num_steps=2,  # 2 for testing
        eval_num_batches=None,  # None for full dataset (divided by batch size), 2 for testing
        checkpoint_every_steps=1,  # 2000 for full dataset, 1 for testing
        summary_every_steps=1,  # 1000 for full dataset, 1 for testing
        eval_every_steps=1,  # 2000 for full dataset, 1 for testing
        shared_config_kwargs=shared_config_kwargs,
        using_curriculum=False,
        shuffle=False,
        checkpoint_dir=str(tmp_path),
        threshold_range=np.concatenate((np.arange(0.001, 0.01, 0.001), np.arange(0.01, 1, 0.01)), axis=0),
    )
    return config


class TestEndToEndPipeline:
    def test_end_to_end_training_only(self, tmp_path):
        tf.keras.backend.clear_session()

        # output_dir = Path("none") / "test_main_output"
        output_dir = Path(tmp_path)

        main(
            [
                "main",
                "training",
                "default",
                "--output_dir",
                str(output_dir),
                "--data",
                "tests/fixtures/test_data_ingest_output",
                "--steps",
                "2",
                "--checkpoint_every",
                "1",
                "--eval_every",
                "1",
                "--summary_every",
                "1",
                "--context",
                "True",
            ]
        )

    def test_end_to_end_training_only_no_cli(
        self, tmp_path, multiple_adverse_outcomes_training_test, multiple_adverse_outcomes_training_test_eval
    ):
        tf.keras.backend.clear_session()

        checkpoint_dir = str(tmp_path)

        multiple_adverse_outcomes_training_test.checkpoint.checkpoint_dir = checkpoint_dir
        multiple_adverse_outcomes_training_test_eval.checkpoint.checkpoint_dir = checkpoint_dir

        multiple_adverse_outcomes_training.run(
            multiple_adverse_outcomes_training_test, multiple_adverse_outcomes_training_test_eval
        )

    def test_end_to_end_pipeline(self, tmpdir):
        tf.keras.backend.clear_session()

        output_dir = Path(tmpdir) / "test_main_log"

        if not Path.exists(output_dir):
            Path.mkdir(output_dir)

        # Ingest
        main(
            [
                "main",
                "ingest",
                "--output_dir",
                str(output_dir),
                "--config",
                "tests/fixtures/mock_config.json",
            ]
        )

        # Helper code to detect ingested data directory
        ingest_output_files = os.listdir(output_dir)
        ingest_output_dir = None
        for v in ingest_output_files:
            print(v)
            if (output_dir / v).is_dir():
                ingest_output_dir = output_dir / v
                print(ingest_output_dir)
                break

        # Metric extraction
        main(
            [
                "main",
                "survey",
                "--output_dir",
                str(output_dir),
                "--data",
                str(ingest_output_dir),
            ]
        )

        # Add missing metadata mapping file
        missing_metadata_mapping = {
            "diagnosis": "diagnosis_",
            "ethnic_origin": "ethnic_origin_nan",
            "method_of_admission": None,
            "sex": None,
            "year_of_birth": None,
        }
        missing_metadata_mapping_path = Path(ingest_output_dir) / "missing_metadata_mapping.json"
        save_dictionary_json(missing_metadata_mapping_path, missing_metadata_mapping)

        # Training
        main(
            [
                "main",
                "training",
                "default",
                "--output_dir",
                str(output_dir),
                "--data",
                str(ingest_output_dir),
                "--steps",
                "100",
                "--checkpoint_every",
                "50",
                "--eval_every",
                "50",
                "--summary_every",
                "10",
                "--context",
                "True",
            ]
        )

        training_output_dir = output_dir / "checkpoints" / "ttl=120d" / "train"

        assert Path.exists(training_output_dir)

        print(os.listdir(training_output_dir))

        # Evaluation

        # Model Selection
        main(
            [
                "main",
                "evaluation",
                "performance_comparison",
                "--output_dir",
                str(output_dir),
                "-a",
                "mortality",
                "-m",
                str(training_output_dir / "metrics-50-0.5.json"),
                str(training_output_dir / "metrics-100-0.5.json"),
            ]
        )

        # Confusion plots
        main(
            [
                "main",
                "evaluation",
                "performance_evaluation",
                "--output_dir",
                str(output_dir),
                "-m",
                str(training_output_dir / "metrics-100-0.5.json"),
            ]
        )

        # Model Metrics on Best Model
        main(
            [
                "main",
                "evaluation",
                "performance_comparison",
                "--output_dir",
                str(output_dir),
                "-a",
                "mortality",
                "-m",
                str(training_output_dir / "metrics-100-0.5.json"),
            ]
        )
