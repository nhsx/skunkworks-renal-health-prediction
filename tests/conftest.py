from pathlib import Path

import pytest

import aki_predictions.ehr_prediction_modeling.config as test_config
from aki_predictions.ehr_prediction_modeling import types


@pytest.fixture()
def curriculum_multiple_adverse_outcomes_config(tmp_path, data_locs_dict, shared_config_kwargs):
    # for updating ndim_dict values - debug to return of _get_mappings in data loader, then
    # ndim_dict = {'pres_s': max([int(el) for el in list(presence_map.values())]) + 1,
    # 'num_s': max([int(el) for el in list(numerical_map.values())]) + 1,
    # 'count_s': max([int(el) for el in list(feature_category_map.values())]) + 1}
    config = test_config.get_config(
        ndim_dict={"pres_s": 2, "num_s": 2, "count_s": 2},
        data_locs_dict=data_locs_dict,
        num_steps=2,
        eval_num_batches=1,
        shared_config_kwargs=shared_config_kwargs,
    )
    config.checkpoint.checkpoint_dir = str(tmp_path)
    return config


@pytest.fixture()
def multiple_adverse_outcomes_no_shuffle_config(tmp_path, data_locs_dict, shared_config_kwargs):
    # for updating ndim_dict values - debug to return of _get_mappings in data loader, then
    # ndim_dict = {'pres_s': max([int(el) for el in list(presence_map.values())]) + 1,
    # 'num_s': max([int(el) for el in list(numerical_map.values())]) + 1,
    # 'count_s': max([int(el) for el in list(feature_category_map.values())]) + 1}
    config = test_config.get_config(
        ndim_dict={"pres_s": 2, "num_s": 2, "count_s": 2},
        data_locs_dict=data_locs_dict,
        num_steps=2,
        eval_num_batches=1,
        shared_config_kwargs=shared_config_kwargs,
        shuffle=False,
    )
    config.checkpoint.checkpoint_dir = str(tmp_path)
    return config


@pytest.fixture()
def multiple_adverse_outcomes_w_context_config(tmp_path, data_locs_dict, shared_config_kwargs):
    data_locs_dict["metadata_mapping"] = "metadata_mapping.json"
    data_locs_dict["missing_metadata_mapping"] = "missing_metadata_mapping.json"
    context_ndim_dict = {"diagnosis": 3, "ethnic_origin": 5, "method_of_admission": 7, "sex": 9, "year_of_birth": 1}
    context_nact_dict = {"diagnosis": 2, "ethnic_origin": 1, "method_of_admission": 1, "sex": 1, "year_of_birth": 1}
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
        eval_num_batches=1,
        shared_config_kwargs=shared_config_kwargs,
        shuffle=False,
    )
    config.checkpoint.checkpoint_dir = str(tmp_path)
    return config


@pytest.fixture()
def curriculum_multiple_adverse_outcomes_no_shuffle_config(tmp_path, data_locs_dict, shared_config_kwargs):
    # for updating ndim_dict values - debug to return of _get_mappings in data loader, then
    # ndim_dict = {'pres_s': max([int(el) for el in list(presence_map.values())]) + 1,
    # 'num_s': max([int(el) for el in list(numerical_map.values())]) + 1,
    # 'count_s': max([int(el) for el in list(feature_category_map.values())]) + 1}
    config = test_config.get_config(
        ndim_dict={"pres_s": 2, "num_s": 2, "count_s": 2},
        data_locs_dict=data_locs_dict,
        num_steps=2,
        eval_num_batches=1,
        shared_config_kwargs=shared_config_kwargs,
        using_curriculum=True,
        shuffle=False,
        curriculum_starting_epoch=2,
    )
    config.checkpoint.checkpoint_dir = str(tmp_path)
    return config


@pytest.fixture()
def data_locs_dict():
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
    return data_locs_dict


@pytest.fixture()
def shared_config_kwargs():
    shared_config_kwargs = {
        "tasks": (types.TaskNames.ITU_OUTCOME, types.TaskNames.DIALYSIS_OUTCOME, types.TaskNames.MORTALITY_OUTCOME)
    }
    return shared_config_kwargs


@pytest.fixture()
def curriculum_multiple_adverse_outcomes_eval_config(tmp_path, data_locs_dict, shared_config_kwargs):
    eval_config = test_config.get_config(
        ndim_dict={"pres_s": 2, "num_s": 2, "count_s": 2},
        data_locs_dict=data_locs_dict,
        num_steps=2,
        eval_num_batches=1,
        shared_config_kwargs=shared_config_kwargs,
        using_curriculum=False,
        shuffle=False,
    )
    eval_config.checkpoint.checkpoint_dir = str(tmp_path)
    return eval_config
