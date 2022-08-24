from pathlib import Path

import pytest
import tensorflow.compat.v1 as tf
import numpy as np

from aki_predictions.ehr_prediction_modeling.data import tf_dataset
from aki_predictions.training.multiple_adverse_outcomes_training import get_task_coordinator
from aki_predictions.ehr_prediction_modeling.utils import occlusion_utils
from aki_predictions import file_operations
from aki_predictions.ehr_prediction_modeling import config
from aki_predictions.ehr_prediction_modeling import types


@pytest.fixture()
def batch_generator_batch(multiple_adverse_outcomes_no_shuffle_config):
    task_coordinator = get_task_coordinator(multiple_adverse_outcomes_no_shuffle_config)
    batch_gen = tf_dataset.JsonlBatchGenerator(
        multiple_adverse_outcomes_no_shuffle_config, True, task_coordinator, "train"
    )
    batch = batch_gen.batch
    return batch


def test_build_batch_tensor_name_dict_gets_names_for_every_tensor(batch_generator_batch):
    batch_tensor_name_dict = occlusion_utils.build_batch_tensor_name_dict(batch_generator_batch)

    for key in batch_generator_batch.context.keys():
        if batch_tensor_name_dict["context"][key]["is_sparse"]:
            assert isinstance(batch_tensor_name_dict["context"][key]["values_name"], str)
            assert isinstance(batch_tensor_name_dict["context"][key]["indices_name"], str)
            assert isinstance(batch_tensor_name_dict["context"][key]["dense_shape_name"], str)
        else:
            assert isinstance(batch_tensor_name_dict["context"][key]["name"], str)
    for key in batch_generator_batch.sequences.keys():
        if batch_tensor_name_dict["sequences"][key]["is_sparse"]:
            assert isinstance(batch_tensor_name_dict["sequences"][key]["values_name"], str)
            assert isinstance(batch_tensor_name_dict["sequences"][key]["indices_name"], str)
            assert isinstance(batch_tensor_name_dict["sequences"][key]["dense_shape_name"], str)
        else:
            assert isinstance(batch_tensor_name_dict["sequences"][key]["name"], str)


def test_write_batch_tensor_names_creates_recoverable_json(tmpdir, batch_generator_batch):
    output_dir = Path(tmpdir)
    batch_tensor_name_dict = occlusion_utils.build_batch_tensor_name_dict(batch_generator_batch)
    json_name = "test.json"
    file_operations.save_dictionary_json(output_dir / json_name, batch_tensor_name_dict, sort_keys=False)
    loaded_batch_tensor_name_dict = file_operations.load_json(output_dir / json_name)
    assert batch_tensor_name_dict == loaded_batch_tensor_name_dict


def test_build_batch_placeholder_defines_all_necessary_tensors_and_names(batch_generator_batch):
    batch_placeholder, placeholder_names = occlusion_utils.build_batch_placeholder(batch_generator_batch)

    def _check_shape_equal(shape_1, shape_2):
        for el_1, el_2 in zip(shape_1, shape_2):
            if el_1 != el_2:
                return False
        return True

    assert batch_placeholder.is_beginning_sequence.dtype == batch_generator_batch.is_beginning_sequence.dtype
    assert _check_shape_equal(
        batch_generator_batch.is_beginning_sequence.shape, batch_placeholder.is_beginning_sequence.shape
    )
    assert batch_placeholder.is_beginning_sequence.name == placeholder_names["is_beginning_sequence"]

    for key, val in batch_generator_batch.context.items():
        assert batch_placeholder.context[key].dtype == val.dtype
        assert _check_shape_equal(batch_placeholder.context[key].shape, val.shape)
        assert batch_placeholder.context[key].name == placeholder_names["context"][key]

    for key, val in batch_generator_batch.sequences.items():
        assert batch_placeholder.sequences[key].dtype == val.dtype
        if isinstance(val, tf.Tensor):
            # sparse_placeholder doesn't need shape specified
            assert _check_shape_equal(batch_placeholder.sequences[key].shape, val.shape)
        if isinstance(val, tf.Tensor):
            assert batch_placeholder.sequences[key].name == placeholder_names["sequences"][key]
        else:
            batch_placeholder.sequences[key].values.name == placeholder_names["sequences"][key]["values"]
            batch_placeholder.sequences[key].indices.name == placeholder_names["sequences"][key]["indices"]
            batch_placeholder.sequences[key].dense_shape.name == placeholder_names["sequences"][key]["dense_shape"]


class TestOcclusionInBatchGenerator:
    @pytest.fixture()
    def batch_generator(self):
        tf.keras.backend.clear_session()
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
        configuration = config.get_config(data_locs_dict=data_locs_dict, entry_change_flag=9.0)

        batch_generator = tf_dataset.JsonlBatchGenerator(
            config=configuration, is_training=False, task_coordinator=None, data_split_name="valid"
        )
        return batch_generator

    @pytest.fixture()
    def batch_generator_wout_giveaways(self):
        tf.keras.backend.clear_session()
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
            "sequence_giveaways": "sequence_giveaways.json",
        }
        configuration = config.get_config(data_locs_dict=data_locs_dict, entry_change_flag=9.0)

        batch_generator_wout_giveaways = tf_dataset.JsonlBatchGenerator(
            config=configuration, is_training=False, task_coordinator=None, data_split_name="valid"
        )
        return batch_generator_wout_giveaways

    def test_using_giveaways_hides_giveaways_from_sequence_tensors(self, batch_generator_wout_giveaways):
        with tf.Session() as sess:
            batch = batch_generator_wout_giveaways.batch
            sess.run(batch_generator_wout_giveaways.iterator.initializer)
            fetches = {"batch": batch}
            giveaway_field = list(batch_generator_wout_giveaways.presence_map.keys())[0]
            sess.run(batch_generator_wout_giveaways.iterator.initializer)
            no_giveaways_fetches_np = sess.run(fetches)
            expected_occluded_val = float(batch_generator_wout_giveaways.presence_map[giveaway_field])
            assert (
                expected_occluded_val
                not in no_giveaways_fetches_np["batch"].sequences["indexes_presence"].values.tolist()
            )

    @pytest.fixture()
    def batch_generator_w_context(self):
        tf.keras.backend.clear_session()
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
        configuration = config.get_config(
            data_locs_dict=data_locs_dict,
            nact_dict=nact_dict,
            shared_config_kwargs=shared_config_kwargs,
        )

        batch_generator = tf_dataset.JsonlBatchGenerator(
            config=configuration, is_training=False, task_coordinator=None, data_split_name="valid"
        )
        return batch_generator

    def test_expected_fields_occluded_in_sequence(self, batch_generator):
        with tf.Session() as sess:
            batch = batch_generator.batch
            sess.run(batch_generator.iterator.initializer)
            fetches = {"batch": batch}
            fetches_np = sess.run(fetches)
            occlusion_field = list(batch_generator.presence_map.keys())[0]
            batch_generator.set_occluded_field(occlusion_field, occlusion_type="sequence")
            sess.run(batch_generator.iterator.initializer)
            occluded_fetches_np = sess.run(fetches)
            expected_unoccluded_val = float(batch_generator.presence_map[occlusion_field])
            assert expected_unoccluded_val in fetches_np["batch"].sequences["indexes_presence"].values.tolist()
            assert (
                expected_unoccluded_val
                not in occluded_fetches_np["batch"].sequences["indexes_presence"].values.tolist()
            )

    def test_expected_fields_occluded_in_context(self, batch_generator_w_context):
        with tf.Session() as sess:
            batch = batch_generator_w_context.batch
            sess.run(batch_generator_w_context.iterator.initializer)
            fetches = {"batch": batch}
            fetches_np = sess.run(fetches)
            context_family = list(batch_generator_w_context.context_map.keys())[0]
            occlusion_field = batch_generator_w_context.context_map[context_family][-1]
            batch_generator_w_context.set_occluded_field(occlusion_field, occlusion_type="context")
            sess.run(batch_generator_w_context.iterator.initializer)
            occluded_fetches_np = sess.run(fetches)
            expected_unoccluded_val = float(occlusion_field)
            assert expected_unoccluded_val in fetches_np["batch"].context[f"indexes_{context_family}"].values.tolist()
            assert (
                expected_unoccluded_val
                not in occluded_fetches_np["batch"].context[f"indexes_{context_family}"].values.tolist()
            )

    def test_measure_occlusion_difference_raises_error_if_timestamps_unequal(self):
        # think timestamp sequences should always be equal, but want to know about it during analysis if they're not
        # because that'd spoil the measure
        with pytest.raises(ValueError):
            num_batches = 2
            timestamps = [np.random.rand(128, 128) for _ in range(num_batches)]
            timestamps_2 = [np.random.rand(128, 128) for _ in range(num_batches)]
            predictions = [
                {key: np.random.rand(128, 128, 1, 8) for key in ["outcome_1", "outcome_2", "outcome_3"]}
                for _ in range(num_batches)
            ]
            occlusion_utils.measure_occlusion_difference(predictions, predictions, timestamps, timestamps_2)

    def test_measure_occlusion_difference_returns_higher_value_when_predictions_are_more_different(self):
        num_batches = 2
        timestamps = [np.random.rand(128, 128) for _ in range(num_batches)]
        predictions = [
            {key: np.random.rand(128, 128, 1, 8) for key in ["outcome_1", "outcome_2", "outcome_3"]}
            for _ in range(num_batches)
        ]
        slightly_changed_predictions = []
        severely_changed_predictions = []
        for prediction_set in predictions:
            slightly_changed_prediction_set = {}
            severely_changed_prediction_set = {}
            for key in prediction_set.keys():
                slightly_changed_prediction_set[key] = prediction_set[key] + 0.01 * np.random.rand(
                    *prediction_set[key].shape
                )
                severely_changed_prediction_set[key] = prediction_set[key] + 0.1 * np.random.rand(
                    *prediction_set[key].shape
                )
                slightly_changed_prediction_set[key] /= slightly_changed_prediction_set[key].max()
                severely_changed_prediction_set[key] /= severely_changed_prediction_set[key].max()
            slightly_changed_predictions.append(slightly_changed_prediction_set)
            severely_changed_predictions.append(severely_changed_prediction_set)

        occlusion_measure_min = occlusion_utils.measure_occlusion_difference(
            predictions, predictions, timestamps, timestamps
        )
        occlusion_measure_slight = occlusion_utils.measure_occlusion_difference(
            predictions, slightly_changed_predictions, timestamps, timestamps
        )
        occlusion_measure_severe = occlusion_utils.measure_occlusion_difference(
            predictions, severely_changed_predictions, timestamps, timestamps
        )
        assert occlusion_measure_slight > occlusion_measure_min
        assert occlusion_measure_severe > occlusion_measure_slight

    def test_measure_occlusion_difference_raises_assertion_error_if_predictions_outside_0_1_range(self):
        num_batches = 2
        timestamps = [np.random.rand(128, 128) for _ in range(num_batches)]
        acceptable_predictions = [
            {key: np.random.rand(128, 128, 1, 8) for key in ["outcome_1", "outcome_2", "outcome_3"]}
            for _ in range(num_batches)
        ]
        too_high_predictions = [
            {key: np.random.rand(128, 128, 1, 8) + 1.0 for key in ["outcome_1", "outcome_2", "outcome_3"]}
            for _ in range(num_batches)
        ]
        too_low_predictions = [
            {key: np.random.rand(128, 128, 1, 8) - 1.0 for key in ["outcome_1", "outcome_2", "outcome_3"]}
            for _ in range(num_batches)
        ]
        with pytest.raises(AssertionError):
            occlusion_utils.measure_occlusion_difference(
                acceptable_predictions, too_high_predictions, timestamps, timestamps
            )
        with pytest.raises(AssertionError):
            occlusion_utils.measure_occlusion_difference(
                acceptable_predictions, too_low_predictions, timestamps, timestamps
            )
        with pytest.raises(AssertionError):
            occlusion_utils.measure_occlusion_difference(
                too_high_predictions, acceptable_predictions, timestamps, timestamps
            )
        with pytest.raises(AssertionError):
            occlusion_utils.measure_occlusion_difference(
                too_low_predictions, acceptable_predictions, timestamps, timestamps
            )


class TestNormaliseCrossEntropies:
    def test_normalise_cross_entropies_returns_dict(self):
        assert isinstance(
            occlusion_utils.normalise_cross_entropies({"unoccluded_unoccluded": 0.0, "sequence_0": 1.0}), dict
        )

    def test_returns_normalised_values(self):
        input_values = {
            "unoccluded_unoccluded": 10.0,
            "sequence_feature_0": 15.0,
            "sequence_feature_1": 12.0,
            "sequence_feature_2": 18.0,
        }
        output_expected = {
            "unoccluded_unoccluded": 0.0,
            "sequence_feature_0": 5.0,
            "sequence_feature_1": 2.0,
            "sequence_feature_2": 8.0,
        }
        normalised = occlusion_utils.normalise_cross_entropies(input_values)
        assert normalised == output_expected

    def test_returns_normalised_values_absolute(self):
        input_values = {
            "unoccluded_unoccluded": 10.0,
            "sequence_feature_0": 9.0,
            "sequence_feature_1": 12.0,
            "sequence_feature_2": 18.0,
        }
        output_expected = {
            "unoccluded_unoccluded": 0.0,
            "sequence_feature_0": 1.0,
            "sequence_feature_1": 2.0,
            "sequence_feature_2": 8.0,
        }
        normalised = occlusion_utils.normalise_cross_entropies(input_values)
        assert normalised == output_expected

    def test_returns_normalised_values_absolute_float(self):
        input_values = {
            "unoccluded_unoccluded": 10.2,
            "sequence_feature_0": 9.4,
            "sequence_feature_1": 12.6,
            "sequence_feature_2": 18.8,
        }
        output_expected = {
            "unoccluded_unoccluded": 0.0,
            "sequence_feature_0": 0.8,
            "sequence_feature_1": 2.4,
            "sequence_feature_2": 8.6,
        }
        normalised = occlusion_utils.normalise_cross_entropies(input_values)
        assert normalised == output_expected
