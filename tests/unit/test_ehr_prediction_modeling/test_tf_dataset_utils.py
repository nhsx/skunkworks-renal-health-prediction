import numpy as np
import tensorflow.compat.v1 as tf

from aki_predictions.ehr_prediction_modeling.data import tf_dataset_utils


def test_ndarray_from_sparse_definition():
    shape = [5, 2]
    indices = np.array([[0, 0], [1, 1]])
    values = np.array([1, 2])
    arr = tf_dataset_utils.ndarray_from_sparse_definition(indices, values, shape)
    expected = np.zeros([5, 2])
    expected[0, 0] = 1
    expected[1, 1] = 2
    np.testing.assert_array_equal(arr, expected)


def test_ndarray_from_sparse_definition_can_handle_none_shape():
    shape = None
    indices = np.array([])
    values = np.array([], dtype=np.int64)
    arr = tf_dataset_utils.ndarray_from_sparse_definition(indices, values, shape)
    expected = np.array([0], dtype=np.int64)
    np.testing.assert_array_equal(arr, expected)


def test_ndarray_from_sparse_definition_can_return_multidimensional_empty_array():
    shape = None
    expected_dims = 3
    indices = np.array([])
    values = np.array([], dtype=np.int64)
    arr = tf_dataset_utils.ndarray_from_sparse_definition(indices, values, shape, expected_dims)
    expected = np.array([[[0]]], dtype=np.int64)
    np.testing.assert_array_equal(arr, expected)


def test_ndarray_from_sparse_definition_can_specify_dimension_1_size_w_none_shape():
    shape = None
    expected_dims = 3
    expected_shape_1 = 5
    indices = np.array([])
    values = np.array([], dtype=np.int64)
    arr = tf_dataset_utils.ndarray_from_sparse_definition(indices, values, shape, expected_dims, expected_shape_1)
    expected = np.zeros([1, 5, 1], dtype=int)
    np.testing.assert_array_equal(arr, expected)


def test_ndarray_from_sparse_definition_can_specify_dimension_1_size():
    shape = [4, 4, 4]
    expected_dims = 3
    expected_shape_1 = 5
    indices = np.array([0, 0, 0])
    values = np.array([1], dtype=np.int64)
    arr = tf_dataset_utils.ndarray_from_sparse_definition(indices, values, shape, expected_dims, expected_shape_1)
    expected = np.zeros([4, 5, 4], dtype=int)
    expected[0, 0, 0] = 1
    np.testing.assert_array_equal(arr, expected)


def test_pad_of_nested_dict_of_lists_of_ndarrays_works():
    dummy_dict = {
        "nested_0": {"nested_0_1": [np.random.rand(1, 2, 3), np.random.rand(2, 3, 4)]},
        "nested_1": [np.random.rand(3, 2, 1), np.random.rand(2, 3, 1)],
    }

    padded_dummy_dict = tf_dataset_utils.pad_nested_dict_of_lists_of_ndarray(dummy_dict)

    assert padded_dummy_dict["nested_0"]["nested_0_1"][0].shape == padded_dummy_dict["nested_0"]["nested_0_1"][1].shape
    assert padded_dummy_dict["nested_1"][0].shape == padded_dummy_dict["nested_1"][1].shape


def test_append_nested_dict_of_lists_works():
    dummy_dict = {"nested_0": {"nested_0_1": []}, "nested_1": []}
    new_items_dict = {"nested_0": {"nested_0_1": 0}, "nested_1": 1}
    expected = {"nested_0": {"nested_0_1": [0]}, "nested_1": [1]}
    tf_dataset_utils.append_nested_dict_of_lists(dummy_dict, new_items_dict)
    assert dummy_dict == expected


def test_concat_nested_dict_of_lists_of_ndarrays_works():
    dummy_dict = {"nested_0": {"nested_1": [np.array([1, 2, 3]), np.array([2, 3, 4])]}}
    expected = np.array([[1, 2, 3], [2, 3, 4]])
    tf_dataset_utils.stack_nested_dict_of_lists_of_ndarrays(dummy_dict)
    np.testing.assert_array_equal(dummy_dict["nested_0"]["nested_1"], expected)


def test_prepend_batch_size_to_nested_dict_of_output_shapes_works():
    dummy_dict = {"nested_0": {"nested_1": tf.TensorShape([1, 2, 3])}}
    tf_dataset_utils.prepend_batch_size_to_nested_dict_of_output_shapes(dummy_dict, batch_size=5)
    expected = tf.TensorShape([5, 1, 2, 3])
    assert dummy_dict["nested_0"]["nested_1"] == expected


def test_pad_nested_dict_of_lists_of_ndarrays_can_do_top_down_via_sequence_lengths():
    dummy_dict = {
        "nested_0": {"nested_0_1": [np.random.rand(1, 2, 3), np.random.rand(2, 3, 4)]},
        "nested_1": [np.random.rand(3, 2, 1), np.random.rand(2, 3, 1)],
        "sequence_lengths": [np.array(np.int64(3)), np.array(np.int64(2))],
    }
    tf_dataset_utils.pad_nested_dict_of_lists_of_ndarray(dummy_dict, use_sequence_lengths=True)
    assert dummy_dict["nested_0"]["nested_0_1"][0].shape[0] == dummy_dict["nested_1"][0].shape[0]
