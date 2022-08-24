import numpy as np

from aki_predictions import data_utils


class TestSplitData:
    def test_returns_four_lists(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert len(data_utils.split_data(data)) == 4

    def test_returns_split_data(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        train, test, val, calib = data_utils.split_data(data)
        np.testing.assert_array_equal(train, [1, 2, 3])
        np.testing.assert_array_equal(test, [4, 5, 6])
        np.testing.assert_array_equal(val, [7, 8, 9])
        np.testing.assert_array_equal(calib, [10, 11, 12])

    def test_returns_split_data_custom_splits(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        train, test, val, calib = data_utils.split_data(data, splits=[0.8, 0.1, 0.05, 0.05])
        np.testing.assert_array_equal(train, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        np.testing.assert_array_equal(test, [17, 18])
        np.testing.assert_array_equal(val, [19])
        np.testing.assert_array_equal(calib, [20])
