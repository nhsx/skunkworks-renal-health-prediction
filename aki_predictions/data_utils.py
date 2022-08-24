import numpy as np


def split_data(data, splits=None):
    """Split list of data entries.

    Output training, testing, validation, and calibration splits.

    Args:
        data (list): list of data entries or keys
        splits (list): list of floats, where each float represents the fraction
         of each data split (length 4), should sum to 1

    Returns:
        (list, list, list, list): (train, test, val, calib) lists of entries
    """
    if splits is None:
        splits = [0.25, 0.25, 0.25, 0.25]
    number_of_records = len(data)
    train, test, val, calib = np.split(
        data,
        [
            int(number_of_records * splits[0]),
            int(number_of_records * np.sum(splits[:2])),
            int(number_of_records * np.sum(splits[:3])),
        ],
    )
    return train, test, val, calib
