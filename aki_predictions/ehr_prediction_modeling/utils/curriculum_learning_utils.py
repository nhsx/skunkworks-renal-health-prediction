import numpy as np


def update_curriculum(
    curriculum,
    num_records,
    previous_record_order,
    losses_by_batch,
    min_prob=0.25,
    max_prob=5.0,
    curr_epoch=0,
    start_epoch=1,
):
    """Update a curriculum of difficulties of elements in a dataset according to they were recently
    sampled and what loss was obtained for them, only updating beyond start_epoch and capping probabilities
    Args:
        curriculum (ndarray): 1D array, values >0, higher values mean higher chance of being sampled
        num_records (int): The number of indices to draw
        previous_record_order (ndarray): Array of ints, indices of which elements of the dataset the losses
        pertain to
        losses_by_batch (list of ndarrays): List detailing the loss encountered per element in each batch in the past
        epoch
        min_prob (float or dict): A minimum value to cap probabilties to, or a dict with keys 'method' and 'thresh'.
        Method 'percentile' currently supported. The value of 'thresh' should be a float in range 0-100
        max_prob (float or dict): A maximum value to cap probabilties to, or a dict with keys 'method' and 'thresh'.
        Method 'percentile' currently supported. The value of 'thresh' should be a float in range 0-100
        curr_epoch (int): the current epoch
        start_epoch (int): the epoch after which to start updating the curriculum, otherwise return existing curriculum
    Returns:
        (ndarray): the new curriculum
    """
    out_curriculum = curriculum.copy()
    if start_epoch > curr_epoch:
        # for early epochs (even only 1 epoch), first losses will be very high, and difficulty measurements won't
        # reflect how hard a sample is at the end of the early epoch, so a curriculum is of little value. Introduce
        # curriculum changes once intra-epoch difficulty changes are less stark
        return out_curriculum

    losses = np.concatenate(losses_by_batch)
    # some overspill in the final batch of an epoch can mean these losses need trimming
    losses = losses[:num_records]

    out_curriculum[previous_record_order[: len(losses)]] = losses
    # cap min and max difficulty (never want data to disappear from curriculum or become overwhelmingly common)
    if isinstance(min_prob, dict):
        if min_prob["method"] == "percentile":
            min_prob = np.percentile(losses, min_prob["thresh"])
        else:
            min_prob = min_prob["thresh"]
    if isinstance(max_prob, dict):
        if max_prob["method"] == "percentile":
            max_prob = np.percentile(losses, max_prob["thresh"])
        else:
            max_prob = max_prob["thresh"]
    out_curriculum[out_curriculum < min_prob] = min_prob
    out_curriculum[out_curriculum > max_prob] = max_prob
    if min_prob < 0:
        # never want curriculum going negative, or zero. So if it occurs, offset
        # minimum to 10% of max above zero
        out_curriculum -= min_prob - 0.1 * np.abs(max_prob)
    if np.mean(out_curriculum) == 0.0:
        # very unlikely, but if model manages to get a perfect score then set all probabilities to one to
        # avoid problems in probabilistic sampling with 0 probability
        out_curriculum = np.ones_like(out_curriculum)
    return out_curriculum


def sample_curriculum(curriculum, num_records, epoch, final_epoch=None):
    """Draw indices with replacement of a dataset according to probabilities defined by its curriculum
    Hard example mining is performed, progressing towards uniform sampling on later epochs
    Args:
        curriculum (np.ndarray): 1D array, values >0, higher values mean higher chance of being sampled
        num_records (int): The number of indices to draw
        epoch (int): the current epoch
        final_epoch (int): the total number of epochs expected
    Returns:
        (ndarray): 1D array if int indices
    """
    if final_epoch is not None:
        # for hard example mining, want to start off by having high probability of hard examples, trending towards
        # uniform selection by end of training

        # just in case one epoch after final_epoch is started, take min between usual completion_factor and 1 to
        # ensure epoch_weighted_curriculum won't go negative
        completion_factor = (final_epoch - epoch) / final_epoch
        if completion_factor < 0.0:
            # beyond final epoch for curriculum, just make all weighting go to uniform ones
            completion_factor = 0.0
        epoch_weighted_curriculum = completion_factor * curriculum + (1 - completion_factor) * np.ones_like(curriculum)
        p = epoch_weighted_curriculum / np.sum(epoch_weighted_curriculum)
    else:
        p = curriculum / np.sum(curriculum)
    return np.random.choice(list(range(len(curriculum))), num_records, replace=True, p=p)
