import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm

from aki_predictions import file_operations
from aki_predictions.ehr_prediction_modeling.utils import batches


def _build_single_tensor_name_dict(tensor):
    """Create a flat dictionary defining named tensors in the tf graph that are needed to recover a tensor. In the
    case of regular tensors, this is just the name, for sparse tensors values, indices and dense_shape names are needed
    Flag up when a sparse tensor is defined via 'is_sparse' key
    Args:
        tensor (tf.Tensor or tf.SparseTensor): The tensor whose names are to be defined in the output dictionary
    Returns:
        (dict): A dictionary defining the names of the required tensors
    """
    out_dict = {}
    if isinstance(tensor, tf.Tensor):
        out_dict["is_sparse"] = False
        out_dict["name"] = tensor.name
    elif isinstance(tensor, tf.SparseTensor):
        out_dict["is_sparse"] = True
        out_dict["values_name"] = tensor.values.name
        out_dict["indices_name"] = tensor.indices.name
        out_dict["dense_shape_name"] = tensor.dense_shape.name
    return out_dict


def build_batch_tensor_name_dict(batch):
    """Build a nested dictionary with 'context', 'sequences', 'is_beginning_sequence' keys, whose bottom level
    values define tensors from a batch. These may be either normal or sparse tensors, and where sparse tensors occur
    a 'is_sparse': True key-val will exist at the bottom level to flag that various tensors needed to define a sparse
    tensor are included
    Args:
        batch (TFBatch): A batch (named tuple with .context, .sequence, .is_beginning_sequence properties)
    Returns:
        (dict): Dictionary defining the names of relevant tensors from the batch as defined in the tf graph
    """
    out_dict = {"context": {}, "sequences": {}}
    for key in batch.context.keys():
        out_dict["context"][key] = _build_single_tensor_name_dict(batch.context[key])
    for key in batch.sequences.keys():
        out_dict["sequences"][key] = _build_single_tensor_name_dict(batch.sequences[key])
    out_dict["is_beginning_sequence"] = batch.is_beginning_sequence.name
    return out_dict


def recover_tensor_dicts(graph, eval_tensor_names_json_path, eval_predictions_tensor_names_json_path):
    """Build two tensor dictionaries that define the input and output nodes needed for ingesting new
    data and recovering predictions, with values
    corresponding to tensors recovered from a tf.Graph
    Args:
        graph (tf.Graph): A graph defining all tensors and operation nodes in a tensorflow training pipeline
        eval_tensor_names_json_path (dict): A nested dictionary with 'context', 'sequence' and 'is_beginning_sequence'
        keys at top level, defining tensors in a graph required to reach a prediction output without running any
        model weight-altering operations (e.g. backprop). Typically expected to be placeholder tensors that follow
        a similar path through the graph as evaluation tensors
        eval_predictions_tensor_names_json_path (dict): A flat dictionary of tensors that define predictions coming
        out of the model, typically for each of the set of outcomes required during training
    Returns:
        (dict, dict): (dictionary defining input tensors, dictionary defining prediction tensors)
    """
    eval_batch_tensor_name_dict = file_operations.load_json(eval_tensor_names_json_path)
    eval_predictions_tensor_names_dict = file_operations.load_json(eval_predictions_tensor_names_json_path)

    eval_batch_tensor_dict = {"context": {}, "sequences": {}}
    eval_predictions_dict = {}

    # tensors in the batch are split up by context, is_beginning_sequence, and sequence tensors, which may then be
    # defined in a sparse
    # fashion or regular tensors. Need to recover values, indices, dense shape tensors to define a sparse tensor from
    # the graph, just the name for a regular tensor
    for tensor_set_key in ["context", "sequences"]:
        for tensor_key, val in eval_batch_tensor_name_dict[tensor_set_key].items():

            if val["is_sparse"] is True:
                eval_batch_tensor_dict[tensor_set_key][tensor_key] = {}
                eval_batch_tensor_dict[tensor_set_key][tensor_key]["is_sparse"] = True
                eval_batch_tensor_dict[tensor_set_key][tensor_key]["values"] = graph.get_tensor_by_name(
                    val["values_name"]
                )
                eval_batch_tensor_dict[tensor_set_key][tensor_key]["indices"] = graph.get_tensor_by_name(
                    val["indices_name"]
                )
                eval_batch_tensor_dict[tensor_set_key][tensor_key]["dense_shape"] = graph.get_tensor_by_name(
                    val["dense_shape_name"]
                )
            else:
                eval_batch_tensor_dict[tensor_set_key][tensor_key] = {}
                eval_batch_tensor_dict[tensor_set_key][tensor_key]["is_sparse"] = False
                eval_batch_tensor_dict[tensor_set_key][tensor_key]["name"] = graph.get_tensor_by_name(val["name"])

    eval_batch_tensor_dict["is_beginning_sequence"] = {
        "is_sparse": False,
        "name": graph.get_tensor_by_name(eval_batch_tensor_name_dict["is_beginning_sequence"]),
    }

    # predictions tensors should all be regular tensors recoverable by name, and were saved more simply
    for key, val in eval_predictions_tensor_names_dict.items():
        eval_predictions_dict[key] = graph.get_tensor_by_name(val)

    return eval_batch_tensor_dict, eval_predictions_dict


def build_nested_feed_dict(tensor_dict, values_dict):
    """Take nested dictionaries with the same key structure defining both normal and sparse tensors (with 'is_sparse'
    signifying a sparse definition at the bottom nested level of tensor_dict), and build a feed dict as required
    by tf.Session-like objects for ingesting new data.
    Args:
        tensor_dict (nested dict of tensors): Dictionary defining expected tensors, and whether the tensors defined
        at the bottom level are sparse
        values_dict (nested dict of arrays): Dictionary defining feedable arrays to the corresponding tensors in
        tensor_dict (where values, indices, dense_shape must be defined separately for sparse tensors)
    Returns:
        (dict): A nested dictionary where keys are tensors and values are arrays
    """
    feed_dict = {}
    for key, val in tensor_dict.items():
        if "is_sparse" not in val:
            # any bottom level dictionary defining a tensor or sparse tensor should have an 'is_sparse' key,
            # otherwise haven't reached bottom level of nested structure yet
            feed_dict[key] = build_nested_feed_dict(val, values_dict[key])
        else:
            if val["is_sparse"] is True:
                feed_dict[val["values"]] = values_dict[key].values
                feed_dict[val["indices"]] = values_dict[key].indices
                feed_dict[val["dense_shape"]] = values_dict[key].dense_shape
            else:
                feed_dict[val["name"]] = values_dict[key]
    return feed_dict


def flatten_feed_dict(nested_feed_dict):
    """Flatten a nested dictionary structure, bringing nested keys and items up to top level. Intended for
    flattening 'feed dicts' used by tensorflow, which in our models case will have a nested structure at generation
    but need to be flat for tf to ingest
    Args:
        nested_feed_dict (dict): Dictionary with potentially nested dictionaries
    Returns:
        (dict): A flat dictionary
    """
    out_dict = {}
    for key, item in nested_feed_dict.items():
        if isinstance(item, dict):
            sub_dict = flatten_feed_dict(item)
            for sub_key, sub_item in sub_dict.items():
                out_dict[sub_key] = sub_item
        else:
            out_dict[key] = item
    return out_dict


def build_batch_placeholder(real_batch):
    """Creates a set of tensor placeholders according to the structure of a real batch of tensors, in order
    to create nodes in a graph that can have new values fed via feed dict, enabling new data ingestion after training
    completes
    Args:
        real_batch (ehr_prediction_modeling.utils.batches.TFBatch): A batch of tensors and sparse tensors, held
        in dictionaries in a named-tuple structure
    Returns:
        (TFBatch, dict): (The batch of placeholder tensors, a dictionary defining the tensors' names in the graph)
    """
    name_prefix = "placeholder"
    placeholder_names = {}
    ctx_placeholder = {}
    seq_placeholder = {}
    is_beginning_sequence_name = f"{name_prefix}_is_beginning_sequence"
    is_beginning_sequence_placeholder = tf.placeholder(
        real_batch.is_beginning_sequence.dtype,
        shape=real_batch.is_beginning_sequence.shape,
        name=is_beginning_sequence_name,
    )
    placeholder_names["is_beginning_sequence"] = is_beginning_sequence_placeholder.name

    # build up dictionary of context placeholders
    context_placeholder_names = {}
    for key, val in real_batch.context.items():
        name = f"{name_prefix}_context_{key}"
        if isinstance(val, tf.Tensor):
            ctx_placeholder[key] = tf.placeholder(dtype=val.dtype, shape=val.shape, name=name)
            context_placeholder_names[key] = ctx_placeholder[key].name
        elif isinstance(val, tf.SparseTensor):
            ctx_placeholder[key] = tf.sparse_placeholder(dtype=val.dtype, name=name)
            context_placeholder_names[key] = {
                "values": ctx_placeholder[key].values.name,
                "indices": ctx_placeholder[key].indices.name,
                "dense_shape": ctx_placeholder[key].dense_shape.name,
            }
    placeholder_names["context"] = context_placeholder_names

    # build up dictionary of sequence placeholders
    sequence_placeholder_names = {}
    for key, val in real_batch.sequences.items():
        name = f"{name_prefix}_sequences_{key}"
        if isinstance(val, tf.Tensor):
            seq_placeholder[key] = tf.placeholder(dtype=val.dtype, shape=val.shape, name=name)
            sequence_placeholder_names[key] = seq_placeholder[key].name
        elif isinstance(val, tf.SparseTensor):
            seq_placeholder[key] = tf.sparse_placeholder(dtype=val.dtype, name=name)
            sparse_names = {
                "values": seq_placeholder[key].values.name,
                "indices": seq_placeholder[key].indices.name,
                "dense_shape": seq_placeholder[key].dense_shape.name,
            }
            sequence_placeholder_names[key] = sparse_names
    placeholder_names["sequences"] = sequence_placeholder_names

    # put everything into a TFBatch
    batch_placeholder = batches.TFBatch(
        context=ctx_placeholder, sequences=seq_placeholder, is_beginning_sequence=is_beginning_sequence_placeholder
    )
    return batch_placeholder, placeholder_names


def _cross_entropy(arr_1, arr_2):
    """Take a cross-entropy measure of how different two arrays are. Normalise across zeroth axis to ensure
    probabilities that sum to 1 (which likely doesn't occur naturally for time series probability predictions, but is
    needed for sensible cross-entropy)
    Args:
        arr_1 (ndarray): Array of floats
        arr_2 (ndarray): Array of floats
    Returns:
        (float): The cross entropy, summed across all axes
    """
    # time series probabilities won't necessarily sum to 1, which is necessary for a sensible cross-entropy measure,
    # so normalise across time axis
    arr_1_offset = arr_1 + 1e-7
    arr_1_normed = arr_1_offset / np.sum(arr_1_offset, axis=0)

    # Resolve issue with 0/negative values
    arr_2_offset = arr_2 + 1e-7
    arr_2_normaed = arr_2_offset / np.sum(arr_2_offset, axis=0)

    # make sure no unrealistic probabilities existed to start with
    assert arr_1_offset.min() > 0.0
    assert arr_2_offset.min() > 0.0
    assert arr_1.max() <= 1.0
    assert arr_2.max() <= 1.0
    # get the cross-entropy of the distributions
    return -np.sum(arr_1_normed * np.log(arr_2_normaed))


def measure_occlusion_difference(
    unoccluded_predictions, occlusion_predictions, unoccluded_timestamps, current_timestamps
):
    """Measure the difference introduced by occluding a field on predictions from the model compared to unoccluded
    predictions. Timestamps are passed as well as the prediction sequences entirely to assure that no time series
    shift has been introduced by any given occlusion (not expected to be possible but would ruin the comparison if it
    occurred)
    Args:
        unoccluded_predictions (list of dictionaries of ndarrays): The predictions when no field is occluded, held in
        a list of outputs per batch, where the outputs per batch are separated by adverse outcome keys. The ndarrays
        defining each adverse outcome prediction are then shaped
        by [time, batch element, unsqueezed dimension, time window dimension]
        occlusion_predictions (list of dictionaries of ndarrays): Similar to unoccluded_predictions, but produced
        via the model when a field has been occluded from the data loading
        unoccluded_timestamps (list of ndarrays): The timestamps (time axis, batch axis) of data from the
        unoccluded data loading
        current_timestamps (list of ndarrays): The timestamps (time axis, batch axis) of data from occluded data
        loading
    Returns:
        (float): A measure of how different the occlusion predictions are from unoccluded predictions
    """
    for unocc_timestamp_batch, curr_timestamp_batch in zip(unoccluded_timestamps, current_timestamps):
        if not np.all(unocc_timestamp_batch == curr_timestamp_batch):
            raise ValueError("Timestamps do not match")

    tot_diff = 0.0
    for unoccluded_prediction_set, occlusion_prediction_set in zip(unoccluded_predictions, occlusion_predictions):
        for key in unoccluded_prediction_set.keys():
            tot_diff += _cross_entropy(unoccluded_prediction_set[key], occlusion_prediction_set[key])

    return tot_diff


def run_occlusion_analysis_all_fields(
    session, batch_generator, batch, tensor_name_json_path, predictions_tensors_json_path, output_path=None
):
    """Iterate through full passes of a batch generator a number of times equal to the number of fields in a batch
    generator's presence_map, occluding a different field each time,
    and comparing a model's output predictions to the unoccluded case each time. Save a jsonl of output comparisons
    Args:
        session (tf.MonitoredTrainingSession): A tensorflow MonitoredTrainingSession which has ingested a checkpoint
        directory containing a trained model with graph structure as defined
        in aki_predictions.training.multiple_adverse_outcomes_training.py
        batch_generator (JsonlBatchGenerator): A batch generator with a batch property, a named-tuple and dictionary
        structure containing tensors as defined in aki_predictions.training.multiple_adverse_outcomes_training.py
        batch (ehr_prediction_modeling.utils.batches.TFBatch): A batch pre-drawn from batch_generator
        tensor_name_json_path (pathlib Path): Path to a json file defining a dictionary of placeholder tensor names
        defined in the model's graph that can be used to define a feed dict for the predictions tensors
        predictions_tensors_json_path (pathlib Path): Path to a json file defining a dictionary of tensor names that
        represent the output predictions of the model without any optimization steps
        output_path (pathlib Path): Path to a json location to append occlusion measurements as they are made
    Returns:
        (dict): The occlusion measurements, un-normalized. An 'unoccluded' key corresponds to the unoccluded-unoccluded
        measurement (which will not be zero when using cross-entropy)
    """
    if output_path is not None:
        output_path.parent.mkdir(exist_ok=True, parents=True)

    # get ndarrays out of the batch
    inference_input = {"batch": batch}

    # build dictionaries of tensors from the session graph needed for feeding to the model and predictions
    inference_batch_tensor_dict, inference_predictions_dict = recover_tensor_dicts(
        session.graph, tensor_name_json_path, predictions_tensors_json_path
    )
    unoccluded_predictions = []
    unoccluded_timestamps = []

    presence_map_keys = list(batch_generator.presence_map.keys())
    context_map_keys = []
    for val in batch_generator.context_map.values():
        for idx in val:
            context_map_keys.append(idx)

    # first run unoccluded, then context, then sequence
    occlusion_sets = {"unoccluded": [None], "context": context_map_keys, "sequence": presence_map_keys}
    per_label_cross_entropies = {key: {} for key in occlusion_sets.keys()}

    for occlusion_top_level_key, occlusion_set in occlusion_sets.items():

        for ignore_label in tqdm(occlusion_set):
            # re-initialize batch generator for each field to be occluded and run through full dataset with it removed
            batch_generator.set_occluded_field(ignore_label, occlusion_type=occlusion_top_level_key)
            session.run(batch_generator.iterator.initializer)
            current_predictions = []
            current_timestamps = []
            while True:
                # extract batches and get predictions until batch generator hits end of dataset
                try:
                    inference_input_np = session.run(inference_input)
                    values_dict = {
                        "context": inference_input_np["batch"].context,
                        "sequences": inference_input_np["batch"].sequences,
                        "is_beginning_sequence": inference_input_np["batch"].is_beginning_sequence,
                    }
                    nested_feed_dict = build_nested_feed_dict(inference_batch_tensor_dict, values_dict)
                    feed_dict = flatten_feed_dict(nested_feed_dict)
                    predictions_np = session.run(inference_predictions_dict, feed_dict)
                    if ignore_label is None:
                        # on first pass build unoccluded set of predictions for later comparison
                        unoccluded_predictions.append(predictions_np)
                        unoccluded_timestamps.append(inference_input_np["batch"].sequences["timestamp"])
                    else:
                        # on latter passes build occluded predictions
                        current_predictions.append(predictions_np)
                        current_timestamps.append(inference_input_np["batch"].sequences["timestamp"])
                except tf.errors.OutOfRangeError:
                    if ignore_label is None:
                        # on first pass prepare to get self-self cross-entropy for measurement normalisations elsewhere
                        per_label_cross_entropy_key = "unoccluded"
                        current_predictions = unoccluded_predictions
                        current_timestamps = unoccluded_timestamps
                    else:
                        per_label_cross_entropy_key = ignore_label

                    # store each measurement in a dictionary for return
                    per_label_cross_entropies[occlusion_top_level_key][
                        per_label_cross_entropy_key
                    ] = measure_occlusion_difference(
                        unoccluded_predictions, current_predictions, unoccluded_timestamps, current_timestamps
                    )

                    if output_path is not None:
                        # append measurement to a jsonl for later processing, prepending the occlusion top level key
                        # e.g. unoccluded, context, sequence to the occluded key
                        jsonl_key = f"{occlusion_top_level_key}_{per_label_cross_entropy_key}"
                        file_operations.append_jsonl(
                            output_path,
                            {
                                jsonl_key: per_label_cross_entropies[occlusion_top_level_key][
                                    per_label_cross_entropy_key
                                ]
                            },
                        )
                    break
    return per_label_cross_entropies


def normalise_cross_entropies(cross_entropy_dict, round_decimals=10):
    """Normalise the cross entropy values within the provided dictionary output.

    Assumed to utilise saved output from run_occlusion_analysis_all_fields

    Args:
        cross_entropy_dict (dict): dictionary of feature indexes vs cross entropy values (including `unoccluded` entry)

    Returns:
        (dict): dictionary of cross entropy values, normalised using the unoccluded value (for easier plotting)
    """
    baseline = cross_entropy_dict["unoccluded_unoccluded"]
    output = {key: round(abs(value - baseline), round_decimals) for (key, value) in cross_entropy_dict.items()}
    return output
