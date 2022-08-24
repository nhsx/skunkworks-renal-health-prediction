import tensorflow.compat.v1 as tf
import numpy as np

from aki_predictions.ehr_prediction_modeling.utils.occlusion_utils import (
    recover_tensor_dicts,
    build_nested_feed_dict,
    flatten_feed_dict,
)
from aki_predictions import file_operations


def extract_record_indices(context):
    """Extract unique record indices from batch context.

    Args:
        context (list): list of arrays containing the sequence record numbers corresponding to the batch predictions.
            for example [array(128, 128, 1), ... (for number of batches)]

    Returns:
        (list): list of unique indices populated in the record context.
    """
    record_indices = []
    for context_set in context:
        for context_seq in context_set:
            for context_val in context_seq:
                if context_val != b"":
                    if context_val not in record_indices:
                        record_indices.append(context_val)
    return record_indices


def run_inference(
    session, batch_generator, batch, tensor_name_json_path, predictions_tensors_json_path, output_path=None
):
    """Run data batches through the model and collect inference predictions.

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
        None (Saves out json dictionry containing predicitons for each outcome at each time interval and timestep.)
    """
    if output_path is not None:
        output_path.parent.mkdir(exist_ok=True, parents=True)

    # get ndarrays out of the batch
    inference_input = {"batch": batch}

    inference_batch_tensor_dict, inference_predictions_dict = recover_tensor_dicts(
        session.graph, tensor_name_json_path, predictions_tensors_json_path
    )

    session.run(batch_generator.iterator.initializer)

    context = []
    predictions = []
    timestamps = []

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
            context.append(inference_input_np["batch"].context["record_number"])
            predictions.append(predictions_np)
            timestamps.append(inference_input_np["batch"].sequences["timestamp"])
        except tf.errors.OutOfRangeError:
            break

    # Determine all records present (should only be one)
    record_indices = extract_record_indices(context)
    assert len(record_indices) == 1, "Only single patient spell allowed for inference."

    # Ideally define this more dynamically. Assumes 8 time intervals.
    intervals = [6, 12, 18, 24, 30, 36, 42, 48]
    output = {
        "MortalityOutcome": {interval: {"values": [], "timestamps": []} for interval in intervals},
        "DialysisOutcome": {interval: {"values": [], "timestamps": []} for interval in intervals},
        "ITUOutcome": {interval: {"values": [], "timestamps": []} for interval in intervals},
    }

    # Process through batches and aggregate predictions for provided spell.
    # Assumes single patient provided.
    for record_context, predictions_set, timestamps in zip(context, predictions, timestamps):
        # Assess current batch
        record_mask = np.where(record_context, record_context == record_indices[0], False)

        # Note: value axis identity: ["time_dim", "batch_dim", "unused", "time_interval"]
        for key, value in predictions_set.items():
            for interval_index, interval in enumerate(intervals):
                # for each interval
                print(f"interval {interval}h, index {interval_index}")
                # Mask by record number (remove empty entries)
                masked_timestamps = timestamps[:, :, 0][record_mask.astype(bool)]
                masked_predictions = value[:, :, 0, interval_index][record_mask.astype(bool)]

                # Append values and timestamps to structure
                for j, (pred_val, time_val) in enumerate(zip(masked_predictions, masked_timestamps)):
                    # Cut off appending values if time value is zero.
                    # Allows first time value to be zero.
                    if j == 0:
                        output[key][interval]["values"].append(float(pred_val))
                        output[key][interval]["timestamps"].append(int(time_val))
                    if j > 0 and time_val != 0:
                        output[key][interval]["values"].append(float(pred_val))
                        output[key][interval]["timestamps"].append(int(time_val))

    # Normalise timestamps
    for key in output.keys():
        for interval_index, interval in enumerate(intervals):
            reference = output[key][interval]["timestamps"][0]
            output[key][interval]["timestamps"] = [val - reference for val in output[key][interval]["timestamps"]]

    # Save predictions output
    file_operations.save_dictionary_json(output_path, output)
