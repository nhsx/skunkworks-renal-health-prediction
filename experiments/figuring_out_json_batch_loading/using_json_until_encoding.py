"""
This script provides a more easily debuggable replication of what occurs in
aki_predictions/ehr_prediction_modeling/experiment.py by refactoring out necessary parts of the code and avoiding
tensorflow processes that disallow eager execution. Eager execution allows us to look at the structure of tensors
throughout their ingestion more clearly.

This script recreates steps from DeepMind's example, but replaces their data loader
with code from a new json dataloader, in order to investigate how the tensors mutate
as they pass steps of our new dataloader and through stages of the remaining DeepMind pipeline.
Much of the new code has been refactored from tf_dataset.JsonDMBatchGenerator to avoid dependence on
class properties, and further updates have been made in that class, so this no longer matches up closely.
However, this may serve as a good example script for debugging the building of another json-ingesting loader
that will need to deal with our custom json structure. Steps follow up until a batch is encoded by
a deep encoder, after which it was found that test_experiment could proceed with the new batch
loader to completion.
"""

import pickle
from pathlib import Path
from aki_predictions.ehr_prediction_modeling.data import tf_dataset_utils
import json
import os
import copy

from aki_predictions.ehr_prediction_modeling import config as experiment_config
from aki_predictions.ehr_prediction_modeling import embeddings
from aki_predictions.ehr_prediction_modeling import encoder_module_base
from aki_predictions.ehr_prediction_modeling import losses
from aki_predictions.ehr_prediction_modeling import types
from aki_predictions.ehr_prediction_modeling.data import tf_dataset
from aki_predictions.ehr_prediction_modeling.eval import metrics_coordinator as metrics
from aki_predictions.ehr_prediction_modeling.models import model_utils
from aki_predictions.ehr_prediction_modeling.models import rnn_model
from aki_predictions.ehr_prediction_modeling.tasks import coordinator
import tensorflow.compat.v1 as tf
import numpy as np

tf.enable_eager_execution()


def get_checkpoint_dir(config, mode):
    """Return checkpoint directory."""
    ttl = "ttl=%sd" % config.ttl
    return os.path.join(config.checkpoint_dir, "checkpoints", ttl, mode)


def get_task_from_config(config, task_name):
    """Returns an instantiated Task based on the provided task name."""
    if task_name not in config.task_configs:
        raise ValueError(
            "Task %s is not present in the list of task configurations: %s." % (task_name, config.task_configs.keys())
        )
    task_config = config.task_configs[task_name]

    if task_config.task_type not in experiment_config.TASK_MAPPING:
        raise ValueError("config.tasks.type unknown: %s" % task_config.task_type)

    task = experiment_config.TASK_MAPPING[task_config.task_type](task_config)
    return task


def get_task_coordinator(config):
    """Return task coordinator object."""
    task_list = [get_task_from_config(config, task_name) for task_name in config.tasks]
    return coordinator.Coordinator(task_list, optimizer_config=config.get("optimizer"))


config = experiment_config.get_config()
for key in config.encoder.ndim_dict.keys():
    # increment ndims to deal with updating maps to never map to 0
    config.encoder.ndim_dict[key] = config.encoder.ndim_dict[key] + 1

task_coordinator = get_task_coordinator(config)

aki_pred_dir = Path(r"C:\Projects\healthcare\aki-predictions")

fake_data_dir = aki_pred_dir / "aki_predictions" / "ehr_prediction_modeling" / "fake_data"

feature_category_map_path = fake_data_dir / "feature_category_map.pkl"
features_ids_per_category_map_path = fake_data_dir / "features_ids_per_category_map.pkl"
presence_map_path = fake_data_dir / "presence_map.pkl"
numerical_map_path = fake_data_dir / "numerical_map.pkl"

tf_records_path = str(
    aki_pred_dir / r"aki_predictions/ehr_prediction_modeling/fake_data/standardize" / "train.tfrecords"
)
tf_records_unit_range_path = str(Path(tf_records_path).parents[1] / "unit_range" / "train.tfrecords")

json_records_path = aki_pred_dir / r"aki_predictions/ehr_prediction_modeling/fake_data" / "fake_patients.json"

adverse_outcome_label_keys = [
    "adverse_outcome_within_12h",
    "adverse_outcome_within_18h",
    "adverse_outcome_within_24h",
    "adverse_outcome_within_36h",
    "adverse_outcome_within_48h",
    "adverse_outcome_within_60h",
    "adverse_outcome_within_6h",
    "adverse_outcome_within_72h",
]

with open(feature_category_map_path, "rb") as f:
    feature_category_map = pickle.load(f)
max_val = 0
for key, item in feature_category_map.items():
    if item > max_val:
        max_val = item
feature_category_map["1"] = max_val + 1

with open(features_ids_per_category_map_path, "rb") as f:
    features_ids_per_category_map = pickle.load(f)

inverse_features_ids_per_category_map = {}
for key in features_ids_per_category_map.keys():
    for item in features_ids_per_category_map[key]:
        inverse_features_ids_per_category_map[item] = key

with open(presence_map_path, "rb") as f:
    presence_map = pickle.load(f)
max_val = 0  # any of these maps mapping to a numerical 0 leads to problems with numpy to tensor to sparse tensor conversions where 0 values get omitted in sparse tensors and later on the number of values and indices in values and indexes sparse tensors need to match
for key, item in presence_map.items():
    if item > max_val:
        max_val = item
presence_map["0"] = max_val + 1

with open(numerical_map_path, "rb") as f:
    numerical_map = pickle.load(f)
max_val = 0
for key, item in numerical_map.items():
    if item > max_val:
        max_val = item
numerical_map["104"] = max_val + 1

with open(json_records_path, "rb") as f:
    dataset_list = json.load(f)["patients"]

extra_keys = ["delta_time", "episode_id", "ignore_label", "segment_mask", "timestamp"]
# define keys for which a lookiup within episode 'labels is needed
adverse_outcome_keys = [
    "adverse_outcome_within_12h",
    "adverse_outcome_within_18h",
    "adverse_outcome_within_24h",
    "adverse_outcome_within_36h",
    "adverse_outcome_within_48h",
    "adverse_outcome_within_60h",
    "adverse_outcome_within_6h",
    "adverse_outcome_within_72h",
]
adverse_outcome_keys_dtypes = [float if "lab_" in key else int for key in adverse_outcome_keys]
label_keys = [
    "lab_42_in_12h",
    "lab_42_in_18h",
    "lab_42_in_24h",
    "lab_42_in_36h",
    "lab_42_in_48h",
    "lab_42_in_60h",
    "lab_42_in_6h",
    "lab_42_in_72h",
    "lab_43_in_12h",
    "lab_43_in_18h",
    "lab_43_in_24h",
    "lab_43_in_36h",
    "lab_43_in_48h",
    "lab_43_in_60h",
    "lab_43_in_6h",
    "lab_43_in_72h",
    "lab_44_in_12h",
    "lab_44_in_18h",
    "lab_44_in_24h",
    "lab_44_in_36h",
    "lab_44_in_48h",
    "lab_44_in_60h",
    "lab_44_in_6h",
    "lab_44_in_72h",
    "lab_45_in_12h",
    "lab_45_in_18h",
    "lab_45_in_24h",
    "lab_45_in_36h",
    "lab_45_in_48h",
    "lab_45_in_60h",
    "lab_45_in_6h",
    "lab_45_in_72h",
    "lab_46_in_12h",
    "lab_46_in_18h",
    "lab_46_in_24h",
    "lab_46_in_36h",
    "lab_46_in_48h",
    "lab_46_in_60h",
    "lab_46_in_6h",
    "lab_46_in_72h",
]
tensor_to_sparse_keys = [
    "indexes_category_counts",
    "indexes_numeric",
    "indexes_presence",
    "values_category_counts",
    "values_numeric",
    "values_presence",
]

label_keys_dtypes = [float if "lab_" in key else int for key in label_keys]
last_dim_1_keys = adverse_outcome_keys + label_keys + extra_keys

ctx, seq = tf_dataset_utils.get_label_dicts(
    task_coordinator, config.data.context_features, config.data.sequential_features
)


def _severity_lookup(feature_idx, feature_value):
    """Get a normal (1), low (2), high (3), very low (4) or very high (5) severity score based on a particular
    feature index and its current value

    Args:
        feature_idx (str): A possible value that can be taken by an entry in a list of entries in an event in a
        patient episode, e.g. present in one of the items in the feature_category_map dictionary (not a
        feature_category_idx)
        feature_value (numeric): The current value taken for the feature_idx in the entry

    Returns:
        (int): A score in the set [1-5] for how severe a value is,
        normal (1), low (2), high (3), very low (4) or very high (5)

    """
    return 1.0  # haven't seen anything else in deepmind tensors yet


def _build_seq_tensor(patient, sequence_length):
    sequences = {}
    label_sequences = {label_key: np.zeros(shape=[1, sequence_length, 1], dtype=np.float32) for label_key in label_keys}
    adverse_outcome_sequences = {
        label_key: np.zeros(shape=[1, sequence_length, 1], dtype=np.int64) for label_key in adverse_outcome_keys
    }
    segment_mask = np.zeros(shape=[1, sequence_length, 1], dtype=np.int64)
    episode_id = np.zeros(shape=[1, sequence_length, 1], dtype=np.str)
    timestamp = np.zeros(shape=[1, sequence_length, 1], dtype=np.int64)
    delta_time = np.zeros(shape=[1, sequence_length, 1], dtype=np.int64)
    ignore_label = np.zeros(shape=[1, sequence_length, 1], dtype=np.int64)
    entry_change_flag = 9.0
    running_event_total = 0
    indexes_category_counts = {"values": [], "indices": [], "dense_shape": []}
    indexes_numeric = {"values": [], "indices": [], "dense_shape": []}
    indexes_presence = {"values": [], "indices": [], "dense_shape": []}
    values_category_counts = {"values": [], "indices": [], "dense_shape": []}
    values_numeric = {"values": [], "indices": [], "dense_shape": []}
    values_presence = {"values": [], "indices": [], "dense_shape": []}
    for episode_i, episode in enumerate(patient["episodes"]):
        if "admission" in episode:
            event_type = "admission"
        elif "outpatient_events" in episode:
            event_type = "outpatient_events"
        else:
            continue

        clinical_events = episode[event_type]["clinical_events"]

        curr_episode_id = episode["episode_id"]

        for event in clinical_events:

            if event_type is "outpatient_events" or (event["time_of_day"] is "0"):
                ignore_label[0, running_event_total, 0] = 1

            # time_of_day = int(event['time_of_day'])
            for label_key, label_key_dtype in zip(label_keys, label_keys_dtypes):
                curr_label_val = event["labels"].get(label_key, None)
                if curr_label_val is not None:
                    label_sequences[label_key][0, running_event_total, 0] = label_key_dtype(curr_label_val)
            for label_key, label_key_dtype in zip(adverse_outcome_keys, adverse_outcome_keys_dtypes):
                curr_label_val = event["labels"].get(label_key, None)
                if curr_label_val is not None:
                    adverse_outcome_sequences[label_key][0, running_event_total, 0] = label_key_dtype(curr_label_val)
            curr_segment_mask_val = event["labels"].get("segment_mask", None)
            segment_mask[0, running_event_total, 0] = curr_segment_mask_val

            episode_id[0, running_event_total, 0] = curr_episode_id
            timestamp[0, running_event_total, 0] = (
                int(event["patient_age"]) * 60 * 60 * 24 + (int(event["time_of_day"]) - 1) * 6 * 60 * 60
            )

            if "entries" in event:
                # categories = []
                categories = {}
                running_entries_total = 0
                running_numeric_entries_total = 0
                # raise ValueError('replace categories with dict of number of encounters, to deal with values_category_coutns?')
                if running_entries_total < config.encoder.ndim_dict["count_s"] - 1:
                    if len(event["entries"]) > 1:
                        indexes_category_counts["values"].append(entry_change_flag)
                        indexes_category_counts["indices"].append([0, running_event_total, running_entries_total])

                        values_category_counts["values"].append(entry_change_flag)
                        values_category_counts["indices"].append([0, running_event_total, running_entries_total])
                        running_entries_total += 1

                for entry_i, entry in enumerate(event["entries"]):
                    if entry["feature_category_idx"] in categories:
                        categories[entry["feature_category_idx"]] += 1
                    else:
                        categories[entry["feature_category_idx"]] = 1
                    if entry_i < config.encoder.ndim_dict["pres_s"] - 1:
                        indexes_presence["values"].append(presence_map[entry["feature_idx"]])
                        indexes_presence["indices"].append([0, running_event_total, entry_i])
                        try:
                            values_presence["values"].append(
                                _severity_lookup(entry["feature_category_idx"], float(entry["feature_value"]))
                            )
                        except:
                            values_presence["values"].append(
                                entry_change_flag
                            )  # ehr-predictions readme states a binary flag exists for whether a value was actually measured, this is only place I can think it makes sense to put that except maybe ignore_label or segment_mask

                        values_presence["indices"].append([0, running_event_total, entry_i])

                    if entry["feature_idx"] in numerical_map:
                        if running_numeric_entries_total < config.encoder.ndim_dict["num_s"] - 1:
                            try:
                                values_numeric["values"].append(float(entry["feature_value"]))
                            except:
                                values_numeric["values"].append(
                                    entry_change_flag
                                )  # ehr-predictions readme states 0 is the numerical imputed value for missing entries, but converting from numpy to tensor to sparse loses 0 values
                            values_numeric["indices"].append([0, running_event_total, running_numeric_entries_total])

                            indexes_numeric["values"].append(numerical_map[entry["feature_idx"]])
                            indexes_numeric["indices"].append([0, running_event_total, running_numeric_entries_total])

                            running_numeric_entries_total += 1
                if running_entries_total < config.encoder.ndim_dict["count_s"] - 1:
                    for cat in categories.keys():
                        indexes_category_counts["values"].append(float(feature_category_map[cat]))
                        indexes_category_counts["indices"].append([0, running_event_total, running_entries_total])

                        values_category_counts["values"].append(categories[cat])
                        values_category_counts["indices"].append([0, running_event_total, running_entries_total])
                        running_entries_total += 1
            running_event_total += 1

    for key in sequences.keys():
        # indexes_numeric and values_numeric can sometimes come through empty,
        # but this leads to empty tensors and errors in tf
        if len(sequences[key]["values"]) == 0:
            sequences[key]["values"].append(entry_change_flag)
            sequences[key]["indices"].append([0, 0, 0])

    sequences["indexes_category_counts"] = indexes_category_counts
    sequences["indexes_numeric"] = indexes_numeric
    sequences["indexes_presence"] = indexes_presence
    sequences["values_category_counts"] = values_category_counts
    sequences["values_numeric"] = values_numeric
    sequences["values_presence"] = values_presence

    for key in sequences.keys():
        sequences[key]["values"] = np.array(sequences[key]["values"], dtype=seq[key].dtype.as_numpy_dtype)
        sequences[key]["indices"] = np.array(sequences[key]["indices"], dtype=np.int64)

        if len(sequences[key]["indices"]) == 0:
            sequences[key]["dense_shape"] = None
            expected_dims = 3
        else:
            sequences[key]["dense_shape"] = np.max(sequences[key]["indices"], axis=0) + 1
            expected_dims = None

        sequences[key] = tf_dataset_utils.ndarray_from_sparse_definition(
            sequences[key]["indices"],
            sequences[key]["values"],
            sequences[key]["dense_shape"],
            expected_dims=expected_dims,
        )

    sequences.update(adverse_outcome_sequences)

    delta_time[0, 1:, 0] = np.diff(timestamp[0, :, 0])
    sequences["delta_time"] = delta_time

    sequences["episode_id"] = episode_id

    sequences["ignore_label"] = ignore_label

    sequences.update(label_sequences)

    sequences["segment_mask"] = segment_mask

    sequences["timestamp"] = timestamp

    for key in sequences.keys():
        # remove batch dimension because it seems like we don't actually want batch dimension yet
        sequences[key] = np.squeeze(sequences[key], axis=0)

    return sequences


def get_tensors_from_single_patient(patient):
    context = {"record_number": np.array(patient["record_number"], np.str_)}
    sequence_length = 0
    for episode in patient["episodes"]:
        if "admission" in episode:
            sequence_length += len(episode["admission"]["clinical_events"])
        elif "outpatient_events" in episode:
            sequence_length += len(episode["outpatient_events"]["clinical_events"])
    sequence_lengths = np.array(sequence_length, dtype=np.int64)
    sequences = _build_seq_tensor(patient, sequence_length)
    out_dict = {"context": context, "sequences": sequences, "sequence_lengths": sequence_lengths}

    return out_dict


def get_dataset_output_dict_tensor_types_and_shapes_and_empty_batch(tensor_dict):
    output_types = {}
    output_shapes = {}
    empty_batch = {}
    for key, item in tensor_dict.items():
        if isinstance(item, dict):
            (
                output_types[key],
                output_shapes[key],
                empty_batch[key],
            ) = get_dataset_output_dict_tensor_types_and_shapes_and_empty_batch(item)
        else:
            output_types[key] = item.dtype
            if key in seq and isinstance(seq[key], tf.FixedLenSequenceFeature):
                tensor_shape_input = [None for i in range(len(item.shape) - 1)] + [seq[key].shape[0]]
                output_shapes[key] = tf.TensorShape(tensor_shape_input)
            elif key in ctx or key == "sequence_lengths":
                output_shapes[key] = tf.TensorShape([])
            else:
                tensor_shape_input = [None for i in range(len(item.shape))]
                output_shapes[key] = tf.TensorShape(tensor_shape_input)
            empty_batch[key] = []
    return output_types, output_shapes, empty_batch


single_tensor_dict = get_tensors_from_single_patient(dataset_list[0])
output_types, output_shapes, empty_batch = get_dataset_output_dict_tensor_types_and_shapes_and_empty_batch(
    single_tensor_dict
)


def data_generator():
    data_gen_i = 0

    while True:
        elem = dataset_list[data_gen_i]
        data_gen_i += 1
        if data_gen_i == len(dataset_list):
            data_gen_i = 0
        yield get_tensors_from_single_patient(elem)


def create_batch_generator(gen, batch_size):
    def batch_generator():
        while True:
            for i, datum in enumerate(gen()):
                if i % batch_size == 0:
                    batch = copy.deepcopy(empty_batch)
                tf_dataset_utils.append_nested_dict_of_lists(batch, datum)
                if (i + 1) % batch_size == 0:
                    break
            tf_dataset_utils.pad_nested_dict_of_lists_of_ndarray(batch)
            for i in range(len(batch["context"]["record_number"])):
                batch_element = tf_dataset_utils.extract_ndarray_at_idx_from_nested_dict_of_lists_of_ndarrays(batch, i)
                yield batch_element

    return batch_generator


batch_generator = create_batch_generator(data_generator, batch_size=5)
dataset = tf.data.Dataset.from_generator(batch_generator, output_types=output_types, output_shapes=output_shapes)

dataset = tf_dataset.transform_dataset(
    dataset,
    task_coordinator=task_coordinator,
    parse_cycle_length=128,
    context_features=config.data.context_features,
    sequential_features=config.data.sequential_features,
    batch_size=128,
    num_unroll=128,
    segment_length=None,
    num_prefetch=16,
    shuffle=False,
    bypass_seqex_to_dict=True,
    tensor_to_sparse_keys=tensor_to_sparse_keys,
)

embedding_classes = {
    types.EmbeddingType.LOOKUP: embeddings.BasicEmbeddingLookup,
    types.EmbeddingType.DEEP_EMBEDDING: embeddings.DeepEmbedding,
}
encoder = encoder_module_base.EncoderModule(config.encoder, embedding_classes)

elements = []
for element in dataset.take(1):
    elements.append(element)
assert True

for element in elements:
    features, time_vect = encoder.embed_batch(element)
assert True
