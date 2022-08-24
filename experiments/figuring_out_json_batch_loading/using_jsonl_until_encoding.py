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
import jsonlines

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
config.encoder.ndim_dict = {"pres_s": 10224, "num_s": 15, "count_s": 21}

# for key in config.encoder.ndim_dict.keys():
#     # increment ndims to deal with updating maps to never map to 0
#     config.encoder.ndim_dict[key] = config.encoder.ndim_dict[key] + 1

task_coordinator = get_task_coordinator(config)

aki_pred_dir = Path(r"C:\Projects\healthcare\aki-predictions")

fake_data_dir = aki_pred_dir / "aki_predictions" / "ehr_prediction_modeling" / "fake_data"

feature_category_map_path = fake_data_dir / "feature_category_map.pkl"
features_ids_per_category_map_path = fake_data_dir / "features_ids_per_category_map.pkl"
presence_map_path = fake_data_dir / "presence_map.pkl"
numerical_map_path = fake_data_dir / "numerical_map.pkl"

records_dir = Path(__file__).parents[2] / "data"

json_records_path = records_dir / "data_ingest_index_testing_full_ingest_records_output_lines.jsonl"

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
extra_keys = ["delta_time", "episode_id", "ignore_label", "segment_mask", "timestamp"]
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

mappings_dir = Path(__file__).parents[2] / "data"
feature_names_to_categories_path = mappings_dir / "data_ingest_index_testing_full_ingest_category_mapping.json"
feature_names_to_feature_idxs_path = mappings_dir / "data_ingest_index_testing_full_ingest_feature_mapping.json"
reduced_stats_path = mappings_dir / "data_ingest_index_testing_full.json"

presence_map = {}
with open(feature_names_to_feature_idxs_path, "rb") as f:
    feature_names_to_feature_idxs = json.load(f)
for key_i, (key, item) in enumerate(feature_names_to_feature_idxs.items()):
    presence_map[str(item)] = str(key_i + 1)  # fairly dumb mapping that just avoids using 0

feature_category_map = {}
with open(feature_names_to_categories_path, "rb") as f:
    feature_names_to_categories = json.load(f)
categories = []
for val in feature_names_to_categories.values():
    categories.append(val)
categories = list(set(categories))
for cat_i, cat in enumerate(categories):
    feature_category_map[str(cat)] = str(cat_i + 1)

numerical_map = {}
with open(reduced_stats_path, "rb") as f:
    reduced_stats = json.load(f)
numerical_feature_names = reduced_stats["mapping"]["numerical_features"]
for feat_i, feat in enumerate(numerical_feature_names):
    feature_idx = feature_names_to_feature_idxs[feat]
    numerical_map[str(feature_idx)] = str(feat_i + 1)

features_ids_per_category_map = {}


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


def _prepare_empty_sequence_dict_arrays(sequence_length):
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
    return (
        sequences,
        label_sequences,
        adverse_outcome_sequences,
        segment_mask,
        episode_id,
        timestamp,
        delta_time,
        ignore_label,
        entry_change_flag,
        running_event_total,
        indexes_category_counts,
        indexes_numeric,
        indexes_presence,
        values_category_counts,
        values_numeric,
        values_presence,
    )


def _build_seq_tensor(patient, sequence_length):
    (
        sequences,
        label_sequences,
        adverse_outcome_sequences,
        segment_mask,
        episode_id,
        timestamp,
        delta_time,
        ignore_label,
        entry_change_flag,
        running_event_total,
        indexes_category_counts,
        indexes_numeric,
        indexes_presence,
        values_category_counts,
        values_numeric,
        values_presence,
    ) = _prepare_empty_sequence_dict_arrays(sequence_length)
    for episode_i, episode in enumerate(patient["episodes"]):
        # try:
        #     # TODO: current jsonl structure does not include 'clinical events' which is a dictionary containing 'entries' which is a time-binned list of dictionaries defining features that occurred in the time bin
        clinical_events = episode["events"]
        # except KeyError:
        #     # get something equivalent from current structure that can act as expected for iterating through events
        #     clinical_events = [{'entries': episode['entries']}]
        curr_episode_id = patient["master_key"]
        for event in clinical_events:
            # try:
            #     # TODO: need some equivalent of 'time_of_day' in our events, and a flag to get the grouped events where time was unknown
            if event["time_of_day"] == "0":
                # 0 time of day flags summaries of entries that had unknown times
                ignore_label[0, running_event_total, 0] = 1
            # except KeyError:
            #     pass  # ignore_label left as 1

            # get the continuous labels (e.g. lab test numericals)
            for label_key, label_key_dtype in zip(label_keys, label_keys_dtypes):
                # try:
                #     # TODO: generate all the numeric labels looking forward in time for each event
                curr_labels = event["labels"]
                # except KeyError:
                #     curr_labels = {}
                curr_label_val = curr_labels.get(label_key, None)
                if curr_label_val is not None:
                    # TODO: likely all label_key_dtypes here will simply be float, so could refactor if we find
                    # this is the case
                    label_sequences[label_key][0, running_event_total, 0] = label_key_dtype(curr_label_val)
                else:
                    label_sequences[label_key][0, running_event_total, 0] = 0.0
            # get the binary adverse outcome labels
            for label_key, label_key_dtype in zip(adverse_outcome_keys, adverse_outcome_keys_dtypes):
                # try:
                #     # TODO: generate all the binary labels looking forward in time for each event
                curr_labels = event["labels"]
                # except KeyError:
                #     curr_labels = {}
                curr_label_val = curr_labels.get(label_key, None)
                if curr_label_val is not None:
                    # TODO: likely all label_key_dtypes here will be float, refactor if we find this is the case
                    adverse_outcome_sequences[label_key][0, running_event_total, 0] = label_key_dtype(curr_label_val)
                else:
                    adverse_outcome_sequences[label_key][0, running_event_total, 0] = 0.0
            try:
                # TODO: make sure segment_mask values actually exist
                curr_segment_mask_val = event["labels"]["segment_mask"]
            except KeyError:
                curr_segment_mask_val = "0"
            segment_mask[0, running_event_total, 0] = float(curr_segment_mask_val)

            episode_id[0, running_event_total, 0] = curr_episode_id
            try:
                # TODO: update this with whatever timestamp key is used once datetimes have been reconciled,
                # should be a total number of seconds since patient birth
                timestamp[0, running_event_total, 0] = (
                    int(event["patient_age"]) * 60 * 60 * 24
                    + (int(event["time_of_day"]) - 1) * config.model.time_bin_length
                )
            except KeyError:
                timestamp[0, running_event_total, 0] = 1

            # run through the entries and extract features
            if "entries" in event:
                _extract_sequences_from_event(
                    event,
                    entry_change_flag,
                    running_event_total,
                    indexes_category_counts,
                    values_category_counts,
                    indexes_presence,
                    values_presence,
                    indexes_numeric,
                    values_numeric,
                )

            running_event_total += 1

        sequences["indexes_category_counts"] = indexes_category_counts
        sequences["indexes_numeric"] = indexes_numeric
        sequences["indexes_presence"] = indexes_presence
        sequences["values_category_counts"] = values_category_counts
        sequences["values_numeric"] = values_numeric
        sequences["values_presence"] = values_presence

        for key in sequences.keys():
            # indexes_numeric and values_numeric can sometimes come through empty,
            # but this leads to empty tensors and errors in tf
            if len(sequences[key]["values"]) == 0:
                sequences[key]["values"].append(entry_change_flag)
                sequences[key]["indices"].append([0, 0, 0])

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
            # Remove batch dimension because after switching to a batch generator that takes in a regular generator
            # it turns out that tensorflow don't want the batch dimension to exist yet
            sequences[key] = np.squeeze(sequences[key], axis=0)

    return sequences


def _extract_sequences_from_event(
    event,
    entry_change_flag,
    running_event_total,
    indexes_category_counts,
    values_category_counts,
    indexes_presence,
    values_presence,
    indexes_numeric,
    values_numeric,
):
    categories = {}
    running_entries_total = 0
    running_numeric_entries_total = 0
    # if running_entries_total < config.encoder.ndim_dict['count_s'] - 1:
    # passed test without this if statement but I'm suspicious
    # that it might need to go back in for very large episodes
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

        indexes_presence["values"].append(presence_map[entry["feature_idx"]])
        indexes_presence["indices"].append([0, running_event_total, entry_i])
        try:
            values_presence["values"].append(
                _severity_lookup(entry["feature_category_idx"], float(entry["feature_value"]))
            )
        except ValueError:
            # ehr-predictions readme states 0 is the numerical imputed value for missing entries
            # but this leads to errors down the line with sparse tensor conversion
            values_presence["values"].append(entry_change_flag)

        values_presence["indices"].append([0, running_event_total, entry_i])

        if entry["feature_idx"] in numerical_map:

            try:
                feat_val = float(entry["feature_value"])
                if feat_val == 0.0:
                    values_numeric["values"].append(
                        -0.1
                    )  # hopefully a reasonable imputation for 0 values that break sparse to dense tensors
                else:
                    values_numeric["values"].append(feat_val)
            except ValueError:
                # ehr-predictions readme states 0 is the numerical imputed value for missing entries
                # but this leads to errors down the line with sparse tensor conversion
                values_numeric["values"].append(entry_change_flag)
            values_numeric["indices"].append([0, running_event_total, running_numeric_entries_total])

            indexes_numeric["values"].append(numerical_map[entry["feature_idx"]])
            indexes_numeric["indices"].append([0, running_event_total, running_numeric_entries_total])
            running_numeric_entries_total += 1
    for cat in categories.keys():
        indexes_category_counts["values"].append(float(feature_category_map[cat]))
        indexes_category_counts["indices"].append([0, running_event_total, running_entries_total])

        values_category_counts["values"].append(categories[cat])
        values_category_counts["indices"].append([0, running_event_total, running_entries_total])
        running_entries_total += 1


def get_tensors_from_single_patient(patient):
    """Build a dictionary of ndarrays to later be converted to tensors for a single patient

    Args:
        patient (dict): A nested dictionary defining a patient's medical history

    Returns:
        (dict): A nested dictionary defining context, sequence and sequence_lengths

    """
    context = {"record_number": np.array(patient["record_number"], np.str_)}
    sequence_length = 0
    for episode in patient["episodes"]:
        sequence_length += len(episode["events"])
    sequence_lengths = np.array(sequence_length, dtype=np.int64)
    sequences = _build_seq_tensor(patient, sequence_length)
    out_dict = {"context": context, "sequences": sequences, "sequence_lengths": sequence_lengths}

    return out_dict


def _read_record_files(records_filename):
    with jsonlines.open(records_filename) as reader:
        single_line = reader.read()
    single_tensor_dict = get_tensors_from_single_patient(single_line)
    output_types, output_shapes, empty_batch = get_dataset_output_dict_tensor_types_and_shapes_and_empty_batch(
        single_tensor_dict
    )

    def data_generator(start_i=0):

        with jsonlines.open(records_filename) as reader:
            # would like to be able to randomly select lines, figure this out later
            while True:
                line = reader.read()
                yield get_tensors_from_single_patient(line)

    # for elem in data_generator():
    #     pass
    #     print('another')

    def create_batch_generator(gen, batch_size):
        def batch_gen():
            start_i = 0
            while True:
                for i, datum in enumerate(gen(start_i=start_i)):
                    if i % batch_size == 0:
                        batch = copy.deepcopy(empty_batch)
                    tf_dataset_utils.append_nested_dict_of_lists(batch, datum)
                    if (i + 1) % batch_size == 0:
                        break
                tf_dataset_utils.pad_nested_dict_of_lists_of_ndarray(batch)
                for i in range(len(batch["context"]["record_number"])):
                    batch_element = tf_dataset_utils.extract_ndarray_at_idx_from_nested_dict_of_lists_of_ndarrays(
                        batch, i
                    )
                    yield batch_element
                start_i += batch_size

        return batch_gen

    batch_generator = create_batch_generator(data_generator, batch_size=config.data.batch_size)

    dataset = tf.data.Dataset.from_generator(batch_generator, output_types=output_types, output_shapes=output_shapes)
    return dataset


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


# batch_generator = create_batch_generator(data_generator, batch_size=5)
# dataset = tf.data.Dataset.from_generator(batch_generator, output_types=output_types, output_shapes=output_shapes)

dataset = _read_record_files(json_records_path)

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
