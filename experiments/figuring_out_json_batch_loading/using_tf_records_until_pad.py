"""
This script provides a more easily debuggable replication of what occurs in
aki_predictions/ehr_prediction_modeling/experiment.py by refactoring out necessary parts of the code and avoiding
tensorflow processes that disallow eager execution. Eager execution allows us to look at the structure of tensors
throughout their ingestion more clearly.

This script recreates steps from DeepMind's example using their tfrecords dataset, up until the dataset
is padded and reshaped, which usually occurs inside tf_dataset.transform_dataset
"""

import pickle
from pathlib import Path
from aki_predictions.ehr_prediction_modeling.data import tf_dataset_utils
import json
import os

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

with open(features_ids_per_category_map_path, "rb") as f:
    features_ids_per_category_map = pickle.load(f)

inverse_features_ids_per_category_map = {}
for key in features_ids_per_category_map.keys():
    for item in features_ids_per_category_map[key]:
        inverse_features_ids_per_category_map[item] = key

with open(presence_map_path, "rb") as f:
    presence_map = pickle.load(f)

with open(numerical_map_path, "rb") as f:
    numerical_map = pickle.load(f)

tf_records_dataset = tf_dataset_utils.read_seqex_dataset(tf_records_path)
tf_unit_ranges = tf_dataset_utils.read_seqex_dataset(tf_records_unit_range_path)


with open(json_records_path, "rb") as f:
    json_records_list = json.load(f)["patients"]


ctx, seq = tf_dataset_utils.get_label_dicts(
    task_coordinator, config.data.context_features, config.data.sequential_features
)

dataset = tf_records_dataset.batch(128, drop_remainder=False)
# Parallelize the parse call
seqex_to_dict = lambda x: tf_dataset_utils.seqex_to_dict(x, ctx, seq)
dataset = dataset.map(seqex_to_dict, num_parallel_calls=128)
dataset = dataset.map(tf_dataset_utils.convert_to_time_major, num_parallel_calls=128)

elements = []
for element in dataset.take(1):
    elements.append(element)

pad_elements = []
for el in dataset.take(1):
    pad_el = tf_dataset_utils.add_beginning_of_sequence_mask_and_pad(el, 128, 128)
assert True
