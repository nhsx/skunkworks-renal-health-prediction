# coding=utf-8
# MIT Licence
#
# Copyright (c) 2022 NHS England
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This project incorporates work covered by the following copyright and permission notice:
#
#     Copyright 2021 Google Health Research.
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#             http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

# Lint as: python3
"""TensorFlow utility functions to extract Tensor and batch from tf protos.
RandomBatchGenerator created by Roke Manor Research for data structure exploration purposes, BatchGenerator
refactored to allow Roke-created classes that inherit from it to overload some behaviour
"""

import os
from typing import List, Optional
import json  # Roke Manor Research
from pathlib import Path  # Roke Manor Research
import pickle  # Roke Manor Research
import copy  # Roke Manor Research
import logging

import tensorflow.compat.v1 as tf
import numpy as np  # Roke Manor Research

from aki_predictions.ehr_prediction_modeling.data import tf_dataset_utils
from aki_predictions.ehr_prediction_modeling.tasks import coordinator
from aki_predictions.ehr_prediction_modeling.utils import batches
from aki_predictions.ehr_prediction_modeling import configdict
from aki_predictions.ehr_prediction_modeling.types import TaskNames
from aki_predictions.ehr_prediction_modeling.utils import label_utils
from aki_predictions.ehr_prediction_modeling.utils import curriculum_learning_utils


logger = logging.getLogger(__name__)


class BatchGenerator(object):
    """Tool for working with a TF dataset."""

    def __init__(
        self,
        config: configdict.ConfigDict,
        is_training: bool,
        task_coordinator: Optional[coordinator.Coordinator],
        data_split_name: str,
        bypass_seqex_to_dict: bool = False,  # Roke Manor Research
        tensor_to_sparse_keys: list = None,  # Roke Manor Research
        debug_mode: bool = False,
    ):
        """Initialise BatchGenerator object."""
        self._config = config
        self._is_training = is_training
        self._task_coordinator = task_coordinator
        self._split_name = data_split_name
        self._iterator = None  # Initialized in create_batch.
        self._bypass_seqex_to_dict = bypass_seqex_to_dict
        self._tensor_to_sparse_keys = tensor_to_sparse_keys
        if not debug_mode:
            self._batch = self._create_batch()

    @property
    def iterator(self) -> tf.data.Iterator:
        """Return iterator property."""
        return self._iterator

    @property
    def batch(self) -> batches.TFBatch:
        """Return batch property."""
        return self._batch

    def _get_filename(self) -> str:
        """Return filename of data records."""
        split_to_filename = {
            "train": self._config.data.train_filename,
            "valid": self._config.data.valid_filename,
            "test": self._config.data.test_filename,
            "calib": self._config.data.calib_filename,
        }
        return os.path.join(self._config.data.records_dirpath, split_to_filename[self._split_name])

    def _read_record_files(self, records_filename):
        return tf_dataset_utils.read_seqex_dataset(records_filename)

    def _create_batch(self) -> batches.TFBatch:
        """Creates a batch of data in time-major format."""
        tfrecords_filename = self._get_filename()
        if not tfrecords_filename:
            raise ValueError(
                "No recordio found for split {} at {}.".format(repr(self._split_name), self._config.recordio_path)
            )
        num_unroll = self._config.data.num_unroll
        batch_size = self._config.data.batch_size
        segment_length = getattr(self._config.data, "segment_length", None)
        if not self._is_training:
            if segment_length is not None:
                logger.info(
                    "Splitting sequences by segment_length is only supported during "
                    "Training. Updating segment_length to None."
                )
                segment_length = None

        parallelism_config = self._config.data.padded_settings

        with tf.device("/cpu"):
            raw_dataset = self._read_record_files(tfrecords_filename)
            dataset = transform_dataset(
                raw_dataset,
                task_coordinator=self._task_coordinator,
                parse_cycle_length=parallelism_config.parse_cycle_length,
                context_features=self._config.data.context_features,
                sequential_features=self._config.data.sequential_features,
                batch_size=batch_size,
                num_unroll=num_unroll,
                segment_length=segment_length,
                num_prefetch=parallelism_config.num_prefetch,
                shuffle=self._config.data.shuffle,
                bypass_seqex_to_dict=self._bypass_seqex_to_dict,  # Roke Manor Research
                tensor_to_sparse_keys=self._tensor_to_sparse_keys,
            )

            if self._is_training:
                # Don't repeat if we are iterating the whole dataset every eval epoch.
                dataset = dataset.repeat(-1)

            self._iterator = tf.data.make_initializable_iterator(dataset)
            tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, self._iterator.initializer)
            return self._iterator.get_next()

    def _debug_prepare_pre_initialized_dataset(self):
        tfrecords_filename = self._get_filename()
        if not tfrecords_filename:
            raise ValueError(
                "No recordio found for split {} at {}.".format(repr(self._split_name), self._config.recordio_path)
            )
        num_unroll = self._config.data.num_unroll
        batch_size = self._config.data.batch_size
        segment_length = getattr(self._config.data, "segment_length", None)
        if not self._is_training:
            if segment_length is not None:
                logger.info(
                    "Splitting sequences by segment_length is only supported during "
                    "Training. Updating segment_length to None."
                )
                segment_length = None

        parallelism_config = self._config.data.padded_settings

        with tf.device("/cpu"):
            raw_dataset = self._read_record_files(tfrecords_filename)
            dataset = transform_dataset(
                raw_dataset,
                task_coordinator=self._task_coordinator,
                parse_cycle_length=parallelism_config.parse_cycle_length,
                context_features=self._config.data.context_features,
                sequential_features=self._config.data.sequential_features,
                batch_size=batch_size,
                num_unroll=num_unroll,
                segment_length=segment_length,
                num_prefetch=parallelism_config.num_prefetch,
                shuffle=self._config.data.shuffle,
                bypass_seqex_to_dict=self._bypass_seqex_to_dict,  # Roke Manor Research
                tensor_to_sparse_keys=self._tensor_to_sparse_keys,  # Roke Manor Research
                debug=True,
            )

            if self._is_training:
                # Don't repeat if we are iterating the whole dataset every eval epoch.
                dataset = dataset.repeat(-1)

            return dataset


def transform_dataset(
    dataset: tf.data.Dataset,
    task_coordinator: Optional[coordinator.Coordinator] = None,
    parse_cycle_length: int = 128,
    context_features: Optional[List[str]] = None,
    sequential_features: Optional[List[str]] = None,
    batch_size: int = 32,
    num_unroll: int = 128,
    segment_length: Optional[int] = None,
    num_prefetch: int = 16,
    shuffle: bool = False,
    bypass_seqex_to_dict=False,
    tensor_to_sparse_keys=None,
    debug=False,
) -> tf.data.Dataset:
    """Transforms the dataset format.

    Args:
      dataset: A Tensorflow Dataset to transform.
      task_coordinator: Coordinator instance with the info about tasks.
      parse_cycle_length: Number of parallel calls to parsing.
      context_features: features to add to the context.
      sequential_features: features to add to the sequence. with lists of indexes
        for each historical feature type.
      batch_size: the batch size.
      num_unroll: the fixed sequence length.
      segment_length: the fixed segments (sub-sequences) length.
      num_prefetch: Number of (batched) sequences to prefetch.
      shuffle: Whether to shuffle the data.

    Returns:
      A TF dataset with context and sequence tensors.
      Each element of this dataset is a Batch object.
    """
    if segment_length is not None and (segment_length % num_unroll) != 0:
        raise ValueError(
            "segment_length should be multiples of num_unroll, "
            "found segment_length={} and num_unroll={}.".format(segment_length, num_unroll)
        )

    if not bypass_seqex_to_dict:  # Roke Manor Research introduced this bypass
        ctx, seq = tf_dataset_utils.get_label_dicts(task_coordinator, context_features, sequential_features)

        dataset = dataset.batch(batch_size, drop_remainder=False)
        # Parallelize the parse call
        seqex_to_dict = lambda x: tf_dataset_utils.seqex_to_dict(x, ctx, seq)
        dataset = dataset.map(seqex_to_dict, num_parallel_calls=parse_cycle_length)
    else:
        mapping_lambda = lambda x: tf_dataset_utils.convert_required_tensors_to_sparse(x, tensor_to_sparse_keys)
        dataset = dataset.map(
            mapping_lambda,
            num_parallel_calls=parse_cycle_length,
        )
        dataset = dataset.batch(batch_size, drop_remainder=False)

    if segment_length is not None:
        dataset = dataset.unbatch()
        dataset = dataset.batch(1, drop_remainder=False)
        # Uncomment the if statement below to enable eager execution scripts to assist debugging.
        # if debug is True:
        #     return dataset

        # Call convert_to_segments with batch_size=1 to reduce graph size.
        mapping_lambda = lambda x: tf_dataset_utils.convert_to_segments(x, segment_length)
        dataset = dataset.map(mapping_lambda, num_parallel_calls=parse_cycle_length)
        dataset = dataset.unbatch()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 64)
        dataset = dataset.batch(batch_size, drop_remainder=False)
    else:
        dataset = dataset.unbatch()
        dataset = dataset.batch(1, drop_remainder=False)
        # Uncomment the if statement below to enable eager execution scripts to assist debugging.
        # if debug:
        #     return dataset
        mapping_lambda = lambda x: tf_dataset_utils.convert_sparse_context_to_segments(x)
        dataset = dataset.map(mapping_lambda, num_parallel_calls=parse_cycle_length)
        dataset = dataset.unbatch()
        dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.map(tf_dataset_utils.convert_to_time_major, num_parallel_calls=parse_cycle_length)
    absmp = tf_dataset_utils.add_beginning_of_sequence_mask_and_pad
    add_beginning_of_sequence_mask_and_pad = lambda x: absmp(x, batch_size, num_unroll)
    dataset = dataset.flat_map(add_beginning_of_sequence_mask_and_pad)

    # Uncomment the if statement below to enable eager execution scripts to assist debugging.
    # if debug is True:
    #     return dataset

    # # Rebatch the data by num_unroll
    dataset = dataset.unbatch()
    dataset = dataset.batch(num_unroll, drop_remainder=False)

    if debug is True:
        return dataset
    if num_prefetch != -1:
        dataset = dataset.prefetch(num_prefetch)

    return dataset


class JsonDMBatchGenerator(BatchGenerator):  # Roke Manor Research
    """A batch generator that can ingest json files generated from the example DeepMind proto buffer dummy data
    files. This was developed to serve as an example for how to ingest further json files generated from other
    healthcare records

    """

    def __init__(
        self,
        config: configdict.ConfigDict,
        is_training: bool,
        task_coordinator: Optional[coordinator.Coordinator],
        data_split_name: str,
        debug_mode: bool = False,
    ):
        """Create a batch generator that ingests a json set of patient records according to jsons generated
        from DeepMind fake data
        """
        (
            self.feature_category_map,
            self.features_ids_per_category_map,
            self.presence_map,
            self.numerical_map,
            self.context_map,
            self.sequence_giveaways,
        ) = self._get_mappings()

        self.extra_keys = ["delta_time", "episode_id", "ignore_label", "segment_mask", "timestamp"]
        self.lookahead_adverse_outcome_label_key_fn_dict = self._get_lookahead_adverse_outcome_label_key_fn_dict(config)
        window_times_sets = {
            task: config.task_configs[task]["window_times"] for task in config.tasks if "Outcome" in task
        }
        # define keys for which a lookup within episode 'labels is needed
        self.adverse_outcome_keys = self._get_adverse_outcome_keys(window_times_sets)
        self.adverse_outcome_keys_dtypes = [float for key in self.adverse_outcome_keys]
        self.label_keys = [
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
            "indexes_diagnosis",
            "values_diagnosis",
            "indexes_ethnic_origin",
            "values_ethnic_origin",
            "indexes_method_of_admission",
            "values_method_of_admission",
            "indexes_sex",
            "values_sex",
            "indexes_year_of_birth",
            "values_year_of_birth",
        ]

        self.label_keys_dtypes = [float if "lab_" in key else int for key in self.label_keys]
        self.last_dim_1_keys = self.adverse_outcome_keys + self.label_keys + self.extra_keys

        self.ctx, self.seq = tf_dataset_utils.get_label_dicts(
            task_coordinator,
            config.data.context_features,
            config.data.sequential_features,
            fixed_len_metadata=config.shared.fixed_len_context_features,
        )

        super().__init__(
            config,
            is_training,
            task_coordinator,
            data_split_name,
            bypass_seqex_to_dict=True,
            tensor_to_sparse_keys=tensor_to_sparse_keys,
            debug_mode=debug_mode,
        )

    def _get_lookahead_adverse_outcome_label_key_fn_dict(self, config):
        return {
            TaskNames.ADVERSE_OUTCOME_RISK: label_utils.get_adverse_outcome_lookahead_label_key,
            TaskNames.ITU_OUTCOME: label_utils.get_itu_outcome_lookahead_label_key,
            TaskNames.DIALYSIS_OUTCOME: label_utils.get_dialysis_outcome_lookahead_label_key,
            TaskNames.MORTALITY_OUTCOME: label_utils.get_mortality_outcome_lookahead_label_key,
        }

    def _get_adverse_outcome_keys(self, window_times_sets):
        out_keys = []
        for key, window_times in window_times_sets.items():
            for window_time in window_times:
                out_keys.append(self.lookahead_adverse_outcome_label_key_fn_dict[key](window_time))

        return out_keys

    def _get_mappings(self):

        fake_data_dir = Path(__file__).parents[1] / "fake_data"

        feature_category_map_path = fake_data_dir / "feature_category_map.pkl"
        features_ids_per_category_map_path = fake_data_dir / "features_ids_per_category_map.pkl"
        presence_map_path = fake_data_dir / "presence_map.pkl"
        numerical_map_path = fake_data_dir / "numerical_map.pkl"
        with open(feature_category_map_path, "rb") as f:
            feature_category_map = pickle.load(f)
        feature_category_map = self._correct_zero_mappings_in_dict(feature_category_map)

        with open(features_ids_per_category_map_path, "rb") as f:
            features_ids_per_category_map = pickle.load(f)

        with open(presence_map_path, "rb") as f:
            presence_map = pickle.load(f)
        presence_map = self._correct_zero_mappings_in_dict(presence_map)

        with open(numerical_map_path, "rb") as f:
            numerical_map = pickle.load(f)
        numerical_map = self._correct_zero_mappings_in_dict(numerical_map)

        context_map = {}  # placeholder for inheriting classes that require context_map

        sequence_giveaways = []  # placeholder for inheriting classes that require sequence_giveaways

        return (
            feature_category_map,
            features_ids_per_category_map,
            presence_map,
            numerical_map,
            context_map,
            sequence_giveaways,
        )

    def _correct_zero_mappings_in_dict(self, input_dict):
        """Some mapping dictionaries will direct strings to a numerical value of 0, which leads to errors in
        retained indices when converting between tensors and sparse tensors. So update any key mapped to 0 to a new
        unique mapping (one greater than highest mapping)

        Args:
            input_dict (dict): string keys and int values

        Returns:
            (dict): string keys and int values

        """
        max_val = 0
        zero_mapping_key = None
        for key, item in input_dict.items():
            if item > max_val:
                max_val = item
            if item == 0:
                zero_mapping_key = key
        new_unique_val = max_val + 1
        if zero_mapping_key is not None:
            input_dict[zero_mapping_key] = new_unique_val
        return input_dict

    def _severity_lookup(self, feature_idx, feature_value):
        """Get a normal (1), low (2), high (3), very low (4) or very high (5) severity score based on a particular
        feature index and its current value

        Args:
            feature_idx (str): A possible value that can be taken by an entry in a list of entries in an event in a
            patient episode, e.g. present in one of the items in the self.feature_category_map dictionary (not a
            feature_category_idx)
            feature_value (numeric): The current value taken for the feature_idx in the entry

        Returns:
            (int): A score in the set [1-5] for how severe a value is,
            normal (1), low (2), high (3), very low (4) or very high (5)

        """
        return 1.0  # haven't seen anything else in deepmind tensors yet

    def _prepare_empty_sequence_dict_arrays(self, sequence_length):
        sequences = {}
        label_sequences = {
            label_key: np.zeros(shape=[1, sequence_length, 1], dtype=np.float32) for label_key in self.label_keys
        }
        adverse_outcome_sequences = {
            label_key: np.zeros(shape=[1, sequence_length, 1], dtype=np.int64)
            for label_key in self.adverse_outcome_keys
        }
        segment_mask = np.zeros(shape=[1, sequence_length, 1], dtype=np.int64)
        episode_id = np.zeros(shape=[1, sequence_length, 1], dtype=np.str)
        timestamp = np.zeros(shape=[1, sequence_length, 1], dtype=np.int64)
        delta_time = np.zeros(shape=[1, sequence_length, 1], dtype=np.int64)
        ignore_label = np.zeros(shape=[1, sequence_length, 1], dtype=np.int64)
        entry_change_flag = self._config.entry_change_flag
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

    def _build_seq_tensor(self, patient, sequence_length):  # noqa: C901
        """Goes through a nested JSON structure entry for a single patient with multiple 'episodes', and multiple
        'entries' within the 'clinical_events' key-value, where each entry occurs at a set time in the sequence.

        Args:
            patient (dict): Dictionary with nested episodes-clinical events-entries structure, with each entry
            being a list of dictionaries, each dictionary containing a feature_category_idx, feature_idx, value, each
            of these mapping to str, str, str, with this last str convertible to int or float
            sequence_length (int): Total number of 'entry' items in the patient throughout the nested structure

        Returns:
            (dict): Dictionary of ndarrays representing the sequence information for various events and metrics
            associated with a patient's history

        """
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
        ) = self._prepare_empty_sequence_dict_arrays(sequence_length)
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

                if event_type == "outpatient_events" or (event["time_of_day"] == "0"):
                    ignore_label[0, running_event_total, 0] = 1

                for label_key, label_key_dtype in zip(self.label_keys, self.label_keys_dtypes):
                    curr_label_val = event["labels"].get(label_key, None)
                    if curr_label_val is not None:
                        label_sequences[label_key][0, running_event_total, 0] = label_key_dtype(curr_label_val)
                for label_key, label_key_dtype in zip(self.adverse_outcome_keys, self.adverse_outcome_keys_dtypes):
                    curr_label_val = event["labels"].get(label_key, None)
                    if curr_label_val is not None:
                        adverse_outcome_sequences[label_key][0, running_event_total, 0] = label_key_dtype(
                            curr_label_val
                        )
                curr_segment_mask_val = event["labels"].get("segment_mask", None)
                segment_mask[0, running_event_total, 0] = float(curr_segment_mask_val)

                episode_id[0, running_event_total, 0] = curr_episode_id
                timestamp[0, running_event_total, 0] = (
                    int(event["patient_age"]) * 60 * 60 * 24
                    + (int(event["time_of_day"]) - 1) * self._config.model.time_bin_length
                )

                if "entries" in event:
                    categories = {}
                    running_entries_total = 0
                    running_numeric_entries_total = 0
                    # if running_entries_total < self._config.encoder.ndim_dict['count_s'] - 1:
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

                        # if entry_i < self._config.encoder.ndim_dict['pres_s'] - 1:
                        # passed test without this if statement but I'm suspicious
                        # that it might need to go back in for very large episodes
                        indexes_presence["values"].append(self.presence_map[entry["feature_idx"]])
                        indexes_presence["indices"].append([0, running_event_total, entry_i])
                        try:
                            values_presence["values"].append(
                                self._severity_lookup(entry["feature_category_idx"], float(entry["feature_value"]))
                            )
                        except ValueError:
                            # ehr-predictions readme states 0 is the numerical imputed value for missing entries
                            # but this leads to errors down the line with sparse tensor conversion
                            values_presence["values"].append(entry_change_flag)

                        values_presence["indices"].append([0, running_event_total, entry_i])

                        if entry["feature_idx"] in self.numerical_map:
                            # if running_numeric_entries_total < self._config.encoder.ndim_dict['num_s'] - 1:
                            # passed test without this if statement but I'm suspicious
                            # that it might need to go back in for very large episodes

                            try:
                                values_numeric["values"].append(float(entry["feature_value"]))
                            except ValueError:
                                # ehr-predictions readme states 0 is the numerical imputed value for missing entries
                                # but this leads to errors down the line with sparse tensor conversion
                                values_numeric["values"].append(entry_change_flag)
                            values_numeric["indices"].append([0, running_event_total, running_numeric_entries_total])

                            indexes_numeric["values"].append(self.numerical_map[entry["feature_idx"]])
                            indexes_numeric["indices"].append([0, running_event_total, running_numeric_entries_total])
                            running_numeric_entries_total += 1
                    # if running_entries_total < self._config.encoder.ndim_dict['count_s'] - 1:
                    # passed test without this if statement but I'm suspicious
                    # that it might need to go back in for very large episodes
                    for cat in categories.keys():
                        indexes_category_counts["values"].append(float(self.feature_category_map[cat]))
                        indexes_category_counts["indices"].append([0, running_event_total, running_entries_total])

                        values_category_counts["values"].append(categories[cat])
                        values_category_counts["indices"].append([0, running_event_total, running_entries_total])
                        running_entries_total += 1
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
            sequences[key]["values"] = np.array(sequences[key]["values"], dtype=self.seq[key].dtype.as_numpy_dtype)
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

    def get_tensors_from_single_patient(self, patient):
        """Build a dictionary of ndarrays to later be converted to tensors for a single patient

        Args:
            patient (dict): A nested dictionary defining a patient's medical history

        Returns:
            (dict): A nested dictionary defining context, sequence and sequence_lengths

        """
        context = {"record_number": np.array(patient["record_number"], np.str_)}
        sequence_length = 0
        for episode in patient["episodes"]:
            if "admission" in episode:
                sequence_length += len(episode["admission"]["clinical_events"])
            elif "outpatient_events" in episode:
                sequence_length += len(episode["outpatient_events"]["clinical_events"])
        sequence_lengths = np.array(sequence_length, dtype=np.int64)
        sequences = self._build_seq_tensor(patient, sequence_length)
        out_dict = {"context": context, "sequences": sequences, "sequence_lengths": sequence_lengths}

        return out_dict

    def get_dataset_output_dict_tensor_types_and_shapes_and_empty_batch(self, tensor_dict):
        """Get the types and shapes of a nested dictionary of ndarrays such that these types and shapes
        can be passed to tf.data.Dataset.from_generator to convert to tensors

        Args:
            tensor_dict (dict): A dictionary of ndarrays that are intended to be converted to tensors later

        Returns:
            (dict, dict, dict): The nested dictionaries of output types and shapes for passing to
            tf.data.Dataset.from_generator, and an empty batch dictionary of a similar structure with empty
            lists instead of ndarrays

        """
        output_types = {}
        output_shapes = {}
        empty_batch = {}
        for key, item in tensor_dict.items():
            if isinstance(item, dict):
                (
                    output_types[key],
                    output_shapes[key],
                    empty_batch[key],
                ) = self.get_dataset_output_dict_tensor_types_and_shapes_and_empty_batch(item)
            else:
                output_types[key] = item.dtype
                if key in self.seq and isinstance(self.seq[key], tf.FixedLenSequenceFeature):
                    tensor_shape_input = [None for i in range(len(item.shape) - 1)] + [self.seq[key].shape[0]]
                    output_shapes[key] = tf.TensorShape(tensor_shape_input)
                elif (
                    (key in self.ctx) and (not isinstance(self.ctx[key], tf.VarLenFeature))
                ) or key == "sequence_lengths":
                    output_shapes[key] = tf.TensorShape([])
                else:
                    tensor_shape_input = [None for i in range(len(item.shape))]
                    output_shapes[key] = tf.TensorShape(tensor_shape_input)
                empty_batch[key] = []
        return output_types, output_shapes, empty_batch

    def _read_record_files(self, records_filename):
        with open(records_filename, "rb") as file:
            dataset_list = json.load(file)["patients"]

        single_tensor_dict = self.get_tensors_from_single_patient(dataset_list[0])
        output_types, output_shapes, empty_batch = self.get_dataset_output_dict_tensor_types_and_shapes_and_empty_batch(
            single_tensor_dict
        )

        def data_generator(start_i=0):
            data_gen_i = start_i
            data_gen_i = data_gen_i % len(dataset_list)
            while True:
                elem = dataset_list[data_gen_i]
                data_gen_i += 1
                if data_gen_i == len(dataset_list):
                    data_gen_i = 0
                yield self.get_tensors_from_single_patient(elem)

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

        batch_generator = create_batch_generator(data_generator, batch_size=self._config.data.batch_size)

        dataset = tf.data.Dataset.from_generator(
            batch_generator, output_types=output_types, output_shapes=output_shapes
        )
        return dataset


class JsonlBatchGenerator(JsonDMBatchGenerator):
    """A batch generator that can ingest jsonl files generated for the C308 project."""

    def __init__(
        self,
        config: configdict.ConfigDict,
        is_training: bool,
        task_coordinator: Optional[coordinator.Coordinator],
        data_split_name: str,
        debug_mode: bool = False,
    ):
        """Create a batch generator that ingests a jsonl set of patient records according to jsonl generated
        for C308 project
        """
        self._config = config
        self.curriculum_updates = []
        self.sequence_occlusion_field = None
        self.context_occlusion_field = None
        super().__init__(
            config,
            is_training,
            task_coordinator,
            data_split_name,
            debug_mode=debug_mode,
        )

    """
    for maps, grab some files from us-topeka/Projects-LA/C308 Renal Health/Data/Processed
    /Intermediate Minimal JSON outputs, put in aki-predictions/data folder
    numerical feature map can be got from data_ingest_index_testing_reduced_statistics.json (real feature names,
    need mapping)
    feature ids can be gotten from data_ingest_index_testing_reduced_statistics_ingest_feature_mapping.json
    feature categories can be gotten from data_ingest_index_testing_reduced_statistics_ingest_category_mapping.json,
    needs matching up from data_ingest_index_testing_reduced_statistics_ingest_feature_mapping.json
    presence map needs to be made up for now, static version will come later
    """

    def set_occluded_field(self, field, occlusion_type=None):
        """Set self.context_occlusion_field and self.sequence_occlusion_field, which causes entries in events to be
        skipped if the feature_idx corresponds to it, with context or sequence fields selected via 'occlusion_type.
        Sets the other occlusion field to None, to avoid occluding
        both a context and sequence field simultaneously, which can have overlapping keys
        Args:
            occlusion_type (str): the family of fields to occlude field within, e.g. 'context' or 'sequence'
            field (str): The field to occlude
        """
        if occlusion_type == "context":
            self.context_occlusion_field = field
            self.sequence_occlusion_field = None
        elif occlusion_type == "sequence":
            self.context_occlusion_field = None
            self.sequence_occlusion_field = field
        else:
            self.sequence_occlusion_field = None
            self.context_occlusion_field = None

    def _get_mappings(self):
        """Currently an overengineered solution that extracts viable mappings out of files - expecting to eventually
        have a static set of jsons to read in directly

        Returns:
            (dict, dict, dict, dict): Various mappings (feature_category_map, features_ids_per_category_map,
            presence_map, numerical_map)

        """
        (
            feature_names_to_categories_path,
            feature_names_to_feature_idxs_path,
            numerical_map_path,
            metadata_map_path,
            missing_metadata_map_path,
        ) = self._get_mapping_locations()

        presence_map = {}
        with open(feature_names_to_feature_idxs_path, "rb") as f:
            feature_names_to_feature_idxs = json.loads(f.read())
        for key_i, (_, item) in enumerate(feature_names_to_feature_idxs.items()):
            presence_map[str(item)] = str(key_i + 1)  # simple mapping that just avoids using 0

        sequence_giveaways = []
        for entry in self._config.sequence_giveaways:
            sequence_giveaways.append(str(feature_names_to_feature_idxs[entry]))

        feature_category_map = {}
        with open(feature_names_to_categories_path, "rb") as f:
            feature_names_to_categories = json.loads(f.read())
        categories = []
        for val in feature_names_to_categories.values():
            categories.append(val)
        categories = list(set(categories))
        for cat_i, cat in enumerate(categories):
            feature_category_map[str(cat)] = str(cat_i + 1)

        numerical_map = {}
        with open(numerical_map_path, "rb") as f:
            numerical_feature_map = json.loads(f.read())
        for feat_i, (_, item) in enumerate(numerical_feature_map.items()):
            numerical_map[str(item)] = str(feat_i + 1)

        features_ids_per_category_map = {}  # don't know if this ever actually is needed

        if metadata_map_path is not None:
            # build context_mappings as needed in get_tensors_from_single_patient
            with open(metadata_map_path, "rb") as f:
                metadata_map = json.loads(f.read())
            self.metadata_map = metadata_map
            # some metadata has existing fields for that metadata being missing, some doesn't - need to build
            # a map of this and get a unique metadata 'missing' value for when no such field exists
            with open(missing_metadata_map_path, "rb") as f:
                self.missing_metadata_map = json.loads(f.read())

            # most metadata has a default value for when the item isn't present, but some don't,
            # so need a default default for those cases
            self.missing_metadata_map_missing_value = 0
            context_map = {}
            for context_feature in self._config.shared.context_features:
                # go through all metadata keys and split to separate types of metadata according to naming convention,
                # eg field names include ethnic_origin_, sex_, method_of_admission_ appended
                curr_context_feature_map = []
                for key, val in metadata_map.items():
                    if context_feature in key:
                        curr_context_feature_map.append(str(val))
                context_map[context_feature] = curr_context_feature_map
        else:
            context_map = {}

        return (
            feature_category_map,
            features_ids_per_category_map,
            presence_map,
            numerical_map,
            context_map,
            sequence_giveaways,
        )

    def _get_mapping_locations(self):
        """Extract and compile mapping paths from data config."""
        category_mapping_path = Path(self._config.data.records_dirpath) / self._config.data.category_mapping
        feature_mapping_path = Path(self._config.data.records_dirpath) / self._config.data.feature_mapping
        numerical_feature_mapping_path = (
            Path(self._config.data.records_dirpath) / self._config.data.numerical_feature_mapping
        )
        if len(self._config.data.metadata_mapping) > 0:
            metadata_mapping_path = Path(self._config.data.records_dirpath) / self._config.data.metadata_mapping
        else:
            metadata_mapping_path = None
        if len(self._config.data.missing_metadata_mapping) > 0:
            missing_metadata_mapping_path = (
                Path(self._config.data.records_dirpath) / self._config.data.missing_metadata_mapping
            )
        else:
            missing_metadata_mapping_path = None
        return (
            category_mapping_path,
            feature_mapping_path,
            numerical_feature_mapping_path,
            metadata_mapping_path,
            missing_metadata_mapping_path,
        )

    def _build_seq_tensor(self, patient, sequence_length):  # noqa: C901
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
        ) = self._prepare_empty_sequence_dict_arrays(sequence_length)
        for episode_i, episode in enumerate(patient["episodes"]):
            clinical_events = episode["events"]
            curr_episode_id = patient["master_key"]
            for event in clinical_events:
                if event["time_of_day"] == "0":
                    # 0 time of day flags summaries of entries that had unknown times
                    ignore_label[0, running_event_total, 0] = 1

                # get the continuous labels (e.g. lab test numericals)
                curr_labels = event["labels"]
                for label_key, label_key_dtype in zip(self.label_keys, self.label_keys_dtypes):

                    curr_label_val = curr_labels.get(label_key, None)
                    if curr_label_val is not None:
                        label_sequences[label_key][0, running_event_total, 0] = label_key_dtype(curr_label_val)
                    else:
                        # DeepMind generates 'unknown_label_mask' on the fly based on whether labs are greater than 0,
                        # we'll use a negative value to generate this mask without the sparse to dense conversion issue
                        label_sequences[label_key][0, running_event_total, 0] = -1.0
                # get the binary adverse outcome labels
                for label_key, label_key_dtype in zip(self.adverse_outcome_keys, self.adverse_outcome_keys_dtypes):

                    curr_label_val = curr_labels.get(label_key, None)
                    if curr_label_val is not None:
                        adverse_outcome_sequences[label_key][0, running_event_total, 0] = label_key_dtype(
                            curr_label_val
                        )
                    else:
                        # DeepMind generates 'unknown_label_mask' on the fly based on whether labs are greater than 0,
                        # we'll use a negative value to generate this mask without the sparse to dense conversion issue
                        adverse_outcome_sequences[label_key][0, running_event_total, 0] = -1.0
                try:
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
                        + (int(event["time_of_day"])) * self._config.model.time_bin_length
                    )
                except KeyError:
                    timestamp[0, running_event_total, 0] = 1

                # run through the entries and extract features
                if "entries" in event:
                    self._extract_sequences_from_event(
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
                sequences[key]["values"] = np.array(sequences[key]["values"], dtype=self.seq[key].dtype.as_numpy_dtype)
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
        self,
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
        if len(event["entries"]) > 1:
            indexes_category_counts["values"].append(entry_change_flag)
            indexes_category_counts["indices"].append([0, running_event_total, running_entries_total])

            values_category_counts["values"].append(entry_change_flag)
            values_category_counts["indices"].append([0, running_event_total, running_entries_total])
            running_entries_total += 1

        for entry_i, entry in enumerate(event["entries"]):

            # check for occlusion and skip entry if feature is occluded
            if entry["feature_idx"] == self.sequence_occlusion_field:
                continue
            # check for giveaways and skip entry if feature is a giveaway
            if entry["feature_idx"] in self.sequence_giveaways:
                continue

            if entry["feature_category_idx"] in categories:
                categories[entry["feature_category_idx"]] += 1
            else:
                categories[entry["feature_category_idx"]] = 1

            indexes_presence["values"].append(self.presence_map[entry["feature_idx"]])
            indexes_presence["indices"].append([0, running_event_total, entry_i])
            try:
                values_presence["values"].append(
                    self._severity_lookup(entry["feature_category_idx"], float(entry["feature_value"]))
                )
            except ValueError:
                # ehr-predictions readme states 0 is the numerical imputed value for missing entries
                # but this leads to errors down the line with sparse tensor conversion
                values_presence["values"].append(entry_change_flag)

            values_presence["indices"].append([0, running_event_total, entry_i])

            if entry["feature_idx"] in self.numerical_map:

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

                indexes_numeric["values"].append(self.numerical_map[entry["feature_idx"]])
                indexes_numeric["indices"].append([0, running_event_total, running_numeric_entries_total])
                running_numeric_entries_total += 1
        for cat in categories.keys():
            indexes_category_counts["values"].append(float(self.feature_category_map[cat]))
            indexes_category_counts["indices"].append([0, running_event_total, running_entries_total])

            values_category_counts["values"].append(categories[cat])
            values_category_counts["indices"].append([0, running_event_total, running_entries_total])
            running_entries_total += 1

    def _build_context_var_len_tensors(self, context_list, patient, metadata_map_item):
        """Build ndarrays or variable length from metadata of a patient that can be ingested into tf datasets
        An indexes and a values array will be given, which is the format required for encoding later in the framework
        Args:
            context_list (list of str): A set of keys corresponding to a particular type of metadata, ie all the keys
            that can be associated with pre-existing diagnoses
            patient (dict): Dictionary whose top level keys may include elements of the context_list
        Returns:
            (ndarray, ndarray): indexes array, values array
        """
        indexes = {"values": [], "indices": []}
        values = {"values": [], "indices": []}
        running_relevant_key_total = 0
        for key in patient.keys():
            if key == self.context_occlusion_field:
                # if current key is the occluded field, don't build it into the tensors
                continue
            if key in context_list:
                indexes["values"].append(np.float32(key))
                indexes["indices"].append([0, running_relevant_key_total])

                values["values"].append(np.float32(patient[key]))
                values["indices"].append([0, running_relevant_key_total])
                running_relevant_key_total += 1

        if len(indexes["indices"]) == 0:
            # can occur with diagnoses data, which aren't necessarily present, therefore put in missing data flag
            # if this occurs, a default metadata_map entry must exist for that context_map_item, denoted with an
            # underscore after the context_map_item, e.g. 'diagnosis_'
            missing_metadata_key = self.missing_metadata_map[metadata_map_item]
            if missing_metadata_key is not None:
                missing_metadata_val = self.metadata_map[missing_metadata_key]
            else:
                missing_metadata_val = self.missing_metadata_map_missing_value
            indexes["values"].append(np.float32(missing_metadata_val))
            indexes["indices"].append([0, running_relevant_key_total])

            values["values"].append(np.float32(1.0))
            values["indices"].append([0, running_relevant_key_total])

        indexes["values"] = np.array(indexes["values"], dtype=np.float32)
        indexes["indices"] = np.array(indexes["indices"], dtype=np.int64)

        values["values"] = np.array(values["values"], dtype=np.float32)
        values["indices"] = np.array(values["indices"], dtype=np.int64)

        # if len(indexes['indices']) == 0:
        #     print('here')

        indexes["dense_shape"] = np.max(indexes["indices"], axis=0) + 1
        values["dense_shape"] = np.max(values["indices"], axis=0) + 1

        indexes = tf_dataset_utils.ndarray_from_sparse_definition(
            indexes["indices"], indexes["values"], indexes["dense_shape"]
        )
        values = tf_dataset_utils.ndarray_from_sparse_definition(
            values["indices"], values["values"], values["dense_shape"]
        )
        return indexes, values

    def get_tensors_from_single_patient(self, patient):
        """Build a dictionary of ndarrays to later be converted to tensors for a single patient

        Args:
            patient (dict): A nested dictionary defining a patient's medical history

        Returns:
            (dict): A nested dictionary defining context, sequence and sequence_lengths

        """
        context = {"record_number": np.array(patient["record_number"], np.str_)}
        additional_context = {}
        for context_item in self._config.shared.context_features:
            if context_item in self._config.shared.var_len_context_features:
                (
                    additional_context[f"indexes_{context_item}"],
                    additional_context[f"values_{context_item}"],
                ) = self._build_context_var_len_tensors(
                    self.context_map[context_item], patient=patient, metadata_map_item=context_item
                )
            else:
                key_matched = False
                for key in patient.keys():
                    if key == self.context_occlusion_field:
                        # if key is to be occluded, don't build it into the context
                        continue
                    if key in self.context_map[context_item]:
                        # context metadata is included as a key-value pair, where the value is always '1' if the key
                        # is present, and at least 1 item from each of self._config.shared_context_features is included
                        # So if a metadata field such as 'diagnosis' is present, the presence of the particular
                        # diagnosis
                        # has a value of 1, and the useful information as to which diagnosis it is is stored in the key
                        # itself, which must be converted to float32
                        additional_context[f"indexes_{context_item}"] = np.array([[np.float32(key)]], dtype=np.float32)
                        additional_context[f"values_{context_item}"] = np.array(
                            [[np.float32(patient[key])]], dtype=np.float32
                        )
                        key_matched = True
                if not key_matched:
                    missing_metadata_key = self.missing_metadata_map[context_item]
                    if missing_metadata_key is not None:
                        missing_metadata_val = np.array([[np.float32(self.metadata_map[missing_metadata_key])]])
                    else:
                        missing_metadata_val = np.array([[np.float32(self.missing_metadata_map_missing_value)]])
                    additional_context[f"indexes_{context_item}"] = missing_metadata_val
                    additional_context[f"values_{context_item}"] = np.array([[1.0]], dtype=np.float32)
        context.update(additional_context)
        sequence_length = 0
        for episode in patient["episodes"]:
            sequence_length += len(episode["events"])
        sequence_lengths = np.array(sequence_length, dtype=np.int64)
        sequences = self._build_seq_tensor(patient, sequence_length)
        out_dict = {"context": context, "sequences": sequences, "sequence_lengths": sequence_lengths}

        return out_dict

    def _read_record_files(self, records_filename):
        with open(records_filename, "r") as file:
            line_dict = json.loads(file.readline())
        single_tensor_dict = self.get_tensors_from_single_patient(line_dict)
        output_types, output_shapes, empty_batch = self.get_dataset_output_dict_tensor_types_and_shapes_and_empty_batch(
            single_tensor_dict
        )

        with open(records_filename, "r") as file:
            self.records = file.readlines()
        self.num_records = len(self.records)
        self.total_epochs = int(np.ceil(self._config.model.num_steps / self.num_records))
        self.curriculum = np.ones(self.num_records, dtype=np.float32)

        def data_generator(start_i=0, records=[], num_records=0, record_order=[]):

            end_reached = False
            while not end_reached:
                line = records[record_order[start_i]]
                record = json.loads(line)
                yield self.get_tensors_from_single_patient(record)
                start_i += 1
                if start_i >= num_records:
                    if self._is_training is True:
                        # start again at first record
                        start_i = 0
                    else:
                        end_reached = True

        def create_batch_generator(gen, batch_size):
            def batch_gen():
                end_reached = False
                start_i = 0
                num_restarts = 0
                initial_record_order = list(range(self.num_records))
                self.record_order = initial_record_order
                # use_sequence_lengths_for_padding = (len(self._config.shared.var_len_context_features) > 0)

                while not end_reached:
                    if start_i >= self.num_records:
                        start_i = 0
                        num_restarts += 1
                        if (
                            self._config.using_curriculum
                            and len(self.curriculum_updates) > 0
                            and num_restarts >= self._config.curriculum_starting_epoch
                        ):
                            self.curriculum = curriculum_learning_utils.update_curriculum(
                                self.curriculum,
                                self.num_records,
                                self.record_order,
                                self.curriculum_updates,
                                curr_epoch=num_restarts,
                                start_epoch=self._config.curriculum_starting_epoch,
                                min_prob=self._config.curriculum_learning_min_prob,
                                max_prob=self._config.curriculum_learning_max_prob,
                            )
                            self.curriculum_updates = []
                            self.record_order = curriculum_learning_utils.sample_curriculum(
                                self.curriculum, self.num_records, num_restarts, self.total_epochs
                            )
                        else:
                            self.record_order = initial_record_order
                    for i, datum in enumerate(
                        gen(
                            start_i=start_i,
                            records=self.records,
                            num_records=self.num_records,
                            record_order=self.record_order,
                        )
                    ):
                        if i % batch_size == 0:
                            batch = copy.deepcopy(empty_batch)
                        tf_dataset_utils.append_nested_dict_of_lists(batch, datum)
                        if (i + 1) % batch_size == 0:
                            break
                    tf_dataset_utils.pad_nested_dict_of_lists_of_ndarray(batch)
                    for i in range(len(batch["context"]["record_number"])):
                        # have to pad tensors in the constructed batch so they can be combined into a batch by
                        # tensorflow later, then extract one batch element at a time
                        batch_element = tf_dataset_utils.extract_ndarray_at_idx_from_nested_dict_of_lists_of_ndarrays(
                            batch, i
                        )
                        # print(f'start_i {start_i}')
                        yield batch_element
                    start_i += batch_size
                    if start_i >= self.num_records and not self._is_training:
                        end_reached = True

            return batch_gen

        batch_generator = create_batch_generator(data_generator, batch_size=self._config.data.batch_size)
        # for el_i, el in enumerate(batch_generator()):
        #     if el_i % 128 == 0:
        #         print(el_i)
        #     pass
        dataset = tf.data.Dataset.from_generator(
            batch_generator, output_types=output_types, output_shapes=output_shapes
        )

        return dataset
