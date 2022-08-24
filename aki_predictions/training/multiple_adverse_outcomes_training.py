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
"""Experiment runner."""
import sys
import os
import logging
import time
from pathlib import Path


import tensorflow.compat.v1 as tf
import numpy as np

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
from aki_predictions.file_operations import save_dictionary_json
from aki_predictions.ehr_prediction_modeling.utils import occlusion_utils
from aki_predictions.inference.inference_utils import run_inference
from aki_predictions import file_operations
from aki_predictions.data_processing import CAP_CENTILES


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


def eval_fn_on_data_split(
    task_coordinator,
    metrics_coordinator,
    session,
    eval_batch_gen,
    eval_task_vars,
    split_name,
    current_step,
    eval_num_batches=None,  # Roke Manor Research
    threshold_range=None,
    root_logger=None,
):
    """Runs evaluation of a given datasplit and logs metrics."""
    # The dataset needs to be re-initialized for each epoch since we iterate the
    # entire data split.
    if root_logger is None:
        root_logger = logging.getLogger()
    if threshold_range is None:
        threshold_range = [0.5]
    session.run(eval_batch_gen.iterator.initializer)
    task_prediction_types = task_coordinator.task_prediction_types
    target_names_list = task_coordinator.target_names_list
    fetches = {
        "task_variables_list": eval_task_vars,
    }

    batch_count = 0
    while True:
        # root_logger.info("Evaluating batches: %s", batch_count)
        if eval_num_batches is not None and batch_count >= eval_num_batches:
            print(
                "Limited eval_num_batches introduces a likely mishandling of the evaluation batches unless"
                "calculated with inter-batch sequences considered, this method is only retained for testing cases,"
                "please set eval_num_batches to None or leave out of config generation in real experiments"
            )
            break
        try:
            fetches_np = session.run(fetches)
            for (target_names, task_type, task_variables) in zip(
                target_names_list, task_prediction_types, fetches_np["task_variables_list"]
            ):
                if task_type == types.TaskType.BINARY_CLASSIFICATION:
                    metrics.add_batch_to_binary_metrics_data(
                        metrics_coordinator=metrics_coordinator,
                        target_names=target_names,
                        predictions=task_variables.predictions,
                        binary_targets=task_variables.targets,
                        eval_mask_dict=task_variables.eval_mask_dict,
                        split_name=split_name,
                    )

                elif task_type == types.TaskType.REGRESSION:
                    metrics.add_batch_to_regression_metrics_data(
                        metrics_coordinator=metrics_coordinator,
                        target_names=target_names,
                        predictions=task_variables.predictions,
                        targets=task_variables.targets,
                        eval_mask_dict=task_variables.eval_mask_dict,
                        split_name=split_name,
                    )
                else:
                    raise ValueError("Unsupported task type for evaluation: %s" % task_type)

        except tf.errors.OutOfRangeError:
            # OutOfRangeError is the normal error thrown when the queue is empty
            # due to the epoch limitation.
            break

        if batch_count % 100 == 0:
            root_logger.info("Evaluated %s batches.", batch_count)

        batch_count += 1

    root_logger.info("Finished evaluating %s batches.", batch_count)

    root_logger.info("Calculating metrics...")
    if len(threshold_range) > 1:
        clear_data = False
    else:
        clear_data = True
    metrics_output = {}
    for threshold in threshold_range:
        root_logger.info(f"Logging metrics for threshold: {threshold}")
        metrics_output[threshold] = metrics_coordinator.log_metrics(
            current_step, clear_data=clear_data, silence=True, class_threshold=threshold
        )

    metrics_coordinator._clear_data()

    return metrics_output


def setup_eval(config, task_coordinator, split, model, encoder):
    """Setup evaluation using batch generator."""
    batch_gen = tf_dataset.JsonlBatchGenerator(config, False, task_coordinator, split)
    batch = batch_gen.batch
    features, time_vect = encoder.embed_batch(batch)
    forward_return = model(features, batch.is_beginning_sequence, time_vect)
    tasks_graph = task_coordinator.get_coordinator_variables(batch, forward_return.model_output)
    return (batch_gen, tasks_graph.task_variables_list, batch)


def make_serialisable(input_dictionary):
    """Recursive method to serialise the metrics output.

    Changes numpy float32 values to float, and ndarray values to lists.

    Args:
        input_dictionary (dict): input dictionary to make serialisable.

    Returns:
        (dict): Serialisable dictionary.
    """
    new_dictionary = input_dictionary.copy()
    for k, v in new_dictionary.items():
        if isinstance(v, np.float32):
            new_dictionary[k] = float(v)
        elif isinstance(v, np.ndarray):
            value_list = v.tolist()
            # Fill nan values in histograms with zeros
            cleaned_values = [value if not np.isnan(value) else str(value) for value in value_list]
            new_dictionary[k] = cleaned_values
        elif isinstance(v, dict):
            new_dictionary[k] = make_serialisable(v)
        elif isinstance(v, list):
            if len(v) > 0 and isinstance(v[0], list):
                # Assumes only two levels of lists. Fills with ones (to indicate inf)
                new_dictionary[k] = [
                    [value if not np.isnan(value) else str(value) for value in value_list] for value_list in v
                ]
            else:
                new_dictionary[k] = [value if not np.isnan(value) else str(value) for value in v]
    return new_dictionary


def run(config, eval_config=None, root_logger=None):  # noqa: C901
    """Build model and runs experiment."""
    if root_logger is None:
        root_logger = logging.getLogger()
    task_coordinator = get_task_coordinator(config)

    tf.random.set_random_seed(config.get("seed", 0))
    root_logger.info(config)

    metrics_coordinator = metrics.MetricsCoordinator()

    embedding_classes = {
        types.EmbeddingType.LOOKUP: embeddings.BasicEmbeddingLookup,
        types.EmbeddingType.DEEP_EMBEDDING: embeddings.DeepEmbedding,
    }
    encoder = encoder_module_base.EncoderModule(config.encoder, embedding_classes)

    model_init_kwargs = {"config": config.model, "embedding_size": encoder.get_total_embedding_size()}
    base_model = rnn_model.RNNModel(**model_init_kwargs)
    model = model_utils.RNNModelWithPersistentState(base_model)
    optimizer = model_utils.get_optimizer_from_config(config.optimizer)

    batch_gen = tf_dataset.JsonlBatchGenerator(config, True, task_coordinator, "train")
    batch = batch_gen.batch
    features, time_vect = encoder.embed_batch(batch)
    forward_return = model(features, batch.is_beginning_sequence, time_vect)
    tasks_graph = task_coordinator.get_coordinator_variables(batch, forward_return.model_output)
    embedding_loss, _ = encoder.get_embedding_loss(batch)

    if config.using_curriculum is True:
        expanded_losses = []
        for task in task_coordinator._task_list:
            # average loss across time and time window axes
            expanded_losses.append(tf.reduce_mean(task.expanded_loss, axis=[0, 3]))

        batch_curriculum_update = tf.add_n(expanded_losses)

    loss = tasks_graph.combined_loss
    loss += encoder.get_embedding_regularization_loss()
    loss += embedding_loss
    loss += model.get_model_regularization_loss()

    losses_per_task = {}
    for task_name, task_vars in zip(task_coordinator.task_names, tasks_graph.task_variables_list):
        losses_per_task[task_name] = task_vars.loss

    loss += task_coordinator.get_task_regularization_losses()

    loss_to_vars = losses.get_loss_to_variables_dict(
        model=model,
        encoder=encoder,
        losses_per_task=losses_per_task,
        all_variables=tf.trainable_variables(),
        total_loss=loss,
    )
    step = model_utils.multiple_loss_optim_fn(optimizer, loss_to_vars, norm_clip=config.optimizer.norm_clip)

    split = config.splits_to_evaluate
    if eval_config is not None:
        eval_batch_gen, eval_task_vars, eval_batch = setup_eval(eval_config, task_coordinator, split, model, encoder)
    else:
        eval_batch_gen, eval_task_vars, eval_batch = setup_eval(config, task_coordinator, split, model, encoder)

    with tf.control_dependencies([step]):
        scalar_loss = tf.reduce_mean(loss)
        step_cnt = tf.train.get_or_create_global_step()
    current_step = 0  # Set to provided number of steps in config?

    checkpoint_dir = get_checkpoint_dir(config.checkpoint, "train")
    checkpoint_dir_path = Path(os.getcwd()) / checkpoint_dir
    checkpoint_dir_path.mkdir(exist_ok=True, parents=True)

    # prepare inference nodes in graph
    inference_batch_placeholder, inference_placeholder_names = occlusion_utils.build_batch_placeholder(eval_batch)
    inference_batch_tensor_name_dict = occlusion_utils.build_batch_tensor_name_dict(inference_batch_placeholder)
    inference_tensor_name_json_path = checkpoint_dir_path / "inference_batch_tensor_name_dict.json"
    file_operations.save_dictionary_json(
        inference_tensor_name_json_path, inference_batch_tensor_name_dict, sort_keys=False
    )
    inf_features, inf_time_vect = encoder.embed_batch(inference_batch_placeholder)
    inference_forward_return = model(inf_features, inference_batch_placeholder.is_beginning_sequence, inf_time_vect)
    inference_tasks_graph = task_coordinator.get_coordinator_variables(
        inference_batch_placeholder, inference_forward_return.model_output
    )
    inference_predictions_list = [el.predictions for el in inference_tasks_graph.task_variables_list]
    inference_predictions_name_dict = {key: el.name for key, el in zip(config.tasks, inference_predictions_list)}
    inference_predictions_tensors_json_path = checkpoint_dir_path / "inference_predictions_tensors_dict.json"
    file_operations.save_dictionary_json(inference_predictions_tensors_json_path, inference_predictions_name_dict)

    if config.run_occlusion_analysis:
        inference_batch_gen = tf_dataset.JsonlBatchGenerator(config, False, task_coordinator, config.splits_to_evaluate)
        inference_batch = inference_batch_gen.batch

    if config.run_inference:
        inference_batch_gen = tf_dataset.JsonlBatchGenerator(
            config, False, task_coordinator, "train"
        )  # Assumes inference data passed as training path.
        inference_batch = inference_batch_gen.batch

    tf.summary.scalar("loss", scalar_loss)

    summary_op = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(save_steps=100, output_dir=checkpoint_dir, summary_op=summary_op)

    # Turn off checkpointing to prevent saving of a new model after reloading.
    if config.run_inference or config.run_occlusion_analysis or config.run_threshold_sweep:
        config.checkpoint.checkpoint_every_steps = None
        config.checkpoint.summary_every_steps = None
        scaffold = None
    else:
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=20))

    with tf.train.MonitoredTrainingSession(
        is_chief=True,
        hooks=[summary_hook],
        checkpoint_dir=checkpoint_dir,
        save_checkpoint_steps=config.checkpoint.checkpoint_every_steps,
        save_summaries_steps=config.checkpoint.summary_every_steps,
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False),
        scaffold=scaffold,
    ) as session:

        previous_first_record_number = None
        if config.using_curriculum is True:
            fetches = {
                "step": step_cnt,
                "loss": scalar_loss,
                "batch_curriculum_update": batch_curriculum_update,
                "batch": batch,
            }
        else:
            fetches = {
                "step": step_cnt,
                "loss": scalar_loss,
            }
        while current_step < config.model.num_steps:

            if config.run_occlusion_analysis:
                # run occlusion analysis, then end loop
                occlusion_output_path = Path(checkpoint_dir) / "occlusion_analysis.jsonl"
                occlusion_utils.run_occlusion_analysis_all_fields(
                    session,
                    inference_batch_gen,
                    inference_batch,
                    inference_tensor_name_json_path,
                    inference_predictions_tensors_json_path,
                    output_path=occlusion_output_path,
                )
                # break once occlusion analysis finishes - no intention to do training/evaluation if occlusion is
                # being done, as that expects the model is already trained
                break

            if config.run_inference:
                inference_output_path = Path(checkpoint_dir) / "inference_predictions.json"
                run_inference(
                    session,
                    inference_batch_gen,
                    inference_batch,
                    inference_tensor_name_json_path,
                    inference_predictions_tensors_json_path,
                    output_path=inference_output_path,
                )
                # Finish once inference is complete.
                break

            if config.run_threshold_sweep:
                # Run evaluation threshold sweep and save out metrics.
                threshold_range = config.threshold_range

                root_logger.info(f"Starting evaluation on data split: {split}, for threshold range {threshold_range}")
                current_metrics = eval_fn_on_data_split(
                    task_coordinator,
                    metrics_coordinator,
                    session,
                    eval_batch_gen,
                    eval_task_vars,
                    split,
                    current_step,
                    threshold_range=threshold_range,
                )

                for threshold, metric_outputs in current_metrics.items():
                    # Make dictionary serialisable
                    root_logger.info("Serialising metrics...")
                    serialised_metrics = make_serialisable(metric_outputs)

                    root_logger.info("Saving metrics...")
                    save_dictionary_json(
                        Path(checkpoint_dir) / f"metrics-{current_step}-{threshold}.json",
                        serialised_metrics,
                        sort_keys=False,
                    )

                break

            fetches_np = session.run(fetches)
            current_step = fetches_np["step"]
            # print(current_step)
            if config.using_curriculum:

                # with long sequences we get them split over time sequences within a batch, which will not line up
                # with the known order used in the curriculum sampling - remove repeats from the batch curriculum update
                record_numbers = fetches_np["batch"].context["record_number"][0, :]
                batch_curriculum_update = np.squeeze(fetches_np["batch_curriculum_update"])
                valid_update_positions = np.ones_like(batch_curriculum_update, dtype=bool)
                for record_num_i in range(len(record_numbers) - 1):
                    if record_numbers[record_num_i + 1] == record_numbers[record_num_i]:
                        valid_update_positions[record_num_i + 1] = False
                batch_curriculum_update = batch_curriculum_update[valid_update_positions]

                # detect whether current batch is due to sequence overspill and do curriculum appending if not
                curr_first_record_number = fetches_np["batch"].context["record_number"][0, 0]
                if previous_first_record_number is None or previous_first_record_number != curr_first_record_number:
                    previous_first_record_number = curr_first_record_number
                    batch_gen.curriculum_updates.append(batch_curriculum_update)

            if current_step % config.model.eval_every_steps == 0:
                root_logger.info("step %s, fetches: %s", current_step, fetches_np)
                root_logger.info("Starting evaluation on data split: %s", split)
                current_metrics = eval_fn_on_data_split(
                    task_coordinator,
                    metrics_coordinator,
                    session,
                    eval_batch_gen,
                    eval_task_vars,
                    split,
                    current_step,
                    threshold_range=[config.class_threshold],
                    root_logger=root_logger,
                )

                for threshold, metrics_output in current_metrics.items():
                    # Make dictionary serialisable
                    root_logger.info("Serialising metrics...")
                    serialised_metrics = make_serialisable(metrics_output)

                    root_logger.info("Saving metrics...")
                    save_dictionary_json(
                        Path(checkpoint_dir) / f"metrics-{current_step}-{threshold}.json",
                        serialised_metrics,
                        sort_keys=False,
                    )


def _get_config(
    data_dir,
    checkpoint_dir="",
    root_logger=None,
    steps=None,
    checkpoint_every=None,
    eval_every=None,
    summary_every=None,
    **kwargs,
):
    if root_logger is None:
        root_logger = logging.getLogger()
    root_logger.info(data_dir)

    if steps is None:
        # Default training run length
        steps = 220000
    if checkpoint_every is None:
        checkpoint_every = 4000
    if eval_every is None:
        eval_every = 4000
    if summary_every is None:
        summary_every = 1000

    if CAP_CENTILES:
        capped_string = ""
    else:
        capped_string = "_uncapped"

    data_locs_dict = {
        "records_dirpath": data_dir,
        "train_filename": f"ingest_records_output_lines_train{capped_string}.jsonl",
        "valid_filename": f"ingest_records_output_lines_validate{capped_string}.jsonl",
        "test_filename": f"ingest_records_output_lines_test{capped_string}.jsonl",
        "calib_filename": f"ingest_records_output_lines_calib{capped_string}.jsonl",
        "category_mapping": "category_mapping.json",
        "feature_mapping": "feature_mapping.json",
        "numerical_feature_mapping": "numerical_feature_mapping.json",
    }

    if kwargs.get("expect_giveaways", True):
        root_logger.info("Training with giveaway fields masked.")
        data_locs_dict["sequence_giveaways"] = "sequence_giveaways.json"
    shared_config_kwargs = {
        "tasks": (types.TaskNames.ITU_OUTCOME, types.TaskNames.DIALYSIS_OUTCOME, types.TaskNames.MORTALITY_OUTCOME)
    }

    config = experiment_config.get_config(
        data_locs_dict=data_locs_dict,
        num_steps=steps,  # 2 for testing
        eval_num_batches=None,  # None for full dataset (divided by batch size), 2 for testing
        checkpoint_every_steps=checkpoint_every,  # 2000 for full dataset, 1 for testing
        summary_every_steps=summary_every,  # 1000 for full dataset, 1 for testing
        eval_every_steps=eval_every,  # 2000 for full dataset, 1 for testing
        shared_config_kwargs=shared_config_kwargs,
        using_curriculum=False,
        shuffle=True,
        checkpoint_dir=checkpoint_dir,
        threshold_range=np.concatenate((np.arange(0.001, 0.01, 0.001), np.arange(0.01, 1, 0.01)), axis=0),
        **kwargs,
    )
    return config


def _get_eval_config(
    data_dir,
    checkpoint_dir="",
    root_logger=None,
    steps=None,
    checkpoint_every=None,
    eval_every=None,
    summary_every=None,
    **kwargs,
):
    if root_logger is None:
        root_logger = logging.getLogger()
    eval_config = _get_config(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        root_logger=root_logger,
        steps=steps,
        checkpoint_every=checkpoint_every,
        eval_every=eval_every,
        summary_every=summary_every,
        **kwargs,
    )
    eval_config.using_curriculum = False
    eval_config.shuffle = False


def main(output_dir, data_dir, steps, checkpoint_every, eval_every, summary_every):
    """Run experiment."""
    root_logger = logging.getLogger(__name__)

    config = _get_config(
        data_dir=data_dir,
        checkpoint_dir=output_dir,
        root_logger=root_logger,
        steps=steps,
        checkpoint_every=checkpoint_every,
        eval_every=eval_every,
        summary_every=summary_every,
    )
    eval_config = _get_eval_config(
        data_dir=data_dir,
        checkpoint_dir=output_dir,
        root_logger=root_logger,
        steps=steps,
        checkpoint_every=checkpoint_every,
        eval_every=eval_every,
        summary_every=summary_every,
    )
    run(config, eval_config, root_logger)


if __name__ == "__main__":
    output_dir = sys.argv[1]
    data_dir = sys.argv[2]
    steps = int(sys.argv[3])
    checkpoint_every = int(sys.argv[4])
    eval_every = int(sys.argv[5])
    summary_every = int(sys.argv[6])

    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parents[2] / "data" / "data_ingest_index_full_2022-07-11-100305")

    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    artifacts_dir = Path(output_dir)
    if artifacts_dir.is_dir() is False:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(artifacts_dir, f"{timestamp}_training_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)

    main(artifacts_dir, data_dir, steps, checkpoint_every, eval_every, summary_every)
