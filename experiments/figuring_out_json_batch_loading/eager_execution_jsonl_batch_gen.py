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
from aki_predictions.data_processing import CAP_CENTILES

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


def _get_config():
    # data_path = str(Path(__file__).parents[2] / "data" / "data_ingest_index_full_2022-07-11-100305")
    data_path = str(Path(r"C:\Projects\healthcare\aki-predictions\tests\fixtures\test_data_ingest_output"))
    print(data_path)

    if CAP_CENTILES:
        capped_string = ""
    else:
        capped_string = "_uncapped"
        # capped_string = ""

    data_locs_dict = {
        "records_dirpath": data_path,
        "train_filename": f"ingest_records_output_lines_train{capped_string}.jsonl",
        "valid_filename": f"ingest_records_output_lines_validate{capped_string}.jsonl",
        "test_filename": f"ingest_records_output_lines_test{capped_string}.jsonl",
        "calib_filename": f"ingest_records_output_lines_calib{capped_string}.jsonl",
        "category_mapping": "category_mapping.json",
        "feature_mapping": "feature_mapping.json",
        "numerical_feature_mapping": "numerical_feature_mapping.json",
        "metadata_mapping": "metadata_mapping.json",
        "missing_metadata_mapping": "missing_metadata_mapping.json",
        "sequence_giveaways": "sequence_giveaways.json",
    }
    # context_ndim_dict = {'diagnosis': 5183, 'ethnic_origin': 19, 'method_of_admission': 37, 'sex': 21,
    #                      'year_of_birth': 1}
    # context_ndim_dict = {'diagnosis': 3, 'ethnic_origin': 5, 'method_of_admission': 7, 'sex': 9,
    #                      'year_of_birth': 1}
    context_nact_dict = {"diagnosis": 2, "ethnic_origin": 1, "method_of_admission": 1, "sex": 1, "year_of_birth": 1}
    nact_dict = {
        types.FeatureTypes.PRESENCE_SEQ: 10,
        types.FeatureTypes.NUMERIC_SEQ: 3,
        types.FeatureTypes.CATEGORY_COUNTS_SEQ: 3,
        **context_nact_dict,
    }
    context_features = list(context_nact_dict.keys())
    var_len_context_features = ["diagnosis"]
    fixed_len_context_features = [
        cont_feat for cont_feat in context_features if cont_feat not in var_len_context_features
    ]

    shared_config_kwargs = {
        "tasks": (types.TaskNames.ITU_OUTCOME, types.TaskNames.DIALYSIS_OUTCOME, types.TaskNames.MORTALITY_OUTCOME),
        "context_features": context_features,
        "fixed_len_context_features": fixed_len_context_features,
        "var_len_context_features": var_len_context_features,
        "identity_lookup_features": [
            types.FeatureTypes.CATEGORY_COUNTS_SEQ,
            types.FeatureTypes.ETHNIC_ORIGIN,
            types.FeatureTypes.METHOD_OF_ADMISSION,
            types.FeatureTypes.SEX,
        ],
        # "encoder_layer_sizes": (2 * [400], 2 * [400], [], [400, 400],
        #                         [context_ndim_dict["ethnic_origin"], 400],
        #                         [context_ndim_dict["method_of_admission"], 400],
        #                         [context_ndim_dict["sex"], 400], [400, 400])
    }
    # for updating ndim_dict values - debug to return of _get_mappings in data loader, then
    # ndim_dict = {'pres_s': max([int(el) for el in list(presence_map.values())]) + 1,
    # 'num_s': max([int(el) for el in list(numerical_map.values())]) + 1,
    # 'count_s': max([int(el) for el in list(feature_category_map.values())]) + 1}
    config = experiment_config.get_config(
        # ndim_dict={"pres_s": 10240, "num_s": 15, "count_s": 22, **context_ndim_dict},
        nact_dict=nact_dict,
        data_locs_dict=data_locs_dict,
        num_steps=200001,  # 101 for testing
        eval_num_batches=None,  # to allow exiting via out of range error
        checkpoint_every_steps=1000,  # 1000 for full dataset, 100 for testing
        summary_every_steps=1000,  # 1000 for full dataset, 100 for testing
        eval_every_steps=2000,  # 1000 for full dataset, 100 for testing
        shared_config_kwargs=shared_config_kwargs,
        shuffle=False,
        run_occlusion_analysis=False,
    )
    return config


config = _get_config()

task_coordinator = get_task_coordinator(config)

batch_gen = tf_dataset.JsonlBatchGenerator(config, True, task_coordinator, "train", debug_mode=True)
# batch_gen = tf_dataset.JsonlBatchGenerator(config, False, task_coordinator, "valid", debug_mode=True)

dataset = batch_gen._debug_prepare_pre_initialized_dataset()

# for el_i, el in enumerate(dataset.take(5)):
#     tf_dataset_utils.convert_sparse_context_to_segments(el)

# for el_i, el in enumerate(dataset.take(5)):
#     pass

#
#
# for el_i, el in enumerate(dataset.take(5)):
#     tf_dataset_utils.convert_required_tensors_to_sparse(el, batch_gen._tensor_to_sparse_keys)
#
# for el_i, el in enumerate(dataset.take(5)):
#     tf_dataset_utils.convert_to_segments(el, batch_gen._config.data.get("segment_length"))
#
# for el_i, el in enumerate(dataset.take(5)):
#     tf_dataset_utils.convert_to_time_major(el)
#
#
#
# for el_i, el in enumerate(dataset.take(5)):
#     print(el_i)
#
# for el_i, el in enumerate(dataset.take(1)):
#     el_padded = tf_dataset_utils.add_beginning_of_sequence_mask_and_pad(el, 128, 128)
#
# absmp = tf_dataset_utils.add_beginning_of_sequence_mask_and_pad
# add_beginning_of_sequence_mask_and_pad = lambda x: absmp(x, 128, 128)
# dataset = dataset.flat_map(add_beginning_of_sequence_mask_and_pad)
#
# # for el_i, el in enumerate(dataset.take(5)):
# #     print(el_i)
#
# # Rebatch the data by num_unroll
# dataset = dataset.unbatch()
# # for el_i, el in enumerate(dataset.take(5)):
# #     print(el_i)
# dataset = dataset.batch(128, drop_remainder=False)
#
# for el_i, el in enumerate(dataset.take(25)):
#     print(el_i)
#
# assert True


embedding_classes = {
    types.EmbeddingType.LOOKUP: embeddings.BasicEmbeddingLookup,
    types.EmbeddingType.DEEP_EMBEDDING: embeddings.DeepEmbedding,
}
encoder = encoder_module_base.EncoderModule(config.encoder, embedding_classes)

model_init_kwargs = {"config": config.model, "embedding_size": encoder.get_total_embedding_size()}
base_model = rnn_model.RNNModel(**model_init_kwargs)
model = model_utils.RNNModelWithPersistentState(base_model)

for el_i, el in enumerate(dataset.take(100)):
    features, time_vect = encoder.embed_batch(el)
    # forward_return = model(features, el.is_beginning_sequence, time_vect)
