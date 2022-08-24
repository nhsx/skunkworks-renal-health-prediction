# coding=utf-8
# MIT Licence
#
# Copyright (c) 2021 NHS England
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
"""Default configuration."""
import json
from pathlib import Path
import logging

from aki_predictions.ehr_prediction_modeling import types
from aki_predictions.ehr_prediction_modeling.tasks import adverse_outcome_task
from aki_predictions.ehr_prediction_modeling.tasks import labs_task
from aki_predictions.ehr_prediction_modeling.tasks import los_task
from aki_predictions.ehr_prediction_modeling.tasks import mortality_task
from aki_predictions.ehr_prediction_modeling.tasks import readmission_task
from aki_predictions.ehr_prediction_modeling.tasks import itu_outcome_task
from aki_predictions.ehr_prediction_modeling.tasks import dialysis_outcome_task
from aki_predictions.ehr_prediction_modeling.tasks import mortality_outcome_task
from aki_predictions.ehr_prediction_modeling import configdict


# Mapping of task type to Class.
TASK_MAPPING = {
    adverse_outcome_task.AdverseOutcomeRisk.task_type: adverse_outcome_task.AdverseOutcomeRisk,
    labs_task.LabsRegression.task_type: labs_task.LabsRegression,
    los_task.LengthOfStay.task_type: los_task.LengthOfStay,
    mortality_task.MortalityRisk.task_type: mortality_task.MortalityRisk,
    readmission_task.ReadmissionRisk.task_type: readmission_task.ReadmissionRisk,
    itu_outcome_task.ITUOutcomeRisk.task_type: itu_outcome_task.ITUOutcomeRisk,
    dialysis_outcome_task.DialysisOutcomeRisk.task_type: dialysis_outcome_task.DialysisOutcomeRisk,
    mortality_outcome_task.MortalityOutcomeRisk.task_type: mortality_outcome_task.MortalityOutcomeRisk,
}


def calculate_ndim_dict(data_locations_dict):
    """Open mapping files and extract feature dimensions."""
    output_dict = None
    if data_locations_dict != {}:
        if Path.exists(Path(data_locations_dict["records_dirpath"])):
            presence_map = {}
            with open(Path(data_locations_dict["records_dirpath"]) / data_locations_dict["feature_mapping"], "rb") as f:
                feature_names_to_feature_idxs = json.loads(f.read())
            for key_i, (_, item) in enumerate(feature_names_to_feature_idxs.items()):
                presence_map[str(item)] = str(key_i + 1)  # fairly dumb mapping that just avoids using 0

            feature_category_map = {}
            with open(
                Path(data_locations_dict["records_dirpath"]) / data_locations_dict["category_mapping"], "rb"
            ) as f:
                feature_names_to_categories = json.loads(f.read())
            categories = []
            for val in feature_names_to_categories.values():
                categories.append(val)
            categories = list(set(categories))
            for cat_i, cat in enumerate(categories):
                feature_category_map[str(cat)] = str(cat_i + 1)

            numerical_map = {}
            with open(
                Path(data_locations_dict["records_dirpath"]) / data_locations_dict["numerical_feature_mapping"], "rb"
            ) as f:
                numerical_feature_map = json.loads(f.read())
            for feat_i, (_, item) in enumerate(numerical_feature_map.items()):
                numerical_map[str(item)] = str(feat_i + 1)

            if "metadata_mapping" in data_locations_dict:
                with open(
                    Path(data_locations_dict["records_dirpath"]) / data_locations_dict["metadata_mapping"], "rb"
                ) as f:
                    metadata_map = json.loads(f.read())
                    prefixes = ["diagnosis", "ethnic_origin", "method_of_admission", "sex", "year_of_birth"]
                    metadata_max_vals = {}
                    for prefix in prefixes:
                        curr_max = 0
                        for key in metadata_map.keys():
                            if prefix in key:
                                if metadata_map[key] >= curr_max:
                                    metadata_max_vals[prefix] = metadata_map[key]
                                    curr_max = metadata_map[key]
                    for key in metadata_max_vals.keys():
                        metadata_max_vals[key] += 1
            else:
                metadata_max_vals = {}

            output_dict = {
                "pres_s": max([int(el) for el in list(presence_map.values())]) + 1,
                "num_s": max([int(el) for el in list(numerical_map.values())]) + 1,
                "count_s": max([int(el) for el in list(feature_category_map.values())]) + 1,
                **metadata_max_vals,
            }
    return output_dict


def load_sequence_giveaway_fields(data_locations_dict):
    """Return a list of field names (not mappings) that are to be ignored during sequence tensor creation, as
    they are classed as 'giveaway' features for the model
    Args:
        data_locations_dict (dict): dictionary with at minimum records_dirpath and sequence_giveaways keys, which
        correspond to directory and file within that directory strings
    Returns:
        (list of strings): the fields to be ignored
    """
    if "sequence_giveaways" in data_locations_dict:
        try:
            with open(
                Path(data_locations_dict["records_dirpath"]) / data_locations_dict["sequence_giveaways"], "rb"
            ) as f:
                sequence_giveaways = json.loads(f.read())["sequence_giveaways"]
        except FileNotFoundError:
            root_logger = logging.getLogger()
            root_logger.warning("Sequence giveaways file not found. Proceeding without giveaway field definitions.")
            sequence_giveaways = []
    else:
        sequence_giveaways = []
    return sequence_giveaways


def get_config(**kwargs):
    """Gets configuration parameters.

    Returns:
      ConfigDict of default hyperparameters.
    """
    config = configdict.ConfigDict()

    shared_config_kwargs = kwargs.get("shared_config_kwargs", {})
    shared = shared_config(**shared_config_kwargs)
    config.shared = shared

    # Data splits over which to run evaluation.
    config.splits_to_evaluate = "valid"

    # Config for dataset reading and parsing.
    config.data = get_data_config(
        config.shared, data_locs_dict=kwargs.get("data_locs_dict", {}), shuffle=kwargs.get("shuffle", True)
    )

    # Config for saving checkpoints.
    checkpoint_every_steps = kwargs.get("checkpoint_every_steps", 100)
    summary_every_steps = kwargs.get("summary_every_steps", 100)
    checkpoint_dir = kwargs.get("checkpoint_dir", "")
    config.checkpoint = get_checkpoint_config(
        checkpoint_every_steps=checkpoint_every_steps,
        summary_every_steps=summary_every_steps,
        checkpoint_dir=checkpoint_dir,
    )

    # Setting the seed for the random initializations.
    config.seed = None

    config.tasks = shared.tasks

    config.using_curriculum = kwargs.get("using_curriculum", False)

    config.run_occlusion_analysis = kwargs.get("run_occlusion_analysis", False)

    config.run_inference = kwargs.get("run_inference", False)

    config.run_threshold_sweep = kwargs.get("run_threshold_sweep", False)

    # Default class threshold range for threshold sweep
    config.threshold_range = kwargs.get("threshold_range", [0.5])

    # model tends to descend loss a lot in first epoch, ie early data will need to be measured for difficulty
    # more accurately in epoch 1, followed by starting to use curriculum in potentially a later epoch
    config.curriculum_starting_epoch = kwargs.get("curriculum_starting_epoch", 2)

    config.curriculum_learning_min_prob = kwargs.get(
        "curriculum_learning_min_prob", {"method": "percentile", "thresh": 10}
    )
    config.curriculum_learning_max_prob = kwargs.get(
        "curriculum_learning_max_prob", {"method": "percentile", "thresh": 90}
    )

    config.entry_change_flag = kwargs.get("entry_change_flag", 1.0)

    task_configs = {}
    for task_type, task_factory in TASK_MAPPING.items():
        if "_outcome" in task_type:
            # outcome tasks will have curriculum generated from their losses
            for task_config in task_factory.default_configs(using_curriculum=config.using_curriculum):
                task_configs[task_config.name] = task_config
        else:
            for task_config in task_factory.default_configs():
                task_configs[task_config.name] = task_config

    # Dict of prediction tasks configurations.
    config.task_configs = task_configs

    # Config for model parameters.
    num_steps = kwargs.get("num_steps", 400000)
    eval_every_steps = kwargs.get("eval_every_steps", 100)
    rnn_cell = kwargs.get("rnn_cell", types.RNNCellType.LSTM)
    config.model = get_model_config(shared, num_steps=num_steps, eval_every_steps=eval_every_steps, rnn_cell=rnn_cell)

    # Config for the encoder module to embed the data.
    ndim_dict = kwargs.get("ndim_dict", calculate_ndim_dict(kwargs.get("data_locs_dict", {})))
    nact_dict = kwargs.get("nact_dict", None)
    config.encoder = get_encoder_config(shared, ndim_dict=ndim_dict, nact_dict=nact_dict)

    # fix first encoder layer sizes in shared having retrieved them
    for (key, new_encoder_layer_size), shared_layer_sizes_i in zip(
        config.encoder.ndim_dict.items(), range(len(config.shared.encoder_layer_sizes))
    ):
        if (
            len(config.shared.encoder_layer_sizes[shared_layer_sizes_i]) > 0
            and key in config.shared.identity_lookup_features
        ):
            config.shared.encoder_layer_sizes[shared_layer_sizes_i][0] = new_encoder_layer_size

    # Config for the optimizer.
    initial_learning_rate = kwargs.get("initial_learning_rate", 0.001)
    config.optimizer = get_optimizer_config(initial_learning_rate=initial_learning_rate)

    # Config class threshold default
    config.class_threshold = kwargs.get("default_class_threshold", 0.5)

    # Roke Manor Research - added simple limit on number of eval batches. DeepMind's method retained if set to None
    config.eval_num_batches = kwargs.get("eval_num_batches", None)

    config.sequence_giveaways = load_sequence_giveaway_fields(kwargs.get("data_locs_dict", {}))

    return config


def get_data_config(shared, **kwargs):
    """Gets config for data loading that is common to all experiments."""
    data_config = configdict.ConfigDict()

    data_locs_dict = kwargs.get("data_locs_dict", {})

    # Add path to directory containing tfrecords.
    data_config.records_dirpath = data_locs_dict.get(
        "records_dirpath", "aki_predictions/ehr_prediction_modeling/fake_data/standardize"
    )

    data_config.train_filename = data_locs_dict.get("train_filename", "train.tfrecords")

    data_config.valid_filename = data_locs_dict.get("valid_filename", "valid.tfrecords")

    data_config.test_filename = data_locs_dict.get("test_filename", "test.tfrecords")

    data_config.calib_filename = data_locs_dict.get("calib_filename", "calib.tfrecords")

    # Mappings for use with JsonlBatchGenerator
    data_config.category_mapping = data_locs_dict.get("category_mapping", "")
    data_config.feature_mapping = data_locs_dict.get("feature_mapping", "")
    data_config.numerical_feature_mapping = data_locs_dict.get("numerical_feature_mapping", "")
    data_config.metadata_mapping = data_locs_dict.get("metadata_mapping", "")
    data_config.missing_metadata_mapping = data_locs_dict.get("missing_metadata_mapping", "")

    # Whether to shuffle the dataset.
    data_config.shuffle = kwargs.get("shuffle", True)

    # The size of the shuffle buffer.
    data_config.shuffle_buffer_size = 8

    # Parallelism settings for the padded data reading implementation
    data_config.padded_settings = configdict.ConfigDict()

    # Number of parallel calls to parsing.
    data_config.padded_settings.parse_cycle_length = 4

    # Number of parallel recordio reads.
    data_config.padded_settings.recordio_cycle_length = 16

    # Number of (batched) sequences to prefetch.
    data_config.padded_settings.num_prefetch = 8

    # Extend config for dataset reading and parsing.
    data_config.context_features = shared.context_features
    data_config.sequential_features = shared.sequential_features
    data_config.batch_size = shared.batch_size
    data_config.num_unroll = shared.num_unroll
    data_config.segment_length = shared.segment_length

    return data_config


def get_model_config(shared, **kwargs):
    """Gets model configuration."""
    model_config = configdict.ConfigDict()

    # One of types.ModelModes
    model_config.mode = shared["mode"]

    # Number of training steps.
    num_steps = kwargs.get("num_steps", 400000)
    model_config.num_steps = num_steps

    eval_every_steps = kwargs.get("eval_every_steps", 100)
    model_config.eval_every_steps = eval_every_steps

    # Length of one time bin in seconds
    model_config.time_bin_length = 6 * 3600

    # List of number of dimensions of the hidden layers of the rnn model.
    # If using types.ModelTypes.SNRNN the input expected is a list of lists where
    # each inner list would have the state dimensions for each cell in the layer.
    model_config.ndim_lstm = [200, 200, 200]

    # RNN activation function: one of types.ActivationFunctions
    model_config.act_fn = types.ActivationFunction.TANH

    # Name of the TensorFlow variable scope for model variables.
    model_config.scope = "model"

    # Whether or not to use the highway connections in the RNN.
    model_config.use_highway_connections = True

    # The choice of None / L1-regularization / L2-regularization for LSTM weights.
    # One of types.RegularizationType.
    model_config.l_regularization = types.RegularizationType.NONE

    # The choice of None / L1-regularization / L2-regularization for logistic
    # weights. The weight applied to these is task specific. One of
    # types.RegularizationType.
    model_config.logistic_l_regularization = types.RegularizationType.NONE

    # The weight used in L1/L2 regularization for LSTM weights.
    model_config.l_reg_factor_weight = 0.0

    # Coefficient for leaky relu activation functions.
    model_config.leaky_relu_coeff = 0.2

    # Cell type for the model: one of types.RNNCellType
    rnn_cell = kwargs.get("rnn_cell", types.RNNCellType.SRU)
    model_config.cell_type = rnn_cell

    # Number of steps for which event sequence will be unrolled.
    model_config.num_unroll = shared.num_unroll

    # Batch size
    model_config.batch_size = shared.batch_size

    # Number of parallel iterations for dynamic RNN.
    model_config.parallel_iterations = 1

    model_config.cell_config = get_cell_config()

    model_config.snr = get_model_snr_config()

    return model_config


def get_optimizer_config(**kwargs):
    """Gets configuration for optimizer."""
    optimizer_config = configdict.ConfigDict()
    # Learning rate scheduling. One of: ["fixed", "exponential_decay"]
    optimizer_config.learning_rate_scheduling = "exponential_decay"

    # Optimization algorithm. One of: ["SGD", "Adam", "RMSprop"].
    optimizer_config.optim_type = "Adam"

    # Adam beta1.
    optimizer_config.beta1 = 0.9

    # Adam beta2.
    optimizer_config.beta2 = 0.999

    # Norm clipping threshold applied for rnn cells (no clip if 0).
    optimizer_config.norm_clip = 0.0

    # Learning rate.
    optimizer_config.initial_learning_rate = kwargs.get("initial_learning_rate", 0.001)

    # The learning rate decay 'epoch' length.
    optimizer_config.lr_decay_steps = 12000

    # The learning rate decay base, applied per epoch.
    optimizer_config.lr_decay_base = 0.85

    # RMSprop decay.
    optimizer_config.decay = 0.9

    # RMSprop moment.
    optimizer_config.mom = 0.0

    return optimizer_config


def get_encoder_config(shared, ndim_dict=None, nact_dict=None):
    """Gets config for encoder module that embeds data."""
    encoder_config = configdict.ConfigDict()
    encoder_config.sequential_features = shared.sequential_features
    encoder_config.context_features = shared.context_features
    encoder_config.identity_lookup_features = shared.identity_lookup_features

    # Name of the TensorFlow variable scope for encoder variables.
    encoder_config.scope = "encoder"

    # Number of dimensions of the embedding layer.
    encoder_config.ndim_emb = 400

    # Dict of median number of active features for each feature type. Updated
    # at runtime.
    if nact_dict is None:
        encoder_config.nact_dict = {
            types.FeatureTypes.PRESENCE_SEQ: 10,
            types.FeatureTypes.NUMERIC_SEQ: 3,
            types.FeatureTypes.CATEGORY_COUNTS_SEQ: 3,
        }
    else:
        encoder_config.nact_dict = nact_dict

    if ndim_dict is None:
        encoder_config.ndim_dict = {
            types.FeatureTypes.PRESENCE_SEQ: 142,
            types.FeatureTypes.NUMERIC_SEQ: 11,
            types.FeatureTypes.CATEGORY_COUNTS_SEQ: 10,
        }
    else:
        encoder_config.ndim_dict = ndim_dict

    # How to combine the initial sparse multiplication. Valid options are
    # ["sum" (default), "sqrtn", "mean"]
    encoder_config.sparse_combine = "sum"

    # Number of steps for which event sequence will be unrolled.
    encoder_config.num_unroll = shared.num_unroll

    # The batch size.
    encoder_config.batch_size = shared.batch_size

    # How to combine embeddings. One of types.EmbeddingCombinationMethod
    encoder_config.embedding_combination_method = types.EmbeddingCombinationMethod.CONCATENATE

    # Embedding type enum as per types.EmbeddingType.
    encoder_config.embedding_type = types.EmbeddingType.DEEP_EMBEDDING

    # Probability of performing embedding dropout. Will not be applied to sparse
    # lookup layers. If types.EmbeddingType.LOOKUP is used, this value will be
    # ignored.
    encoder_config.embedding_dropout_prob = 0.0

    # Probability of performing embedding dropout on the sparse lookup layer.
    # If types.EmbeddingType.LOOKUP is used, this is the only dropout_prob that
    # will be used, embedding_dropout_prob will be ignored.
    encoder_config.sparse_lookup_dropout_prob = 0.0

    # Coefficient for leaky relu activation functions.
    encoder_config.leaky_relu_coeff = 0.2

    # The choice of None / L1-regularization / L2-regularization for the sparse
    # lookup embedding weights. One of types.RegularizationType.
    encoder_config.sparse_lookup_regularization = types.RegularizationType.L1

    # The weight used in L1/L2 regularization for the sparse lookup embedding
    # weights.
    encoder_config.sparse_lookup_regularization_weight = 0.00001

    # The choice of regularization for the encoder (fc or residual) weights. One
    # of types.RegularizationType.
    encoder_config.encoder_regularization = types.RegularizationType.L1

    # The weight used in L1/L2 regularization for the encoder weights.
    encoder_config.encoder_regularization_weight = 0.0

    # The weight of the loss for embeddings with a reconstruction loss.
    encoder_config.embedding_loss_weight = 1.0

    # The config for deep embeddings.
    encoder_config.deep = deep_embedding_config(shared)

    # One of types.ModelModes
    encoder_config.mode = shared["mode"]

    return encoder_config


def get_cell_config():
    """Gets the config for a RNN cell."""
    cell_config = configdict.ConfigDict()

    cell_config.leak = 0.001

    return cell_config


def shared_config(**kwargs):
    """Configuration that is needed in more than one field of main config."""
    shared = configdict.ConfigDict()

    # Features from the context of the TF context example to use. By default is
    # empty.
    shared.context_features = kwargs.get("context_features", [])
    shared.fixed_len_context_features = kwargs.get("fixed_len_context_features", [])
    shared.var_len_context_features = kwargs.get("var_len_context_features", [])

    # Features from the sequence of the TF sequence example to use. By default is
    # set to the default sequence features.
    shared.sequential_features = [
        types.FeatureTypes.PRESENCE_SEQ,
        types.FeatureTypes.NUMERIC_SEQ,
        types.FeatureTypes.CATEGORY_COUNTS_SEQ,
    ]

    # Features for which to just use a lookup embedding.
    shared.identity_lookup_features = kwargs.get("identity_lookup_features", [types.FeatureTypes.CATEGORY_COUNTS_SEQ])

    # Number of steps for which event sequence will be unrolled.
    shared.num_unroll = 128

    # The batch size.
    shared.batch_size = 128

    # The mode: one of types.ModelMode
    shared.mode = [types.ModelMode.TRAIN]

    # Each event sequence will chopped into segments of this number of steps.
    # This should be a multiple of num_unroll and is only used in "fast" training
    # mode with pad_sequences=True. RNN states will not be propagated across
    # segment boundaries; as far as the model is concerned, each segment is a
    # complete sequence.
    shared.segment_length = shared.num_unroll * 2

    # Tasks to include in the experiment.
    shared.tasks = kwargs.get("tasks", (types.TaskNames.ADVERSE_OUTCOME_RISK, types.TaskNames.LAB_REGRESSION))

    shared.encoder_layer_sizes = kwargs.get(
        "encoder_layer_sizes", (2 * [400], 2 * [400], [], [400, 400], [19, 400], [37, 400], [21, 400], [400, 400])
    )

    return shared


def deep_embedding_config(shared):
    """Options for Deep embeddings."""
    config = configdict.ConfigDict()

    # Embedding activation function.
    # One of: ["tanh", "relu", "lrelu", "swish", "elu", "selu", "elish",
    # "hard_elish", "sigmoid", "hard_sigmoid", "tanh_pen"].
    config.embedding_act = "tanh"

    # Type of encoder to use: types.EmbeddingEncoderType
    config.encoder_type = types.EmbeddingEncoderType.RESIDUAL

    # Encoder layer sizes for presence, numeric and category_counts features in
    # order.
    # Will be mapped to a dict by get_sizes_for_all_features()
    # For FC and residual encoder, the architecture for each feature type is
    # defined by a list specifying the number of units per layer.
    config.encoder_layer_sizes = shared.get("encoder_layer_sizes")

    config.arch_args = configdict.ConfigDict()
    config.arch_args.use_highway_connection = True
    config.arch_args.use_batch_norm = False
    config.arch_args.activate_final = False

    config.tasks = shared.tasks

    # Set configs for SNREncoder
    config.snr = snr_config()

    return config


def get_model_snr_config():
    """SNR related parameters used in the model."""
    config = configdict.ConfigDict()

    # Set parameters for the hard sigmoid
    config.zeta = 3.0
    config.gamma = -1.0
    config.beta = 1.0

    # Set the regularization weight for SNRConnections.
    config.subnetwork_conn_l_reg_factor_weight = 0.0001

    # When set to true it will pass all RNN cell outputs to the tasks.
    # If set to false then it passes only the outputs from the cells on the last
    # layer.
    config.should_pass_all_cell_outputs = True

    # Specify the type of connection between two sub-networks. One of
    # types.SubNettoSubNetConnType.
    config.subnetwork_to_subnetwork_conn_type = types.SubNettoSubNetConnType.BOOL

    # When set to true, it will create a unique routing connection for each input
    # and each task for all RNN cells.
    config.use_task_specific_routing = False

    # Specify how to combine inputs to subnetworks.
    config.input_combination_method = types.SNRInputCombinationType.CONCATENATE

    return config


def snr_config():
    """Options for SNREncoder."""
    config = configdict.ConfigDict()

    # Set parameters for the hard sigmoid
    config.zeta = 3.0
    config.gamma = -1.0
    config.beta = 1.0

    # Set parameters for regularizing the SNREncoder
    config.subnetwork_weight_l_reg_factor_weight = 0.0
    config.subnetwork_weight_l_reg = types.RegularizationType.L2

    config.subnetwork_conn_l_reg_factor_weight = 0.0001

    # Whether to use skip connections in SNREncoder
    config.use_skip_connections = True

    # Whether to use activation before aggregation
    config.activation_before_aggregation = False

    # Specify the type of unit to use in SNREncoder. One of
    # types.SNRBlockConnType
    config.snr_block_conn_type = types.SNRBlockConnType.NONE

    # Specify the type of connection between two sub-networks. One of
    # types.SubNettoSubNetConnType.
    config.subnetwork_to_subnetwork_conn_type = types.SubNettoSubNetConnType.BOOL

    # When set to true, it will create a unique routing connection for each input
    # and each task in all subnetworks of the encoder.
    config.use_task_specific_routing = False

    # Specify how to combine inputs to subnetworks.
    config.input_combination_method = types.SNRInputCombinationType.CONCATENATE

    return config


def get_checkpoint_config(**kwargs):
    """Gets configuration for checkpointing."""
    checkpoint_config = configdict.ConfigDict()

    # Directory for writing checkpoints.
    checkpoint_config.checkpoint_dir = kwargs.get("checkpoint_dir", "")

    # How frequently to checkpoint the model (in seconds).
    checkpoint_config.checkpoint_every = 3600  # 1 hour

    checkpoint_every_steps = kwargs.get("checkpoint_every_steps", 100)
    checkpoint_config.checkpoint_every_steps = checkpoint_every_steps

    summary_every_steps = kwargs.get("summary_every_steps", 100)
    checkpoint_config.summary_every_steps = summary_every_steps

    # TTL for the training model checkpoints in days.
    checkpoint_config.ttl = 120

    return checkpoint_config
