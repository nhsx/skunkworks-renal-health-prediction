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

# Note/Changes:
#   Requires reference to the base_task_layer class in a different location
#   to the original codebase.
"""Modeling layers used by tasks."""
from typing import TypeVar

import sonnet as snt
import tensorflow.compat.v1 as tf

from aki_predictions.ehr_prediction_modeling import types
from aki_predictions.ehr_prediction_modeling.models import model_utils
from aki_predictions.ehr_prediction_modeling.models.nets import snr
from aki_predictions.ehr_prediction_modeling.tasks import base_task_layer
from aki_predictions.ehr_prediction_modeling import configdict


TaskLayers = TypeVar("TaskLayers", bound=base_task_layer.BaseTaskLayer)


class MLPLayer(base_task_layer.BaseTaskLayer):
    """MLP task modeling layer."""

    def _get_task_layer(self) -> snt.BatchApply:
        """Returns a function for creating the task layer."""
        return snt.BatchApply(snt.nets.MLP(activate_final=False, **self.layer_kwargs))


class SNRMLPLayer(base_task_layer.BaseTaskLayer):
    """Sub Network Routine MLP task modeling layer."""

    def get_regularization_loss(self) -> tf.Tensor:
        """Gets the regularization loss on task-specific layers."""
        return self._task_layer.get_layer_regularization_loss()

    def _get_task_layer(self) -> snr.SNRTaskLayer:
        """See base class."""
        return snr.SNRTaskLayer(
            regularizer=model_utils.get_regularizer(self._regularization_type, self._regularization_weight),
            is_training=self._config.get("is_training", False),
            snr_config=self._config.get("snr_config", None),
            **self.layer_kwargs,
        )


def get_default_snr_config():
    """SNR configuration related variables."""
    config = configdict.ConfigDict()

    # Set parameters for the hard sigmoid
    config.zeta = 3.0
    config.gamma = -1.0
    config.beta = 1.0

    # Set parameters for regularizing the connections.
    config.subnetwork_conn_l_reg_factor_weight = 0.0001

    # Specify the type of connection between two sub-networks. One of
    # types.SubNettoSubNetConnType.
    config.subnetwork_to_subnetwork_conn_type = types.SubNettoSubNetConnType.BOOL

    # Specify how to combine inputs to subnetworks
    config.input_combination_method = types.SNRInputCombinationType.CONCATENATE

    return config


def get_task_layer(config: configdict.ConfigDict, num_task_targets: int) -> TaskLayers:
    """Returns the task layer."""
    task_layer_mapping = {
        types.TaskLayerTypes.MLP: MLPLayer,
        types.TaskLayerTypes.SNRMLP: SNRMLPLayer,
    }
    layer_type = config.get("task_layer_type", types.TaskLayerTypes.MLP)
    if layer_type not in task_layer_mapping:
        raise ValueError(f"Unknown task layer type specified: {layer_type}")
    return task_layer_mapping[layer_type](config, num_task_targets)
