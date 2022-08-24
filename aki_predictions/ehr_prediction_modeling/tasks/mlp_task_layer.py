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
"""Modeling layers used by tasks."""
from typing import Any, Mapping

import sonnet as snt
import tensorflow.compat.v1 as tf
import tf_slim as slim
import tree

from aki_predictions.ehr_prediction_modeling import types
from aki_predictions.ehr_prediction_modeling.models import model_utils
from aki_predictions.ehr_prediction_modeling.utils import activations
from aki_predictions.ehr_prediction_modeling import configdict


HiddenTaskLayerType = snt.BatchApply


class MLPLayer:
    """MLP for dealing with task modeling layers."""

    def __init__(self, config: configdict.ConfigDict, num_task_targets: int) -> None:
        """Initialise MLPLayer object."""
        self._config = config
        self._task_layer = None
        self._num_targets = num_task_targets
        self._task_layer_sizes = self._config.get("task_layer_sizes", []).copy()
        self._regularization_type = self._config.get("regularization_type", types.RegularizationType.NONE)
        self._regularization_weight = self._config.get("regularization_weight", 0.0)
        self._init_task_layer()

    def _init_task_layer(self) -> None:
        """Initializes the fully connected task-specific layer of the model."""
        self._task_layer = snt.BatchApply(snt.nets.MLP(activate_final=False, **self.layer_kwargs))

    @property
    def layer_kwargs(self) -> Mapping[str, Any]:
        """Returns mapping of kwargs used for layer construction."""
        layers = self._task_layer_sizes + [self._num_targets]
        w_initializer = slim.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        kwargs = {
            "output_sizes": layers,
            "initializers": {"w": w_initializer, "b": tf.zeros_initializer},
            "activation": activations.get_activation(self._config.get("activation", "relu")),
            "name": self._config.name,
        }
        return kwargs

    def get_hidden_layer(self) -> HiddenTaskLayerType:
        """Fetches the task layer this class manages."""
        return self._task_layer

    def get_regularization_loss(self) -> tf.Tensor:
        """Gets the regularization loss on task-specific layers."""
        regularizer = model_utils.get_regularizer(self._regularization_type, self._regularization_weight)
        if not regularizer:
            return tf.constant(0.0)
        return slim.apply_regularization(regularizer, self._task_layer.trainable_variables)

    def _layer_logits(self, model_output: tf.Tensor) -> tf.Tensor:
        """Passes model output through task-specific layer to get logits."""
        return self._task_layer(model_output)

    def get_logits(self, model_output: tf.Tensor) -> tf.Tensor:
        """Passes model output through task-specific layers and formats output.

        Args:
          model_output: tensor of shape [num_unroll, batch_size, dim_model_output]
            or List[num_unroll, batch_size, dim_model_output] when using SNRNN as a
            model.

        Returns:
          Tensor of shape [num_unroll, batch_size, num_targets].
        """
        logits = self._layer_logits(model_output)
        num_unroll, batch_size, _ = tree.flatten(model_output)[0].shape
        logits.shape.assert_is_compatible_with([num_unroll, batch_size, self._num_targets])

        # Reshape the tensor from wnt -> wnct [num_unroll, batch_size, num_targets]
        # -> [num_unroll, batch_size, channel, num_targets]
        logits = tf.expand_dims(logits, axis=2)
        return logits
