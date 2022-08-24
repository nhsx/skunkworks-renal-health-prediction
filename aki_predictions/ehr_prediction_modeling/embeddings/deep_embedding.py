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
"""Deep feedforward embedding with dependencies."""

import typing
from typing import Dict

import tensorflow.compat.v1 as tf

from aki_predictions.ehr_prediction_modeling import types
from aki_predictions.ehr_prediction_modeling.embeddings import input_embedding_base
from aki_predictions.ehr_prediction_modeling.models.nets import deep_encoders
from aki_predictions.ehr_prediction_modeling.utils import activations


if typing.TYPE_CHECKING:
    from aki_predictions.ehr_prediction_modeling import configdict


class DeepEmbedding(input_embedding_base.InputEmbeddingBase):
    """An Embedding with more non-linear transformations."""

    def __init__(self, encoder_config: "configdict.ConfigDict", emb_dim_dict: Dict[str, int]):
        """Initialize the deep embedding object. See base class for Args."""
        super().__init__(encoder_config, emb_dim_dict)
        self._deep_emb_config = self._config.deep
        self._act_fn = activations.get_activation(
            act_fn=self._deep_emb_config.embedding_act, lrelu_coeff=self._config.leaky_relu_coeff
        )

        # Assign functions for specified encoder architecture
        self._encoder_fn = deep_encoders.get_encoder(self._deep_emb_config.encoder_type)
        self._n_encoder_hidden = self._get_layer_sizes()

        with tf.variable_scope(None, default_name=self._name) as variable_scope:
            self.variable_scope = variable_scope

            # Initialize all the weights.
            self._initialize_weights()

    def _get_layer_sizes(self):
        """Get encoder layer sizes for all feature types based on attributes in the config object."""
        (
            pres_size,
            num_size,
            dom_size,
            diagnosis_size,
            ethnic_origin_size,
            method_of_admission_size,
            sex_size,
            year_of_birth_size,
        ) = self._deep_emb_config.encoder_layer_sizes
        return {
            types.FeatureTypes.PRESENCE_SEQ: pres_size,
            types.FeatureTypes.NUMERIC_SEQ: num_size,
            types.FeatureTypes.CATEGORY_COUNTS_SEQ: dom_size,
            types.FeatureTypes.DIAGNOSIS: diagnosis_size,
            types.FeatureTypes.ETHNIC_ORIGIN: ethnic_origin_size,
            types.FeatureTypes.METHOD_OF_ADMISSION: method_of_admission_size,
            types.FeatureTypes.SEX: sex_size,
            types.FeatureTypes.YEAR_OF_BIRTH: year_of_birth_size,
        }

    def _initialize_weights(self):
        """Initialize the weights for embedding computations."""
        # Create networks for each type of sequence and feature type
        for feat_type in self._all_features:
            self._encoders[feat_type] = self._encoder_fn(
                self._embed_dim_dict[feat_type],
                self._config.ndim_dict[feat_type],
                emb_config=self._deep_emb_config,
                enc_config=self._config,
                n_act=self._config.nact_dict[feat_type],
                n_encoder_hidden=self._n_encoder_hidden[feat_type],
                act_fn=self._act_fn,
                name="enc_{}".format(feat_type),
                identity_lookup=(feat_type in self._config.identity_lookup_features),
            )
