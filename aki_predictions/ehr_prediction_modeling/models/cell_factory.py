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
"""Factory for creation of cells."""
import functools
import typing

import sonnet as snt
import tensorflow.compat.v1 as tf
from tensorflow import contrib as tf_contrib

from aki_predictions.ehr_prediction_modeling import types
from aki_predictions.ehr_prediction_modeling.models import reset_core


if typing.TYPE_CHECKING:
    from aki_predictions.ehr_prediction_modeling import configdict


def _map_cell_type(config: "configdict.ConfigDict") -> snt.AbstractModule:
    """Returns cell corresponding to cell_type string."""
    cells_not_allowing_wrappers = [
        types.RNNCellType.SRU,
    ]
    cell_construction_mapping = {
        types.RNNCellType.LSTM: functools.partial(
            tf.nn.rnn_cell.LSTMCell, num_units=config.ndim_lstm, activation=config.activation_fn
        ),
        # Simple Recurrent Unit. Faster training.
        types.RNNCellType.SRU: functools.partial(
            tf_contrib.rnn.SRUCell, num_units=config.ndim_lstm, activation=config.activation_fn
        ),
        types.RNNCellType.UGRNN: functools.partial(
            tf_contrib.rnn.UGRNNCell, num_units=config.ndim_lstm, activation=config.activation_fn
        ),
    }

    if config.cell_type not in cell_construction_mapping:
        raise ValueError(f"Invalid cell_type: {config.cell_type}")

    cell = cell_construction_mapping[config.cell_type]()

    if config.cell_type in cells_not_allowing_wrappers:
        # Certain cells do not support wrappers.
        return cell

    if config.use_highway_connections:
        cell = tf_contrib.rnn.HighwayWrapper(cell)
    return cell


def init_deep_cell(cell_config: "configdict.ConfigDict") -> reset_core.ResetCore:
    """Initializes and wraps a cell in a ResetCore.

    Args:
      cell_config: Cell ConfigDict.

    Returns:
      An RNNCore object.
    """
    cell = _map_cell_type(cell_config)
    return reset_core.ResetCore(cell)
