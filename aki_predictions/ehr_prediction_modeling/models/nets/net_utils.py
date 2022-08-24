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
"""Defines utilities used by modules in nets."""

from typing import Callable, Dict, Optional

import sonnet as snt
import tensorflow.compat.v1 as tf
import tf_slim as slim

_W_INITIALIZER = tf.random_normal_initializer(stddev=0.02)
_B_INITIALIZER = tf.zeros_initializer()
INITIALIZER_DICT = {"w": _W_INITIALIZER, "b": _B_INITIALIZER}


def res_block(
    x: tf.Tensor,
    nhidden: int,
    highway: bool,
    use_bias: bool = True,
    dropout_prob: float = 0.0,
    is_training: bool = True,
) -> tf.Tensor:
    """Builds a residual layer to be chained."""
    h = snt.Linear(output_size=nhidden, use_bias=use_bias)(x)

    if highway:
        # If highway, then mix data and transform using convex combination
        w = snt.Linear(output_size=nhidden, initializers=INITIALIZER_DICT)(x)
        w = tf.sigmoid(w)
        out = tf.multiply(h, w) + tf.multiply(x, 1 - w)
    else:
        # Standard residual connection
        out = h + x

    if dropout_prob > 0:
        out = slim.dropout(out, keep_prob=(1 - dropout_prob), is_training=is_training)

    return out


def activation_with_batch_norm(
    x: tf.Tensor,
    act_fn: Callable[[tf.Tensor], tf.Tensor],
    context: Optional[Dict[str, bool]] = None,
    use_bn: bool = False,
) -> tf.Tensor:
    """Apply batch normalization and activation function.

    Args:
      x: The input tensor of shape [batch_size, num_features].
      act_fn: The nonlinear activation function (e.g. ReLU or tanh).
      context: An optional dictionary with unique key 'is_training' that changes
        the behavior of batch normalization.
      use_bn: Whether to use batch normalization or not.

    Returns:
      act_fn(BN(x)) if use_bn else act_fn(x)
    """
    if not context:
        context = {}
    if use_bn:
        is_training = context.get("is_training", False)
        x = snt.BatchNorm()(x, is_training=is_training)
    return act_fn(x)
