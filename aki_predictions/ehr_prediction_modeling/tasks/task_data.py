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
"""Classes for working with task data."""
from typing import Dict, List, Mapping

import attr
import tensorflow.compat.v1 as tf


@attr.s(auto_attribs=True)
class TaskVariables(object):
    """Variables returned by a task with the tf API."""

    targets: tf.Tensor
    train_mask: tf.Tensor
    eval_mask_dict: Dict[str, tf.Tensor]
    loss: tf.Tensor = None
    predictions: tf.Tensor = None
    eval_losses: Dict[str, tf.Tensor] = {}
    target_names: tf.Tensor = []
    auxiliary_losses: Mapping[str, tf.Tensor] = {}
    training_metrics: Mapping[str, tf.Tensor] = {}


@attr.s(auto_attribs=True)
class CoordinatorVariables(object):
    """Variables returned by a task coordinator with the tf API."""

    combined_loss: tf.Tensor
    combined_eval_loss: tf.Tensor
    task_variables_list: List[TaskVariables]
    task_loss_weight_dict: Dict[str, tf.Tensor]
