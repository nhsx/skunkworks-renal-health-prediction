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
"""Metric calculator for regression tasks."""

import math
from typing import List, Union

import numpy as np

from aki_predictions.ehr_prediction_modeling.metrics.calculators import base


ListOrArray = Union[List[Union[float, int]], np.ndarray]


class RegressionTaskMetricCalculator(base.MetricCalculatorBase):
    """Calculator class for evaluation metrics for regression tasks."""

    ALL_METRICS = ["l1", "l1_percentage", "std", "nb_ex", "target_mean", "target_std"]

    def __init__(self, preds: ListOrArray, targets: ListOrArray, weights: ListOrArray) -> None:
        """Initialize the auxiliary metrics calculator.

        Args:
          preds: The prediction values.
          targets: The target values.
          weights: The target weights.
        """
        self._preds = np.asarray(preds)
        self._targets = np.asarray(targets)
        self._abs_pred_target_diff = np.absolute(self._preds - self._targets)
        self._abs_percentage_pred_target_diff = np.absolute((self._preds - self._targets) / self._targets)
        self._weights = weights

    def nb_ex(self) -> int:
        """Get the total number of examples over which to compute metrics."""
        return np.sum(self._weights)

    def l1(self) -> float:
        """l1 calculation."""
        if np.sum(self._weights) > 0:
            return np.average(self._abs_pred_target_diff, weights=self._weights)
        return np.nan

    def l1_percentage(self) -> float:
        """l1 percentage calculation."""
        if np.sum(self._weights) > 0:
            return np.average(self._abs_percentage_pred_target_diff, weights=self._weights)
        return np.nan

    def std(self) -> float:
        """Standard deviation calculation."""
        if np.sum(self._weights) > 0:
            variance = np.average(self._abs_pred_target_diff**2, weights=self._weights)
            return math.sqrt(variance)
        return np.nan

    def target_mean(self) -> float:
        """Mean calculation for target."""
        return np.average(self._targets)

    def target_std(self) -> float:
        """Standard deviation calculation for target."""
        return np.std(self._targets)
