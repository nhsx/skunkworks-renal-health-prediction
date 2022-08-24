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
"""Classes to handle different types of data used for metrics in evaluation."""

import abc
from typing import Any, List, Mapping, Optional, TypeVar
import logging

import six

from aki_predictions.ehr_prediction_modeling.metrics.calculators import classification_task
from aki_predictions.ehr_prediction_modeling.metrics.calculators import regression_task


logger = logging.getLogger(__name__)

MAX_METRICS_KEY_LENGTH = 64

MetricsTypes = TypeVar("MetricsTypes", bound="MetricsData")


@six.add_metaclass(abc.ABCMeta)
class MetricsData:
    """Base class for writing metrics data."""

    @abc.abstractmethod
    def get_metrics(self) -> Mapping[str, Any]:
        """Returns metrics calculated from data."""

    @abc.abstractmethod
    def add_data(self):
        """Extends object's lists of data."""

    def log_metrics(self, target_name: str, current_step: int, silence=True, class_threshold=0.5) -> None:
        """Logs metrics calculated from data.

        Args:
          target_name: The unique name of the target the current data represents.
          current_step: Step metrics were calculated at.

        Returns:
          (dict): Metrics dict.
        """
        metrics = self.get_metrics(class_threshold=class_threshold)
        if not silence:
            logger.info("Metrics for target (%s) at step %s:\n%s", target_name, current_step, metrics)
        return metrics


class BinaryMetricsData(MetricsData):
    """Accumulates data for metric calculation from binary predictions."""

    def __init__(
        self,
        positive_predictions: Optional[List[float]] = None,
        negative_predictions: Optional[List[float]] = None,
        positive_weights: Optional[List[float]] = None,
        negative_weights: Optional[List[float]] = None,
    ):
        """Initialise BinaryMetricsData object."""
        self.positive_predictions = positive_predictions or []
        self.negative_predictions = negative_predictions or []
        self.positive_weights = positive_weights or []
        self.negative_weights = negative_weights or []

    def __eq__(self, other):
        """Return logical equality comparison for predictions and weights."""
        if not isinstance(other, BinaryMetricsData):
            # Don't attempt to compare against unrelated types.
            return NotImplemented

        return (
            self.positive_predictions == other.positive_predictions
            and self.negative_predictions == other.negative_predictions
            and self.positive_weights == other.positive_weights
            and self.negative_weights == other.negative_weights
        )

    def __str__(self):
        """Return string representation of predictions and weights."""
        return str(
            {
                "positive_predictions": self.positive_predictions,
                "negative_predictions": self.negative_predictions,
                "positive_weights": self.positive_weights,
                "negative_weights": self.negative_weights,
            }
        )

    def add_data(self, positive_predictions, negative_predictions, positive_weights, negative_weights):
        """Extends object's lists of data.

        Args:
          positive_predictions: list of floats (between 0 and 1), predictions where
            the target is 1.
          negative_predictions: list of floats (between 0 and 1), predictions where
            the target is 0.
          positive_weights: list of floats, weights to associate with positive
            predictions.
          negative_weights: list of floats, weights to associate with negative
            predictions.

        Raises:
          ValueError if the length of positive_predictions doesn't match the length
             of positive_weights; or if the length of negative_predictions doesn't
             match the length of negative_weights.
        """
        if len(positive_predictions) != len(positive_weights):
            raise ValueError(
                "Length of positive predictions and weights passed to "
                "BinaryMetricsData do not match: lengths %d versus %d "
                % (len(positive_predictions), len(positive_weights))
            )
        if len(negative_predictions) != len(negative_weights):
            raise ValueError(
                "Length of negative predictions and weights passed to "
                "BinaryMetricsData do not match: lengths %d, %d " % (len(negative_predictions), len(negative_weights))
            )
        self.positive_predictions.extend(positive_predictions)
        self.negative_predictions.extend(negative_predictions)
        self.positive_weights.extend(positive_weights)
        self.negative_weights.extend(negative_weights)

    def get_metrics(self, class_threshold=0.5) -> Mapping[str, Any]:
        """See base class."""
        metric_calculator = classification_task.ClassificationTaskMetricCalculator(
            self.positive_predictions,
            self.negative_predictions,
            self.positive_weights,
            self.negative_weights,
            class_threshold=class_threshold,
        )
        return metric_calculator.get_metrics_dict()


class RegressionMetricsData(MetricsData):
    """Accumulates data for metric calculation from regression predictions."""

    def __init__(
        self,
        predictions: Optional[List[float]] = None,
        targets: Optional[List[float]] = None,
        weights: Optional[List[float]] = None,
    ) -> None:
        """Initialise RegressionMetricsData object."""
        self.predictions = predictions or []
        self.targets = targets or []
        self.weights = weights or []

    def __eq__(self, other):
        """Equality comparison magic method."""
        if not isinstance(other, RegressionMetricsData):
            # Don't attempt to compare against unrelated types.
            return NotImplemented
        return self.predictions == other.predictions and self.targets == other.targets and self.weights == other.weights

    def __str__(self):
        """String formatting magic method."""
        return str({"predictions": self.predictions, "targets": self.targets, "weights": self.weights})

    def add_data(self, predictions, targets, weights):
        """Extends object's lists of data.

        Args:
          predictions: list of floats, predictions for a regression target (may have
            value >1).
          targets: list of floats, targets for a regression target (may have value
            >1).
          weights: list of floats, weights to associate with predictions.

        Raises:
          ValueError if the length of predictions, targets, and weights are not all
            equal.
        """
        if not all(len(elt) == len(predictions) for elt in [targets, weights]):
            raise ValueError(
                "Lengths of predictions, targets, and weights passed to "
                "RegressionMetricsData do not match: lengths %d, %d, %d "
                % (len(predictions), len(targets), len(weights))
            )
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        self.weights.extend(weights)

    def get_metrics(self) -> Mapping[str, Any]:
        """See base class."""
        metric_calculator = regression_task.RegressionTaskMetricCalculator(self.predictions, self.targets, self.weights)
        return metric_calculator.get_metrics_dict()
