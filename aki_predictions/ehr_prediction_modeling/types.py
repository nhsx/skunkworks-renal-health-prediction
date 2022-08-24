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
"""Definitions for namedtuples and enums used throughout experiments."""
import collections


class ForwardFunctionReturns(
    collections.namedtuple(
        "ForwardFunctionReturns",
        ["model_output", "hidden_states", "inputs", "activations"],
    )
):
    """The variables returned by the forward function."""


class EmbeddingType(object):
    """Available embedding types."""

    LOOKUP = "Lookup"
    DEEP_EMBEDDING = "DeepEmbedding"


class EmbeddingEncoderType(object):
    """Available encoders for DeepEmbedding."""

    RESIDUAL = "residual"
    FC = "fc"
    SNR = "snr"


class ActivationFunction(object):
    """Available activation functions for cells in lstm_model."""

    TANH = "tanh"
    RELU = "relu"
    LRELU = "lrelu"
    SWISH = "swish"
    ELU = "elu"
    SLU = "selu"
    ELISH = "elish"
    HARD_ELISH = "hard_elish"
    SIGMOID = "sigmoid"
    HARD_SIGMOID = "hard_sigmoid"
    TANH_PEN = "tanh_pen"


class RegularizationType(object):
    """Available types of regularization for model and embedding weights."""

    NONE = "None"
    L1 = "L1"
    L2 = "L2"


class TaskType(object):
    """Available task types."""

    BINARY_CLASSIFICATION = "BinaryClassification"
    REGRESSION = "Regression"


class BestModelMetric(object):
    """Best model metric object."""

    PRAUC = "prauc"
    ROCAUC = "rocauc"
    L1 = "l1"


class TaskNames(object):
    """Names of available tasks."""

    ADVERSE_OUTCOME_RISK = "AdverseOutcomeRisk"
    LAB_REGRESSION = "Labs"
    BINARIZED_LOS = "BinarizedLengthOfStay"
    REGRESSION_LOS = "RegressionLengthOfStay"
    MORTALITY = "MortalityRisk"
    READMISSION = "ReadmissionRisk"
    ITU_OUTCOME = "ITUOutcome"
    DIALYSIS_OUTCOME = "DialysisOutcome"
    MORTALITY_OUTCOME = "MortalityOutcome"


class TaskTypes(object):
    """Type of tasks available for modeling."""

    ADVERSE_OUTCOME_RISK = "ao_risk"
    LAB_REGRESSION = "labs_regression"
    LOS = "length_of_stay"
    MORTALITY = "mortality"
    READMISSION = "readmission"
    ITU_OUTCOME = "itu_outcome"
    DIALYSIS_OUTCOME = "dialysis_outcome"
    MORTALITY_OUTCOME = "mortality_outcome"


class TaskLossType(object):
    """Task loss type object."""

    CE = "CE"
    MULTI_CE = "MULTI_CE"
    BRIER = "Brier"
    L1 = "L1"
    L2 = "L2"


class RNNCellType(object):
    """Available cell types for rnn_model."""

    SRU = "SRU"
    LSTM = "LSTM"
    UGRNN = "UGRNN"


class FeatureTypes(object):
    """Types of features available in the data representation."""

    PRESENCE_SEQ = "pres_s"  # Sequential presence features
    NUMERIC_SEQ = "num_s"  # Sequential numerical features
    CATEGORY_COUNTS_SEQ = "count_s"  # Sequential category counts features
    DIAGNOSIS = "diagnosis"
    ETHNIC_ORIGIN = "ethnic_origin"
    METHOD_OF_ADMISSION = "method_of_admission"
    SEX = "sex"
    YEAR_OF_BIRTH = "year_of_birth"


# Mappings from feature names in the model to names in tf.SequenceExample.
FEAT_TO_NAME = {
    "pres_s": "presence",
    "num_s": "numeric",
    "count_s": "category_counts",
    "diagnosis": "diagnosis",
    "ethnic_origin": "ethnic_origin",
    "method_of_admission": "method_of_admission",
    "sex": "sex",
    "year_of_birth": "year_of_birth",
}


class Optimizers(object):
    """Optimizers object definitions."""

    SGD = "SGD"
    ADAM = "Adam"
    RMSPROP = "RMSprop"


class LearningRateScheduling(object):
    """Learning rate scheduling definitions."""

    FIXED = "fixed"
    EXPONENTIAL_DECAY = "exponential_decay"


class EmbeddingCombinationMethod(object):
    """Available embedding combination methods."""

    CONCATENATE = "concatenate"
    SUM_ALL = "sum_all"
    SUM_BY_SUFFIX = "sum_by_suffix"
    COMBINE_SNR_OUT = "combine_snr_out"


class SNRBlockConnType(object):
    """Available unit types for SNREncoder."""

    FC = "fc"
    HIGHWAY = "highway"
    RESIDUAL = "residual"
    NONE = "None"


class SubNettoSubNetConnType(object):
    """Available connection types between subnetworks."""

    BOOL = "bool"
    SCALAR_WEIGHT = "scalar_weight"


class SNRInputCombinationType(object):
    """Available combination methods for subnetwork input."""

    CONCATENATE = "concatenate"
    SUM_ALL = "sum"


class LossCombinationType(object):
    """Loss combination type object."""

    SUM_ALL = "SUM_ALL"
    UNCERTAINTY_WEIGHTED = "UNCERTAINTY_WEIGHTED"


class ModelTypes(object):
    """Model types object."""

    RNN = "RNN"
    SNRNN = "SNRNN"


class TaskLayerTypes(object):
    """Task layer types object."""

    NONE = "none"
    MLP = "MLP"
    SNRMLP = "SNRMLP"


class ModelMode(object):
    """Available modes for the model."""

    TRAIN = "train"
    EVAL = "eval"
    PREDICT = "predict"

    @classmethod
    def is_train(cls, mode: str) -> bool:
        """Class method check if mode is TRAIN."""
        return mode == ModelMode.TRAIN
