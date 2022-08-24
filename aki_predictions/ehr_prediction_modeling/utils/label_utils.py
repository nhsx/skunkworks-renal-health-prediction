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
"""Utility functions used in label computations."""
from typing import List

# Open Source Labels
ADVERSE_OUTCOME_IN_ADMISSION = "adverse_outcome_in_admission"
MORTALITY_IN_ADMISSION_LABEL = "mortality_in_admission"
MORTALITY_LOOKAHEAD_LABEL_BASE = "mortality_in"
READMISSION_LABEL_BASE = "readmission_within"
TIME_UNTIL_NEXT_ADMISSION = "time_until_next_admission"
ITU_OUTCOME_IN_ADMISSION = "itu_outcome_in_admission"
DIALYSIS_OUTCOME_IN_ADMISSION = "dialysis_outcome_in_admission"
MORTALITY_OUTCOME_IN_ADMISSION = "mortality_outcome_in_admission"

SEGMENT_LABEL = "segment_mask"
CENSORED_PATIENT_LABEL = "patient_mask"

LAB_LOOKAHEAD_REGEX_BASE = r"lab_[0-9]+_value_within"

LOS_LABEL = "length_of_stay"

TIMESTAMP_KEY = "timestamp"
TSA_LABEL = "time_since_admission"
IGNORE_LABEL = "ignore_label"
TOD_LABEL = "time_of_day_label"
DISCHARGE_LABEL = "discharge_label"

# In the early stages of the project the focus was on longer time windows. Due
# to the clinical relevance we have now extended this list to be more finely
# granular on the low-end, with multiple shorter time windows 6h apart. For
# backwards compatibility, we still compute the longer windows - in order to be
# able to compare to our past performance. However, note that these should not
# really be used in models/evaluation going forward.
DEFAULT_LOOKAHEAD_WINDOWS = [6, 12, 18, 24, 36, 48, 60, 72]

C308_LOOKAHEAD_WINDOWS = [6, 12, 18, 24, 30, 36, 42, 48]


def get_lab_label_lookahead_key(lab_number: str, time_window_hours: int, suffix=None) -> str:
    """Return lab label lookahead key."""
    if not suffix:
        return f"lab_{lab_number}_in_{time_window_hours}h"
    else:
        return f"lab_{lab_number}_in_{time_window_hours}h_{suffix}"


def get_adverse_outcome_lookahead_label_key(time_window_hours: int) -> str:
    """Returns the lookahead label key for the provided time window in hours."""
    return f"adverse_outcome_within_{time_window_hours}h"


def get_itu_outcome_lookahead_label_key(time_window_hours: int) -> str:
    """Returns the lookahead label key for the provided time window in hours."""
    return f"adverse_outcome_itu_within_{time_window_hours}h"


def get_dialysis_outcome_lookahead_label_key(time_window_hours: int) -> str:
    """Returns the lookahead label key for the provided time window in hours."""
    return f"adverse_outcome_dialysis_within_{time_window_hours}h"


def get_mortality_outcome_lookahead_label_key(time_window_hours: int) -> str:
    """Returns the lookahead label key for the provided time window in hours."""
    return f"adverse_outcome_mortality_within_{time_window_hours}h"


def get_readmission_label_keys(time_windows: List[int]) -> List[str]:
    """Get label keys for readmission.

    Args:
      time_windows: list<int> of the considered time windows (in days) for
                    readmission.

    Returns:
      list<str> of labels for readmission within X days
    """
    return [f"{READMISSION_LABEL_BASE}_{t}_days" for t in time_windows]
