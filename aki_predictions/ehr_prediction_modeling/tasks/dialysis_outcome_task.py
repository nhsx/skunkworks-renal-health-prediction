from typing import List

from aki_predictions.ehr_prediction_modeling import types
from aki_predictions.ehr_prediction_modeling.utils import label_utils
from aki_predictions.ehr_prediction_modeling import configdict
from aki_predictions.ehr_prediction_modeling.tasks import adverse_outcome_task


class DialysisOutcomeRisk(adverse_outcome_task.AdverseOutcomeRisk):
    """Task for running dialysis outcome risk prediction."""

    task_type = types.TaskTypes.DIALYSIS_OUTCOME

    @property
    def lookahead_label_key_fn(self):
        """Return lookahead label keys method"""
        return label_utils.get_dialysis_outcome_lookahead_label_key

    def _get_during_stay_label(self):
        """Return outcome label."""
        return label_utils.DIALYSIS_OUTCOME_IN_ADMISSION

    def __init__(self, config: configdict.ConfigDict):
        """Initialise dialysis outcome risk class."""
        super(DialysisOutcomeRisk, self).__init__(config)

    @classmethod
    def default_configs(cls, using_curriculum=False) -> List[configdict.ConfigDict]:
        """Generates a default config object for AdverseOutcomeRisk."""
        return [
            DialysisOutcomeRisk.config(
                window_times=label_utils.C308_LOOKAHEAD_WINDOWS,
                adverse_outcome_level=1,
                train_mask="base_train",
                eval_masks=["base_eval"],
                scale_pos_weight=1.0,
                adverse_outcome_during_remaining_stay=False,
                loss_weight=1.0,
                name=types.TaskNames.DIALYSIS_OUTCOME,
                overwrite_task_type=DialysisOutcomeRisk.task_type,
                using_curriculum=using_curriculum,
            )
        ]
