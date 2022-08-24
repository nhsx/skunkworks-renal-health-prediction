from typing import List

from aki_predictions.ehr_prediction_modeling import types
from aki_predictions.ehr_prediction_modeling.utils import label_utils
from aki_predictions.ehr_prediction_modeling import configdict
from aki_predictions.ehr_prediction_modeling.tasks import adverse_outcome_task


class ITUOutcomeRisk(adverse_outcome_task.AdverseOutcomeRisk):
    """Task for running ITU outcome risk prediction."""

    task_type = types.TaskTypes.ITU_OUTCOME

    @property
    def lookahead_label_key_fn(self):
        """Return lookahead label keys method"""
        return label_utils.get_itu_outcome_lookahead_label_key

    def _get_during_stay_label(self):
        """Return outcome label."""
        return label_utils.ITU_OUTCOME_IN_ADMISSION

    def __init__(self, config: configdict.ConfigDict):
        """Initialise ITU outcome risk class."""
        super(ITUOutcomeRisk, self).__init__(config)
        # self.task_type = types.TaskTypes.ITU_OUTCOME

    @classmethod
    def default_configs(cls, using_curriculum=False) -> List[configdict.ConfigDict]:
        """Generates a default config object for AdverseOutcomeRisk."""
        return [
            ITUOutcomeRisk.config(
                window_times=label_utils.C308_LOOKAHEAD_WINDOWS,
                adverse_outcome_level=1,
                train_mask="base_train",
                eval_masks=["base_eval"],
                scale_pos_weight=1.0,
                adverse_outcome_during_remaining_stay=False,
                loss_weight=1.0,
                name=types.TaskNames.ITU_OUTCOME,
                overwrite_task_type=ITUOutcomeRisk.task_type,
                using_curriculum=using_curriculum,
            )
        ]
