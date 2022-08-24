import mock
import tensorflow.compat.v1 as tf

from aki_predictions.training import multiple_adverse_outcomes_w_curriculum
from aki_predictions.training import multiple_adverse_outcomes_training
from aki_predictions.training import multiple_adverse_outcomes_w_context_training
from aki_predictions.ehr_prediction_modeling.utils import curriculum_learning_utils


def test_single_iteration_w_multiple_adverse_outcomes_completes(multiple_adverse_outcomes_no_shuffle_config):
    tf.keras.backend.clear_session()
    multiple_adverse_outcomes_training.run(multiple_adverse_outcomes_no_shuffle_config)


def test_curriculum_updated_during_experiment(
    curriculum_multiple_adverse_outcomes_eval_config,
    curriculum_multiple_adverse_outcomes_no_shuffle_config,
    monkeypatch,
):
    tf.keras.backend.clear_session()
    update_patch = mock.Mock(wraps=curriculum_learning_utils.update_curriculum)
    monkeypatch.setattr(curriculum_learning_utils, "update_curriculum", update_patch)

    sample_patch = mock.Mock(wraps=curriculum_learning_utils.sample_curriculum)
    monkeypatch.setattr(curriculum_learning_utils, "sample_curriculum", sample_patch)

    multiple_adverse_outcomes_w_curriculum.run(
        curriculum_multiple_adverse_outcomes_no_shuffle_config, curriculum_multiple_adverse_outcomes_eval_config
    )
    assert update_patch.call_count > 0
    assert sample_patch.call_count > 0
    assert update_patch.call_args_list[0].kwargs["min_prob"]["thresh"] == 10
    assert update_patch.call_args_list[0].kwargs["max_prob"]["thresh"] == 90
    assert update_patch.call_args_list[0].kwargs["start_epoch"] == 2


def test_training_w_context_completes(multiple_adverse_outcomes_w_context_config):
    tf.keras.backend.clear_session()
    multiple_adverse_outcomes_w_context_training.run(multiple_adverse_outcomes_w_context_config)
