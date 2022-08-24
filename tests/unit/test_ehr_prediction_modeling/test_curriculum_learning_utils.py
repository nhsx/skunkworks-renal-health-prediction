import pytest
import numpy as np

from aki_predictions.ehr_prediction_modeling.utils import curriculum_learning_utils


class TestUpdateCurriculum:
    @pytest.mark.parametrize(
        "previous_record_order, expected",
        [(np.array([0, 1]), np.array([0.9, 1.1])), (np.array([1, 0]), np.array([1.1, 0.9]))],
    )
    def test_curriculum_elements_change_as_expected(self, previous_record_order, expected):
        curriculum = np.array([1.0, 1.0])
        losses_by_batch = [np.array([0.9, 1.1])]
        num_records = len(curriculum)
        new_curriculum = curriculum_learning_utils.update_curriculum(
            curriculum, num_records, previous_record_order, losses_by_batch, curr_epoch=1
        )
        np.testing.assert_allclose(new_curriculum, expected)

    def test_curriculum_update_can_handle_long_losses_by_batch(self):
        curriculum = np.array([1.0, 1.0])
        losses_by_batch = [np.array([0.9, 1.1]), np.array([1.0, 1.0])]
        num_records = len(curriculum)
        previous_record_order = np.array([0, 1])
        new_curriculum = curriculum_learning_utils.update_curriculum(
            curriculum, num_records, previous_record_order, losses_by_batch, curr_epoch=1
        )
        expected = np.array([0.9, 1.1])
        np.testing.assert_allclose(new_curriculum, expected)

    def test_curriculum_stays_in_bounds(self):
        curriculum = np.array([1.0, 1.0])
        losses_by_batch = [np.array([0.01, 1000])]
        num_records = len(curriculum)
        previous_record_order = np.array([0, 1])
        min_prob = 0.1
        max_prob = 3.0
        new_curriculum = curriculum_learning_utils.update_curriculum(
            curriculum,
            num_records,
            previous_record_order,
            losses_by_batch,
            min_prob=min_prob,
            max_prob=max_prob,
            curr_epoch=1,
        )
        assert np.min(new_curriculum) >= min_prob
        assert np.max(new_curriculum) <= max_prob

    @pytest.mark.parametrize("offset, factor", ([0.0, 1.0], [-1.0, 1.0], [0.0, 0.0]))
    def test_curriculum_stays_in_bounds_w_auto_bounds(self, offset, factor):
        curriculum = np.ones(1000)
        losses_by_batch = [(np.random.rand(1000) + offset) * factor]
        num_records = len(curriculum)
        previous_record_order = list(range(1000))
        min_prob = {"method": "percentile", "thresh": 10}
        max_prob = {"method": "percentile", "thresh": 90}
        min_prob_expected = np.percentile(losses_by_batch, 10) - offset
        max_prob_expected = np.percentile(losses_by_batch, 90) - offset
        new_curriculum = curriculum_learning_utils.update_curriculum(
            curriculum,
            num_records,
            previous_record_order,
            losses_by_batch,
            min_prob=min_prob,
            max_prob=max_prob,
            curr_epoch=1,
        )
        if offset >= 0 and factor > 0:
            # sensible case where losses haven't gone negative
            assert np.min(new_curriculum) >= min_prob_expected
            assert np.max(new_curriculum) <= max_prob_expected
        assert np.min(new_curriculum) > 0
        assert np.max(new_curriculum) > 0

    def test_no_curriculum_change_if_below_start_epoch(self):
        curriculum = np.array([1.0, 1.0])
        losses_by_batch = [np.array([0.01, 1000])]
        num_records = len(curriculum)
        previous_record_order = np.array([0, 1])
        new_curriculum = curriculum_learning_utils.update_curriculum(
            curriculum, num_records, previous_record_order, losses_by_batch, curr_epoch=0, start_epoch=1
        )
        np.testing.assert_array_equal(curriculum, new_curriculum)


class TestSampleCurriculum:
    def test_roughly_correct_samples_drawn(self):
        curriculum = np.array([1, 9])
        num_samples = 100000
        samples = curriculum_learning_utils.sample_curriculum(curriculum, num_samples, epoch=0)
        sample_nums = [sum(samples == el) for el in [0, 1]]
        np.testing.assert_almost_equal(sample_nums[0] / num_samples, curriculum[0] / sum(curriculum), decimal=2)
        np.testing.assert_almost_equal(sample_nums[1] / num_samples, curriculum[1] / sum(curriculum), decimal=2)

    @pytest.mark.parametrize("epoch, expected", [(0, 0.1), (5, 1 / 6), (10, 0.5)])
    def test_roughly_correct_samples_drawn_with_training_progress_provided(self, epoch, expected):
        curriculum = np.array([1, 9])
        num_samples = 100000
        samples = curriculum_learning_utils.sample_curriculum(curriculum, num_samples, epoch=epoch, final_epoch=10)
        sample_nums = [sum(samples == el) for el in [0, 1]]
        np.testing.assert_almost_equal(sample_nums[0] / num_samples, expected, decimal=2)
