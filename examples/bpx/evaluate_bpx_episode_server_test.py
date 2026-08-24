import numpy as np
import pytest

import evaluate_bpx_episode_server as evaluation


def test_select_query_indices_covers_episode_when_limited():
    np.testing.assert_array_equal(evaluation.select_query_indices(100, 10, None), np.arange(0, 100, 10))
    selected = evaluation.select_query_indices(100, 10, 4)
    assert selected[0] == 0
    assert selected[-1] == 90
    assert len(selected) == 4


def test_compute_metrics_masks_short_final_horizon_and_normalizes():
    targets = np.asarray([[[1.0, 2.0], [2.0, 4.0]], [[3.0, 6.0], [np.nan, np.nan]]])
    predictions = np.asarray([[[1.5, 1.0], [1.0, 5.0]], [[2.0, 8.0], [100.0, 100.0]]])
    full_actions = np.asarray([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]])
    metrics = evaluation.compute_metrics(predictions, targets, full_actions)
    assert metrics["aggregate_mae"] == pytest.approx(1.0833333333333333)
    assert len(metrics["per_joint_mae"]) == 2
    assert len(metrics["per_horizon_mae"]) == 2
    assert np.isfinite(metrics["aggregate_normalized_mae"])


@pytest.mark.parametrize("frames,stride,max_queries", [(0, 1, None), (10, 0, None), (10, 1, 0)])
def test_select_query_indices_rejects_invalid_arguments(frames, stride, max_queries):
    with pytest.raises(ValueError):
        evaluation.select_query_indices(frames, stride, max_queries)
