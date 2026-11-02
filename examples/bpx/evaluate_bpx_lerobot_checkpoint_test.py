"""Run on the server; no model loading in these unit specifications."""
import json

import evaluate_bpx_lerobot_checkpoint as evaluation
from compare_bpx_offline_evaluations import PAIRING_KEYS, check_pairing
import numpy as np
import pytest


def test_query_frames_include_last_complete_chunk_and_never_padding():
    np.testing.assert_array_equal(evaluation.query_frames(42, 30, 5), [0, 5, 10, 12])
    np.testing.assert_array_equal(evaluation.query_frames(30, 30, 5), [0])


@pytest.mark.parametrize("values", [(10, 30, 5), (42, 0, 5), (42, 30, 0)])
def test_bad_query_contract_fails(values):
    with pytest.raises(ValueError):
        evaluation.query_frames(*values)


def test_noise_is_paired_but_varies_with_origin_and_seed():
    first = evaluation.fixed_noise(101, 20, 5, (30, 32))
    np.testing.assert_array_equal(first, evaluation.fixed_noise(101, 20, 5, (30, 32)))
    assert first.dtype == np.float32
    assert not np.array_equal(first, evaluation.fixed_noise(202, 20, 5, (30, 32)))
    assert not np.array_equal(first, evaluation.fixed_noise(101, 20, 10, (30, 32)))


def test_metrics_do_not_hide_bad_seeds_by_averaging_before_scoring():
    predictions = np.asarray([[[[1.]], [[-1.]]]])
    metrics = evaluation.action_errors(predictions, np.zeros((1, 1, 1)))
    assert metrics["per_joint_mae"] == [1.]
    assert metrics["per_joint_rmse"] == [1.]


def test_metrics_reject_nonfinite():
    with pytest.raises(ValueError, match="Non-finite"):
        evaluation.action_errors(np.full((1, 1, 1, 1), np.nan), np.zeros((1, 1, 1)))


def test_recommended_path_resolves_without_requiring_optimizer(tmp_path):
    candidate = tmp_path / "selected_checkpoints/18000"
    (candidate / "params").mkdir(parents=True)
    (candidate / "assets").mkdir()
    recommendation = tmp_path / "recommended.json"
    recommendation.write_text(json.dumps({"checkpoint_path": str(candidate)}))
    assert evaluation.resolve_checkpoint(recommendation) == candidate.resolve()
    assert not (candidate / "train_state").exists()


@pytest.mark.parametrize("key", PAIRING_KEYS)
def test_comparison_rejects_changed_data_prompt_noise_or_norm(key):
    left = dict.fromkeys(PAIRING_KEYS, "fixed")
    right = {**left, key: "changed"}
    with pytest.raises(ValueError, match=key):
        check_pairing([left, right])
