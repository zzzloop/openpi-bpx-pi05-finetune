"""CPU regression specifications; run on the server, not the local desktop."""
import pathlib

import numpy as np
import pytest

from openpi.training.checkpoint_selection import SelectionHistory, choose_candidates, external_metrics
from openpi.training.selection_probes import counterfactual_squared_l2


def _row(step, **changes):
    return {"step": step, "available": True, "checkpoint_complete": True, "action_mse": .1,
            "val_loss": .1, "val_stderr": .03, "train_eval_loss": .1, "jerk_excess": .0,
            "s_cf": 2., "s_cf_active": 1., "s_cf_active_se": .01,
            "independent_groups": 4, "semantic_reviewed": False, **changes}


def test_formula_is_squared_sum_not_rms_and_padding_is_explicit():
    left = np.asarray([[1., 2., 100.], [3., 4., 100.]])
    right = np.zeros_like(left)
    assert counterfactual_squared_l2(left, right) == 20030.
    assert counterfactual_squared_l2(left, right, [0, 1]) == 30.
    assert counterfactual_squared_l2(left, left) == 0.


def test_missing_benchmarks_never_claims_vlm_retention():
    decision = choose_candidates([_row(5000)], keep=3)
    assert decision["status"] == "provisional_offline_candidate"
    assert any("semantic_retention_unconfirmed" in message for message in decision["warnings"])


def test_smoother_early_candidate_can_beat_lowest_validation_loss():
    earlier = _row(5000, val_loss=.1, jerk_excess=0.)
    later = _row(200000, val_loss=.08, jerk_excess=100.)
    decision = choose_candidates([earlier, later], keep=2)
    assert decision["recommended_step"] == 5000
    assert len(decision["retained_steps"]) <= 2


def test_no_completed_checkpoint_no_recommendation():
    decision = choose_candidates([_row(-1, available=False, checkpoint_complete=False)], keep=3)
    assert decision["recommended_step"] is None
    assert decision["retained_steps"] == []


def test_explicit_adaptation_gate_can_reject_every_checkpoint():
    decision = choose_candidates([_row(5000)], keep=3, gates={"max_action_mse": .001})
    assert decision["status"] == "no_candidate_meets_user_gates"


def test_decline_uses_uncertainty_and_history_not_fixed_half_ratio():
    baseline = _row(-1, available=False, s_cf_active=1., s_cf_active_se=.01)
    rows = [baseline]+[_row(step, s_cf_active=.8, s_cf_active_se=.01) for step in (5000, 10000, 15000)]
    decision = choose_candidates(rows, keep=3)
    assert any("instruction_response_decline" in message for message in decision["warnings"])


def test_archived_inference_params_survive_original_file_deletion(tmp_path, monkeypatch):
    run = tmp_path / "run"
    source = run / "5000"
    for item in ("params", "assets"):
        (source / item).mkdir(parents=True)
        (source / item / "sample").write_text(item)
    history = SelectionHistory(run, {"suite_sha256": "fixture"}, keep=1, start_step=0, resume=False)
    monkeypatch.setattr(history, "render", lambda: None)
    decision = history.append(5000, _row(5000), [], {}, optimizer_step=5001, checkpoint_complete=True)
    archived = pathlib.Path(decision["checkpoint_path"])
    assert (archived / "params/sample").read_text() == "params"
    (source / "params/sample").unlink()
    assert (archived / "params/sample").read_text() == "params"
    assert not (archived / "train_state").exists()


def test_resume_rejects_different_probe_protocol(tmp_path):
    SelectionHistory(tmp_path, {"suite_sha256": "first"}, keep=3, start_step=0, resume=False)
    with pytest.raises(ValueError, match="protocol changed"):
        SelectionHistory(tmp_path, {"suite_sha256": "different"}, keep=3, start_step=100, resume=True)


def test_external_benchmark_requires_matching_baseline_and_evidence(tmp_path):
    counts = {"correct": 7, "total": 10, "baseline_correct": 8, "baseline_total": 10,
              "baseline_id": "base-v1", "protocol_id": "fixed-scenes-v1", "evidence_uri": "report.json"}
    evidence = {"schema_version": 1, "suite_sha256": "fixture", "run_dir": str(tmp_path),
                "records": [{"checkpoint_step": 5000, "operator_reviewed": True, "vlm_retention": counts}]}
    metrics = external_metrics(evidence, suite_sha256="fixture", run_dir=tmp_path)
    assert metrics[5000]["vlm_difference_lower"] < metrics[5000]["vlm_difference_upper"]
    evidence["records"].append({"checkpoint_step": 10000, "operator_reviewed": True,
        "vlm_retention": {**counts, "baseline_id": "different"}})
    with pytest.raises(ValueError, match="baseline changed"):
        external_metrics(evidence, suite_sha256="fixture", run_dir=tmp_path)


def test_unreviewed_external_metrics_cannot_promote_vlm_claims(tmp_path):
    evidence = {"schema_version": 1, "suite_sha256": "fixture", "run_dir": str(tmp_path),
                "records": [{"checkpoint_step": 5000, "operator_reviewed": False,
                             "vlm_retention": {"correct": 100, "total": 100}}]}
    assert external_metrics(evidence, suite_sha256="fixture", run_dir=tmp_path) == {5000: {}}
