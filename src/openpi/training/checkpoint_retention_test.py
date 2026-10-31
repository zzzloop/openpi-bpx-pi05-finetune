"""Server-only retention regressions; tiny arrays, no model/data conversion."""
import pathlib

import numpy as np
import pytest

from openpi.training.checkpoint_selection import SelectionHistory
from openpi.training.checkpoints import initialize_checkpoint_dir, retention_budget


@pytest.mark.parametrize(
    ("limit", "candidates", "expected"),
    [(None, None, (1, 25000, None)), (None, 3, (1, 25000, 3)),
     (3, None, (3, None, None)), (3, 3, (1, None, 2)),
     (3, 20, (1, None, 2)), (3, 1, (1, None, 1)),
     (2, 3, (1, None, 1)), (1, None, (1, None, None))],
)
def test_total_budget_overrides_periodic_retention_only_when_enabled(limit, candidates, expected):
    assert retention_budget(limit, 25000, candidates) == expected


@pytest.mark.parametrize(("limit", "candidates"), [(0, None), (-1, 3), (True, None), (1, 3), (3, 0)])
def test_invalid_budget_fails_before_modifying_directories(limit, candidates):
    with pytest.raises(ValueError):
        retention_budget(limit, 25000, candidates)


@pytest.mark.parametrize("select", [False, True])
def test_actual_orbax_saves_and_archives_share_total_three_budget(tmp_path, monkeypatch, select):
    run = tmp_path / "new_run"
    old_run = tmp_path / "old_run"
    old_run.mkdir()
    (old_run / "do_not_touch").write_text("old experiment")
    manager, resuming = initialize_checkpoint_dir(
        run, keep_period=10, overwrite=False, resume=False,
        checkpoint_limit=3, selection_keep=3 if select else None,
    )
    assert not resuming
    history = None
    if select:
        history = SelectionHistory(
            run, {"suite_sha256": "retention_fixture", "checkpoint_limit": 3},
            keep=2, start_step=0, resume=False,
        )
        monkeypatch.setattr(history, "render", lambda: None)
    try:
        for step in (5, 10, 15, 20, 25):
            manager.save(step, items={
                "params": {"params": {"weights": np.asarray([step], dtype=np.float32)}},
                "train_state": {"step": np.asarray(step)},
                "assets": lambda directory: (directory / "fixture.txt").write_text("normalization"),
            })
            manager.wait_until_finished()
            if history is not None:
                decision = history.append(step, {
                    "action_mse": .1, "val_loss": .1-step*.0001, "val_stderr": .03,
                    "train_eval_loss": .1, "jerk_excess": step*.001,
                    "s_cf_active": 1., "s_cf": 2.,
                }, [], {}, optimizer_step=step+1, checkpoint_complete=True)
                assert pathlib.Path(decision["checkpoint_path"]).is_dir()
            archives = [path for path in (run / "selected_checkpoints").glob("*") if path.name.isdigit()]
            ordinary = list(manager.all_steps())
            assert step == max(ordinary)
            assert (run / str(step) / "train_state").is_dir()
            assert len(ordinary) + len(archives) <= 3
        if select:
            assert (run / "selected_checkpoints/5/params").is_dir()
            assert not (run / "5").exists()
            assert len(manager.all_steps()) == 1
        else:
            assert sorted(manager.all_steps()) == [15, 20, 25]
        assert (old_run / "do_not_touch").read_text() == "old experiment"
    finally:
        manager.close()
