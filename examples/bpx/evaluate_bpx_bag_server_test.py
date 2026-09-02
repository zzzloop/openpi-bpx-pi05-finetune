import json

import evaluate_bpx_bag_server as evaluation
import pytest


def test_resolve_prompt_prefers_explicit_value(tmp_path):
    bag = tmp_path / "unknown" / "episode.bag"
    bag.parent.mkdir()
    prompt, source = evaluation.resolve_prompt(bag, "  Pick up the red block.  ", None)
    assert prompt == "Pick up the red block."
    assert source == "explicit"


def test_resolve_prompt_from_audited_task_directory(tmp_path):
    audit = tmp_path / "_bpx_audit"
    audit.mkdir()
    mapping = audit / "task_language.json"
    mapping.write_text(
        json.dumps({"抓红色方块": "Pick up the red block and place it in the bucket."}),
        encoding="utf-8",
    )
    bag = tmp_path / "抓红色方块" / "0.bag"
    bag.parent.mkdir()
    prompt, source = evaluation.resolve_prompt(bag, None, None)
    assert prompt == "Pick up the red block and place it in the bucket."
    assert source == str(mapping)


def test_resolve_prompt_requires_explicit_or_audited_mapping(tmp_path):
    bag = tmp_path / "task" / "0.bag"
    bag.parent.mkdir()
    with pytest.raises(FileNotFoundError, match="pass --prompt"):
        evaluation.resolve_prompt(bag, None, None)
