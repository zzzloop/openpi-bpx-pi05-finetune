"""Immutable, episode-grouped holdouts. No model/GPU dependencies here."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import random
import re


def inventory(root: pathlib.Path) -> tuple[list[dict], str]:
    meta = root / "meta"
    info = (meta / "info.json").read_bytes()
    episodes = (meta / "episodes.jsonl").read_bytes()
    records = [json.loads(line) for line in episodes.decode("utf-8").splitlines() if line.strip()]
    if [r["episode_index"] for r in records] != list(range(len(records))):
        raise ValueError("Validation requires contiguous LeRobot v2.x episode indices")
    if not records or any(int(r["length"]) <= 0 or len(r.get("tasks", [])) != 1 for r in records):
        raise ValueError("Each validation episode must have positive length and exactly one task")
    return records, hashlib.sha256(info + b"\n" + episodes).hexdigest()


def source_group(record: dict) -> str:
    if record.get("group_id"):
        return str(record["group_id"])
    source = str(record["path"]).replace("\\", "/")
    if record["source_kind"] == "lerobot":
        return f"{source}#episode={int(record['source_index'])}"
    # An original bag and ALL its photometric copies must stay on the same side.
    return re.sub(r"__aug_(?:dark|bright)(?=\.[^/]+$)", "", source)


def build_split(
    dataset_root: pathlib.Path,
    output: pathlib.Path,
    *,
    source_manifest: pathlib.Path | None = None,
    independent_episodes: bool = False,
    fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    if not 0 < fraction < 0.5:
        raise ValueError("Validation fraction must be between 0 and 0.5")
    records, fingerprint = inventory(dataset_root)
    if source_manifest is not None:
        sources = json.loads(source_manifest.read_text(encoding="utf-8"))["episodes"]
        if len(sources) != len(records):
            raise ValueError("Source manifest must describe this exact converted dataset in output order")
        groups = []
        for index, (episode, source) in enumerate(zip(records, sources, strict=True)):
            if (source["episode_index"] != index or source["frames"] != episode["length"]
                    or source["task"] != episode["tasks"][0]):
                raise ValueError(f"Source manifest does not match output episode {index}")
            groups.append(source_group(source))
    elif independent_episodes:
        groups = [f"episode:{r['episode_index']}" for r in records]
    else:
        raise ValueError("Provide a source manifest to group augmentations, or explicitly confirm independent episodes")

    by_task: dict[str, set[str]] = {}
    group_tasks: dict[str, str] = {}
    for episode, group in zip(records, groups, strict=True):
        task = episode["tasks"][0]
        if group in group_tasks and group_tasks[group] != task:
            raise ValueError("One source group has conflicting task labels")
        group_tasks[group] = task
        by_task.setdefault(task, set()).add(group)
    rng = random.Random(seed)
    heldout: set[str] = set()
    for task, task_groups in sorted(by_task.items()):
        candidates = sorted(task_groups)
        if len(candidates) < 2:
            raise ValueError(f"Task needs at least two independent groups for a holdout: {task}")
        rng.shuffle(candidates)
        count = min(len(candidates) - 1, max(1, round(len(candidates) * fraction)))
        heldout.update(candidates[:count])
    validation = [i for i, group in enumerate(groups) if group in heldout]
    train = [i for i, group in enumerate(groups) if group not in heldout]
    payload = {
        "schema_version": 1, "metadata_sha256": fingerprint,
        "dataset_root": str(dataset_root.resolve()), "fraction": fraction, "seed": seed,
        "source_manifest": str(source_manifest.resolve()) if source_manifest else None,
        "grouping": "source_manifest" if source_manifest else "confirmed_independent_episodes",
        "groups": groups, "train_episodes": train, "validation_episodes": validation,
        "train_frames": sum(records[i]["length"] for i in train),
        "validation_frames": sum(records[i]["length"] for i in validation),
    }
    if output.exists():
        raise FileExistsError(f"Split is immutable; choose a new output path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def load_split(path: str | pathlib.Path) -> dict:
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported validation split schema")
    train, val, groups = payload["train_episodes"], payload["validation_episodes"], payload["groups"]
    if (not train or not val or any(type(i) is not int for i in [*train, *val])
            or sorted([*train, *val]) != list(range(len(groups)))):
        raise ValueError("Split must partition every episode exactly once")
    if {groups[i] for i in train} & {groups[i] for i in val}:
        raise ValueError("Train/validation source groups overlap")
    return payload


def split_digest(path: str | pathlib.Path) -> str:
    canonical = json.dumps(load_split(path), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def configure(config):
    """Use separate TRAIN-ONLY norm assets, never overwrite an existing full-data run."""
    if not config.validation_manifest:
        return config
    digest = split_digest(config.validation_manifest)
    base = pathlib.Path(config.assets_base_dir)
    if base.name != f"holdout-{digest}":
        base = base / f"holdout-{digest}"
    factory = dataclasses.replace(config.data, assets=dataclasses.replace(config.data.assets, assets_dir=None))
    return dataclasses.replace(config, data=factory, assets_base_dir=str(base))


def select_data_config(config, data_config, split: str = "train"):
    if not config.validation_manifest:
        if split != "train":
            raise ValueError("A validation manifest is required for evaluation")
        return data_config
    # Import lazily: split creation/audit should not import torch, JAX or LeRobot.
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    payload = load_split(config.validation_manifest)
    records, fingerprint = inventory(pathlib.Path(HF_LEROBOT_HOME) / data_config.repo_id)
    if fingerprint != payload["metadata_sha256"] or len(records) != len(payload["groups"]):
        raise ValueError("Dataset metadata changed since split creation; do not resume with another dataset")
    indices = payload["validation_episodes" if split == "validation" else "train_episodes"]
    return dataclasses.replace(data_config, episode_indices=tuple(indices),
                               balance_by_task=data_config.balance_by_task and split == "train")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--source-manifest", type=pathlib.Path)
    parser.add_argument("--independent-episodes", action="store_true")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    result = build_split(**vars(parser.parse_args()))
    print(f"SPLIT PASS: train={len(result['train_episodes'])} episodes, "
          f"validation={len(result['validation_episodes'])} episodes", flush=True)


if __name__ == "__main__":
    main()
