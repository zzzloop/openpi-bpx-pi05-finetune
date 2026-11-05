"""Stage-2 dataset contract and sampler, deliberately independent of JAX and ROS."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

from openpi.training import validation_split

CONFIGS = {
    "pi05_bpx_cf_naive_full": "naive_full",
    "pi05_bpx_cf_policy_cfg": "policy_cfg",
    "pi05_bpx_cf_partial_ft": "partial_ft",
    "pi05_bpx_cf_ours": "ccfa_ours",
}
BASE_CONFIG = next(iter(CONFIGS))
HORIZON = 30
SAMPLING_PROTOCOL = "alternate_full_frame_permutation_and_aligned_quad_v2"
SEMANTIC_EDGES = ((0, 1), (2, 3))
SPATIAL_EDGES = ((0, 2), (1, 3))
SWAP_EDGES = ((0, 3), (1, 2))
INITIAL_CHECKPOINT = (
    "pi05_bpx_full_low_mem_v1_base_estimated/bpx_v2_humanoid_bs2_v001_new/selected_checkpoints/18000"
)


@dataclass(frozen=True)
class Quad:
    group_id: str
    family: str
    episodes: tuple[int, ...]
    windows: tuple[tuple[int, int], ...]
    attention: tuple[bool, ...]
    split_group: str


def _attention_mask(pairs, valid_edges, group_id):
    if not isinstance(pairs, list):
        raise ValueError(f"{group_id}: attention pairs must be a list")
    edges = []
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2 or any(type(i) is not int for i in pair)
                or tuple(pair) not in valid_edges or tuple(pair) in edges):
            raise ValueError(f"{group_id}: invalid or repeated attention edge: {pair}")
        edges.append(tuple(pair))
    return tuple(edge in edges for edge in valid_edges)


def load_quads(path: Path, dataset_root: Path):
    """Validate roles, real (unpadded) action windows, and exact dataset coverage."""
    records, fingerprint = validation_split.inventory(dataset_root)
    if (dataset_root / "meta/bpx_feedback_recovery.json").exists():
        raise ValueError("Stage-2 requires newly collected demonstrations, not estimated-feedback Stage-1 data")
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    features = info.get("features", {})
    if info.get("fps") != 30 or any(features.get(k, {}).get("shape") != [18]
                                    for k in ("observation.state", "action")):
        raise ValueError("Stage-2 requires the BPX effective-18 state/action schema at 30 Hz")
    for camera in ("left_eye", "left_wrist", "right_wrist"):
        if features.get(f"observation.images.{camera}", {}).get("dtype") not in {"image", "video"}:
            raise ValueError(f"Stage-2 camera feature missing: {camera}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("groups"), list):
        raise ValueError("Unsupported Stage-2 quadruple manifest")
    if payload.get("metadata_sha256") != fingerprint:
        raise ValueError("Stage-2 manifest metadata_sha256 does not match the LeRobot dataset")
    seen_groups, seen_episodes = set(), set()
    quads = []
    for row in payload["groups"]:
        group_id, family = row["group_id"], row["family"]
        split_group = row.get("split_group", group_id)
        if any(not isinstance(s, str) or not s.strip() for s in (group_id, family, split_group)):
            raise ValueError("group_id, family and split_group must be nonempty strings")
        if group_id in seen_groups:
            raise ValueError(f"Duplicate quadruple: {group_id}")
        seen_groups.add(group_id)
        ids = row["episodes"]
        if (not isinstance(ids, list) or len(ids) != 4
                or any(type(i) is not int or not 0 <= i < len(records) for i in ids)
                or len(set(ids)) != 4 or seen_episodes.intersection(ids)):
            raise ValueError(f"{group_id}: four distinct episode indices required in role order")
        seen_episodes.update(ids)
        prompts = [records[i]["tasks"][0].strip() for i in ids]
        if not prompts[0] or not prompts[1] or prompts[0] == prompts[1] or prompts[:2] != prompts[2:]:
            raise ValueError(f"{group_id}: roles must be [Oa/I1, Oa/I2, Ob/I1, Ob/I2] with matching task labels")
        windows = row["windows"]
        if not isinstance(windows, list) or len(windows) != 4:
            raise ValueError(f"{group_id}: four reviewed [first,last] reference-frame windows are required")
        for episode, window in zip(ids, windows, strict=True):
            if (not isinstance(window, list) or len(window) != 2 or any(type(x) is not int for x in window)
                    or not 0 <= window[0] <= window[1] <= int(records[episode]["length"]) - HORIZON):
                raise ValueError(f"{group_id}: episode {episode} window exceeds real {HORIZON}-frame action chunks")
        attention = _attention_mask(row.get("attention_semantic", []), SEMANTIC_EDGES, group_id)
        attention += _attention_mask(row.get("attention_swap", []), SWAP_EDGES, group_id)
        if family == "task3" and attention[2]:
            raise ValueError(f"{group_id}: Task 3 does not align the (0,3) target locations")
        quads.append(Quad(group_id, family, tuple(ids), tuple(map(tuple, windows)), attention, split_group))
    if seen_episodes != set(range(len(records))):
        raise ValueError("Every Stage-2 episode must belong to exactly one quadruple; use a separate repo for extras")
    return tuple(quads), records, fingerprint, hashlib.sha256(raw).hexdigest()


def build_split(dataset_root: Path, manifest: Path, output: Path, fraction: float = 0.1, seed: int = 42):
    if not 0 < fraction < 0.5:
        raise ValueError("Validation fraction must be between 0 and 0.5")
    quads, records, fingerprint, digest = load_quads(manifest, dataset_root)
    by_family = {}
    family_by_split_group = {}
    groups = [""] * len(records)
    for quad in quads:
        if family_by_split_group.setdefault(quad.split_group, quad.family) != quad.family:
            raise ValueError("A split_group may not span task families")
        by_family.setdefault(quad.family, set()).add(quad.split_group)
        for i in quad.episodes:
            groups[i] = quad.split_group
    rng = random.Random(seed)
    heldout = set()
    for family, keys in sorted(by_family.items()):
        keys = sorted(keys)
        if len(keys) < 2:
            raise ValueError(f"{family}: need at least two independent layout groups for holdout")
        rng.shuffle(keys)
        heldout.update(keys[:min(len(keys) - 1, max(1, round(len(keys) * fraction)))])
    val = [i for i, group in enumerate(groups) if group in heldout]
    train = [i for i, group in enumerate(groups) if group not in heldout]
    payload = {
        "schema_version": 1, "metadata_sha256": fingerprint, "dataset_root": str(dataset_root.resolve()),
        "fraction": fraction, "seed": seed, "grouping": "bpx_counterfactual_quads",
        "quad_manifest_sha256": digest, "groups": groups,
        "train_episodes": train, "validation_episodes": val,
        "train_frames": sum(records[i]["length"] for i in train),
        "validation_frames": sum(records[i]["length"] for i in val),
        "heldout_quads": [
            {"family": q.family, "group_id": q.group_id, "episodes": list(q.episodes),
             "anchor_frames": [lo for lo, _ in q.windows]}
            for q in quads if q.split_group in heldout
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def bound_quads(manifest: Path, dataset_root: Path, split_path: Path, *, require_attention=False):
    quads, records, fingerprint, digest = load_quads(manifest, dataset_root)
    split = validation_split.load_split(split_path)
    if (split.get("grouping") != "bpx_counterfactual_quads" or split.get("quad_manifest_sha256") != digest
            or split["metadata_sha256"] != fingerprint or len(split["groups"]) != len(records)):
        raise ValueError("Stage-2 split must be generated from this exact quadruple manifest and dataset")
    if any(split["groups"][i] != q.split_group for q in quads for i in q.episodes):
        raise ValueError("Stage-2 split_group binding has changed")
    train = set(split["train_episodes"])
    selected = tuple(q for q in quads if q.episodes[0] in train)
    if any(not set(q.episodes).issubset(train) for q in selected):
        raise ValueError("A training quadruple crosses the holdout boundary")
    if require_attention and not all(any(any(q.attention[start:start+2]) for q in selected)
                                     for start in (0, 2)):
        raise ValueError("CCFA requires reviewed semantic and swap attention edges in the training split")
    return selected, records


def is_vlm_parameter(path, _value=None):
    """Gemma names the action expert submodules with a _1 suffix."""
    if len(path) < 2 or path[0] != "PaliGemma":
        return False
    return path[1] == "img" or (path[1] == "llm" and not any(str(p).endswith("_1") for p in path[2:]))


def heldout_probe_cases(split, records, max_cases):
    """Use the other instruction from the SAME layout, not an unrelated task."""
    by_family = {}
    for q in split["heldout_quads"]:
        by_family.setdefault(q["family"], []).append(q)
    ordered = []
    for offset in range(max(map(len, by_family.values()), default=0)):
        ordered.extend(rows[offset] for _, rows in sorted(by_family.items()) if offset < len(rows))
    result = []
    for role in (0, 2, 1, 3):
        for q in ordered:
            index = q["episodes"][role]
            peer = q["episodes"][role ^ 1]
            result.append({
                "id": f"episode_{index}", "episode_index": index, "frame_index": q["anchor_frames"][role],
                "counterfactual_prompt": records[peer]["tasks"][0],
                "paraphrase_prompt": None, "semantic_reviewed": False,
            })
            if len(result) == max_cases:
                return result
    return result


def is_aligned_step(step: int) -> bool:
    """The same schedule is used for all four recipes, including on resume."""
    return step % 2 == 1


class QuadBatchSampler:
    """Alternate whole-trajectory CFM batches and aligned counterfactual quads."""

    def __init__(self, quads, records, episode_indices, seed):
        self.quads = quads
        self.seed = seed
        self.start_step = 0
        self.offsets = {}
        offset = 0
        for i in episode_indices:
            self.offsets[i] = offset
            offset += int(records[i]["length"])
        self.frame_count = offset
        if not self.quads or self.frame_count < 4:
            raise ValueError("Stage-2 requires complete training quads and at least four training frames")

    def __iter__(self):
        step = self.start_step
        cached_epoch = -1
        permutation = []
        while True:
            if is_aligned_step(step):
                rng = random.Random(f"bpx-stage2-quad:{self.seed}:{step}")
                quad = self.quads[rng.randrange(len(self.quads))]
                # Event-aligned windows, not the same fraction of unrelated whole episodes.
                phase = rng.random()
                yield [(self.offsets[i] + lo + int(phase * (hi - lo + 1)), quad.attention)
                       for i, (lo, hi) in zip(quad.episodes, quad.windows, strict=True)]
            else:
                # The first N whole-trajectory draws cover all N training frames, including
                # prelude, transitions and termination. LeRobot owns normal tail padding.
                batch = []
                for draw in range(4 * (step // 2), 4 * (step // 2) + 4):
                    epoch, position = divmod(draw, self.frame_count)
                    if epoch != cached_epoch:
                        permutation = list(range(self.frame_count))
                        random.Random(f"bpx-stage2-full:{self.seed}:{epoch}").shuffle(permutation)
                        cached_epoch = epoch
                    batch.append((permutation[position], None))
                yield batch
            step += 1


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--fingerprint", action="store_true", help="Print the metadata_sha256 for a new manifest")
    args = parser.parse_args()
    if args.fingerprint:
        print(validation_split.inventory(args.dataset_root)[1])
        return
    if args.manifest is None or args.output is None:
        parser.error("--manifest and --output are required unless using --fingerprint")
    result = build_split(args.dataset_root, args.manifest, args.output, args.fraction)
    print(f"SPLIT PASS: train={len(result['train_episodes'])} episodes, "
          f"validation={len(result['validation_episodes'])} episodes", flush=True)


if __name__ == "__main__":
    main()
