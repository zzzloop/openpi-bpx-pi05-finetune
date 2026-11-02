"""Read-only, fixed-noise BPX LeRobot v2.1 replay; never connects to a robot.

Reuse the runtime Policy input/output transforms, including checkpoint assets,
delta reversal, and raw23 -> effective18 conversion. Predictions are absolute
action targets, NOT simulated or measured robot feedback.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import time

import numpy as np

from openpi.policies import bpx_constants as constants


def _json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _numpy(value):
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)


def query_frames(frames: int, horizon: int, stride: int) -> np.ndarray:
    if horizon < 1 or stride < 1 or frames < horizon:
        raise ValueError("Positive stride/horizon and at least one complete action chunk are required")
    # Include the last complete chunk, never score padded/repeated final actions.
    return np.unique(np.append(np.arange(0, frames-horizon+1, stride), frames-horizon)).astype(np.int64)


def fixed_noise(seed: int, episode: int, frame: int, shape: tuple[int, int]) -> np.ndarray:
    """Same noise for the same origin across checkpoints; independent of call order."""
    return np.random.default_rng(np.random.SeedSequence([seed, episode, frame])).standard_normal(shape).astype(np.float32)


def action_errors(samples: np.ndarray, targets: np.ndarray) -> dict:
    # Score EVERY seed, not the error of a seed-averaged trajectory (which can
    # hide a mixture of individually poor predictions).
    if samples.ndim != 4 or not samples.size or targets.shape != (samples.shape[0], *samples.shape[2:]):
        raise ValueError("Expected samples[origin,seed,horizon,joint] and targets[origin,horizon,joint]")
    if not np.isfinite(samples).all() or not np.isfinite(targets).all():
        raise ValueError("Non-finite prediction/target")
    error = samples.astype(np.float64)-targets[:, None].astype(np.float64)
    return {"per_joint_mae": np.abs(error).mean(axis=(0, 1, 2)).tolist(),
            "per_joint_rmse": np.sqrt((error**2).mean(axis=(0, 1, 2))).tolist(),
            "per_horizon_joint_mae": np.abs(error).mean(axis=(0, 1)).tolist(),
            "first_step_per_joint_mae": np.abs(error[:, :, 0]).mean(axis=(0, 1)).tolist()}


def resolve_checkpoint(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser().resolve()
    if path.is_file():
        path = pathlib.Path(_json(path)["checkpoint_path"]).expanduser().resolve()
    if not (path / "params").is_dir() or not (path / "assets").is_dir():
        raise FileNotFoundError(f"Complete params/assets required: {path}")
    return path


def _plot_cameras(output, dataset, origins):
    from matplotlib import pyplot as plt
    from validate_bpx_lerobot import _to_rgb_uint8

    selected = np.unique(np.asarray(origins)[np.linspace(0, len(origins)-1, 3).round().astype(int)])
    figure, axes = plt.subplots(len(selected), 3, figsize=(14, 4*len(selected)), squeeze=False)
    for row, frame in enumerate(selected):
        item = dataset[int(frame)]
        for column, camera in enumerate(constants.BPX_IMAGE_KEYS):
            axes[row, column].imshow(_to_rgb_uint8(item[f"observation.images.{camera}"]))
            axes[row, column].set_title(f"frame {frame}: {camera}")
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(output / "camera_contact_sheet.png", dpi=120)
    plt.close(figure)


def evaluate(args):
    checkpoint = resolve_checkpoint(args.checkpoint_dir)
    source = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() or output.is_relative_to(source) or output.is_relative_to(checkpoint):
        raise ValueError("Use a NEW output directory outside the source dataset and checkpoint")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("Use distinct nonnegative noise seeds")
    if args.stride < 1 or args.num_steps < 1 or not args.episode_indices:
        raise ValueError("Positive stride/num_steps and at least one episode are required")
    if len(set(args.episode_indices)) != len(args.episode_indices) or min(args.episode_indices) < 0:
        raise ValueError("Use distinct nonnegative episode indices")
    info = _json(source / "meta/info.json")
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        if tuple(feature["shape"]) != (18,) or tuple(feature.get("names", [])) != constants.BPX_EFFECTIVE_JOINT_NAMES:
            raise ValueError(f"Not the canonical FIRST-generation BPX effective18 contract: {key}")
    if float(info["fps"]) != 30:
        raise ValueError("This first-generation evaluator requires the recorded 30Hz contract")
    for camera in constants.BPX_IMAGE_KEYS:
        if tuple(info["features"][f"observation.images.{camera}"]["shape"]) != (360, 640, 3):
            raise ValueError(f"Expected RGB 640x360: {camera}")

    # Local replay must not download/repair a dataset from the Hub.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    print("[BPX REPLAY] Loading local LeRobot/ML dependencies; no robot or policy server connection", flush=True)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.policies import policy_config
    from openpi.training import config, validation_split
    import evaluate_bpx_episode_server as plots
    from smoke_test_bpx_checkpoint import validate_output_contract
    from validate_bpx_lerobot import _to_rgb_uint8

    train_config = config.get_config(args.config_name)
    if not isinstance(train_config.data, config.LeRobotBpxDataConfig):
        raise ValueError("Use a BPX policy config, not another embodiment")
    records, fingerprint = validation_split.inventory(source)
    heldout = set()
    if args.validation_manifest:
        split = validation_split.load_split(args.validation_manifest)
        if pathlib.Path(split["dataset_root"]).resolve() != source or split["metadata_sha256"] != fingerprint:
            raise ValueError("Validation manifest does not match this dataset")
        heldout = set(split["validation_episodes"])
    available = {int(row["episode_index"]) for row in records}
    if set(args.episode_indices)-available:
        raise ValueError("Requested episode is not in this dataset")
    asset_id = train_config.data.assets.asset_id or train_config.data.repo_id
    norm = checkpoint / "assets" / asset_id / "norm_stats.json"
    if not norm.is_file():
        raise FileNotFoundError(f"Checkpoint normalization asset missing: {norm}")
    output.mkdir(parents=True)
    print(f"Loading ONLY this checkpoint: {checkpoint}", flush=True)
    policy = policy_config.create_trained_policy(train_config, checkpoint, sample_kwargs={"num_steps": args.num_steps})
    print("[BPX REPLAY] Checkpoint loaded; normalization comes from checkpoint assets", flush=True)
    horizon, dim = train_config.model.action_horizon, train_config.model.action_dim
    contract = {"config_name": args.config_name, "dataset_root": str(source), "dataset_fingerprint": fingerprint,
                "norm_sha256": _sha(norm), "stride": args.stride, "noise_seeds": args.seeds,
                "noise_rule": "numpy PCG64 SeedSequence([seed, episode_index, frame_index]); float32",
                "action_horizon": horizon, "num_steps": args.num_steps, "fps": 30,
                "dataset_contains_estimated_state": "estimated" in str((train_config.policy_metadata or {}).get("state_provenance", "")),
                "scope": "Recorded-observation open-loop action prediction; NOT simulated or real robot execution"}
    for episode in args.episode_indices:
        print(f"[BPX REPLAY] Loading episode {episode}", flush=True)
        dataset = LeRobotDataset(train_config.data.repo_id, root=source, episodes=[episode],
                                 download_videos=False, video_backend="pyav")
        numeric = dataset.hf_dataset
        action = np.stack([_numpy(value) for value in numeric["action"]]).astype(np.float32)
        if action.shape != (int(records[episode]["length"]), constants.BPX_EFFECTIVE_DIM):
            raise ValueError("Source action shape does not match the episode/18-dimensional contract")
        frames = np.asarray([int(_numpy(value).item()) for value in numeric["frame_index"]])
        ep_ids = {int(_numpy(value).item()) for value in numeric["episode_index"]}
        if ep_ids != {episode} or not np.array_equal(frames, np.arange(len(frames))):
            raise ValueError("LeRobot frame order/episode filter does not match the source")
        if not np.isfinite(action).all():
            raise ValueError("Non-finite source action")
        origins = query_frames(len(action), horizon, args.stride)
        targets = np.stack([action[frame:frame+horizon] for frame in origins])
        print(f"[BPX REPLAY] episode={episode}, queries={len(origins)}, seeds={args.seeds}; first prediction may compile", flush=True)
        samples, prompts, latency = [], [], []
        for position, frame in enumerate(origins):
            item = dataset[int(frame)]
            task = dataset.meta.tasks[int(_numpy(item["task_index"]).item())]
            if task != records[episode]["tasks"][0]:
                raise ValueError("Per-frame task does not match the authoritative episode language")
            observation = {"qpos": _numpy(item["observation.state"]).astype(np.float32),
                           "images": {name: _to_rgb_uint8(item[f"observation.images.{name}"])
                                      for name in constants.BPX_IMAGE_KEYS}, "prompt": task}
            per_seed = []
            for seed in args.seeds:
                noise = fixed_noise(seed, episode, int(frame), (horizon, dim))
                start = time.monotonic()
                result = policy.infer(observation, noise=noise)
                latency.append((time.monotonic()-start)*1000)
                raw = np.asarray(result["actions"], dtype=np.float32)
                validate_output_contract(raw, (horizon, constants.BPX_RAW_DIM))
                per_seed.append(constants.to_effective_joints(raw))
            samples.append(per_seed)
            prompts.append(task)
            print(f"episode={episode} {position+1}/{len(origins)} frame={frame}", flush=True)
        samples = np.asarray(samples, dtype=np.float32)
        destination = output / f"episode_{episode:06d}"
        destination.mkdir()
        parquet = source / dataset.meta.get_data_file_path(episode)
        report = {**contract, "checkpoint_path": str(checkpoint), "episode_index": episode,
                  "source_parquet_sha256": _sha(parquet), "queries": origins.tolist(), "prompts": prompts,
                  "heldout_episode": episode in heldout, "query_count": len(origins),
                  "split_role": ("not_specified" if not args.validation_manifest else
                                 "validation_used_for_selection" if episode in heldout else "training"),
                  "round_trip_ms_including_first_compile": latency, **action_errors(samples, targets),
                  "uncertainty_note": "Plot bands are noise-seed SD, not confidence intervals; overlapping chunks are dependent",
                  "ground_truth_note": "Target is recorded teleop action; observation.state may include estimated Feetech feedback, not a robot rollout"}
        (destination / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        np.savez_compressed(destination / "predictions.npz", query_indices=origins, samples=samples, targets=targets,
                            full_episode_actions=action)
        with (destination / "per_joint_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["index", "joint", "unit", "mae_all_seeds", "rmse_all_seeds", "first_step_mae"])
            for joint, name in enumerate(constants.BPX_EFFECTIVE_JOINT_NAMES):
                writer.writerow([joint, name, "legacy_gripper_coordinate" if joint in (7, 15) else "rad",
                                 report["per_joint_mae"][joint], report["per_joint_rmse"][joint],
                                 report["first_step_per_joint_mae"][joint]])
        mean, sd = samples.mean(axis=1), samples.std(axis=1)
        plots.plot_joint_curves(destination, origins, mean, sd, action)
        plots.plot_chunk_overlay(destination, origins, mean, sd, targets)
        _plot_cameras(destination, dataset, origins)
        print(f"OFFLINE REPLAY PASS: episode={episode}; output={destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", default="pi05_bpx_full_low_mem_v1_base_estimated")
    parser.add_argument("--dataset-root", type=pathlib.Path, default=pathlib.Path("/scratch/bpx_datasets/lerobot/bpx/bpx_v1_base_estimated_v001"))
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--episode-indices", type=int, nargs="+", default=[20, 61])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--validation-manifest", type=pathlib.Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
