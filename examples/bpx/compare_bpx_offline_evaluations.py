"""Overlay paired BPX replay outputs; no model loading and no new ranking."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import matplotlib
import numpy as np

from openpi.policies import bpx_constants as constants

matplotlib.use("Agg")
from matplotlib import pyplot as plt


PAIRING_KEYS = ("config_name", "dataset_root", "dataset_fingerprint", "norm_sha256", "stride", "noise_seeds",
                "noise_rule", "action_horizon", "num_steps", "fps", "source_parquet_sha256", "queries", "prompts")


def check_pairing(reports):
    reference = reports[0]
    for report in reports[1:]:
        for key in PAIRING_KEYS:
            if report.get(key) != reference.get(key):
                raise ValueError(f"Not paired evaluations: {key} changed")


def compare(root: pathlib.Path, steps: list[int]) -> None:
    if len(steps) < 2 or len(set(steps)) != len(steps):
        raise ValueError("At least two distinct checkpoint steps required")
    groups = [{p.name for p in (root / str(step)).glob("episode_*") if (p / "metrics.json").is_file()} for step in steps]
    if not groups[0] or any(group != groups[0] for group in groups[1:]):
        raise ValueError("Every step must have the same completed episodes; do not silently drop missing cases")
    output = root / "comparison"
    if output.exists():
        raise FileExistsError(output)
    # Validate ALL reports before producing comparison outputs.
    all_reports = {}
    for episode in sorted(groups[0]):
        reports = [json.loads((root / str(step) / episode / "metrics.json").read_text(encoding="utf-8")) for step in steps]
        check_pairing(reports)
        all_reports[episode] = reports
    output.mkdir()
    with (output / "paired_per_joint.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "episode", "joint", "unit", "mae", "rmse", "first_step_mae"])
        for episode, reports in all_reports.items():
            arrays = []
            for step in steps:
                with np.load(root / str(step) / episode / "predictions.npz", allow_pickle=False) as data:
                    arrays.append({key: data[key] for key in data.files})
            if any(not np.array_equal(data["targets"], arrays[0]["targets"]) for data in arrays[1:]):
                raise ValueError("Targets differ despite matching metadata")
            for view in ("first_step", "chunk"):
                figure, axes = plt.subplots(6, 3, figsize=(18, 19), constrained_layout=True)
                for joint, axis in enumerate(axes.flat):
                    if view == "first_step":
                        target = arrays[0]["full_episode_actions"][:, joint]
                        axis.plot(np.arange(len(target))/30, target, color="black", linewidth=1, label="teleop target")
                    else:
                        middle = len(arrays[0]["query_indices"])//2
                        axis.plot(arrays[0]["targets"][middle, :, joint], color="black", label="teleop target")
                    for step, data in zip(steps, arrays, strict=True):
                        if view == "first_step":
                            samples = data["samples"][:, :, 0, joint]
                            x = data["query_indices"]/30
                        else:
                            samples = data["samples"][middle, :, :, joint].T
                            x = np.arange(samples.shape[0])
                        mean, sd = samples.mean(axis=1), samples.std(axis=1)
                        line, = axis.plot(x, mean, label=f"step {step}", linewidth=1)
                        axis.fill_between(x, mean-sd, mean+sd, color=line.get_color(), alpha=.1)
                    unit = "legacy gripper" if joint in (7, 15) else "rad"
                    axis.set_title(f"{joint}: {constants.BPX_EFFECTIVE_JOINT_NAMES[joint]}", fontsize=8)
                    axis.set_ylabel(unit)
                    axis.grid(alpha=.2)
                axes.flat[0].legend(fontsize=8)
                figure.supxlabel("episode time (seconds)" if view == "first_step" else "action horizon index")
                figure.suptitle(f"{episode}: absolute ACTION prediction vs teleop target; bands = noise SD, not CI")
                figure.savefig(output / f"{episode}_{view}.png", dpi=140)
                plt.close(figure)
            for step, report in zip(steps, reports, strict=True):
                for joint, name in enumerate(constants.BPX_EFFECTIVE_JOINT_NAMES):
                    writer.writerow([step, report["episode_index"], name, "legacy_gripper_coordinate" if joint in (7, 15) else "rad",
                                     report["per_joint_mae"][joint], report["per_joint_rmse"][joint],
                                     report["first_step_per_joint_mae"][joint]])
    print(f"PAIRED COMPARISON PASS: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[10000, 18000, 19999])
    args = parser.parse_args()
    compare(args.root, args.steps)
