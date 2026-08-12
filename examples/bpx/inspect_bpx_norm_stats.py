"""Validate and print all 18 dimensions of BPX OpenPI normalization statistics."""

import pathlib

import numpy as np
import tyro

from openpi.policies import bpx_constants
from openpi.shared import normalize


def _validated_vector(stats, field: str, key: str) -> np.ndarray:
    value = getattr(stats, field)
    if value is None:
        raise ValueError(f"'{key}.{field}' is missing.")
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (bpx_constants.BPX_EFFECTIVE_DIM,):
        raise ValueError(f"'{key}.{field}' must have shape (18,), got {vector.shape}.")
    if not np.isfinite(vector).all():
        raise ValueError(f"'{key}.{field}' contains NaN or infinity.")
    return vector


def main(
    norm_stats_path: pathlib.Path = pathlib.Path("assets/pi05_bpx_full/bpx/bpx_blocks/norm_stats.json"),
    *,
    small_std_threshold: float = 1e-6,
    small_quantile_range_threshold: float = 1e-6,
    fail_on_small_range: bool = False,
) -> None:
    """Print state/action q01, q99, mean and std, flagging nearly constant dimensions."""
    norm_stats_path = pathlib.Path(norm_stats_path)
    if norm_stats_path.name != "norm_stats.json":
        raise ValueError("norm_stats_path must point to a norm_stats.json file.")
    stats_by_key = normalize.load(norm_stats_path.parent)
    if set(stats_by_key) != {"state", "actions"}:
        raise ValueError(f"Expected norm-stat keys {{'state', 'actions'}}, got {sorted(stats_by_key)}.")

    flagged: list[str] = []
    for key in ("state", "actions"):
        stats = stats_by_key[key]
        mean = _validated_vector(stats, "mean", key)
        std = _validated_vector(stats, "std", key)
        q01 = _validated_vector(stats, "q01", key)
        q99 = _validated_vector(stats, "q99", key)
        if np.any(std < 0.0) or np.any(q99 < q01):
            raise ValueError(f"'{key}' contains a negative std or reversed quantiles.")

        print(f"\n{key} (effective BPX order)")
        print("idx  joint                                      q01          q99         mean          std  flags")
        for index, name in enumerate(bpx_constants.BPX_EFFECTIVE_JOINT_NAMES):
            flags = []
            if std[index] <= small_std_threshold:
                flags.append("LOW_STD")
            if q99[index] - q01[index] <= small_quantile_range_threshold:
                flags.append("LOW_QRANGE")
            flag_text = ",".join(flags) or "-"
            if flags:
                flagged.append(f"{key}[{index}] {name}: {flag_text}")
            print(
                f"{index:2d}   {name:40s} {q01[index]:11.6f} {q99[index]:11.6f} "
                f"{mean[index]:11.6f} {std[index]:11.6f}  {flag_text}"
            )

    if flagged:
        print("\nReview required for nearly constant dimensions:")
        for message in flagged:
            print(f"  - {message}")
        if fail_on_small_range:
            raise ValueError(f"Found {len(flagged)} nearly constant norm-stat dimension(s).")
    else:
        print("\nPASS: all 18 state and action dimensions are finite and exceed the configured range thresholds.")


if __name__ == "__main__":
    tyro.cli(main)
