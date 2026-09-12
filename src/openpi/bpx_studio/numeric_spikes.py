from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SpikeRepairResult:
    repaired: np.ndarray
    mask: np.ndarray
    unresolved_jump_mask: np.ndarray

    @property
    def count(self) -> int:
        return int(self.mask.sum())


def repair_isolated_spikes(
    values: np.ndarray,
    *,
    jump_thresholds: np.ndarray,
    neighbor_thresholds: np.ndarray,
) -> SpikeRepairResult:
    """Repair single-frame excursions, preserving steps and ambiguous runs.

    Thresholds are absolute changes per sample in each feature's native units.
    Remaining large transitions are flagged, not automatically smoothed.
    """
    source = np.asarray(values)
    if source.ndim != 2 or not source.shape[0] or not source.shape[1]:
        raise ValueError("values must have non-empty shape (T, D)")
    if not np.issubdtype(source.dtype, np.floating) or not np.all(np.isfinite(source)):
        raise ValueError("values must be finite floating-point data")
    jumps = np.asarray(jump_thresholds, dtype=np.float64)
    neighbors = np.asarray(neighbor_thresholds, dtype=np.float64)
    if jumps.shape != (source.shape[1],) or neighbors.shape != (source.shape[1],):
        raise ValueError("threshold vectors must match the data dimension")
    if not np.all(np.isfinite(jumps)) or not np.all(np.isfinite(neighbors)):
        raise ValueError("thresholds must be finite")
    if np.any(jumps <= 0) or np.any(neighbors < 0):
        raise ValueError("jump thresholds must be positive and neighbor thresholds non-negative")
    working = source.astype(np.float64)
    previous, center, following = working[:-2], working[1:-1], working[2:]
    isolated = (
        (np.abs(center - previous) > jumps)
        & (np.abs(center - following) > jumps)
        & (np.abs(previous - following) <= neighbors)
        & (((center > previous) & (center > following)) | ((center < previous) & (center < following)))
    )
    # Adjacent candidates cannot serve as reliable anchors for one another.
    conflicting = np.zeros_like(isolated)
    conflicting[1:] |= isolated[:-1]
    conflicting[:-1] |= isolated[1:]
    isolated &= ~conflicting
    mask = np.zeros(source.shape, dtype=bool)
    mask[1:-1] = isolated
    repaired = source.copy()
    midpoint = (previous.astype(np.float64) + following.astype(np.float64)) * 0.5
    repaired[1:-1] = np.where(isolated, midpoint, center).astype(source.dtype)
    unresolved = np.zeros(source.shape, dtype=bool)
    unresolved[1:] = np.abs(np.diff(repaired.astype(np.float64), axis=0)) > jumps
    return SpikeRepairResult(repaired=repaired, mask=mask, unresolved_jump_mask=unresolved)
