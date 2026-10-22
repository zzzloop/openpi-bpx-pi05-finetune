"""Display-only robust trends. Raw losses and checkpoint decisions never change."""

from __future__ import annotations

import numpy as np


def robust_trend(values, *, window: int = 9, alpha: float = 0.2):
    if window < 3 or window % 2 != 1 or not 0 < alpha <= 1:
        raise ValueError("Trend window must be odd and >=3; alpha must be in (0,1]")
    raw = np.asarray(values, dtype=np.float64)
    clean = raw.copy()
    outliers = np.zeros(raw.size, dtype=bool)
    radius = window // 2
    for index, value in enumerate(raw):
        if not np.isfinite(value):
            continue
        neighbors = np.concatenate((raw[max(0, index-radius):index], raw[index+1:index+radius+1]))
        neighbors = neighbors[np.isfinite(neighbors)]
        # Need evidence on both sides; never invent a final "good" loss.
        if index == 0 or index == raw.size - 1 or neighbors.size < 4:
            continue
        center = np.median(neighbors)
        mad = np.median(np.abs(neighbors - center))
        threshold = max(3.0 * 1.4826 * mad, abs(center) * 0.05, 1e-12)
        if abs(value - center) > threshold:
            clean[index], outliers[index] = center, True
    smoothed = clean.copy()
    previous = None
    for index, value in enumerate(clean):
        if not np.isfinite(value):
            continue
        previous = value if previous is None else alpha * value + (1-alpha) * previous
        smoothed[index] = previous
    return smoothed, outliers
