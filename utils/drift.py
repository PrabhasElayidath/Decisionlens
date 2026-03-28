"""Simple training-distribution drift flags for base feature inputs."""

from __future__ import annotations

from typing import Any

# Z-score beyond this (using training mean/std) triggers a flag
Z_THRESHOLD = 3.0


def check_feature_drift(
    base_input: dict[str, Any],
    feature_stats: dict[str, dict[str, float]] | None,
) -> tuple[bool, list[str]]:
    """
    Compare base features to training-set statistics saved in metadata.

    Returns (drift_alert, human-readable notes).
    """
    if not feature_stats:
        return False, []

    notes: list[str] = []
    for name, stats in feature_stats.items():
        if name not in base_input:
            continue
        try:
            v = float(base_input[name])
        except (TypeError, ValueError):
            continue
        lo = stats.get("min")
        hi = stats.get("max")
        mean = stats.get("mean")
        std = stats.get("std", 0.0) or 0.0
        if lo is not None and hi is not None and (v < lo or v > hi):
            notes.append(f"{name}={v:.4g} is outside training range [{lo:.4g}, {hi:.4g}]")
        elif std > 0 and mean is not None:
            z = abs(v - mean) / std
            if z > Z_THRESHOLD:
                notes.append(f"{name} is {z:.1f}σ from training mean (>{Z_THRESHOLD}σ)")

    return (len(notes) > 0, notes)
