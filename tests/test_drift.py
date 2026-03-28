"""Drift helper tests."""

from __future__ import annotations

from utils.drift import check_feature_drift


def test_no_stats_no_alert():
    alert, notes = check_feature_drift({"Age": 99}, None)
    assert alert is False
    assert notes == []


def test_out_of_range():
    stats = {"Age": {"mean": 35.0, "std": 10.0, "min": 18.0, "max": 70.0}}
    alert, notes = check_feature_drift({"Age": 5.0}, stats)
    assert alert is True
    assert any("outside training range" in n for n in notes)
