"""Ensure trained artifacts exist for integration tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def ensure_models() -> Path:
    meta = ROOT / "models" / "metadata.json"
    if meta.exists():
        return ROOT
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run(
        [sys.executable, str(ROOT / "data" / "train.py")],
        cwd=str(ROOT),
        env=env,
        check=True,
        timeout=600,
    )
    return ROOT
