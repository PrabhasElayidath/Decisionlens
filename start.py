"""Launch API + Streamlit together (development convenience)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT)

print("Starting FastAPI backend...")
api_process = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ],
    env=env,
    cwd=str(ROOT),
)

print("Waiting for API (model load / optional auto-train)...")
time.sleep(8)

print("Starting Streamlit dashboard...")
ui_process = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "run_streamlit.py",
        "--server.port",
        "8511",
        "--server.headless",
        "true",
    ],
    env=env,
    cwd=str(ROOT),
)

print("Open http://localhost:8511 (API docs: http://localhost:8000/docs)")

try:
    api_process.wait()
    ui_process.wait()
except KeyboardInterrupt:
    print("Shutting down...")
    api_process.terminate()
    ui_process.terminate()
