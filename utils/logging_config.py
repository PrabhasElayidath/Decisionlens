"""Rotating file logging for the API (optional)."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_api_logging(name: str = "decisionlens") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log

    log.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)

    log_dir = Path(__file__).resolve().parent.parent / "logs" / "logs"
    if os.getenv("DECISIONLENS_LOG_FILE", "true").lower() in ("1", "true", "yes"):
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                log_dir / "decisionlens.log",
                maxBytes=1_000_000,
                backupCount=5,
                encoding="utf-8",
            )
            fh.setFormatter(fmt)
            log.addHandler(fh)
        except OSError:
            pass

    return log
