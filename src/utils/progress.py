"""Progress-bar / log-noise helpers.

tqdm ``\\r`` updates bloat ``tee``/agent terminal logs (MB-scale). Default OFF.
Enable interactively with ``TQDM_ENABLE=1``; force off with ``TQDM_DISABLE=1``.
"""

from __future__ import annotations

import os
import sys


def want_tqdm() -> bool:
    """Whether to show tqdm bars (default: no)."""
    disable = os.environ.get("TQDM_DISABLE", "").strip().lower()
    if disable in ("1", "true", "yes", "on"):
        return False
    enable = os.environ.get("TQDM_ENABLE", "").strip().lower()
    if enable in ("1", "true", "yes", "on"):
        return True
    return False


def configure_quiet_logs(*, verbose_progress: bool = False) -> None:
    """Apply process-wide quiet defaults (call early from run_pipeline / entrypoints)."""
    if verbose_progress:
        os.environ.pop("TQDM_DISABLE", None)
        os.environ["TQDM_ENABLE"] = "1"
        return
    enable = os.environ.get("TQDM_ENABLE", "").strip().lower()
    if enable in ("1", "true", "yes", "on"):
        return
    os.environ.setdefault("TQDM_DISABLE", "1")
    # Stable-Baselines3 / gym noise
    os.environ.setdefault("SB3_LOG_LEVEL", "ERROR")
