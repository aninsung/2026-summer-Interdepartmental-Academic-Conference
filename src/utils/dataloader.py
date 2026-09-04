"""DataLoader worker helpers (Windows-safe defaults)."""

from __future__ import annotations

import os


def resolve_num_workers(num_workers: int | None = None, default_cap: int = 4) -> int:
    """Pick a worker count. None → 0 on Windows (shared-memory stability), else min(cap, cpu)."""
    if num_workers is not None:
        return max(0, int(num_workers))
    if os.name == "nt":
        return 0
    return min(default_cap, os.cpu_count() or 2)


def loader_kwargs(num_workers: int | None = None, *, pin_memory: bool = True) -> dict:
    """
    Common DataLoader kwargs. When workers > 0, enable persistent_workers + prefetch
    so the GPU is not stalled on single-threaded I/O (num_workers=0).
    """
    n = resolve_num_workers(num_workers)
    kw: dict = {"num_workers": n, "pin_memory": pin_memory}
    if n > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return kw
