"""PyTorch device selection with CUDA compatibility probe."""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def get_torch_device(prefer_cuda: bool = True) -> torch.device:
    """Return cuda if available and a probe kernel succeeds, else cpu."""
    if not prefer_cuda or not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda").add_(1)
        return torch.device("cuda")
    except RuntimeError as exc:
        log.warning("CUDA 사용 불가 (%s) → CPU로 학습/추론합니다.", exc)
        return torch.device("cpu")
