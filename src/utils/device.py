"""PyTorch device selection — pipeline defaults to CUDA-only."""

from __future__ import annotations

import logging
import os
import sys

import torch

log = logging.getLogger(__name__)


def get_torch_device(prefer_cuda: bool = True, require_cuda: bool = False) -> torch.device:
    """Return cuda if available and a probe kernel succeeds.

    If require_cuda=True, exit when CUDA is unavailable instead of falling back to CPU.
    """
    if not prefer_cuda:
        if require_cuda:
            log.error("require_cuda=True 인데 prefer_cuda=False 입니다.")
            sys.exit(1)
        return torch.device("cpu")

    if not torch.cuda.is_available():
        msg = "CUDA를 사용할 수 없습니다. GPU 환경에서 실행하세요."
        if require_cuda:
            log.error(msg)
            sys.exit(1)
        log.warning("%s → CPU로 진행합니다.", msg)
        return torch.device("cpu")

    try:
        torch.zeros(1, device="cuda").add_(1)
        torch.cuda.synchronize()
        return torch.device("cuda")
    except RuntimeError as exc:
        msg = f"CUDA 프로브 실패 ({exc})"
        if require_cuda:
            log.error("%s — GPU 전용 모드라 중단합니다.", msg)
            sys.exit(1)
        log.warning("%s → CPU로 학습/추론합니다.", msg)
        return torch.device("cpu")


def require_cuda_device() -> torch.device:
    """Pipeline entry: CUDA만 허용. 실패 시 즉시 종료."""
    # 자식 프로세스/라이브러리가 CPU로 새지 않도록 기본값 정리
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    device = get_torch_device(prefer_cuda=True, require_cuda=True)
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log.info("GPU 전용 모드: %s (%.1f GiB) device=%s", name, mem_gb, device)
    return device
