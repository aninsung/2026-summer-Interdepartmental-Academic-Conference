"""Load checkpoints whose conv in/out channels differ (2.5D, multi-head)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _adapt_tensor(src: torch.Tensor, shape: torch.Size) -> Optional[torch.Tensor]:
    if tuple(src.shape) == tuple(shape):
        return src
    if src.ndim == 4 and src.shape[0] == shape[0] and src.shape[2:] == shape[2:]:
        old_c, new_c = src.shape[1], shape[1]
        if new_c % old_c == 0:
            reps = new_c // old_c
            return src.repeat(1, reps, 1, 1) / float(reps)
        if old_c % new_c == 0:
            grouped = src.view(src.shape[0], new_c, old_c // new_c, *src.shape[2:])
            return grouped.mean(dim=2)
    if src.ndim == 4 and src.shape[1:] == shape[1:]:
        old_o, new_o = src.shape[0], shape[0]
        if new_o % old_o == 0:
            return src.repeat(new_o // old_o, 1, 1, 1)
        if old_o % new_o == 0:
            return src.view(new_o, old_o // new_o, *src.shape[1:]).mean(dim=1)
    if src.ndim == 1 and shape[0] % src.shape[0] == 0:
        return src.repeat(shape[0] // src.shape[0])
    return None


def load_adapted_state_dict(
    model: nn.Module,
    ckpt_path: str,
    device,
    logger=None,
) -> None:
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_state = model.state_dict()
    adapted = {}
    n_adapt = 0
    n_skip = 0
    for k, v in raw.items():
        if k not in model_state:
            n_skip += 1
            continue
        target_shape = model_state[k].shape
        if tuple(v.shape) == tuple(target_shape):
            adapted[k] = v
            continue
        new_v = _adapt_tensor(v, target_shape)
        if new_v is None:
            n_skip += 1
            if logger is not None:
                logger.warning(
                    "Skip %s: checkpoint %s vs model %s",
                    k, tuple(v.shape), tuple(target_shape),
                )
            continue
        adapted[k] = new_v
        n_adapt += 1
        if logger is not None:
            logger.warning(
                "Adapted %s: %s -> %s",
                k, tuple(v.shape), tuple(target_shape),
            )
    missing, unexpected = model.load_state_dict(adapted, strict=False)
    if logger is not None:
        logger.info(
            "Loaded %s (adapted=%d, skipped=%d, missing=%d, unexpected=%d)",
            ckpt_path, n_adapt, n_skip, len(missing), len(unexpected),
        )
