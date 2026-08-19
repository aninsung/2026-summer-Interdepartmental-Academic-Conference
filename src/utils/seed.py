"""전역 시드 고정.

시드를 고정하지 않으면 모델 초기화와 DataLoader 셔플이 실행마다 달라져
동일 설정에서도 Stage 1 정확도가 0.83~0.85 사이로 흔들린다.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

log = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = False) -> int:
    """python/numpy/torch 의 전역 RNG 를 고정한다.

    deterministic=True 면 cuDNN 벤치마크를 끄고 결정적 알고리즘을 요구한다.
    재현성은 올라가지만 학습이 느려지고, 일부 연산은 결정적 구현이 없어
    런타임 오류가 날 수 있으므로 기본값은 False 다.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        log.warning("torch 를 찾을 수 없어 python/numpy 시드만 고정했습니다.")
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError) as e:
            log.warning("결정적 알고리즘 강제 실패 (%s). 시드만 적용됩니다.", e)

    log.info("시드 고정: %d (deterministic=%s)", seed, deterministic)
    return seed
