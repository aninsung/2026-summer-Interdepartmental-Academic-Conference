"""베이스라인 학습/평가가 공유하는 유틸.

가장 중요한 역할은 환자 분할 파일을 보호하는 것이다.
`load_or_create_patient_split` 은 max_patients / root_dir / seed / val_ratio 중
하나라도 캐시와 다르면 분할 파일을 **덮어쓰고 재생성**한다. 파이프라인이 이미
그 분할로 학습을 마친 상태에서 이런 일이 벌어지면 비교 자체가 무의미해지므로,
설정이 어긋나면 실행 전에 멈춘다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

log = logging.getLogger(__name__)


def check_split_compatibility(
    train_root: str,
    max_patients: Optional[int],
    split_path: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict:
    """기존 분할 파일과 설정이 일치하는지 확인한다.

    Raises:
        SystemExit: 설정이 어긋나 분할 파일이 덮어써질 상황이면 안내와 함께 종료한다.
    """
    path = Path(split_path)
    if not path.exists():
        log.warning(
            "분할 파일이 없어 새로 생성됩니다: %s. "
            "파이프라인과 비교하려면 파이프라인이 쓰는 분할 파일을 지정하세요.",
            path,
        )
        return {}

    with open(path, encoding="utf-8") as f:
        split = json.load(f)

    mismatches: List[str] = []
    for key, want in (
        ("root_dir", train_root),
        ("max_patients", max_patients),
        ("val_ratio", val_ratio),
        ("seed", seed),
    ):
        got = split.get(key)
        if got != want:
            mismatches.append(f"  - {key}: 분할 파일={got!r} vs 요청={want!r}")

    if mismatches:
        raise SystemExit(
            f"\n[중단] 요청한 설정이 {path} 와 달라 분할 파일이 덮어써질 상황입니다.\n"
            + "\n".join(mismatches)
            + "\n\n분할이 바뀌면 파이프라인 결과와 더 이상 비교할 수 없습니다. 해결 방법:\n"
            f"  1) 분할 파일과 같은 설정으로 실행 (max_patients={split.get('max_patients')})\n"
            "  2) 다른 실험을 하려면 --patient_split 에 별도 경로를 지정\n"
        )

    log.info(
        "분할 확인 완료: %s (train=%d명, val=%d명)",
        path,
        len(split.get("train", [])),
        len(split.get("val", [])),
    )
    return split


def simple_collate(batch):
    """이미지와 정답 마스크만 묶는다(증강 없음)."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "gt_mask": torch.stack([b["gt_mask"] for b in batch]),
    }
