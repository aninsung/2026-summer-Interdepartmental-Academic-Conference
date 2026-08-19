"""베이스라인 학습에 쓰는 손실 함수.

각 방법이 논문에서 명시한 손실을 그대로 옮긴다.
    - KAIST(1위): BCE + batch Dice, deep supervision 가중 합.
      batch Dice 는 nnU-Net 의 BraTS 설정으로, 샘플별로 Dice 를 구해 평균하지 않고
      배치 전체를 하나의 덩어리로 보고 한 번 계산한다. 종양이 없는 슬라이스에서
      샘플별 Dice 가 불안정해지는 문제를 막아 준다.
    - NVAUTO(2위): soft Dice + Barlow Twins(불변성 + 중복 감소), lambda=0.005.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    batch_dice: bool = True,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """soft Dice 손실.

    Args:
        batch_dice: True 면 배치 전체를 하나로 묶어 Dice 를 한 번 계산한다(nnU-Net BraTS 설정).
            False 면 샘플별로 계산한 뒤 평균한다.
    """
    prob = torch.sigmoid(logits)
    if batch_dice:
        p = prob.reshape(-1)
        g = target.reshape(-1)
        inter = (p * g).sum()
        denom = p.sum() + g.sum()
    else:
        p = prob.reshape(prob.shape[0], -1)
        g = target.reshape(target.shape[0], -1)
        inter = (p * g).sum(dim=1)
        denom = p.sum(dim=1) + g.sum(dim=1)

    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


class BceBatchDiceLoss(nn.Module):
    """nnU-Net BraTS 설정의 BCE + batch Dice 복합 손실."""

    def __init__(self, bce_weight: float = 0.5, batch_dice: bool = True):
        super().__init__()
        self.bce_weight = bce_weight
        self.batch_dice = batch_dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = soft_dice_loss(logits, target, batch_dice=self.batch_dice)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice


class DeepSupervisionLoss(nn.Module):
    """해상도별 출력에 가중치를 주어 손실을 합산한다.

    nnU-Net 과 같이 해상도가 절반씩 낮아질 때마다 가중치도 절반으로 줄이고,
    전체 가중치 합이 1이 되도록 정규화한다. 정답은 각 출력 해상도로 내려서 맞춘다.
    """

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(
        self, outputs: torch.Tensor | Sequence[torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        if torch.is_tensor(outputs):
            return self.base_loss(outputs, target)

        weights = [0.5 ** i for i in range(len(outputs))]
        total = sum(weights)
        loss = outputs[0].new_zeros(())
        for w, out in zip(weights, outputs):
            if out.shape[-2:] != target.shape[-2:]:
                tgt = F.interpolate(target, size=out.shape[-2:], mode="nearest")
            else:
                tgt = target
            loss = loss + (w / total) * self.base_loss(out, tgt)
        return loss


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins 손실: 교차상관 행렬의 대각을 1로, 비대각을 0으로 만든다.

    대각 항은 두 증강 뷰 사이의 불변성을, 비대각 항은 특징 차원 간 중복 감소를
    담당한다. 원 논문과 같이 lambda=0.005 를 기본값으로 쓴다.
    """

    def __init__(self, lambda_coeff: float = 0.005, eps: float = 1e-5):
        super().__init__()
        self.lambda_coeff = lambda_coeff
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        n, d = z1.shape
        if n < 2:
            return z1.new_zeros(())

        # 배치 축(= 영역 축) 기준 표준화
        z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + self.eps)
        z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + self.eps)

        c = (z1.T @ z2) / n
        on_diag = (torch.diagonal(c) - 1.0).pow(2).sum()
        off_diag = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()
        return on_diag + self.lambda_coeff * off_diag


def region_confidence(prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """NVAUTO 논문의 confidence heuristic: 분할된 영역의 평균 확률값.

    확률맵이 확신에 찰수록 분할 영역 내부 확률이 1에 가까워진다는 관찰에 기반한다.
    분할 영역이 비면 0을 반환한다.

    Args:
        prob: (B, C, H, W) 시그모이드 확률맵.
    Returns:
        (B,) 샘플별 confidence.
    """
    mask = (prob > threshold).float()
    area = mask.flatten(1).sum(dim=1)
    conf = (prob.flatten(1) * mask.flatten(1)).sum(dim=1) / area.clamp(min=1.0)
    return torch.where(area > 0, conf, torch.zeros_like(conf))


def confidence_ensemble(
    probs: List[torch.Tensor], top_fraction: float = 0.5, threshold: float = 0.5
) -> torch.Tensor:
    """confidence 상위 N/2 확률맵만 동일 가중으로 평균한다(NVAUTO 앙상블).

    Args:
        probs: 모델별 (B, C, H, W) 확률맵 리스트.
    """
    if len(probs) == 1:
        return probs[0]

    stacked = torch.stack(probs, dim=0)  # (M, B, C, H, W)
    confs = torch.stack([region_confidence(p, threshold) for p in probs], dim=0)  # (M, B)

    k = max(1, int(round(len(probs) * top_fraction)))
    idx = confs.topk(k, dim=0).indices  # (k, B)

    m, b = confs.shape
    select = torch.zeros(m, b, device=stacked.device, dtype=stacked.dtype)
    select.scatter_(0, idx, 1.0)
    weights = select / select.sum(dim=0, keepdim=True)
    return (stacked * weights.view(m, b, 1, 1, 1)).sum(dim=0)
