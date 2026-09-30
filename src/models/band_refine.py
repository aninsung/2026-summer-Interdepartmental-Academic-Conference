"""경계 띠 픽셀 보정 네트워크.

Stage 2 마스크를 통째로 다시 그리지 않는다. 공유 합성곱이 픽셀마다
끄기/유지/켜기를 내고, 띠 밖은 Stage 2를 복사한다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.utils.band import OFF, ON, apply_actions, apply_flair_guard, edit_band, flair_z_map

__all__ = [
    "BandRefineNet", "BandActorCritic", "build_band_refine", "constrain_logits",
    "reconstruct_mask", "refine_batch", "predict_pair", "predict_slice",
]


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class BandRefineNet(nn.Module):
    """소형 U-Net. 출력 3채널은 끄기, 유지, 켜기."""

    def __init__(self, in_channels: int = 4, width: int = 32):
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, width)
        self.enc2 = _ConvBlock(width, width * 2)
        self.enc3 = _ConvBlock(width * 2, width * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec2 = _ConvBlock(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec1 = _ConvBlock(width * 2, width)
        self.head = nn.Conv2d(width, 3, 1)
        nn.init.zeros_(self.head.weight)
        # 학습 시작은 거의 전부 유지. Stage 2 마스크에서 출발한다.
        nn.init.constant_(self.head.bias, 0.0)
        nn.init.constant_(self.head.bias[1], 4.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        return self.dec1(torch.cat([self.up1(d2), e1], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def build_band_refine(in_channels: int = 4, width: int = 32) -> BandRefineNet:
    return BandRefineNet(in_channels=in_channels, width=width)


class BandActorCritic(nn.Module):
    """띠 보정 네트워크에 픽셀 가치 머리를 붙인 하나의 정책."""

    def __init__(self, actor: BandRefineNet):
        super().__init__()
        self.actor = actor
        self.value_head = nn.Conv2d(actor.head.in_channels, 1, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward_with_value(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.actor.encode(x)
        return self.actor.head(feat), self.value_head(feat).squeeze(1)


def constrain_logits(
    logits: torch.Tensor,
    band: torch.Tensor,
    z: torch.Tensor | None = None,
    guard: bool = False,
) -> torch.Tensor:
    """띠 밖은 유지. guard면 어두운 픽셀은 켜기, 밝은 픽셀은 끄기를 막는다."""
    neg = logits.new_full((), -1.0e9)
    editable = band.to(device=logits.device, dtype=torch.bool)
    out = logits.clone()
    out[:, OFF] = torch.where(editable, out[:, OFF], neg)
    out[:, ON] = torch.where(editable, out[:, ON], neg)
    if guard and z is not None:
        zz = z.to(device=logits.device)
        out[:, ON] = torch.where(editable & (zz <= 0), neg, out[:, ON])
        out[:, OFF] = torch.where(editable & (zz >= 0), neg, out[:, OFF])
    return out


def _apply_actions_torch(mask: torch.Tensor, actions: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    out = mask.clone()
    plane = out[:, 0]
    editable = band.to(device=mask.device, dtype=torch.bool)
    plane[editable & (actions == ON)] = 1.0
    plane[editable & (actions == OFF)] = 0.0
    return out


@torch.no_grad()
def refine_batch(
    actor: nn.Module,
    image: torch.Tensor,
    prob: torch.Tensor,
    mask: torch.Tensor,
    flair: torch.Tensor,
    n_steps: int,
    prob_lo: float,
    prob_hi: float,
    radius: int,
    guard_last: bool = True,
) -> torch.Tensor:
    """결정적 argmax로 n_steps번 띠를 고친다. 가드는 마지막 스텝에만 건다."""
    current = mask.clone()
    for step in range(n_steps):
        logits = actor(torch.cat([image, current, prob], dim=1))
        bands, zs = [], []
        prob_np = prob[:, 0].detach().cpu().numpy()
        mask_np = current[:, 0].detach().cpu().numpy()
        flair_np = flair[:, 0].detach().cpu().numpy()
        for index in range(current.shape[0]):
            band = edit_band(prob_np[index], mask_np[index], prob_lo, prob_hi, radius)
            bands.append(band)
            zs.append(flair_z_map(flair_np[index], band))
        band_t = torch.from_numpy(np.stack(bands)).to(current.device)
        z_t = torch.from_numpy(np.stack(zs)).to(current.device)
        logits = constrain_logits(logits, band_t, z_t, guard=guard_last and step == n_steps - 1)
        current = _apply_actions_torch(current, logits.argmax(dim=1), band_t)
    return current


def reconstruct_mask(logits: torch.Tensor, stage2: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """소프트 마스크. 끄기=0, 유지=Stage 2, 켜기=1. 띠 밖은 Stage 2."""
    prob = torch.softmax(logits, dim=1)
    soft = prob[:, 2:3] + prob[:, 1:2] * stage2
    return torch.where(band > 0.5, soft, stage2)


@torch.no_grad()
def predict_pair(
    model: nn.Module,
    entry,
    device: torch.device,
    prob_lo: float,
    prob_hi: float,
    radius: int,
    flair_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """가드 없는 마스크와 FLAIR 가드를 통과한 마스크."""
    prob = np.asarray(entry.probability, dtype=np.float32)
    band = edit_band(prob, entry.stage2, prob_lo, prob_hi, radius)
    image = torch.from_numpy(entry.model_input()).unsqueeze(0).to(device)
    actions = model(image).argmax(dim=1).squeeze(0).cpu().numpy()
    raw = apply_actions(entry.stage2, actions, band)
    flair = np.asarray(entry.image[flair_index], dtype=np.float32)
    guarded = apply_flair_guard(flair, entry.stage2, actions, band)
    return raw, guarded


@torch.no_grad()
def predict_slice(
    model: nn.Module,
    entry,
    device: torch.device,
    prob_lo: float,
    prob_hi: float,
    radius: int,
    flair_index: int,
    use_guard: bool,
) -> np.ndarray:
    """한 슬라이스의 보정 마스크. use_guard면 FLAIR 상대 밝기 조건을 통과한 뒤집기만 적용."""
    raw, guarded = predict_pair(model, entry, device, prob_lo, prob_hi, radius, flair_index)
    return guarded if use_guard else raw
