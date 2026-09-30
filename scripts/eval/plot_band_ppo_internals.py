"""Stage 3 경계 띠 PPO 내부 경로를 코드 기준으로 그린다.

추론은 refine_batch, 학습은 train_band_ppo.py, 네트워크는 BandRefineNet이다.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FONT = "/tmp/fonts/NotoSansKR.otf"
OUT = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "band_ppo_agent_internals.png"
)


def _font(size: float):
    return font_manager.FontProperties(fname=FONT, size=size * 1.65)


def rbox(ax, x, y, w, h, fc, ec, lw=1.15, radius=0.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    ))


def card(ax, x, y, w, h, fc, ec, title, lines, title_size=11, body=9):
    rbox(ax, x, y, w, h, fc, ec)
    ax.text(x + 1.1, y + h - 1.5, title, fontproperties=_font(title_size),
            color="#1c2430", va="top", ha="left")
    ax.text(x + 1.1, y + h - 4.6, "\n".join(lines), fontproperties=_font(body),
            color="#2a3340", va="top", ha="left", linespacing=1.38)


def pipeline_card(ax, x, y, w, title, lines, fc, ec):
    """전체 파이프라인에서 Stage 3까지의 작은 단계 카드."""
    rbox(ax, x, y, w, 13, fc, ec, lw=1.05, radius=0.45)
    ax.text(x + 1, y + 11.6, title, fontproperties=_font(8.4),
            color="#1c2430", va="top")
    ax.text(x + 1, y + 8.2, "\n".join(lines), fontproperties=_font(7.2),
            color="#34404c", va="top", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color="#3d4754"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
        linewidth=1.2, color=color, shrinkA=1, shrinkB=1,
        clip_on=False,
    ))


def routed_arrow(ax, points, color, label, label_xy):
    """패널 사이의 학습 연결을 직각 점선 화살표로 표시한다."""
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]], [start[1], end[1]],
            color=color, linewidth=1.35, linestyle=(0, (4, 2)),
            clip_on=False, zorder=8,
        )
    start, end = points[-2], points[-1]
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        linewidth=1.35, linestyle=(0, (4, 2)), color=color,
        shrinkA=0, shrinkB=0, clip_on=False, zorder=9,
    ))
    ax.text(
        label_xy[0], label_xy[1], label,
        fontproperties=_font(7.2), color=color, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                  edgecolor=color, linewidth=0.7),
        zorder=10,
    )


def main() -> None:
    fig = plt.figure(figsize=(22.4, 14.4), dpi=150, facecolor="white")
    ax = fig.add_axes([0.012, 0.012, 0.976, 0.976])
    ax.set_xlim(0, 224)
    ax.set_ylim(0, 144)
    ax.axis("off")

    ax.text(1, 141.2, "TRIO 파이프라인과 PPO 내부 경로", fontproperties=_font(18),
            color="#15202b", va="top")
    ax.text(
        76, 140.4,
        "Stage 1 분류 → Stage 2 Expert → Stage 3 경계 띠 PPO 하나 → Stage 4 확정 평가",
        fontproperties=_font(10), color="#52606d", va="top",
    )

    pipeline_card(ax, 3, 121, 28, "입력", [
        "T1ce + FLAIR · 128²",
        "Small Expert만 2.5D",
    ], "#e8eef6", "#3d5a80")
    pipeline_card(ax, 35, 121, 36, "Stage 1 · Expert 라우팅", [
        "확정 평가: ResNet18 분류",
        "PPO 학습·선택: GT 면적",
    ], "#edf1f7", "#536b8a")
    pipeline_card(ax, 75, 121, 48, "Stage 2 · 크기별 Expert", [
        "Small CaraNet · Medium UNet++",
        "Large SegResNet (ED+TC→WT)",
    ], "#eef6f1", "#2d6a4f")
    pipeline_card(ax, 127, 121, 50, "Stage 2 · 확률과 마스크", [
        "원본+좌우+상하 flip TTA 평균",
        "임계값 .80/.80/.50 · CC 0/15/25",
        "Small 예측<80px: 임계값 0.05씩↓",
    ], "#fff6dc", "#9a7209")
    pipeline_card(ax, 181, 121, 40, "Stage 3 입력", [
        "중심 2채널 + Expert 확률",
        "+ 현재 Stage 2 마스크",
    ], "#fdecef", "#9b2335")
    arrow(ax, 31, 127.5, 35, 127.5)
    arrow(ax, 71, 127.5, 75, 127.5)
    arrow(ax, 123, 127.5, 127, 127.5)
    arrow(ax, 177, 127.5, 181, 127.5)

    rbox(ax, 1, 62, 222, 56, "#f7f8fa", "#d5d9e0", lw=1.0, radius=0.8)
    ax.text(3, 116.2, "추론    refine_batch  ·  argmax  ·  5스텝",
            fontproperties=_font(13), color="#1c2430", va="top")

    card(ax, 3, 78, 32, 34, "#e8eef6", "#3d5a80", "관측  4채널 · 128²", [
        "T1ce",
        "FLAIR",
        "현재 마스크",
        "Expert 확률",
        "",
        "영상·확률은 고정",
        "마스크만 갱신",
        "정답은 관측에 없음",
    ])
    rbox(ax, 39, 70, 70, 42, "#eef6f1", "#2d6a4f", radius=0.7)
    ax.text(41, 110.2, "BandRefineNet   폭 32", fontproperties=_font(11),
            color="#1c2430", va="top")
    blocks = [
        (42, "enc1", "32", "128²"),
        (54, "enc2", "64", "64²"),
        (66, "enc3", "128", "32²"),
        (80, "dec2", "64", "64²"),
        (92, "dec1", "32", "128²"),
    ]
    for x, name, ch, res in blocks:
        rbox(ax, x, 90, 10, 16, "white", "#2d6a4f", lw=1.0, radius=0.45)
        ax.text(x + 5, 102.6, name, fontproperties=_font(8.5), ha="center", va="center", color="#1b4332")
        ax.text(x + 5, 98.2, ch, fontproperties=_font(12), ha="center", va="center", color="#1c2430")
        ax.text(x + 5, 94.2, res, fontproperties=_font(8), ha="center", va="center", color="#52606d")
    for x1, x2 in ((90, 92), (102, 104), (114, 118), (128, 130)):
        arrow(ax, x1, 98, x2, 98, "#2d6a4f")
    ax.annotate(
        "", xy=(85, 89), xytext=(59, 89),
        arrowprops=dict(arrowstyle="-|>", color="#b08900", lw=1.05,
                        connectionstyle="arc3,rad=0.0"),
    )
    ax.annotate(
        "", xy=(97, 86.2), xytext=(47, 86.2),
        arrowprops=dict(arrowstyle="-|>", color="#b08900", lw=1.05,
                        connectionstyle="arc3,rad=0.0"),
    )
    ax.text(72, 87.4, "skip enc2→dec2", fontproperties=_font(7.2), color="#8a6a00", ha="center", va="bottom")
    ax.text(72, 82.6, "skip enc1→dec1", fontproperties=_font(7.2), color="#8a6a00", ha="center", va="top")
    ax.text(41, 75.5, "디코더는 dec2, dec1 두 블록", fontproperties=_font(8),
            color="#3d4f44", va="center")
    ax.text(41, 72.4, "입력 4 = 영상 2 + 마스크 1 + 확률 1", fontproperties=_font(8),
            color="#3d4f44", va="center")

    card(ax, 113, 82, 34, 30, "#fff6dc", "#9a7209", "편집 띠 + 로짓 제한", [
        "확률 ∈ [0.35, 0.65] 또는",
        "현재 마스크 경계 ±2px",
        "매 스텝 새 마스크로 재계산",
        "",
        "띠 밖 OFF·ON 로짓 −1e9",
        "→ KEEP만 선택 가능",
        "확률 구간은 경계 밖도 포함",
    ], title_size=10.2, body=8.0)
    card(ax, 151, 82, 34, 30, "#fdecef", "#9b2335", "마지막 스텝 FLAIR 가드", [
        "행동 선택 전 로짓 마스킹",
        "z = (밝기−띠 평균) / 표준편차",
        "z ≤ 0  → 켜기 금지",
        "z ≥ 0  → 끄기 금지",
        "띠 < 8px 또는 분산 0",
        "  → z=0, 켜기·끄기 금지",
        "1–4스텝의 변경은 유지",
        "절대 밝기 임계값 없음",
    ], title_size=10.2, body=7.8)
    card(ax, 189, 82, 32, 30, "#fde8d8", "#c45c26", "행동 선택", [
        "3채널: 끄기 · 유지 · 켜기",
        "",
        "추론  argmax",
        "학습  Categorical",
        "",
        "띠 밖은 유지",
    ], title_size=10.5, body=8.2)
    card(ax, 113, 64, 34, 16, "#eee8f7", "#5c4d7a", "가치 머리  1채널", [
        "dec1 특징에서 1×1",
        "픽셀 가치 · 학습 GAE 전용",
        "마스크 갱신에는 사용하지 않음",
    ], title_size=10, body=8.0)

    arrow(ax, 35, 95, 39, 95)
    arrow(ax, 109, 102, 113, 102, "#9a7209")
    arrow(ax, 147, 102, 151, 102, "#9b2335")
    arrow(ax, 185, 102, 189, 102, "#c45c26")
    arrow(ax, 109, 82, 113, 72, "#5c4d7a")
    routed_arrow(
        ax,
        [(201, 121), (201, 119.6), (18, 119.6), (18, 112)],
        "#3d5a80", "중심 영상 + Stage 2 확률·마스크", (109.5, 119.6),
    )
    routed_arrow(
        ax,
        [(35, 82), (35, 78.5), (130, 78.5), (130, 82)],
        "#9a7209", "고정 확률 + 갱신 마스크", (82.5, 78.5),
    )

    rbox(ax, 39, 64, 70, 5.2, "white", "#3d5a80", lw=1.0, radius=0.4)
    ax.text(
        74, 66.6,
        "스텝 1→5    마스크만 갱신    확률·영상 고정    DSC가 내려도 초기 마스크로 되돌리지 않음",
        fontproperties=_font(7.1), color="#1c2430", ha="center", va="center",
    )
    arrow(ax, 18, 69.2, 18, 78, "#3d5a80")
    rbox(ax, 189, 64, 32, 15.5, "white", "#9b2335", lw=1.0, radius=0.4)
    ax.text(205, 73.2, "마스크 갱신", fontproperties=_font(10), color="#9b2335",
            ha="center", va="center")
    ax.text(205, 68.5, "5번째 결과 = 최종 마스크", fontproperties=_font(7.8),
            color="#52606d", ha="center", va="center")
    arrow(ax, 205, 82, 205, 79.5, "#9b2335")

    rbox(ax, 1, 1.5, 222, 58, "#f7f8fa", "#d5d9e0", lw=1.0, radius=0.8)
    ax.text(3, 57.6, "학습    train_band_ppo.py  ·  행동은 샘플",
            fontproperties=_font(13), color="#1c2430", va="top")

    steps = [
        (3, "#e8eef6", "#3d5a80", "1  데이터", [
            "개발 train의 Stage 2",
            "확률·마스크를 TTA로 생성",
            "Medium 지도학습 actor로 시작",
            "클래스 0·1·2를 한 풀에서 셔플",
        ]),
        (40, "#fff6dc", "#9a7209", "2  롤아웃", [
            "배치 16",
            "5스텝, 띠·가드는 추론과 같음",
            "Categorical에서 샘플",
            "로그확률을 PPO에 저장",
        ]),
        (77, "#fde8d8", "#c45c26", "3  보상", [
            "뒤집은 픽셀만",
            "GT와 같으면 +1, 다르면 −1",
            "HD95를 계산할 수 있으면",
            "픽셀마다 +0.25×(이전−이후)",
        ]),
        (114, "#eee8f7", "#5c4d7a", "4  GAE · PPO", [
            "γ 0.9 ,  λ 0.95",
            "이득은 띠 픽셀만 정규화",
            "clip 0.1",
            "entropy 0.001 , value 0.5",
            "PPO epoch 2 , grad clip 0.5",
        ]),
        (151, "#eef6f1", "#2d6a4f", "5  AdamW", [
            "weight decay 0",
            "행동 lr  1e-5",
            "가치 lr  1e-4",
            "학습 에폭 6",
        ]),
        (188, "#fdecef", "#9b2335", "6  저장", [
            "macro DSC ≥ Stage 2",
            "그리고 macro HD95가 최저",
            "actor만 저장 (value 제외)",
            "DSC 0.8972 / HD95 3.558",
            "Stage 2는 0.8741 / 3.838",
        ]),
    ]
    for i, (x, fc, ec, title, lines) in enumerate(steps):
        card(ax, x, 30.5, 33, 22.5, fc, ec, title, lines, title_size=10.5, body=8.3)
        if i < len(steps) - 1:
            arrow(ax, x + 33, 41.5, x + 36.2, 41.5)

    # 추론과 학습은 별도 모델이 아니라 같은 actor/value 네트워크를 공유한다.
    routed_arrow(
        ax,
        [(100, 70), (100, 60.8), (56.5, 60.8), (56.5, 53)],
        "#2f6690", "공유 actor 로짓 → 학습 샘플", (77.5, 60.8),
    )
    routed_arrow(
        ax,
        [(130, 64), (130, 58.5), (130.5, 58.5), (130.5, 53)],
        "#6a4c93", "픽셀 가치 → GAE", (132, 58.5),
    )
    routed_arrow(
        ax,
        [(167.5, 53), (178, 53), (178, 61.8), (108, 61.8), (108, 70)],
        "#2d6a4f", "AdamW: actor + value 갱신", (153, 61.8),
    )

    ax.text(
        3, 27.6,
        "Stage 4 확정 평가  ·  개발 400명을 제외한 851명  ·  분류기 라우팅  ·  50,010장  ·  환자 평균",
        fontproperties=_font(8.4), color="#52606d", va="center",
    )
    headers = ["정답 크기", "환자", "Stage 2 DSC", "PPO DSC", "짝 차이 (95% CI)"]
    rows = [
        ["전체", "851", "0.8337", "0.8604", "+0.0267 (0.0256–0.0278)"],
        ["300px 미만", "851", "0.7579", "0.7933", "+0.0354 (0.0338–0.0370)"],
        ["300–700px", "757", "0.8700", "0.8952", "+0.0252 (0.0239–0.0265)"],
        ["700px 이상", "408", "0.8981", "0.9169", "+0.0188 (0.0173–0.0205)"],
    ]
    col_x = [5, 48, 75, 116, 157]
    col_w = [40, 24, 38, 38, 61]
    head_y = 22.2
    rbox(ax, 3, 3.2, 218, 22.2, "white", "#d5d9e0", lw=1.0, radius=0.45)
    for x, w, text in zip(col_x, col_w, headers):
        if not text:
            continue
        ax.text(x + w / 2, head_y, text, fontproperties=_font(8.6),
                color="#1c2430", ha="center", va="center")
    ax.plot([3.6, 220.4], [19.6, 19.6], color="#e4e7ec", lw=0.8)
    for i, row in enumerate(rows):
        yy = 17.2 - i * 3.5
        for x, w, text in zip(col_x, col_w, row):
            ax.text(x + w / 2, yy, text, fontproperties=_font(8.6),
                    color="#1c2430", ha="center", va="center")

    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(os.path.abspath(OUT))


if __name__ == "__main__":
    main()
