"""한국정보과학회 학술대회 심사용 논문. 저자 정보 없음. 2~3쪽."""

from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]
FONT = "/tmp/fonts/NotoSansKR-VF.ttf"
OUT = Path(__file__).resolve().parent / "kiise_review_blind.pdf"
PIPE = ROOT / "results" / "pipeline_overview.jpg"
RESULT = ROOT / "results" / "band_ppo_delta_matched" / "delta_matched_comparison.png"

pdfmetrics.registerFont(TTFont("Noto", FONT))

PAGE_W, PAGE_H = A4
LEFT, RIGHT, TOP, BOTTOM = 10 * mm, 10 * mm, 30 * mm, 20 * mm
GAP = 4 * mm
COL_W = (PAGE_W - LEFT - RIGHT - GAP) / 2
FULL_W = PAGE_W - LEFT - RIGHT


def style(name, size, leading, align=TA_JUSTIFY, space_before=0, space_after=2):
    return ParagraphStyle(
        name,
        fontName="Noto",
        fontSize=size,
        leading=leading,
        alignment=align,
        textColor=colors.black,
        spaceBefore=space_before,
        spaceAfter=space_after,
        firstLineIndent=0,
    )


S_TITLE = style("title", 14, 18, TA_CENTER, 0, 3)
S_EN = style("en", 11, 14, TA_CENTER, 1, 4)
S_ABS_H = style("absh", 10, 13, TA_CENTER, 2, 2)
S_ABS = style("abs", 9, 12, TA_JUSTIFY, 0, 2)
S_H = style("h", 10, 13, TA_LEFT, 3, 1)
S_BODY = style("body", 9, 12, TA_JUSTIFY, 0, 1.5)
S_CAP = style("cap", 9, 11, TA_CENTER, 1, 2)
S_REF = style("ref", 9, 11, TA_LEFT, 0, 1)


class Paper:
    def __init__(self):
        self.c = canvas.Canvas(str(OUT), pagesize=A4)
        self.page = 1
        self.col = 0
        self.y = PAGE_H - TOP
        self.col_top = self.y

    def _x(self):
        return LEFT if self.col == 0 else LEFT + COL_W + GAP

    def new_page(self):
        self.c.showPage()
        self.page += 1
        self.col = 0
        self.y = PAGE_H - TOP
        self.col_top = self.y

    def _advance_column(self):
        if self.col == 0:
            self.col = 1
            self.y = self.col_top
        else:
            self.new_page()

    def ensure_pair_start(self):
        """전체 폭 항목 앞에 빈 오른쪽 단이 남지 않게 단 쌍을 마친다."""
        if self.col == 1:
            self.new_page()

    def draw_para(self, text, sty, width=None):
        width = COL_W if width is None else width
        full = abs(width - COL_W) > 1
        if full:
            self.ensure_pair_start()
        elif sty is S_H and (self.y - BOTTOM) < 16 * mm:
            self._advance_column()
        p = Paragraph(text, sty)
        w, h = p.wrap(width, 2000)
        if self.y - h < BOTTOM:
            if full:
                self.new_page()
            else:
                self._advance_column()
                if self.y - h < BOTTOM:
                    self.new_page()
        p.drawOn(self.c, LEFT if full else self._x(), self.y - h)
        self.y -= h
        return h

    def draw_full(self, flowable_height, drawer):
        self.ensure_pair_start()
        if self.y - flowable_height < BOTTOM:
            self.new_page()
        drawer(LEFT, self.y - flowable_height, FULL_W, flowable_height)
        self.y -= flowable_height
        self.col = 0
        self.col_top = self.y

    def image(self, path, max_h):
        from reportlab.lib.utils import ImageReader
        img = ImageReader(str(path))
        iw, ih = img.getSize()
        scale = min(FULL_W / iw, max_h / ih)
        w, h = iw * scale, ih * scale
        pad = 1 * mm

        def drawer(x, y, _w, _h):
            self.c.drawImage(str(path), x + (FULL_W - w) / 2, y + pad, w, h - pad, preserveAspectRatio=True, mask="auto")

        self.draw_full(h + pad, drawer)

    def table(self):
        header = ["구간", "환자", "PPO 이전", "PPO 이후", "짝 차이 (95% CI)"]
        rows = [
            ["전체", "851", "0.8337", "0.8604", "+0.0267 (0.0256–0.0278)"],
            ["300px 미만", "851", "0.7579", "0.7933", "+0.0354 (0.0338–0.0370)"],
            ["300–700px", "757", "0.8700", "0.8952", "+0.0252 (0.0239–0.0265)"],
            ["700px 이상", "408", "0.8981", "0.9169", "+0.0188 (0.0173–0.0205)"],
        ]
        data = [header] + rows
        sty = ParagraphStyle("td", fontName="Noto", fontSize=9, leading=11, alignment=TA_CENTER)
        sty_l = ParagraphStyle("tdl", fontName="Noto", fontSize=9, leading=11, alignment=TA_LEFT)
        wrapped = []
        for r, row in enumerate(data):
            wrapped.append([Paragraph(cell, sty_l if c == 0 else sty) for c, cell in enumerate(row)])
        col_w = [28 * mm, 16 * mm, 28 * mm, 28 * mm, 90 * mm]
        t = Table(wrapped, colWidths=col_w, hAlign="CENTER")
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Noto"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.93, 0.93, 0.93)),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        w, h = t.wrap(FULL_W, 400)

        def drawer(x, y, _w, _h):
            t.drawOn(self.c, x + (FULL_W - w) / 2, y)

        self.draw_full(h + 1 * mm, drawer)

    def save(self):
        self.c.save()


def main():
    p = Paper()
    p.draw_para("크기별 전문가와 경계 띠 정책 보정을 이용한 뇌종양 MRI 분할", S_TITLE, FULL_W)
    p.draw_para(
        "Size-Specific Experts and Boundary-Band Policy Refinement for Brain Tumor MRI",
        S_EN, FULL_W,
    )
    p.draw_para("요 약", S_ABS_H, FULL_W)
    p.draw_para(
        "뇌종양 MRI의 whole tumor는 단면 크기와 경계 양상이 달라 하나의 분할 모델로 맞추기 어렵다. "
        "본 연구는 T1ce와 FLAIR를 128×128로 줄인 축상 단면에서 종양 크기를 분류하고, 크기별 전문가로 초기 마스크를 만든 뒤, "
        "하나의 PPO가 정해진 경계 띠 안의 픽셀만 켜거나 끈다. 300픽셀 미만은 2.5D CaraNet, 300 이상 700 미만은 UNet++, "
        "700 이상은 부종과 종양핵을 나눠 예측하는 SegResNet이다. "
        "BraTS 2021에서 개발에 쓴 400명을 뺀 851명, 종양 슬라이스 50,010장을 슬라이스 DSC의 환자 평균으로 평가했다. "
        "분류기가 전문가를 고르고 임계값을 0.80/0.80/0.50으로 고정한 초기 분할 대비 DSC는 0.8337에서 0.8604로 올랐다"
        "(환자 짝 차이 +0.0267, 95% CI 0.0256–0.0278). 본 수치는 3차원 BraTS 점수가 아니다.",
        S_ABS, FULL_W,
    )
    p.col_top = p.y
    p.col = 0
    p.image(PIPE, 34 * mm)
    p.draw_para("그림 1 파이프라인. 크기 분류, 크기별 전문가, 경계 띠 PPO, 확정 평가.", S_CAP, FULL_W)
    p.col_top = p.y
    p.col = 0

    p.draw_para("1. 서 론", S_H)
    p.draw_para(
        "뇌종양 영역 분할은 종양 부담을 정량화하고 치료 범위를 정하는 데 쓰인다. "
        "MRI에서 종양의 크기와 부종 경계는 슬라이스마다 달라, 단일 백본이 아주 작은 단면과 큰 단면을 함께 맞추기 어렵다. "
        "U-Net 계열은 의료영상 분할의 기본 구조이지만[1], 크기별로 오차 형태가 다르면 초기 마스크의 경계에 과소분할이 남는다.",
        S_BODY,
    )
    p.draw_para(
        "본 연구는 초기 분할과 경계 보정을 나눈다. 분류기가 슬라이스 크기를 고르면 해당 전문가가 확률 맵을 만들고, "
        "하나의 PPO가 애매한 확률 구간과 마스크 경계 주변만 수정한다. "
        "이 세 단계를 TRIO라 부른다. TRIO는 크기 분류, 크기별 전문가 분할, 경계 띠 PPO를 가리킨다. "
        "기여는 크기별 라우팅과, 띠 안의 픽셀을 켜고 끄는 PPO 하나이다. "
        "중형 슬라이스만으로 학습한 띠 보정 네트워크는 이 PPO의 초기 가중치이며, 그 지도학습 보정 자체는 기여가 아니다.",
        S_BODY,
    )
    p.draw_para("2. 방 법", S_H)
    p.draw_para(
        "그림 1은 전체 파이프라인이다. 입력은 BraTS 2021의 T1ce와 FLAIR이다[6]. "
        "슬라이스는 128×128로 맞추고, 뇌 영역 안에서 z-score 정규화한 뒤 1–99 퍼센타일로 잘라 0–1로 맞춘다. "
        "종양 픽셀 비율이 0.2% 이상인 단면만 학습과 평가에 넣는다. Whole tumor는 분할 라벨이 0이 아닌 픽셀이다. "
        "크기는 정답 면적으로 나눈다. 300픽셀 미만은 소형, 300 이상 700 미만은 중형, 700 이상은 대형이다.",
        S_BODY,
    )
    p.draw_para(
        "분류기는 ImageNet으로 초기화한 ResNet-18이다. 추론에서 분류기가 전문가를 고른다. "
        "소형은 여섯 채널 2.5D CaraNet[3]이고, 중형은 두 채널 UNet++[2]이며, "
        "대형은 SegResNet[4]이 부종과 종양핵을 따로 예측한 뒤 "
        "1−(1−p<sub>ED</sub>)(1−p<sub>TC</sub>)로 합친다. "
        "확률은 원본과 좌우·상하 반전의 평균이다. 이진화 임계값은 0.80, 0.80, 0.50이고 연결요소 최소 크기는 0, 15, 25픽셀이다.",
        S_BODY,
    )
    p.draw_para(
        "보정 정책은 하나다. 폭 32의 소형 U-Net이 T1ce, FLAIR, 현재 마스크, 전문가 확률을 보고 "
        "픽셀마다 끄기, 유지, 켜기 중 하나를 고른다. 수정 범위는 전문가 확률 0.35–0.65인 픽셀이거나 마스크 경계 ±2픽셀이다. "
        "다섯 스텝 동안 확률 맵은 고정하고 마스크만 넘긴다. FLAIR 상대 밝기 제약은 마지막 스텝에만 둔다. "
        "정답은 학습 보상에만 쓰고 추론 관측에는 넣지 않는다. "
        "뒤집은 픽셀이 정답과 같으면 +1, 다르면 −1이며, HD95가 줄면 그 감소에 0.25를 곱해 더한다. "
        "학습은 PPO[5]이고 클립은 0.1이다. "
        "BraTS의 다중 기관 영상과 전문가 라벨은 Menze 등[7]과 Bakas 등[8]이 정리했고, "
        "nnU-Net[9]은 그 계열에서 전처리와 모델 선택을 데이터에 맞추는 기준이다. "
        "본 연구는 그 3차원 프로토콜 대신 2차원 단면의 경계 띠만 수정한다.",
        S_BODY,
    )

    p.draw_para("3. 실 험", S_H)
    p.draw_para(
        "환자 단위로 나눈다. 1,251명 중 시드 42로 고른 개발 400명은 학습 280, 검증 60, 방법 선택 60이다. "
        "나머지 851명은 가중치 학습과 방법 선택에 쓰지 않았다. "
        "다만 고정 임계값과 연결요소 기준은 이전 210명 풀에서 정했고, 그 중 148명이 이 851명에 들어 있다. "
        "학습과 에폭 선택은 정답 면적으로 전문가를 고르고, 표 1의 평가는 분류기 라우팅이다. "
        "평가 슬라이스의 19.9%는 정답 크기와 다른 전문가의 마스크를 보정한다.",
        S_BODY,
    )
    p.draw_para(
        "주 지표는 종양 슬라이스 DSC의 환자 평균이다. 환자 안과 환자 사이 모두 같은 가중을 쓴다. "
        "표 1은 고정 임계값의 초기 분할과 경계 띠 PPO를 같은 마스크에서 짝비교한 결과이다. "
        "크기 행은 소형 종양만 있는 환자군이 아니다. 그 구간의 단면이 한 장이라도 있는 환자를 모은 것이다.",
        S_BODY,
    )
    p.draw_para("표 1 미사용 851명. 분류기 라우팅. 슬라이스 DSC의 환자 평균.", S_CAP, FULL_W)
    p.table()
    p.col_top = p.y

    p.draw_para(
        "전체 환자 짝 차이 +0.0267의 95% 신뢰구간은 0을 포함하지 않는다. 차이는 300픽셀 미만 구간에서 가장 크고, "
        "700픽셀 이상에서는 +0.0188로 가장 작다. 대형 구간의 초기 DSC가 이미 0.8981이라 띠 안에서 더 올릴 여지가 작다. "
        "같은 평가에서 빈 마스크를 뺀 HD95 평균은 4.888픽셀에서 4.609픽셀로 줄었다. "
        "빠지는 슬라이스가 서로 달라 이 HD95는 짝비교로 쓰지 않는다.",
        S_BODY,
    )
    p.draw_para(
        "검증 60명에서 임계값을 0.45/0.80/0.20으로 다시 고른 초기 분할의 DSC는 0.8411이다. "
        "같은 PPO와의 짝 차이는 +0.0193(95% CI 0.0184–0.0202)으로 남는다. "
        "따라서 표 1의 이득은 임계값을 낮춘 결과만은 아니다.",
        S_BODY,
    )

    p.image(RESULT, 46 * mm)
    p.draw_para(
        "그림 2 개발 집합에 없는 단면. 위는 정답(초록), 가운데는 PPO 이전, 아래는 PPO 이후. "
        "칸의 DSC와 HD95는 그 슬라이스만의 값이며 표 1의 환자 평균이 아니다. "
        "여섯 장 모두 DSC는 올랐다. 중형 첫째는 정답 면적이 중형인데 CaraNet으로 라우팅되었고 HD95는 늘었다.",
        S_CAP, FULL_W,
    )
    p.col_top = p.y

    p.draw_para("4. 결 론", S_H)
    p.draw_para(
        "크기별 전문가와 하나의 경계 띠 PPO를 미사용 851명에 적용하면, "
        "고정 임계값의 초기 분할보다 2차원 슬라이스 DSC의 환자 평균이 0.8337에서 0.8604로 오른다. "
        "수정 범위는 경계 ±2픽셀과 확률 0.35–0.65인 픽셀이다. "
        "이 결과는 3차원 BraTS 점수나 빈 슬라이스를 포함한 평가가 아니고, 학습은 시드 하나의 실행이다. "
        "HD95는 픽셀 단위이며 한쪽 마스크가 비면 그 슬라이스를 평균에서 뺀다.",
        S_BODY,
    )
    p.draw_para("참 고 문 헌", S_H)
    refs = [
        "[1] O. Ronneberger, P. Fischer, T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI, pp. 234–241, 2015.",
        "[2] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, J. Liang, “UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation,” IEEE Trans. Med. Imaging, vol. 39, no. 6, pp. 1856–1867, 2020.",
        "[3] A. Lou, S. Guan, M. Loew, “CaraNet: Context Axial Reverse Attention Network for Segmentation of Small Medical Objects,” J. Med. Imaging, vol. 10, no. 1, 014005, 2023.",
        "[4] A. Myronenko, “3D MRI Brain Tumor Segmentation Using Autoencoder Regularization,” BrainLes, MICCAI, pp. 311–320, 2018.",
        "[5] J. Schulman et al., “Proximal Policy Optimization Algorithms,” arXiv:1707.06347, 2017.",
        "[6] U. Baid et al., “The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification,” arXiv:2107.02314, 2021.",
        "[7] B. H. Menze et al., “The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS),” IEEE Trans. Med. Imaging, vol. 34, no. 10, pp. 1993–2024, 2015.",
        "[8] S. Bakas et al., “Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features,” Scientific Data, vol. 4, 170117, 2017.",
        "[9] F. Isensee et al., “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature Methods, vol. 18, no. 2, pp. 203–211, 2021.",
    ]
    for ref in refs:
        p.draw_para(ref, S_REF)
    p.save()
    print(OUT, "pages", p.page)


if __name__ == "__main__":
    main()
