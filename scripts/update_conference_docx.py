"""Sync the conference Word paper with the latest paper_draft_ko.md text."""
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

SRC = Path(
    r"C:\Users\a3426\OneDrive\Desktop"
    r"\연구_야호_TRIO_A_Tri-Scale_Hybrid_Framework_Combining_Size-Matched_Experts_and_Tailored_PPO_Refinement.docx"
)

FONT = "Times New Roman"


def set_run_font(run, *, size=10, bold=None, italic=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement_safe()
        rpr.append(rfonts)
    for key in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(key), FONT)


def OxmlElement_safe():
    from docx.oxml import OxmlElement
    return OxmlElement("w:rFonts")


def set_text(para, text, *, bold=None, italic=None, size=10):
    runs = para.runs
    if not runs:
        run = para.add_run(text)
        set_run_font(run, size=size, bold=bold, italic=italic)
        return
    first = runs[0]
    for extra in list(runs[1:]):
        extra._element.getparent().remove(extra._element)
    first.text = text
    set_run_font(first, size=size, bold=bold, italic=italic)


def main():
    doc = Document(str(SRC))
    p = doc.paragraphs

    set_text(
        p[11].runs[0]._parent if False else p[11],
        "Keywords  뇌종양 분할, 자기공명영상, 강화학습, PPO, 동적 라우팅, TRIO, 의료영상",
        bold=False,
        italic=True,
        size=10,
    )
    # Rebuild Keywords so the label stays bold.
    para = p[11]
    for extra in list(para.runs):
        extra._element.getparent().remove(extra._element)
    r0 = para.add_run("Keywords  ")
    set_run_font(r0, size=10, bold=True, italic=False)
    r1 = para.add_run("뇌종양 분할, 자기공명영상, 강화학습, PPO, 동적 라우팅, TRIO, 의료영상")
    set_run_font(r1, size=10, bold=False, italic=True)

    abstract = (
        "본 연구는 T1ce와 FLAIR MRI에서 종양 크기를 먼저 세 구간으로 분류하고, "
        "크기별 전문가 분할 모델로 초기 마스크를 만든 뒤 남은 경계를 PPO로 미세 조정하는 "
        "삼중 스케일 파이프라인 TRIO를 제안한다. 전문가가 전역 형태를 담당하고, "
        "크기마다 다른 잔여 경계 오차는 순차 보정으로 다루려 한 설계이다. "
        "소형은 CaraNet, 중형은 UNet++, 대형은 SegResNet을 배정한다. "
        "추론에서는 GT-free 면적 게이트와 STOP만 쓰고 단조 DSC 게이트·best-of-N은 쓰지 않는다. "
        "BraTS 2021 환자 단위 검증 42명(2,434 슬라이스)에서 Stage 2 DSC는 0.8948, HD95는 1.7336 px이다. "
        "Stage 3 배포형 수치는 STOP 재학습 후 동일 프로토콜로 보고한다."
    )
    set_text(doc.tables[1].cell(0, 0).paragraphs[1], abstract, size=10)

    set_text(
        p[14],
        "뇌종양의 정확한 영역 분할은 종양 부담의 정량화, 치료 반응 평가 및 방사선 치료 계획에 중요한 기반 정보를 제공한다. "
        "그러나 MRI에서 종양은 크기, 조영 양상, 경계의 선명도 및 주변 부종의 형태가 환자마다 달라 자동 분할 결과의 경계 부근에 "
        "미세한 오차가 남기 쉽다. U-Net·ViT 계열의 딥러닝 분할기는 전역 구조와 대략적인 영역을 잘 파악하지만[1], "
        "국소 경계에서는 왜곡과 불확실성이 남는다. 또한 하나의 모델로 작은 병변과 넓은 병변을 동시에 최적화하는 데에도 한계가 있다.",
    )
    set_text(
        p[15],
        "연구 초반에는 단일 딥러닝 백본으로 초기 마스크(rough mask)를 만든 뒤 PPO 에이전트로 경계를 보정하는 단순 파이프라인을 시도했다. "
        "이 설정에서는 소형 종양(면적 300 px 이하)에서 평균 DSC가 하락하고 HD95 오차가 커지는 현상이 관찰되었다. "
        "이에 본 연구는 입력이 들어올 때 종양 크기로 클래스를 나누고, 클래스마다 다른 전문가 모델로 rough mask를 생성한 뒤 "
        "크기별 PPO로 보정하는 TRIO를 제안한다. 고정된 형태학 연산과 달리, 강화학습 에이전트는 현재 마스크 상태와 영상 단서를 보고 "
        "보정 행동을 선택할 수 있다.",
    )
    set_text(
        p[17],
        "1. 종양 크기 분류 → 전문가 분할 → PPO 보정 → DSC·HD95 평가로 이어지는 4단계 동적 라우팅 파이프라인(TRIO)을 설계하였다.",
    )
    set_text(
        p[18],
        "2. 영상, 현재 마스크, 확률 맵 및 에지 정보를 활용하는 PPO 기반 경계 보정 환경을 구성하였다. "
        "소형은 확대·연속 행동, 중형은 전역·이산 행동, 대형은 HD95 억제에 가중치를 둔다.",
    )

    set_text(
        p[22],
        "U-Net은 인코더-디코더 구조와 skip connection을 통해 위치 정보와 의미 정보를 결합하여 의료영상 분할의 대표적 기반 모델로 자리 잡았다[1]. "
        "이후 UNet++는 중첩된 dense skip connection으로 객체의 경계 분할 정확도를 높였으며[2], "
        "SegResNet은 Residual Learning 블록을 Encoder-Decoder에 결합해 3차원 의료영상 분할의 표현력을 높였다[3]. "
        "CaraNet은 Axial Attention과 Reverse Attention을 결합하여 크기가 매우 작은 객체를 찾는 데 특화되어 있다[4].",
    )
    set_text(
        p[24],
        "BraTS 2021에서 Luu와 Park은 nnU-Net의 인코더를 비대칭으로 확장하고 GroupNorm과 axial attention을 도입하여 최종 테스트 1위를 기록했다[7]. "
        "Myronenko 등(NVAUTO)은 SegResNet에 Barlow Twins 기반 중복 감소 학습을 결합해 2위를 기록했으며, 1위와의 차이는 통계적으로 유의하지 않았다[8]. "
        "본 연구는 이 두 방법의 핵심 설계를 2차원 이진 분할 조건에 맞춰 재구현하여 비교 기준으로 삼는다.",
    )

    set_text(p[30], "그림 1. TRIO 전체 파이프라인. Stage 1 분류 → Stage 2 크기별 Expert → Stage 3 크기별 PPO → Stage 4 평가. 파랑=분류기, 보라·청록=백본, 초록=RL 에이전트, 실선=데이터 흐름, 점선=동적 라우팅이다.")

    set_text(
        p[36],
        "Stage 1은 입력 슬라이스의 종양 규모를 세 구간으로 나눈다. ImageNet으로 사전학습한 ResNet18 Shape Classifier가 "
        "T1ce+FLAIR(2채널)를 받아 Small / Medium / Large 로짓을 출력한다. 학습 레이블은 GT 마스크 면적으로 정의하며, "
        "구간은 Small <300 px, Medium 300–700 px, Large ≥700 px이다. 추론 시에는 분류기 예측으로 Stage 2·3 Expert/PPO를 선택하고, "
        "Expert·PPO 학습 시 클래스 필터는 GT 면적을 사용해 라벨 누출을 줄인다. 검증 정확도는 0.8512였으며, "
        "300/700 px 경계 부근의 모호한 슬라이스가 오분류의 주원인이다.",
    )
    set_text(
        p[40],
        "Small → CaraNet (2.5D): 인접 슬라이스(prev/center/next)를 채널로 붙여 작은 병변의 문맥을 보강한다. "
        "미세 파편은 학습 시 4배 오버샘플하고, 추론 시 zoom-crop으로 재추론한다.",
    )
    set_text(
        p[41],
        "Medium → UNet++: 중첩 dense skip으로 중형 병변의 경계·내부 채움을 안정적으로 학습한다.",
    )
    set_text(
        p[42],
        "Large → SegResNet: ED·TC 2채널을 예측한 뒤 Whole Tumor(WT)로 합친다. "
        "Large Expert는 크기 필터 없이 전 구간으로 학습하고, 추론 때만 Large 라우팅 슬라이스에 사용한다.",
    )
    set_text(
        p[43],
        "확률 맵은 클래스별 임계값 0.80 / 0.80 / 0.50(Small/Medium/Large)으로 이진화한다. "
        "Large는 과소분할 경향이 있어 임계값을 낮춘다. Small에서 높은 임계값으로 극소 종양이 사라질 수 있는 경우, "
        "면적 하한에 따라 임계값을 단계적으로 낮추는 micro-threshold ladder를 적용한다. "
        "이어서 클래스별 최소 연결요소 필터로 잔여 파편을 정리한다.",
    )

    set_text(p[45], "3.4 Stage 3: 맞춤형 PPO 경계 보정", bold=True)
    set_text(
        p[46],
        "Stage 3는 Stage 2 rough mask를 상태로 두고, 크기별 PPO 에이전트가 경계를 반복 수정한다. "
        "세 에이전트는 관측·행동·보상의 스케일만 달리 하고, 마스크를 갱신하는 기하 연산은 공유한다. "
        "관측에는 정답 마스크가 포함되지 않으며, GT는 학습 보상 산출에만 쓰인다.",
    )
    set_text(
        p[47],
        "공통 갱신. 현재 마스크에서 가장 큰 연결요소의 질량 중심을 잡고, 그 중심 기준 각도로 8개 방위 섹터를 나눈다. "
        "마스크의 signed distance field(SDF; 내부 양수, 외부 음수)에 섹터별 이동량을 더한 뒤 SDF + shift ≥ 0인 화소를 새 마스크로 삼아, "
        "형태학 팽창·침식보다 작은 서브픽셀 이동을 근사한다. 면적이 20 px를 넘으면 closing 후 opening으로 위상을 정리하고, "
        "초기 rough 주변 ±8 px 밖으로 나가지 못하게 클립한다. 보상은 전체 DSC 변화량, 정답 경계 ±3 px 대역의 DSC 변화량, "
        "HD95 감소량, 행동 비용으로 구성한다. 지표가 나빠지면 개선분의 2배를 감점하고, 에피소드 시작 DSC보다 하락하면 추가 패널티(−5.0)를 준다. "
        "종양 면적에 따른 배율은 size_scale = clip(300 / GT면적, 0.5, 3.0)이다. 학습 에피소드는 최대 30스텝이며, "
        "목표 DSC를 1.0으로 두어 조기 종료는 사실상 끈다.",
    )
    set_text(
        p[48],
        "Small (확대 + 연속 행동). 소형 병변은 전체 128×128에서 수 픽셀에 불과하므로, 가장 큰 연결요소 중심으로 64×64를 잘라 확대해 본다. "
        "상태는 T1ce(첫 채널)·현재 마스크·확률 맵·Sobel 에지의 4채널이다. "
        "정책은 8차원 연속 이동 Box(−2, +2)와 STOP 채널을 샘플링한다. 절대값 0.1 이하는 방위 유지(Keep)이며 Keep은 에피소드를 끝내지 않는다. "
        "보상은 HD95 가중 ×0.2, size_scale 1.0–3.0이며, 목표 DSC 보너스(+50)는 STOP을 고를 때만 지급한다. "
        "평가 시 마스크 면적이 35 px 미만이면 수축을 0으로 자른다.",
    )
    set_text(
        p[50],
        "Medium (전역 + 이산 행동). 상태는 T1ce·현재 마스크·확률 맵의 3채널이다. "
        "행동 공간은 MultiDiscrete 5×8(+STOP)이며 각 방위는 강수축·약수축·유지·약팽창·강팽창(−1.0/−0.4/0/+0.4/+1.0 px)이다. "
        "HD95 가중 ×0.1, size_scale 0.5–1.0이다. 목표 DSC 보너스는 STOP 시에만 지급한다.",
    )
    set_text(
        p[52],
        "Large (전역 + HD95 강화). 상태·섹터 행동은 Medium과 같고 HD95 가중만 ×0.5로 올린다. "
        "size_scale은 0.5로 고정된다. 목표 DSC 보너스는 STOP 시에만 지급한다.",
    )
    set_text(
        p[53],
        "학습·추론. 학습 데이터는 실제 Expert 예측 50%와 형태학 노이즈 rough 50%를 섞고, 클래스 필터는 GT 면적만 사용한다. "
        "PPO는 크기별 약 300,000 스텝(병렬 환경 8, 롤아웃 1,024, 학습률 1×10⁻⁴, clip 0.2)이며 학습 조기종료(Stop Motion)는 끈다. "
        "추론에서는 최대 15스텝 또는 STOP까지 돌린 마지막 마스크를 쓰고, GT로 스텝을 고르거나 단조 게이트로 되돌리지 않는다. "
        "배포 안전 장치는 면적 게이트(0.2×–4×)뿐이다.",
        size=10,
    )

    set_text(
        p[54],
        "그림 2. Small 클래스 PPO 경계 보정 루프. 에이전트는 마스크 중심을 기준으로 확대한 4채널 64×64 상태"
        "(영상·현재 마스크·확률 맵·Sobel 에지)를 관측하고, Gaussian 정책이 [−2, +2] 구간의 8차원 연속 행동"
        "(방위별 픽셀 이동 거리)을 출력한다. 보상은 HD95 가중(×0.2), 소형 병변용 size_scale(1.0–3.0), "
        "DSC ≥0.85 보너스로 구성되며, 학습 에피소드는 최대 30스텝이다.",
        size=10,
        italic=True,
    )
    set_text(
        p[58],
        "그림 3. Medium 클래스 PPO 경계 보정 루프. 상태는 전체 슬라이스의 3채널 128×128(영상·현재 마스크·확률 맵)이며, "
        "Categorical 정책 8개가 MultiDiscrete 5×8 행동을 출력한다. 각 방위는 강수축·약수축·유지·약팽창·강팽창"
        "(−1.0 / −0.4 / 0 / +0.4 / +1.0 px) 중 하나를 선택한다. 보상은 HD95 가중(×0.1), size_scale(0.5–1.0), "
        "DSC ≥0.95 보너스로 구성되며, 학습 에피소드는 최대 30스텝이다.",
        size=10,
        italic=True,
    )
    set_text(
        p[61],
        "그림 4. Large 클래스 PPO 경계 보정 루프. 상태·행동 공간은 Medium과 같다(3채널 128×128, MultiDiscrete 5×8). "
        "보상만 대형 병변에 맞추어 HD95 가중을 ×0.5로 5배 강화하고 size_scale을 0.5로 고정하며, "
        "DSC ≥0.95이면 보너스를 준다. 학습 에피소드는 최대 30스텝이다.",
        size=10,
        italic=True,
    )

    set_text(p[68], "4. 실험 및 결과", bold=True)
    set_text(
        p[83],
        "표 3. TRIO Stage 2와 BraTS 2021 상위 입상 방법(KAIST, NVAUTO)의 2차원 각색 비교. "
        "TRIO 열은 Expert 초기 분할이며, 과거 GT 단조 게이트 Final(0.9031)은 쓰지 않는다.",
        italic=True,
        size=10,
    )
    set_text(
        p[86],
        "그림 5. Stage 2 초기 분할(Rough)과 Stage 3 PPO 보정(RL Refined)의 정성 비교. "
        "열은 크기 클래스별 대표 슬라이스 2장씩(Small: CaraNet, Medium: UNet++, Large: SegResNet)이며, "
        "행은 위부터 MRI+정답(초록 윤곽), Stage 2 Rough(빨간 반투명 마스크), Stage 3 RL(하늘색 윤곽)이다. "
        "각 칸 하단의 Rough DSC·RL DSC·Δ는 해당 슬라이스의 겹침 지표와 보정 전후 변화량이다.",
        italic=True,
        size=10,
    )
    set_text(
        p[88],
        "그림 6. 클래스 평균 DSC 부근의 대표 슬라이스에서 TRIO·KAIST·NVAUTO를 같은 장으로 비교한다"
        "(제목: Representative slices near class-mean DSC · dashed green = GT). "
        "행은 방법(위→아래: TRIO 청록, KAIST 파랑, NVAUTO 주황), 열은 Small 1–2 · Medium 1–2 · Large 1–2이며, "
        "초록 점선은 정답 윤곽, 칸 하단은 해당 슬라이스 DSC이다. Small·Medium에서는 TRIO가 정답 윤곽에 더 밀착하고, "
        "Large에서는 세 방법의 차이가 작다.",
        italic=True,
        size=10,
    )
    set_text(
        p[90],
        "그림 7. RL 보정 후 마스크만 비교한 3×4 격자. 행은 Small/Medium/Large, 열은 TRIO·UNet·UNet++·SegResNet이다. "
        "행마다 동일 슬라이스를 쓰고, 각 칸 DSC가 표 4의 클래스 평균에 가깝도록 골랐다(청록=예측, 초록 점선=GT). "
        "TRIO가 전 구간에서 경계가 GT에 더 가깝고, Small에서 단일 백본과의 차이가 가장 잘 드러난다.",
        italic=True,
        size=10,
    )

    set_text(
        p[98],
        "본 연구는 뇌종양 MRI 분할을 위해 크기 기반 동적 라우팅과 PPO 기반 경계 보정을 결합한 TRIO를 제안했다. "
        "종양 크기 분류(Stage 1) → 크기별 전문가 분할(Stage 2) → 맞춤형 PPO 보정(Stage 3, STOP) → 면적 게이트·DSC·HD95 평가(Stage 4)로 구성된다. "
        "BraTS 2021 환자 단위 검증에서 Stage 2 DSC는 0.8948(HD95 1.7336 px)이며, "
        "같은 조건의 2차원 각색 베이스라인과 경쟁한다. Stage 3 배포형 수치는 단조 게이트·best-of-N 없이 재측정하여 보고한다.",
    )
    set_text(
        p[102],
        "향후에는 첫째, 정답 마스크 없이 사용할 수 있는 불확실성 기반 안전 게이트를 설계해야 한다. "
        "둘째, SOTA 대비 격차를 더 줄이기 위해 PPO 정책을 고도화하고, 2차원 슬라이스가 아닌 3차원 볼륨(voxel) 단위의 정책으로 확장해 "
        "슬라이스 간 일관성을 확보할 필요가 있다. 셋째, 소형 병변의 과소분할과 분류기 오분류를 줄이는 방향으로 라우팅을 재설계하고, "
        "기관 외부 데이터에서 일반화 성능을 검증해야 한다.",
    )

    doc.save(str(SRC))
    print("saved", SRC)


if __name__ == "__main__":
    main()
