from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "RL_Refiner_논문_장문초안.docx"
FIGURE_1 = ROOT / "results" / "pipeline_sample_comparison.png"
FIGURE_2 = ROOT / "results" / "subregion_pipeline_comparison.png"

FONT = "Malgun Gothic"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
GRAY = "666666"
BLACK = "000000"


def set_font(run, size=None, bold=None, color=None, italic=None):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:ascii"), FONT)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    if italic is not None:
        run.italic = italic


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_width(cell, width_dxa):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table_pr = table._tbl.tblPr
    tbl_w = table_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        table_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = OxmlElement("w:tblInd")
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    table_pr.append(tbl_ind)
    tbl_layout = table_pr.first_child_found_in("w:tblLayout")
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        table_pr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            set_cell_width(cell, width)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            tc_pr = cell._tc.get_or_add_tcPr()
            margins = tc_pr.first_child_found_in("w:tcMar")
            if margins is None:
                margins = OxmlElement("w:tcMar")
                tc_pr.append(margins)
            for side, value in (("top", "80"), ("bottom", "80"), ("start", "120"), ("end", "120")):
                elem = margins.find(qn(f"w:{side}"))
                if elem is None:
                    elem = OxmlElement(f"w:{side}")
                    margins.append(elem)
                elem.set(qn("w:w"), value)
                elem.set(qn("w:type"), "dxa")


def add_page_field(paragraph):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)
    set_font(run, size=9, color=GRAY)


def apply_style(style, font_size, color, before, after, line_spacing, bold=False, alignment=None):
    style.font.name = FONT
    style._element.rPr.rFonts.set(qn("w:ascii"), FONT)
    style._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    style.font.size = Pt(font_size)
    style.font.color.rgb = RGBColor.from_string(color)
    style.font.bold = bold
    pf = style.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line_spacing
    if alignment is not None:
        pf.alignment = alignment


def add_paragraph(doc, text, style=None, first_line=True, keep_with_next=False):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.keep_with_next = keep_with_next
    if first_line:
        p.paragraph_format.first_line_indent = Inches(0.25)
    r = p.add_run(text)
    set_font(r, size=11, color=BLACK)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_font(r, size={1: 16, 2: 13, 3: 12}[level], bold=True, color={1: BLUE, 2: BLUE, 3: DARK_BLUE}[level])
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    set_font(r, size=9.5, italic=True, color=GRAY)
    return p


def add_reference(doc, number, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.28)
    p.paragraph_format.first_line_indent = Inches(-0.28)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(f"[{number}] {text}")
    set_font(r, size=10.5, color=BLACK)


def add_result_table(doc):
    add_caption(doc, "표 1. 기록된 동적 라우팅 파이프라인의 클래스별 보정 전후 성능")
    table = doc.add_table(rows=1, cols=6)
    table.style = "Table Grid"
    widths = [1400, 1500, 900, 1350, 1350, 2860]
    set_table_geometry(table, widths)
    headers = ["크기 클래스", "전문가 모델", "선택 슬라이스", "초기 DSC", "최종 DSC", "HD95 (px), 초기 -> 최종"]
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(text)
        set_font(r, size=8.5, bold=True, color=BLACK)
        set_cell_shading(cell, "F4F6F9")
    rows = [
        ("소형 (<300 px)", "CaraNet", "100", "0.8233", "0.8234", "2.4205 -> 2.4205"),
        ("중형 (300-700 px)", "UNet++", "100", "0.9327", "0.9328", "0.6717 -> 0.6622"),
        ("대형 (>=700 px)", "SegResNet", "100", "0.9530", "0.9530", "0.4795 -> 0.4795"),
    ]
    for values in rows:
        cells = table.add_row().cells
        for index, (cell, text) in enumerate(zip(cells, values)):
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if index >= 2 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text)
            set_font(r, size=9.2, color=BLACK)
    doc.add_paragraph()
    note = doc.add_paragraph()
    note.paragraph_format.space_before = Pt(4)
    note.paragraph_format.space_after = Pt(8)
    note.paragraph_format.line_spacing = 1.15
    r = note.add_run("주: 구현의 기본 설정은 클래스당 최대 100개 슬라이스를 선택한다. 따라서 표의 클래스별 수치는 총 300개 선택 슬라이스에 대한 결과이며, 입력으로 읽힌 20명 환자의 1,171개 유효 슬라이스 전체에 대한 무작위 독립 검증 결과로 해석해서는 안 된다.")
    set_font(r, size=9.3, color=GRAY, italic=True)


def make_document():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    normal = doc.styles["Normal"]
    apply_style(normal, 11, BLACK, 0, 8, 1.333, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY)
    apply_style(doc.styles["Heading 1"], 16, BLUE, 18, 10, 1.0, bold=True)
    apply_style(doc.styles["Heading 2"], 13, BLUE, 12, 6, 1.0, bold=True)
    apply_style(doc.styles["Heading 3"], 12, DARK_BLUE, 8, 4, 1.0, bold=True)

    header = section.header
    header_p = header.paragraphs[0]
    header_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header_p.paragraph_format.space_after = Pt(0)
    run = header_p.add_run("2026 연합학술제 연구논문 초안 | RL-Refiner")
    set_font(run, size=9, color=GRAY)

    footer = section.footer
    footer_p = footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_p.paragraph_format.space_before = Pt(0)
    run = footer_p.add_run("- ")
    set_font(run, size=9, color=GRAY)
    add_page_field(footer_p)
    run = footer_p.add_run(" -")
    set_font(run, size=9, color=GRAY)

    # Title block: editorial_cover pattern, intentionally restrained for a research manuscript.
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(28)
    p.paragraph_format.space_after = Pt(9)
    r = p.add_run("강화학습 기반 동적 라우팅을 이용한\n뇌종양 MRI 분할 경계 보정 연구")
    set_font(r, size=22, bold=True, color=BLACK)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("RL-Refiner: Size-aware Expert Segmentation and PPO-based Boundary Refinement")
    set_font(r, size=11, italic=True, color=GRAY)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(26)
    r = p.add_run("작성자: [성명 입력]    |    소속: [학과/학교 입력]    |    제출일: 2026년 8월")
    set_font(r, size=10.5, color=BLACK)

    add_heading(doc, "초록", 1)
    abstract = (
        "뇌종양 자기공명영상(Magnetic Resonance Imaging, MRI) 분할은 종양의 위치와 범위를 정량화하고 치료 계획을 보조하는 핵심 기술이다. "
        "그러나 병변의 크기와 경계 선명도, 주변 조직의 신호 특성이 환자별로 다르기 때문에 하나의 분할 모델만으로 모든 병변에서 균일한 성능을 얻기 어렵다. "
        "본 연구는 T1ce와 FLAIR의 2채널 MRI를 입력으로 사용하고, 병변 크기에 따라 전문가 분할 모델을 선택한 뒤 강화학습 기반 보정기를 적용하는 3단계 파이프라인을 제안한다. "
        "소형, 중형, 대형 병변에는 각각 CaraNet, UNet++, SegResNet을 배정하고, Proximal Policy Optimization(PPO) 에이전트가 현재 마스크, 영상, 확률 맵 및 에지 단서를 관찰하여 8개 방위의 경계를 미세 조정한다. "
        "구현에 기록된 실험에서 클래스당 최대 100개씩 선택된 300개 슬라이스는 보정 전후 DSC 0.8233에서 0.8234, 0.9327에서 0.9328, 0.9530에서 0.9530을 각각 보였다. 중형 병변에서는 HD95가 0.6717 px에서 0.6622 px로 감소했다. "
        "다만 현재 평가 구현은 정답 마스크의 면적으로 전문가 모델을 선택하고, 후보 임계값 및 안전 게이트의 일부 판단에도 정답 마스크를 사용한다. 따라서 본 결과는 완전한 임상 추론 성능이 아니라 oracle-aided 조건에서의 탐색적 상한 성능으로 해석해야 한다. "
        "본 초안은 현재 코드와 실험 기록을 근거로 연구의 구조, 관찰된 성능, 재현성 위험 및 향후 검증 계획을 함께 제시한다."
    )
    add_paragraph(doc, abstract, first_line=False)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(15)
    r = p.add_run("주요어: 뇌종양 분할, 의료영상, 자기공명영상, 강화학습, PPO, 동적 라우팅, 경계 보정")
    set_font(r, size=10.5, bold=True, color=BLACK)

    add_heading(doc, "1. 서론", 1)
    add_paragraph(doc, "뇌종양은 병변의 위치, 침윤 양상, 크기 및 주변 부종의 범위가 환자마다 크게 다르다. MRI는 연조직 대비가 우수하고 여러 영상 시퀀스를 함께 사용할 수 있어 뇌종양 평가에 널리 활용되지만, 사람이 모든 슬라이스에서 종양 경계를 직접 표시하는 작업은 시간이 오래 걸리고 관찰자 간 차이도 발생할 수 있다. 특히 치료 계획이나 경과 관찰을 위해 종양 부피와 경계 변화를 반복적으로 측정해야 하는 상황에서는 자동 또는 반자동 분할 기술의 필요성이 커진다.")
    add_paragraph(doc, "합성곱 신경망 기반 분할 모델은 영상에서 종양의 대략적인 위치와 형태를 추정하는 데 높은 성능을 보인다. 대표적으로 U-Net은 인코더-디코더 구조와 skip connection을 통해 의미 정보와 세부 위치 정보를 결합하며, 다양한 의료영상 분할 문제의 강력한 기준선이 되었다[1]. 이후 UNet++와 같은 확장 구조는 다단계 특징 융합을 강화했고[2], 잔차 연결을 활용하는 SegResNet은 의료영상의 복잡한 공간적 특징을 안정적으로 학습하는 데 사용되었다[3]. 그럼에도 병변의 크기가 매우 작거나 경계가 불명확한 경우, 초기 예측 마스크에는 누락, 과분할, 경계 울퉁불퉁함과 같은 오류가 남을 수 있다.")
    add_paragraph(doc, "이 문제에 대한 단순한 접근은 모든 예측 마스크에 동일한 침식, 팽창, 열기 또는 닫기 연산을 적용하는 것이다. 그러나 고정된 형태학 연산은 병변의 실제 경계, 현재 예측의 불확실성, 병변 크기를 충분히 반영하지 못한다. 한 슬라이스에 유리한 연산이 다른 슬라이스에서는 종양을 과도하게 줄이거나 주변 조직까지 포함할 수 있다. 따라서 후처리는 하나의 고정 규칙보다, 현재 상태와 영상 단서를 고려해 수정 강도와 방향을 선택하는 순차적 의사결정 문제로 볼 수 있다.")
    add_paragraph(doc, "본 연구는 이러한 관점에서 RL-Refiner를 구성하였다. 제안 시스템은 먼저 병변 크기에 맞는 전문가 분할 모델을 통해 초기 확률 맵과 마스크를 생성하고, 이후 PPO 정책이 경계 주변을 단계적으로 보정한다. 소형 병변에는 작은 객체의 문맥과 역방향 주의 기제를 활용하는 CaraNet을, 중형 병변에는 중첩 skip connection을 갖는 UNet++를, 대형 병변에는 잔차 기반 SegResNet을 활용한다. 보정기는 영상, 현재 마스크와 확률 정보를 상태로 받아 8개 방위의 경계를 수축, 유지 또는 팽창시키는 행동을 출력한다.")
    add_paragraph(doc, "다만 연구 결과를 해석할 때에는 구현과 평가 절차의 경계를 분명히 해야 한다. 현재 평가 스크립트는 정답 마스크의 면적을 이용해 크기 클래스를 산출하여 전문가 모델을 선택한다. 또한 중형 및 대형 컴포넌트의 임계값과 형태학 후보를 정답 마스크 기준 DSC로 선택하고, 최종 결과가 초기 결과보다 나빠지면 정답 마스크 기준으로 되돌리는 안전 게이트를 사용한다. 이 절차는 보정 메커니즘을 분석하는 데는 유용할 수 있으나, 정답이 없는 실제 배포 환경을 재현하지는 못한다. 본 논문은 이러한 한계를 숨기지 않고, 현 단계의 결과를 oracle-aided 탐색 실험으로 정리한다.")
    add_paragraph(doc, "본 연구의 목적은 세 가지이다. 첫째, 병변 크기에 따른 전문가 모델 선택이 초기 분할의 적합성을 높일 수 있는지 확인한다. 둘째, 강화학습 기반의 방향별 경계 보정이 초기 성능을 크게 훼손하지 않으면서 HD95와 같은 경계 지표를 개선할 수 있는지 검토한다. 셋째, 임상 적용을 위해 반드시 제거하거나 대체해야 할 정답 의존 요소를 식별하고, 재현 가능한 후속 실험 설계를 제안한다.")

    add_heading(doc, "2. 관련 연구 분석", 1)
    add_heading(doc, "2.1 딥러닝 기반 의료영상 분할", 2)
    add_paragraph(doc, "U-Net은 수축 경로에서 문맥 정보를 추출하고 확장 경로에서 공간 해상도를 복원하며, 같은 단계의 특징을 skip connection으로 연결한다[1]. 이 구조는 작은 데이터셋에서도 데이터 증강과 함께 학습할 수 있고, 경계 정보를 보존하는 데 유리해 의료영상 분할의 표준 구조로 널리 채택되었다. 그러나 일반 U-Net의 단일 해상도 특징 결합만으로는 서로 다른 크기와 모양의 병변을 동시에 처리하는 데 제약이 있을 수 있다.")
    add_paragraph(doc, "UNet++는 인코더와 디코더 사이의 skip connection을 중첩된 경로로 재설계해 저수준과 고수준 특징의 의미적 간극을 줄이고자 했다[2]. 복수 해상도의 특징을 점진적으로 정제할 수 있어 복잡한 경계에서 이점을 보일 수 있다. SegResNet은 잔차 블록과 정규화를 기반으로 깊은 네트워크의 학습 안정성을 높이며, BraTS 계열 과제에서 널리 활용된 구조이다[3]. 본 연구는 이 두 모델을 병변 크기에 따라 서로 다른 전문가로 사용했다.")
    add_heading(doc, "2.2 작은 객체 분할과 크기 적응형 모델 선택", 2)
    add_paragraph(doc, "작은 병변은 전체 영상에서 차지하는 비율이 작아 배경 클래스의 영향이 상대적으로 크고, 몇 픽셀의 오류가 DSC와 경계 거리 지표에 큰 영향을 줄 수 있다. CaraNet은 작은 의료 객체의 문맥 정보와 세부 특징을 강화하기 위해 context axial reverse attention과 channel-wise feature pyramid를 도입한 구조다[4]. 본 프로젝트에서 소형 클래스에 CaraNet을 연결한 이유는 전체 영상을 단일한 방식으로 처리하기보다 작은 객체에 특화된 특징 추출기를 사용하려는 데 있다.")
    add_paragraph(doc, "크기 적응형 라우팅은 mixture-of-experts와 유사한 관점을 의료영상 분할에 적용한 것이다. 모든 입력을 같은 모델에 전달하는 대신, 분류기가 입력 특성에 따라 모델을 선택하면 각 전문가가 자신이 강점을 가지는 데이터 부분에 집중할 수 있다. 그러나 이러한 구조의 유효성을 입증하려면 분류기 자체의 정확도, 잘못된 라우팅에 대한 강건성, 전문가 선택 비용 및 단일 모델 대비 순이득을 별도로 측정해야 한다. 현재 구현의 oracle routing은 이 가운데 전문가의 이상적 선택 효과만 관찰하는 조건이므로, 실제 Shape Classifier 성능을 포함한 end-to-end 평가는 후속 과제로 남는다.")
    add_heading(doc, "2.3 강화학습 기반 경계 보정", 2)
    add_paragraph(doc, "강화학습은 상태를 관찰한 에이전트가 행동을 수행하고 보상을 통해 정책을 개선하는 틀이다. PPO는 정책 변화의 폭을 제한하는 목적함수를 사용해 비교적 안정적인 학습을 지원하는 대표적인 on-policy 알고리즘이다[5]. 분할 경계 보정에서는 현재 마스크와 영상의 차이를 상태로 보고, 지역적 팽창 또는 수축을 행동으로 정의할 수 있다. 보상은 예측과 정답의 일치도 변화량, 경계 품질 변화, 행동 비용 등으로 구성할 수 있다.")
    add_paragraph(doc, "강화학습 후처리의 장점은 하나의 정적 필터가 아니라, 샘플별로 다른 조정 행동을 선택할 수 있다는 점이다. 반면 보상 함수가 정답 마스크에 의존하므로 학습과 평가의 구분이 중요하다. 학습에서는 정답을 사용해 보상을 계산하는 것이 가능하지만, 시험과 배포 단계에서는 정답 없이 정책을 실행해야 한다. 따라서 시험 단계의 후보 선택, 모델 라우팅, 안전 게이트가 정답 정보를 사용하면 평가 낙관 편향이 발생할 수 있다. 본 연구의 분석은 이 원칙을 기준으로 현재 구현을 검토한다.")
    add_heading(doc, "2.4 BraTS 데이터셋과 평가 지표", 2)
    add_paragraph(doc, "BraTS 2021은 다기관 다중 파라미터 MRI와 종양 하위영역 레이블을 제공하는 공개 벤치마크다[6]. 프로젝트 데이터 로더는 T1ce와 FLAIR를 결합해 2채널 입력을 구성하며, 레이블에서 0이 아닌 모든 값을 하나의 이진 Whole Tumor 마스크로 변환한다. 따라서 현재 파이프라인의 결과는 enhancing tumor, tumor core, whole tumor 하위영역을 각각 따로 예측한 결과가 아니라, 2차원 슬라이스 수준의 이진 종양 마스크 성능으로 이해해야 한다.")
    add_paragraph(doc, "본 연구에서는 영역 중첩도를 나타내는 Dice Similarity Coefficient(DSC)와 경계 오차를 나타내는 95% Hausdorff distance(HD95)를 사용한다. DSC는 예측 마스크 P와 정답 마스크 G에 대해 DSC = 2|P intersect G| / (|P| + |G|)로 정의되며 1에 가까울수록 좋다. HD95는 두 경계점 집합의 거리 분포에서 극단값의 영향을 줄이기 위해 95번째 백분위수를 사용한 거리 지표로, 낮을수록 경계가 가깝다는 의미다. 현재 구현에서 HD95는 리사이즈된 128 x 128 슬라이스의 픽셀 단위로 계산되므로, 물리적 mm 단위의 임상 거리로 직접 해석할 수 없다.")

    add_heading(doc, "3. 연구 방법론", 1)
    add_heading(doc, "3.1 데이터 전처리와 분석 단위", 2)
    add_paragraph(doc, "입력 데이터는 BraTS 형식의 NIfTI 영상이며, 각 환자 폴더에서 T1ce와 FLAIR 시퀀스를 읽는다. 각 모달리티는 뇌 영역 내 강도 분포를 기준으로 정규화한 뒤 0과 1 사이로 조정되며, 128 x 128 해상도의 2차원 슬라이스로 변환된다. 데이터 로더는 종양 면적 비율이 0.002 이상인 슬라이스만 유효 슬라이스로 선택한다. 이러한 선택은 배경만 존재하는 슬라이스가 지표 평균을 과도하게 높이는 문제를 줄이지만, 전체 MRI 볼륨을 대상으로 한 실제 사용 시나리오와는 구분되어야 한다.")
    add_paragraph(doc, "현 구현의 레이블은 0이 아닌 모든 BraTS 레이블을 종양으로 처리한다. 따라서 본문에서 사용하는 종양 크기는 환자 수준의 3차원 체적이 아니라, 리사이즈된 2차원 이진 마스크의 픽셀 면적을 뜻한다. 소형은 300 px 미만, 중형은 300 px 이상 700 px 미만, 대형은 700 px 이상으로 나뉜다. 이 임계값은 모델과 정책을 선택하기 위한 구현상 구간이며, 임상적 종양 크기 분류와 동등하지 않다.")
    add_heading(doc, "3.2 3단계 동적 라우팅 구조", 2)
    add_paragraph(doc, "첫 번째 단계는 Shape Classifier를 이용한 병변 크기 분류다. 설계 의도상 ResNet 기반 분류기가 T1ce + FLAIR 입력을 받아 소형, 중형, 대형 클래스를 예측하고, 이 예측값이 다음 단계의 전문가 선택에 사용된다. 두 번째 단계에서는 소형 클래스에 CaraNet, 중형 클래스에 UNet++, 대형 클래스에 SegResNet을 배정한다. 각 모델은 하나의 확률 맵을 출력하고, 임계값 처리로 초기 이진 마스크를 만든다. 이와 같은 구조는 하나의 백본이 모든 크기의 병변을 처리할 때 발생할 수 있는 표현력 불균형을 줄이려는 시도다.")
    add_paragraph(doc, "하지만 현재 평가 스크립트는 Shape Classifier의 예측을 사용하지 않는다. 각 슬라이스의 정답 마스크 면적을 계산해 true_c를 산출한 뒤, 이를 pipeline 함수의 true_class_preds 인자로 전달한다. 즉 실험에서 관찰된 라우팅은 classifier-driven routing이 아니라 ground-truth area-driven oracle routing이다. 따라서 현 시점에서 주장 가능한 것은 '정답 크기 클래스가 주어졌을 때의 전문가 조합 성능'이며, '입력 영상만으로 자동으로 최적 전문가를 선택한다'는 주장은 독립적인 분류기 평가가 완료된 뒤에만 가능하다.")
    add_heading(doc, "3.3 강화학습 보정 환경", 2)
    add_paragraph(doc, "MaskRefinementEnv는 Gymnasium 인터페이스로 구현되었다. 중형과 대형 병변에서 관측은 MRI 영상, 현재 이진 마스크, 분할 확률 맵으로 구성된 3채널 텐서다. 소형 병변에서는 현재 마스크의 가장 큰 연결요소 중심을 기준으로 64 x 64 관심영역을 잘라내고, 영상, 마스크, 확률 맵, Sobel 에지 맵을 포함한 4채널 관측을 사용한다. 이 확대 관찰은 소형 병변의 경계 변화를 더 세밀하게 반영하기 위한 장치다.")
    add_paragraph(doc, "행동은 현재 마스크의 중심을 기준으로 나눈 8개 각도 섹터에 대해 정의된다. 중형과 대형 정책은 각 섹터마다 강수축, 약수축, 유지, 약팽창, 강팽창 중 하나를 선택하는 MultiDiscrete 행동을 사용한다. 구현상 이산 행동은 -1.0, -0.4, 0.0, 0.4, 1.0 픽셀의 SDF 이동값으로 매핑된다. 소형 정책은 각 섹터에 대해 -2.0에서 2.0 사이의 연속 이동값을 출력한다. 이 방식은 단순한 1픽셀 형태학 연산보다 작은 경계 이동을 표현하기 위한 근사다.")
    add_paragraph(doc, "마스크의 갱신에는 signed distance field를 사용한다. 현재 마스크 내부에서는 양수, 외부에서는 음수가 되도록 거리장을 계산한 뒤, 섹터별 행동으로 만든 shift map을 더하고 0을 기준으로 다시 이진화한다. 그 후 작은 구조요소를 이용한 닫기와 열기 연산으로 위상적 잡음을 줄이고, 초기 마스크를 침식 및 팽창한 범위 안에서만 수정이 발생하도록 제약한다. 이 탐색 범위 제약은 에이전트가 초기 예측에서 지나치게 멀어지는 것을 방지하는 보수적 장치다.")
    add_heading(doc, "3.4 보상 함수와 종료 조건", 2)
    add_paragraph(doc, "보상은 전체 DSC 변화량, 경계 대역에서의 DSC 변화량, HD95 감소량, 행동 비용 및 목표 성능 보너스를 조합한다. 새로운 마스크의 DSC가 직전 상태보다 낮아지면 감소폭에 더 큰 가중치를 주고, 에피소드 시작 시점의 DSC보다 낮아지는 행동에는 추가 패널티를 부과한다. 소형 병변의 경우 DSC 0.85 이상, 중형과 대형 병변은 DSC 0.95 이상에 도달하면 보너스를 제공하며, 최대 단계 수에 도달하면 에피소드가 종료된다. 이러한 설계는 작은 객체에서의 과도한 축소와 큰 객체에서의 경계 과확장을 모두 줄이려는 목적을 가진다.")
    add_paragraph(doc, "보상 계산에 정답 마스크를 사용한다는 사실 자체는 지도형 시뮬레이터에서 PPO를 훈련하는 일반적 방식과 양립할 수 있다. 문제는 평가 단계에서도 정답을 이용해 후보를 선택하거나 결과를 되돌리는 경우다. 본 프로젝트의 후속 실험은 정책 학습 보상에는 정답을 사용할 수 있지만, 평가 단계에서는 고정된 임계값, 검증 세트에서 사전 결정한 규칙, 혹은 예측 확률과 영상 경계만으로 계산한 신뢰도 지표만 사용하는 방식으로 분리되어야 한다.")
    add_heading(doc, "3.5 컴포넌트별 보정 및 테스트 시 증강", 2)
    add_paragraph(doc, "평가 스크립트는 초기 마스크의 연결요소를 분리하고, 면적이 5 px 이상인 각 컴포넌트를 독립적으로 보정한 뒤 결과를 병합한다. 컴포넌트 면적에 따라 다시 소형, 중형, 대형 정책을 배정하는 component-level routing도 수행한다. 또한 수평 및 수직 반전 입력의 확률 맵을 원래 확률 맵과 평균내는 test-time augmentation(TTA)을 적용한다. 분리된 작은 파편이 중심점 계산 문제로 소실되는 현상을 줄이기 위해 가장 큰 연결요소의 중심을 기준점으로 사용한다.")
    add_paragraph(doc, "그러나 중형과 대형 컴포넌트에 대한 현재 코드는 여러 임계값과 형태학 후보를 만든 뒤 정답 마스크와의 DSC가 가장 높은 후보를 선택한다. 이는 TTA 자체의 효과가 아니라, 정답을 아는 상태에서 후보 중 최선의 결과를 선택한 효과가 함께 반영된다는 뜻이다. 실제 추론에서는 이러한 선택이 불가능하므로, 후보별 평균 확률, 예측 불확실성, 영상 에지 정렬도 또는 검증 세트에서 고정한 단일 임계값으로 대체해야 한다.")
    add_heading(doc, "3.6 안전 게이트의 역할과 한계", 2)
    add_paragraph(doc, "최종 단계의 dual monotonic safety gate는 보정 결과의 DSC가 초기 마스크보다 낮거나 HD95가 높아지면 초기 마스크로 복원한다. 이 장치는 오프라인 분석에서 보정기가 성능을 악화시키는 빈도를 측정하거나, 초기 분할을 보존하는 상한을 구하는 데 유용하다. 하지만 DSC와 HD95 모두 정답 마스크가 있어야 계산할 수 있으므로, 실제 환자 영상에 적용하는 안전 장치가 될 수 없다. 논문에서 '성능 하락 0%' 또는 '단조 증가 보장'을 표현할 때에는 반드시 '정답 기반 오프라인 안전 게이트가 적용된 평가 조건'이라는 한정을 붙여야 한다.")

    add_heading(doc, "4. 실험 및 결과", 1)
    add_heading(doc, "4.1 실험 설정과 보고 범위", 2)
    add_paragraph(doc, "프로젝트 기록에 따르면 BraTS 2021 Task 1 데이터와 T1ce + FLAIR 2채널 입력을 사용했으며, 20명 환자에서 추출된 1,171개 유효 슬라이스를 읽도록 구성했다. 한편 evaluate_pipeline.py의 기본 인자는 max_samples_per_class = 100으로, 각 oracle 크기 클래스에서 최대 100개 슬라이스만 성능 목록에 추가한다. 따라서 코드가 출력하는 'Total Slices Evaluated: 1,171'은 데이터 로더가 순회한 후보 슬라이스 수를 의미하지만, 평균 DSC와 HD95는 최대 300개 선택 슬라이스에 대해 계산된다. 이 불일치를 해소하지 않은 채 1,171개 전체 슬라이스 전수 평가라고 보고하는 것은 적절하지 않다.")
    add_paragraph(doc, "아래 결과는 저장된 최종 보고서와 스크립트의 기본 선택 수를 함께 반영해 클래스당 100개씩 총 300개 선택 슬라이스에 대한 기록값으로 제시한다. 이 표는 프로젝트의 현재 작동 상태를 정리한 것이며, 환자 단위의 독립 검증이나 통계적 유의성 검정을 완료한 최종 임상 성능표가 아니다. 후속 재현 실험에서는 환자 단위 train/validation/test 분할을 고정하고, 모든 테스트 슬라이스를 평가하며, 각 환자 평균과 신뢰구간을 함께 보고해야 한다.")
    add_result_table(doc)
    add_heading(doc, "4.2 클래스별 결과 해석", 2)
    add_paragraph(doc, "소형 클래스에서 CaraNet 기반 초기 마스크의 DSC는 0.8233이었고, 보정 후 0.8234로 0.0001 증가했다. HD95는 2.4205 px로 유지되었다. 이 결과는 소형 병변에서 보정기가 큰 폭의 성능 향상을 만들었다기보다, 초기 마스크를 훼손하지 않는 보수적 조정 또는 초기 상태로의 복원이 주로 일어났을 가능성을 시사한다. 작은 병변은 면적 자체가 작아 몇 픽셀의 변화가 지표를 크게 흔들 수 있으므로, 평균값뿐 아니라 개선된 사례 비율, 악화된 사례 비율, 빈 마스크 발생률을 별도로 제시해야 한다.")
    add_paragraph(doc, "중형 클래스의 UNet++ 결과는 초기 DSC 0.9327, 최종 DSC 0.9328로 매우 작은 상승을 보였다. HD95는 0.6717 px에서 0.6622 px로 0.0095 px 감소했다. 세 클래스 중 경계 거리의 감소가 가장 뚜렷하게 나타난 결과지만, 그 절대량은 작다. 또한 현 평가가 정답 기반 후보 선택과 안전 게이트를 포함하므로, 이 감소를 PPO 정책만의 순수 효과로 해석할 수 없다. 공정한 비교를 위해서는 초기 분할, 고정 형태학 후처리, TTA만 적용한 방법, PPO만 적용한 방법, 전체 보정 방법을 같은 non-oracle 조건에서 비교하는 ablation이 필요하다.")
    add_paragraph(doc, "대형 클래스의 SegResNet은 초기 DSC 0.9530, 최종 DSC 0.9530으로 성능이 유지됐고 HD95도 0.4795 px로 유지됐다. 초기 성능이 높을수록 추가 보정의 잠재 이득은 작아지고, 잘못된 경계 이동의 위험은 상대적으로 커진다. 따라서 대형 병변에서는 공격적인 보정보다 불확실도가 높은 경계 구간에만 선택적으로 보정기를 적용하거나, 보정 전후가 사실상 동일한 경우 정책을 조기에 종료하는 전략이 더 적절할 수 있다.")
    add_heading(doc, "4.3 소형 병변의 층화 분석", 2)
    add_paragraph(doc, "최종 보고서의 소형 클래스 층화 분석에서는 면적 50 px 이상인 active tumor 93개가 초기 DSC 0.8541에서 최종 DSC 0.8543으로 변화했고, HD95는 1.8954 px로 유지됐다. 50 px 미만의 micro fragment 7개는 초기 및 최종 DSC가 0.4138, HD95가 9.3956 px로 보고됐다. 미세 파편의 성능이 낮은 이유는 작은 구조에서 한두 픽셀의 누락 또는 추가가 상대적으로 큰 비율 변화를 만들고, 낮은 해상도에서 경계 정보가 충분하지 않기 때문이다.")
    add_paragraph(doc, "소형 병변에 64 x 64 확대 관찰과 연속 행동을 적용한 설계는 타당한 출발점이다. 다만 현재 확대 중심은 예측 마스크의 가장 큰 연결요소에 기반하므로, 초기 마스크가 비어 있거나 다수의 작은 병변이 떨어져 있는 경우에는 관심영역 자체가 불완전할 수 있다. 향후에는 확률 맵의 다중 peak, 불확실성 지도, 3차원 연결요소 정보를 함께 사용해 여러 후보 ROI를 생성하고, 각 ROI의 결과를 정답 없이 융합하는 방식이 필요하다.")
    add_heading(doc, "4.4 정성적 결과", 2)
    if FIGURE_1.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(FIGURE_1), width=Inches(6.25))
        add_caption(doc, "그림 1. 크기별 동적 라우팅 파이프라인의 예시 시각화. 원본 MRI, 정답 마스크, 초기 마스크 및 보정 마스크의 관계를 확인하기 위한 내부 결과 이미지이다.")
    add_paragraph(doc, "그림 1은 저장된 파이프라인 비교 이미지를 본문에 배치한 것이다. 이 그림은 제안 방법이 어떻게 초기 마스크와 보정 마스크를 비교하는지 직관적으로 보여 주지만, 정성적 예시는 통계적 성능을 대신할 수 없다. 특히 높은 초기 DSC를 가진 샘플만 선택해 제시하면 보정 효과가 과대하게 보일 수 있으므로, 향후 결과 그림에는 무작위 환자 기준 선택, 개선 사례와 악화 사례의 동시 제시, 그리고 각 그림의 DSC 및 HD95를 함께 표기해야 한다.")
    if FIGURE_2.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(FIGURE_2), width=Inches(6.25))
        add_caption(doc, "그림 2. 저장된 하위영역 파이프라인 비교 시각화. 현재 데이터 로더의 이진 Whole Tumor 처리와 시각화의 하위영역 표기는 구분하여 해석해야 한다.")
    add_paragraph(doc, "그림 2의 하위영역 표기는 시각화 또는 별도 실험 산출물일 수 있으나, 본문에서 분석한 evaluate_pipeline.py의 데이터 로더는 0이 아닌 모든 레이블을 하나의 Whole Tumor 마스크로 이진화한다. 그러므로 해당 시각화를 현재 정량 결과와 직접 같은 과제로 간주해서는 안 된다. 향후 논문 버전에서는 각 그림이 어떤 레이블 정의, 데이터 분할, 모델 체크포인트 및 평가 스크립트로 생성됐는지를 캡션에 명시해 정량 결과와 정성 결과 사이의 추적 가능성을 확보해야 한다.")
    add_heading(doc, "4.5 결과의 타당성과 재현성 검토", 2)
    add_paragraph(doc, "현재 결과의 가장 큰 해석상 위험은 oracle 정보의 사용이다. 첫째, 전문가 모델 선택에 정답 면적을 사용한다. 둘째, 일부 임계값 및 형태학 후보 선택에 정답 DSC를 사용한다. 셋째, 최종 안전 게이트가 정답 DSC와 HD95로 되돌림을 결정한다. 이 세 요소가 결합되면 '강화학습 정책이 영상만으로 일반화했다'는 주장보다 높은 성능이 관찰될 수 있다. 따라서 현재 수치는 제안된 보정 공간과 전문가 조합의 탐색 가능성을 보여 주는 upper-bound 실험으로 제한해야 한다.")
    add_paragraph(doc, "두 번째 위험은 표본 수와 평가 단위다. 기본 설정은 클래스당 100개로 제한돼 있고, 선택 순서가 환자 또는 슬라이스 정렬 순서에 영향을 받을 수 있다. 같은 환자의 인접 슬라이스는 서로 매우 유사하므로, 슬라이스 단위 무작위 분할은 데이터 누수를 만들 수 있다. 재현 가능한 실험을 위해서는 환자 ID 기준으로 train, validation, test를 분리하고, test 환자 전체의 유효 슬라이스를 평가하며, 환자별 평균 DSC 및 HD95의 평균과 95% 신뢰구간을 계산해야 한다.")
    add_paragraph(doc, "세 번째 위험은 물리적 해석이다. 보고된 HD95는 128 x 128으로 리사이즈된 2차원 이미지에서 계산한 픽셀 거리다. 원본 MRI의 voxel spacing이 환자마다 다를 수 있고, 2차원 슬라이스는 상하 방향의 불연속성을 반영하지 못한다. 따라서 0.6622 px 또는 0.4795 px를 곧바로 millimeter 수준의 임상 정확도라고 해석해서는 안 된다. 실제 임상적 비교를 위해서는 원본 spacing을 보존한 3차원 HD95(mm), volumetric Dice, surface Dice at tolerance 등을 함께 보고해야 한다.")

    add_heading(doc, "5. 결론 및 향후 연구", 1)
    add_paragraph(doc, "본 연구는 뇌종양 MRI 분할에서 병변 크기에 맞춰 전문가 모델을 선택하고, PPO 기반 보정 정책으로 경계를 미세 조정하는 RL-Refiner 구조를 제안했다. CaraNet, UNet++, SegResNet을 소형, 중형, 대형 병변에 각각 배정하고, SDF 기반 섹터별 경계 이동과 컴포넌트별 보정을 결합했다. 기록된 oracle-aided 평가에서는 중형 병변의 HD95가 0.6717 px에서 0.6622 px로 감소했고, 소형 및 대형 병변에서는 초기 성능이 보존되는 결과가 관찰됐다. 이는 병변 크기와 초기 예측 상태를 반영하는 보수적 후처리의 가능성을 보여 준다.")
    add_paragraph(doc, "그러나 현 단계에서 시스템을 '완전 자동 임상 분할 파이프라인'으로 결론내릴 수는 없다. 평가에서 정답 기반 라우팅, 정답 기반 후보 선택, 정답 기반 안전 게이트가 사용되며, 기본 코드의 성능 평균도 전수 1,171개가 아니라 클래스당 최대 100개 선택 슬라이스에서 계산된다. 따라서 본 논문의 핵심 결론은 임상 성능의 확정이 아니라, 정답 의존 요소를 제거한 엄격한 재평가가 필요한 유망한 설계 초안이라는 것이다.")
    add_heading(doc, "5.1 향후 연구 과제", 2)
    add_paragraph(doc, "첫째, 실제 Shape Classifier의 성능을 환자 단위 검증 세트에서 측정해야 한다. 정확도, macro F1, confusion matrix를 보고하고, 잘못된 클래스 라우팅이 최종 DSC와 HD95에 미치는 영향을 분석해야 한다. 또한 한 전문가의 예측 확률이 낮을 때 두 전문가를 함께 실행해 확률을 융합하는 soft routing도 비교할 수 있다.")
    add_paragraph(doc, "둘째, 시험 단계에서의 모든 정답 의존 연산을 제거해야 한다. 임계값은 validation set에서 하나로 고정하거나, predicted probability calibration에 기반해 결정해야 한다. 안전 게이트는 정답 DSC 대신 분할 엔트로피, augmentations 간 예측 일치도, 영상 에지와 마스크 경계의 정렬도, 또는 별도로 학습한 품질 예측기를 사용해야 한다. 정답 기반 안전 게이트는 별도 oracle upper-bound 실험으로만 분리해 보고하는 것이 바람직하다.")
    add_paragraph(doc, "셋째, 공정한 ablation study가 필요하다. 최소한 (a) 단일 SegResNet 또는 UNet++ 기준선, (b) oracle 없이 예측 클래스만 사용한 동적 라우팅, (c) 고정 형태학 후처리, (d) TTA만 적용한 방법, (e) PPO만 적용한 방법, (f) 전체 제안 방법을 같은 test set에서 비교해야 한다. 각 방법에는 환자별 DSC, HD95(mm), surface Dice, 실행 시간, 빈 마스크 비율을 함께 제시해야 한다.")
    add_paragraph(doc, "넷째, 2차원 슬라이스 기반 보정을 3차원 볼륨 기반으로 확장할 필요가 있다. 현재 방법은 한 슬라이스에서 경계가 좋아져도 인접 슬라이스 간 마스크가 불연속적일 수 있다. 3차원 U-Net 계열 전문가, 3차원 SDF, 볼륨 단위 또는 연속 슬라이스 상태를 사용하는 정책을 도입하면 이러한 한계를 줄일 수 있다. 다만 계산 비용과 정책 학습의 안정성도 함께 검토해야 한다.")
    add_paragraph(doc, "마지막으로 외부 데이터 검증과 사람 중심 평가가 필요하다. 서로 다른 장비, 병원, MRI 프로토콜에서 성능이 유지되는지 확인하고, 방사선종양학자나 영상의학 전문의가 보정 결과를 검토해야 한다. 최종 목표는 특정 수치의 최대화만이 아니라, 의료진의 수동 수정 시간을 줄이면서도 신뢰할 수 있는 오류 경고를 제공하는 보조 도구를 만드는 것이다.")

    add_heading(doc, "6. 참고문헌", 1)
    add_reference(doc, 1, "O. Ronneberger, P. Fischer, and T. Brox, \"U-Net: Convolutional Networks for Biomedical Image Segmentation,\" in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015.")
    add_reference(doc, 2, "Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, \"UNet++: A Nested U-Net Architecture for Medical Image Segmentation,\" IEEE Transactions on Medical Imaging, vol. 39, no. 6, pp. 1856-1867, 2020.")
    add_reference(doc, 3, "A. Myronenko, \"3D MRI Brain Tumor Segmentation Using Autoencoder Regularization,\" in BrainLes: Brain Tumor Segmentation Challenge, 2018.")
    add_reference(doc, 4, "A. Lou, S. Guan, and M. Loew, \"CaraNet: Context Axial Reverse Attention Network for Segmentation of Small Medical Objects,\" Journal of Medical Imaging, vol. 10, no. 1, 014005, 2023.")
    add_reference(doc, 5, "J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, \"Proximal Policy Optimization Algorithms,\" arXiv:1707.06347, 2017.")
    add_reference(doc, 6, "U. Baid et al., \"The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification,\" arXiv:2107.02314, 2021.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.core_properties.title = "강화학습 기반 동적 라우팅을 이용한 뇌종양 MRI 분할 경계 보정 연구"
    doc.core_properties.author = "[성명 입력]"
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    make_document()
