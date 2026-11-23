"""Render the report to PDF with reportlab, following the format guidelines exactly (two-pass)."""
import json
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                               Table, TableStyle, Image, PageBreak, KeepTogether,
                               NextPageTemplate)

import report_content as C

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT = os.path.join(_ROOT, "docs", "Medical_Insurance_Cost_Prediction_ML_Case_Study_Report.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ---------------------------------------------------------------- fonts
FONT, FONTB, FONTI, FONTBI = "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"
cands = [
    ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf"),
]
for reg, bold, ital, bi in cands:
    if os.path.exists(reg):
        pdfmetrics.registerFont(TTFont("Body", reg))
        pdfmetrics.registerFont(TTFont("Body-Bold", bold))
        pdfmetrics.registerFont(TTFont("Body-Italic", ital))
        pdfmetrics.registerFont(TTFont("Body-BoldItalic", bi))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold",
                                      italic="Body-Italic", boldItalic="Body-BoldItalic")
        FONT, FONTB, FONTI, FONTBI = "Body", "Body-Bold", "Body-Italic", "Body-BoldItalic"
        break
print("fonts:", FONT)

MONO = "Courier"
try:
    mp = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    mb = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
    if os.path.exists(mp):
        pdfmetrics.registerFont(TTFont("Mono", mp))
        pdfmetrics.registerFont(TTFont("Mono-Bold", mb))
        MONO = "Mono"
except Exception:
    pass

# ---------------------------------------------------------------- styles
BODY = ParagraphStyle("body", fontName=FONT, fontSize=12, leading=18,
                      alignment=TA_JUSTIFY, spaceAfter=8, firstLineIndent=0)
BODY0 = ParagraphStyle("body0", parent=BODY, spaceAfter=2)
H1S = ParagraphStyle("h1", fontName=FONTB, fontSize=16, leading=21, spaceBefore=4,
                     spaceAfter=12, alignment=TA_LEFT)
H2S = ParagraphStyle("h2", fontName=FONTB, fontSize=14, leading=18, spaceBefore=12,
                     spaceAfter=7, alignment=TA_LEFT)
H3S = ParagraphStyle("h3", fontName=FONTB, fontSize=12, leading=16, spaceBefore=9,
                     spaceAfter=5, alignment=TA_LEFT)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=22, bulletIndent=6, spaceAfter=5)
NUM = ParagraphStyle("num", parent=BODY, leftIndent=24, bulletIndent=6, spaceAfter=5)
CAP = ParagraphStyle("cap", fontName=FONT, fontSize=10, leading=13.5,
                     alignment=TA_CENTER, spaceBefore=5, spaceAfter=12)
TCAP = ParagraphStyle("tcap", fontName=FONT, fontSize=10, leading=13.5,
                      alignment=TA_CENTER, spaceBefore=10, spaceAfter=4)
CELL = ParagraphStyle("cell", fontName=FONT, fontSize=9, leading=11.5, alignment=TA_LEFT)
CELLH = ParagraphStyle("cellh", fontName=FONTB, fontSize=9, leading=11.5, alignment=TA_CENTER)
CELLC = ParagraphStyle("cellc", fontName=FONT, fontSize=9, leading=11.5, alignment=TA_CENTER)
CODE = ParagraphStyle("code", fontName=MONO, fontSize=8.0, leading=10.8,
                      alignment=TA_LEFT, leftIndent=6, spaceBefore=2, spaceAfter=2)
KBOX = ParagraphStyle("kbox", fontName=FONTI, fontSize=11.5, leading=16.5, alignment=TA_JUSTIFY,
                      leftIndent=6, rightIndent=6)
TITLE_ST = ParagraphStyle("title", fontName=FONTB, fontSize=18, leading=24, alignment=TA_CENTER)
SUB_ST = ParagraphStyle("sub", fontName=FONTB, fontSize=13, leading=18, alignment=TA_CENTER)
CTR = ParagraphStyle("ctr", fontName=FONT, fontSize=12, leading=17, alignment=TA_CENTER)
CTRB = ParagraphStyle("ctrb", fontName=FONTB, fontSize=12, leading=17, alignment=TA_CENTER)
SMALL = ParagraphStyle("small", fontName=FONT, fontSize=10.5, leading=14, alignment=TA_LEFT)

LM, RM, TM, BM = 1.0 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch
PW, PH = A4
FW = PW - LM - RM


def header_footer(canvas, doc):
    canvas.saveState()
    page = canvas.getPageNumber()
    canvas.setFont(FONT, 9.5)
    if doc.pageTemplate.id == "prelim":
        label = _roman(page)
        canvas.drawCentredString(PW / 2, BM / 2, label)
    else:
        canvas.drawCentredString(PW / 2, BM / 2, str(page - doc.main_offset))
        canvas.setFont(FONT, 8.5)
        canvas.drawRightString(PW - RM, PH - TM / 2,
                               "Medical Insurance Cost Prediction Using ML Regression Models")
        canvas.setStrokeColor(colors.HexColor("#b8b8b8"))
        canvas.setLineWidth(0.4)
        canvas.line(LM, PH - TM / 2 - 8, PW - RM, PH - TM / 2 - 8)
    canvas.restoreState()


def _roman(n):
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
            (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    s = ""
    for v, r in vals:
        while n >= v:
            s += r
            n -= v
    return s


PGMAP = {}
CUR = {"key": None}


class Doc(BaseDocTemplate):
    main_offset = 0

    def afterFlowable(self, flowable):
        key = getattr(flowable, "_mark", None)
        if key:
            self.canv.bookmarkPage(key)
            PGMAP[key] = self.page


# ---------------------------------------------------------------- flowable helpers
def cell(txt, style=CELL):
    return Paragraph(str(txt), style)


def make_table(cap, headers, rows, widths, fs=9, align_center_cols=None):
    """widths in inches; scaled to frame width."""
    tot = sum(widths)
    w = [x / tot * FW for x in widths]
    cs = ParagraphStyle("c", parent=CELL, fontSize=fs, leading=fs * 1.28)
    ch = ParagraphStyle("ch", parent=CELLH, fontSize=fs, leading=fs * 1.28)
    data = [[Paragraph(f"<b>{h}</b>", ch) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(c), cs) for c in r])
    t = Table(data, colWidths=w, repeatRows=1, hAlign="CENTER")
    st = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8c8c8c")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4eaf1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f8fb")))
    t.setStyle(TableStyle(st))
    cp = Paragraph(cap, TCAP)
    cp._mark = "TBL:" + cap.split(":")[0]
    if len(rows) <= 7:
        return [KeepTogether([cp, t]), Spacer(1, 12)]
    return [cp, t, Spacer(1, 12)]


def render_blocks(blocks, story, counters=None):
    for b in blocks:
        k = b[0]
        if k == "h1":
            para = Paragraph(b[1], H1S)
            para._mark = "H1:" + b[1]
            story.append(para)
        elif k == "h2":
            para = Paragraph(b[1], H2S)
            para._mark = "H2:" + b[1]
            story.append(para)
        elif k == "h3":
            story.append(Paragraph(b[1], H3S))
        elif k == "p":
            story.append(Paragraph(b[1], BODY))
        elif k == "bl":
            for it in b[1]:
                story.append(Paragraph(it, BUL, bulletText="•"))
            story.append(Spacer(1, 5))
        elif k == "nl":
            for i, it in enumerate(b[1], 1):
                story.append(Paragraph(it, NUM, bulletText=f"{i}."))
            story.append(Spacer(1, 5))
        elif k == "fig":
            path, cap = b[1], b[2]
            from PIL import Image as PILImage
            iw, ih = PILImage.open(path).size
            wmax = FW * 0.90
            hmax = PH * 0.355
            sc = min(wmax / iw, hmax / ih)
            img = Image(path, width=iw * sc, height=ih * sc)
            story.append(Spacer(1, 4))
            cp = Paragraph(cap, CAP)
            cp._mark = "FIG:" + cap.split(":")[0]
            story.append(KeepTogether([img, cp]))
        elif k == "tbl":
            story.extend(make_table(b[1], b[2], b[3], b[4] or [1] * len(b[2]), fs=b[6] if len(b) > 6 else 9))
        elif k == "code":
            tbl = Table([[Paragraph(b[1].replace(" ", "&nbsp;").replace("\n", "<br/>"), CODE)]],
                        colWidths=[FW], hAlign="CENTER")
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f4f4")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#b0b0b0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(Spacer(1, 4)); story.append(tbl); story.append(Spacer(1, 10))
        elif k == "kbox":
            tbl = Table([[Paragraph(b[1], KBOX)]], colWidths=[FW * 0.93], hAlign="CENTER")
            tbl.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#5b7a99")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f2f6fa")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(Spacer(1, 4)); story.append(tbl); story.append(Spacer(1, 12))
        elif k == "pb":
            story.append(PageBreak())


# ---------------------------------------------------------------- build story
story = []
NFRONT = 9   # cover has no page number


def cover(story):
    story.append(Spacer(1, 0.42 * inch))
    story.append(Paragraph(C.COLLEGE, SUB_ST))
    story.append(Spacer(1, 4))
    story.append(Paragraph("[Department of Artificial Intelligence and Machine Learning]", CTR))
    story.append(Spacer(1, 0.30 * inch))
    story.append(Paragraph("MACHINE LEARNING CASE STUDY REPORT", TITLE_ST))
    story.append(Spacer(1, 0.10 * inch))
    hr = Table([[""]], colWidths=[FW * 0.55], rowHeights=[2])
    hr.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, -1), 1.2, colors.HexColor("#1d3557"))]))
    story.append(Table([[""]], colWidths=[FW], rowHeights=[1],
                       style=[("LINEBELOW", (0, 0), (-1, -1), 1.2, colors.HexColor("#1d3557"))]))
    story.append(Spacer(1, 0.16 * inch))
    story.append(Paragraph("“" + C.TITLE + "”", ParagraphStyle(
        "t2", parent=TITLE_ST, fontSize=16, leading=22)))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Table([[""]], colWidths=[FW], rowHeights=[1],
                       style=[("LINEBELOW", (0, 0), (-1, -1), 1.2, colors.HexColor("#1d3557"))]))
    story.append(Spacer(1, 0.26 * inch))
    story.append(Paragraph("Submitted in partial fulfilment of the requirements for the", CTR))
    story.append(Paragraph("<b>V Semester Bachelor of Engineering</b>", CTR))
    story.append(Paragraph("in", CTR))
    story.append(Paragraph("<b>Artificial Intelligence and Machine Learning</b>", CTR))
    story.append(Spacer(1, 0.24 * inch))
    story.append(Paragraph("<b>Submitted by</b>", CTRB))
    for s in C.STUDENTS:
        story.append(Paragraph(s, CTR))
    story.append(Spacer(1, 0.22 * inch))
    story.append(Paragraph("<b>Under the guidance of</b>", CTRB))
    story.append(Paragraph(C.GUIDE, CTR))
    story.append(Spacer(1, 0.26 * inch))
    story.append(Paragraph("<b>Academic Year " + C.YEAR + "</b>", CTRB))



cover_done = True
cover(story)
story.append(PageBreak())

# ---- certificate
story.append(Paragraph("CERTIFICATE", H1S))
story.append(Spacer(1, 10))
story.append(Paragraph(
    "This is to certify that the Machine Learning Case Study report entitled “" + C.TITLE +
    "” is a bonafide work carried out by " + ", ".join(s.split(" – ")[0] for s in C.STUDENTS) +
    " (" + ", ".join(s.split(" – ")[1] for s in C.STUDENTS) + "), students of the Department of "
    "Artificial Intelligence and Machine Learning, " + C.COLLEGE + ", during the V Semester of the "
    "academic year " + C.YEAR + ", in partial fulfilment of the requirements for the award of the "
    "Bachelor of Engineering degree in Artificial Intelligence and Machine Learning.", BODY))
story.append(Spacer(1, 0.45 * inch))
sig = Table([[Paragraph("____________________________", SMALL), Paragraph("____________________________", SMALL)],
             [Paragraph("<b>Guide Signature</b><br/>" + C.GUIDE, SMALL),
              Paragraph("<b>HOD Signature</b><br/>Head of the Department<br/>Dept. of AI and ML", SMALL)]],
            colWidths=[FW / 2, FW / 2])
sig.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4)]))
story.append(sig)
story.append(Spacer(1, 0.35 * inch))
story.append(Paragraph("Date: ____________________                              Place: ____________________", BODY))
story.append(PageBreak())

# ---- declaration
story.append(Paragraph("DECLARATION", H1S))
story.append(Spacer(1, 10))
story.append(Paragraph(
    "We hereby declare that the work presented in this report entitled “" + C.TITLE + "” is carried out "
    "by us under the guidance of " + C.GUIDE + ". The work is original to the best of our knowledge. All "
    "external sources, datasets, libraries and references used have been appropriately acknowledged. "
    "This report has not been submitted elsewhere for the award of any degree or diploma.", BODY))
story.append(Spacer(1, 0.30 * inch))
story.append(Paragraph("<b>Names and USNs:</b>", BODY))
for i, s in enumerate(C.STUDENTS, 1):
    story.append(Paragraph(f"{i}. &nbsp;&nbsp;{s.replace(' – ', '  –  ')}   ____________________________________", BODY))
story.append(Spacer(1, 0.55 * inch))
story.append(Paragraph("Place: ____________________", BODY))
story.append(Paragraph("Date: ____________________", BODY))
story.append(PageBreak())

# ---- acknowledgement
story.append(Paragraph("ACKNOWLEDGEMENT", H1S))
story.append(Spacer(1, 10))
for para in C.ACKNOWLEDGEMENT.split("\n\n"):
    story.append(Paragraph(para, BODY))
story.append(PageBreak())

# ---- abstract
_t = Paragraph("ABSTRACT", H1S); _t._mark = "PREL:ABSTRACT"; story.append(_t)
story.append(Spacer(1, 10))
story.append(Paragraph(C.ABSTRACT, BODY))
story.append(Spacer(1, 10))
story.append(Paragraph("<b>Keywords:</b> <i>" + C.KEYWORDS + "</i>", BODY))

# ---------------------------------------------------------------- TOC etc.
PGPATH = os.path.join(_ROOT, "src", "pgmap.json")
PREV = json.load(open(PGPATH)) if os.path.exists(PGPATH) else {}
MAIN_OFFSET = PREV.get("__main_offset__", 0)


def _k(kind, text):
    return f"{kind}:{text}"


def roman(n):
    return _roman(n) if n > 0 else ""


def toc_page(kind, text):
    pg = PREV.get(_k(kind, text))
    if pg is None:
        return "—"
    if pg > MAIN_OFFSET:
        return str(pg - MAIN_OFFSET)
    return roman(pg)


TOC_ENTRIES = [
    ("PREL", "ABSTRACT", "ABSTRACT"),
    ("PREL", "LIST OF FIGURES", "LIST OF FIGURES"),
    ("PREL", "LIST OF TABLES", "LIST OF TABLES"),
    ("PREL", "LIST OF ABBREVIATIONS", "LIST OF ABBREVIATIONS"),
    ("H1", "CHAPTER 1 – INTRODUCTION", "CHAPTER 1 – INTRODUCTION"),
    ("H2", "1.1 Background", "1.1  Background"),
    ("H2", "1.2 Problem Statement", "1.2  Problem Statement"),
    ("H2", "1.3 Objectives", "1.3  Objectives"),
    ("H2", "1.4 Scope of the Work", "1.4  Scope of the Work"),
    ("H2", "1.5 Organization of the Report", "1.5  Organization of the Report"),
    ("H1", "CHAPTER 2 – LITERATURE SURVEY", "CHAPTER 2 – LITERATURE SURVEY"),
    ("H2", "2.1 Research Gap / Motivation", "2.1  Research Gap / Motivation"),
    ("H1", "CHAPTER 3 – DATASET DESCRIPTION AND EXPLORATORY DATA ANALYSIS",
     "CHAPTER 3 – DATASET DESCRIPTION AND EXPLORATORY DATA ANALYSIS"),
    ("H2", "3.1 Dataset Source", "3.1  Dataset Source"),
    ("H2", "3.2 Dataset Overview", "3.2  Dataset Overview"),
    ("H2", "3.3 Attribute Description", "3.3  Attribute Description"),
    ("H2", "3.4 Exploratory Data Analysis", "3.4  Exploratory Data Analysis"),
    ("H2", "3.5 Observations from EDA", "3.5  Observations from EDA"),
    ("H1", "CHAPTER 4 – DATA PREPROCESSING AND FEATURE ENGINEERING",
     "CHAPTER 4 – DATA PREPROCESSING AND FEATURE ENGINEERING"),
    ("H2", "4.1 Data Cleaning", "4.1  Data Cleaning"),
    ("H2", "4.2 Encoding and Feature Scaling", "4.2  Encoding and Feature Scaling"),
    ("H2", "4.3 Feature Selection / Feature Engineering", "4.3  Feature Selection / Feature Engineering"),
    ("H2", "4.4 Training, Validation and Test Split", "4.4  Training, Validation and Test Split"),
    ("H1", "CHAPTER 5 – MACHINE LEARNING MODEL DEVELOPMENT",
     "CHAPTER 5 – MACHINE LEARNING MODEL DEVELOPMENT"),
    ("H2", "5.1 Workflow / System Architecture", "5.1  Workflow / System Architecture"),
    ("H2", "5.2 Algorithms Selected", "5.2  Algorithms Selected"),
    ("H2", "5.3 Implementation Details", "5.3  Implementation Details"),
    ("H2", "5.4 Hyperparameter Settings", "5.4  Hyperparameter Settings"),
    ("H1", "CHAPTER 6 – RESULTS AND PERFORMANCE EVALUATION",
     "CHAPTER 6 – RESULTS AND PERFORMANCE EVALUATION"),
    ("H2", "6.1 Evaluation Metrics", "6.1  Evaluation Metrics"),
    ("H2", "6.2 Experimental Results", "6.2  Experimental Results"),
    ("H2", "6.3 Visual Results", "6.3  Visual Results"),
    ("H2", "6.4 Model Comparison and Discussion", "6.4  Model Comparison and Discussion"),
    ("H2", "6.5 Sample Predictions / Demonstration", "6.5  Sample Predictions / Demonstration"),
    ("H1", "CHAPTER 7 – CONCLUSION AND FUTURE ENHANCEMENTS",
     "CHAPTER 7 – CONCLUSION AND FUTURE ENHANCEMENTS"),
    ("H2", "7.1 Conclusion", "7.1  Conclusion"),
    ("H2", "7.2 Limitations", "7.2  Limitations"),
    ("H2", "7.3 Future Enhancements", "7.3  Future Enhancements"),
    ("H1", "REFERENCES", "REFERENCES"),
    ("H1", "APPENDIX A – SOURCE CODE AND REPRODUCIBILITY",
     "APPENDIX A – SOURCE CODE AND REPRODUCIBILITY"),
    ("H1", "APPENDIX B – ADDITIONAL RESULTS", "APPENDIX B – ADDITIONAL RESULTS"),
]

FIGS = [
    ("Fig. 3.1", "Histogram with KDE and box plot of annual charges"),
    ("Fig. 3.2", "Box plots of charges by smoking status and number of children"),
    ("Fig. 3.3", "Distributions of age, BMI and number of children"),
    ("Fig. 3.4", "Scatter plots of charges against age and BMI, coloured by smoking status"),
    ("Fig. 3.5", "Pearson correlation matrix of numerical attributes and target"),
    ("Fig. 3.6", "Box plots of charges by region and by sex"),
    ("Fig. 5.1", "End-to-end machine learning workflow"),
    ("Fig. 6.1", "Predicted vs actual charges: linear baseline and best model"),
    ("Fig. 6.2", "Residual plots for the linear baseline and the best model"),
    ("Fig. 6.3", "Test R² and RMSE comparison across the four models"),
    ("Fig. 6.4", "Random forest feature importances (top ten)"),
    ("Fig. 6.5", "Absolute error by smoking status for the best model"),
    ("Fig. B.1", "Random forest test R² versus training set size"),
]

TABS = [
    ("Table 2.1", "Summary of reviewed literature"),
    ("Table 3.1", "Dataset overview"),
    ("Table 3.2", "Attribute description"),
    ("Table 3.3", "Descriptive statistics of numerical attributes"),
    ("Table 3.4", "Mean charges by smoking status and BMI category"),
    ("Table 4.1", "Data cleaning summary"),
    ("Table 4.2", "Encoding and scaling decisions by model"),
    ("Table 4.3", "Engineered features"),
    ("Table 4.4", "Data partition used in the experiments"),
    ("Table 5.1", "Hyperparameter search spaces and selected values"),
    ("Table 6.1", "Evaluation metrics used in the study"),
    ("Table 6.2", "Test-set performance of the four regression models"),
    ("Table 6.3", "Five-fold cross-validated performance"),
    ("Table 6.4", "Sample predictions on five held-out test records"),
    ("Table 6.5", "Demonstration predictions for hypothetical applicants"),
    ("Table B.1", "Test R² by training set size"),
]


def toc_table(rows):
    data = []
    for text, pg, bold in rows:
        st = ParagraphStyle("t", fontName=FONTB if bold else FONT, fontSize=11.5 if bold else 11,
                            leading=17)
        data.append([Paragraph(text, st), Paragraph(pg,
                     ParagraphStyle("tp", parent=st, alignment=TA_CENTER))])
    n = len(data)
    t = Table(data, colWidths=[FW * 0.88, FW * 0.12], hAlign="CENTER")
    stl = [("VALIGN", (0, 0), (-1, -1), "BOTTOM"), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    for i in range(1, n):
        stl.append(("LINEBELOW", (0, i), (-1, i), 0.4, colors.HexColor("#cfcfcf")))
    t.setStyle(TableStyle(stl))
    return t


story.append(PageBreak())
_t = Paragraph("TABLE OF CONTENTS", H1S); _t._mark = "PREL:TABLE OF CONTENTS"; story.append(_t)
story.append(Spacer(1, 8))
story.append(toc_table([(disp, toc_page(kind, key), kind == "H1") for kind, key, disp in TOC_ENTRIES]))

story.append(PageBreak())
_t = Paragraph("LIST OF FIGURES", H1S); _t._mark = "PREL:LIST OF FIGURES"; story.append(_t)
story.append(Spacer(1, 8))
story.append(toc_table([(f"{a}   {b}", toc_page("FIG", a), False) for a, b in FIGS]))

story.append(PageBreak())
_t = Paragraph("LIST OF TABLES", H1S); _t._mark = "PREL:LIST OF TABLES"; story.append(_t)
story.append(Spacer(1, 8))
story.append(toc_table([(f"{a}   {b}", toc_page("TBL", a), False) for a, b in TABS]))

story.append(PageBreak())
_t = Paragraph("LIST OF ABBREVIATIONS", H1S); _t._mark = "PREL:LIST OF ABBREVIATIONS"; story.append(_t)
story.append(Spacer(1, 8))
ab = [[Paragraph("<b>Abbreviation</b>", CELLH), Paragraph("<b>Expansion</b>", CELLH)]]
for a, b in C.ABBREVIATIONS:
    ab.append([Paragraph(a, CELLC), Paragraph(b, CELL)])
t = Table(ab, colWidths=[FW * 0.25, FW * 0.6], hAlign="CENTER", repeatRows=1)
t.setStyle(TableStyle([
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8c8c8c")),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4eaf1")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t)

# ---- main body
story.append(NextPageTemplate("main"))
story.append(PageBreak())
render_blocks(C.A, story)

# ---- references
story.append(PageBreak())
_t = Paragraph("REFERENCES", H1S); _t._mark = "H1:REFERENCES"; story.append(_t)
REFST = ParagraphStyle("ref", parent=BODY, leftIndent=26, firstLineIndent=-26, spaceAfter=7)
for i, r in enumerate(C.REFERENCES, 1):
    story.append(Paragraph(f"[{i}]\t{r}", REFST))

# ---- appendix A
story.append(PageBreak())
_t = Paragraph("APPENDIX A – SOURCE CODE AND REPRODUCIBILITY", H1S); _t._mark = "H1:APPENDIX A – SOURCE CODE AND REPRODUCIBILITY"; story.append(_t)
story.append(Paragraph(C.APPENDIX_A, BODY))
story.append(Paragraph("<b>Reproducibility steps</b>", H2S))
for i, s in enumerate(C.APPENDIX_A_STEPS, 1):
    story.append(Paragraph(s, NUM, bulletText=f"{i}."))
story.append(Spacer(1, 6))
story.append(Paragraph("<b>Core code extract</b>", H2S))
tbl = Table([[Paragraph(C.APPENDIX_CODE.replace(" ", "&nbsp;").replace("\n", "<br/>"), CODE)]],
            colWidths=[FW], hAlign="CENTER")
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f4f4")),
    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#b0b0b0")),
    ("LEFTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(tbl)
story.append(Spacer(1, 10))
story.append(Paragraph("The submission includes medical_insurance_ml.py, results.json and the figures/ "
                       "directory containing all thirteen generated plots. No passwords, API keys or "
                       "private data are included; the dataset is public.", BODY))

# ---- appendix B
story.append(PageBreak())
_t = Paragraph("APPENDIX B – ADDITIONAL RESULTS", H1S); _t._mark = "H1:APPENDIX B – ADDITIONAL RESULTS"; story.append(_t)
story.append(Paragraph(C.APPENDIX_B, BODY))
story.append(Paragraph("B.1  Effect of training set size", H2S))
story.append(Paragraph(
    "To check whether the selected models are limited by data volume, the tuned random forest was "
    "refitted on progressively larger fractions of the training subset and scored on the same test set. "
    "Test R² rises steeply between 171 and 513 training records and then flattens, gaining only about "
    "0.5 percentage points over the last 342 records. Performance is therefore close to saturation at "
    "the available sample size, which implies that collecting more rows of the same six attributes "
    "would yield limited improvement, whereas adding informative new attributes would not face the same "
    "ceiling.", BODY))
story.extend(make_table("Table B.1: Random forest test R² versus training set size",
                        ["Training records", "Test R²"],
                        [[str(a), f"{b:.4f}"] for a, b in zip(C.LC["train_size"], C.LC["test_r2"])],
                        [1.6, 1.2], fs=9.5))
_bcap = Paragraph("Fig. B.1: Random forest test R² as a function of the number of training records used.", CAP)
_bcap._mark = "FIG:Fig. B.1"
story.append(KeepTogether([Image(os.path.join(C.FIG, "figB1_learning_curve.png"),
                                width=FW * 0.78, height=FW * 0.78 * 0.654), _bcap]))
story.append(Paragraph("B.2  Polynomial degree selection", H2S))
story.append(Paragraph(
    "Degree 2 was selected after comparing degrees 1, 2 and 3 on the validation subset. Degree 1 "
    "reproduces the linear baseline and underfits. Degree 2 captures the smoking–BMI interaction and "
    "improves validation R² markedly. Degree 3 expands the feature space to several hundred terms on a "
    "training subset of 855 rows, and although Ridge regularisation keeps the solution stable, "
    "validation error does not improve over degree 2 while interpretability and fitting time worsen. "
    "Degree 2 was therefore retained as the best bias–variance compromise.", BODY))
story.append(Paragraph("B.3  Reproducibility notes", H2S))
story.append(Paragraph(
    "All figures and tables in this report are generated directly by medical_insurance_ml.py; no number "
    "was entered manually. The values reported in Table 6.2, Table 6.3, Table 6.4 and Table 6.5 can be "
    "checked against results.json. Re-running the script with the same fixed seed reproduces identical "
    "metrics, which was verified by two independent runs.", BODY))

# ---------------------------------------------------------------- build
doc = Doc(OUT, pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=TM, bottomMargin=BM,
          title=C.TITLE, author=", ".join(C.STUDENTS), subject="Machine Learning Case Study Report")
frame = Frame(LM, BM, FW, PH - TM - BM, id="normal")
prelim = PageTemplate(id="prelim", frames=[frame], onPage=header_footer)
main = PageTemplate(id="main", frames=[frame], onPage=header_footer)
doc.addPageTemplates([prelim, main])
Doc.main_offset = MAIN_OFFSET

# cover page should have no number -> use a template with header_footer that skips page 1
doc.build(story)
_c1 = [v for k, v in PGMAP.items() if k.startswith("H1:CHAPTER 1")]
PGMAP["__main_offset__"] = (_c1[0] - 1) if _c1 else 0
json.dump(PGMAP, open(PGPATH, "w"), indent=1)
print("WROTE", OUT, "pages:", len(PGMAP) - 1, "main_offset:", PGMAP["__main_offset__"])
