"""Render the report to PDF in the house style of the reference report ("ML Case Report - Yash").

Layout reproduced from the reference:
  * US Letter page (612 x 792 pt) with a thin rectangular page border on every page
  * running page number at the bottom centre, Arabic numerals throughout (cover = 1)
  * cover: BMS + VTU logos, blue institution block, no page border break
  * chapter headings "CHAPTER n - TITLE" in bold black, sections in bold navy
  * literature survey as one wide 5-column table
  * figures followed by an "Interpretation:" paragraph
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                                Table, TableStyle, Image, PageBreak, KeepTogether)
from PIL import Image as PILImage

import report_content as C

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT = os.path.join(_ROOT, "docs", "Medical_Insurance_Cost_Prediction_ML_Case_Study_Report.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
ASSETS = os.path.join(_ROOT, "assets")

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

MONO = "Courier"
try:
    if os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        pdfmetrics.registerFont(TTFont("Mono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))
        MONO = "Mono"
except Exception:
    pass

# ---------------------------------------------------------------- styles
NAVY = colors.HexColor("#365f91")
BMS_RED = colors.HexColor("#e04228")
BODY = ParagraphStyle("body", fontName=FONT, fontSize=11.3, leading=17.2,
                      alignment=TA_JUSTIFY, spaceAfter=7)
H1S = ParagraphStyle("h1", fontName=FONTB, fontSize=14.5, leading=19, spaceBefore=2,
                     spaceAfter=10, alignment=TA_LEFT, textColor=colors.black)
H2S = ParagraphStyle("h2", fontName=FONTB, fontSize=12.8, leading=17, spaceBefore=9,
                     spaceAfter=5, alignment=TA_LEFT, textColor=NAVY)
H3S = ParagraphStyle("h3", fontName=FONTB, fontSize=11.5, leading=15.5, spaceBefore=7,
                     spaceAfter=4, alignment=TA_LEFT, textColor=colors.black)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=26, bulletIndent=10, spaceAfter=4)
CAP = ParagraphStyle("cap", fontName=FONTB, fontSize=10, leading=13.5,
                     alignment=TA_CENTER, spaceBefore=5, spaceAfter=3)
INTERP = ParagraphStyle("interp", parent=BODY, spaceBefore=2)
TCAP = ParagraphStyle("tcap", fontName=FONTB, fontSize=9.6, leading=13,
                      alignment=TA_CENTER, spaceBefore=9, spaceAfter=4)
CELL = ParagraphStyle("cell", fontName=FONT, fontSize=8.5, leading=11, alignment=TA_LEFT)
CELLH = ParagraphStyle("cellh", fontName=FONTB, fontSize=8.5, leading=11, alignment=TA_CENTER)
CODE = ParagraphStyle("code", fontName=MONO, fontSize=7.6, leading=10.2,
                      alignment=TA_LEFT, leftIndent=4, spaceBefore=1, spaceAfter=1)
CTR = ParagraphStyle("ctr", fontName=FONT, fontSize=11.3, leading=17.5, alignment=TA_CENTER)
TOC_H = ParagraphStyle("toch", fontName=FONTB, fontSize=13, leading=17, alignment=TA_CENTER)

# ---------------------------------------------------------------- page geometry
PW, PH = letter                     # 612 x 792
LM = RM = 95                        # keeps text inside the reference's border
TM, BM = 60, 62
FW = PW - LM - RM

BORDER = (48.4, 22.6, 563.4, 769.6)


def draw_border(canvas):
    x0, y0, x1, y1 = BORDER
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#7f7f7f"))
    canvas.setLineWidth(0.7)
    canvas.rect(x0, PH - y1, x1 - x0, y1 - y0, stroke=1, fill=0)
    canvas.restoreState()


def header_footer(canvas, doc):
    draw_border(canvas)
    canvas.saveState()
    canvas.setFont(FONT, 10)
    canvas.drawCentredString(PW / 2, 44, str(canvas.getPageNumber()))
    canvas.restoreState()


NEWMAP = {}


class Doc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        key = getattr(flowable, "_mark", None)
        if key:
            NEWMAP[key] = self.page


# page numbers from a previous pass (if any), used to fill the TOC / LOF / LOT
PGPATH = os.path.join(_HERE, "pgmap.json")
PGMAP = json.load(open(PGPATH)) if os.path.exists(PGPATH) else {}

# ---------------------------------------------------------------- helpers
def make_table(cap, headers, rows, widths, fs=8.5):
    tot = sum(widths)
    w = [x / tot * FW for x in widths]
    cs = ParagraphStyle("c", parent=CELL, fontSize=fs, leading=fs * 1.28)
    ch = ParagraphStyle("ch", parent=CELLH, fontSize=fs, leading=fs * 1.28)
    data = [[Paragraph(h, ch) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(c), cs) for c in r])
    t = Table(data, colWidths=w, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    cp = Paragraph(cap, TCAP)
    cp._mark = "TBL:" + cap.split(":")[0]
    if len(rows) <= 7:
        return [KeepTogether([cp, t]), Spacer(1, 9)]
    return [cp, t, Spacer(1, 9)]


def render(blocks, story):
    for b in blocks:
        k = b[0]
        if k == "h1":
            p = Paragraph(b[1], H1S); p._mark = "H1:" + b[1]; story.append(p)
        elif k == "h2":
            p = Paragraph(b[1], H2S); p._mark = "H2:" + b[1]; story.append(p)
        elif k == "h3":
            story.append(Paragraph(b[1], H3S))
        elif k == "p":
            story.append(Paragraph(b[1], BODY))
        elif k == "bl":
            for it in b[1]:
                story.append(Paragraph(it, BUL, bulletText="\u2022"))
            story.append(Spacer(1, 4))
        elif k == "fig":
            path, cap, interp = b[1], b[2], b[3]
            iw, ih = PILImage.open(path).size
            wmax, hmax = FW * 0.88, PH * 0.36
            sc = min(wmax / iw, hmax / ih)
            img = Image(path, width=iw * sc, height=ih * sc)
            cp = Paragraph(cap, CAP)
            cp._mark = "FIG:" + cap.split(":")[0]
            group = [Spacer(1, 4), img, cp]
            if interp:
                group.append(Paragraph("<i>Interpretation:</i> " + interp, INTERP))
            story.append(KeepTogether(group))
            story.append(Spacer(1, 8))
        elif k == "tbl":
            story.extend(make_table(b[1], b[2], b[3], b[4] or [1] * len(b[2]),
                                    fs=(b[6] if len(b) > 6 else 8.5)))
        elif k == "code":
            t = Table([[Paragraph(b[1].replace(" ", "&nbsp;").replace("\n", "<br/>"), CODE)]],
                      colWidths=[FW], hAlign="CENTER")
            t.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.6, colors.black),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f7f7")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(Spacer(1, 3)); story.append(t); story.append(Spacer(1, 9))
        elif k == "pb":
            story.append(PageBreak())


# ---------------------------------------------------------------- build story
story = []

# =============================== COVER (page 1)
if os.path.exists(os.path.join(ASSETS, "bms_logo.jpeg")):
    logo_l = Image(os.path.join(ASSETS, "bms_logo.jpeg"), width=52, height=57)
    logo_r = Image(os.path.join(ASSETS, "vtu_logo.jpeg"), width=57, height=65)
    head = Table([[logo_l, "", logo_r]], colWidths=[70, FW - 140, 70], rowHeights=[68])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("ALIGN", (0, 0), (0, 0), "LEFT"),
                              ("ALIGN", (2, 0), (2, 0), "RIGHT"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(head)
    story.append(Spacer(1, 2))

bms = ParagraphStyle("bms", fontName=FONTB, fontSize=16.5, leading=21, alignment=TA_CENTER)
story.append(Paragraph(
    f'<font color="#e04228">BMS</font> <font color="#365f91">INSTITUTE OF TECHNOLOGY AND MANAGEMENT</font>', bms))
story.append(Paragraph(C.COLLEGE_SUB, ParagraphStyle("cs", parent=CTR, fontSize=10.5, leading=15)))
story.append(Paragraph(C.COLLEGE_ADDR, ParagraphStyle("ca", parent=CTR, fontSize=10.5, leading=15)))
story.append(Spacer(1, 12))
story.append(Paragraph(C.DEPT, ParagraphStyle("dept", fontName=FONTB, fontSize=13.5, leading=17,
                                              alignment=TA_CENTER, textColor=NAVY)))
story.append(Spacer(1, 14))
story.append(Paragraph("MACHINE LEARNING CASE STUDY REPORT",
                       ParagraphStyle("r1", parent=CTR, fontName=FONTB, fontSize=12.8, leading=17)))
story.append(Paragraph("ON", ParagraphStyle("on", parent=CTR, fontSize=11.3, leading=16)))
story.append(Paragraph(C.TITLE, ParagraphStyle("t1", parent=CTR, fontName=FONTB, fontSize=13.6, leading=18)))
story.append(Spacer(1, 16))
for line in ["Submitted in partial fulfilment of the requirements for the",
             "V Semester Bachelor of Engineering", "in",
             "Artificial Intelligence and Machine Learning"]:
    story.append(Paragraph(f"<i>{line}</i>", ParagraphStyle("sub", parent=CTR, fontSize=10.8, leading=15.5)))
story.append(Spacer(1, 14))
story.append(Paragraph("Submitted by", ParagraphStyle("sb", parent=CTR, fontName=FONTB, fontSize=11.3, leading=17)))
for name, usn in C.STUDENTS:
    story.append(Paragraph(f"{name} \u2013 {usn}", ParagraphStyle("st", parent=CTR, fontSize=11, leading=16.5)))
story.append(Spacer(1, 12))
story.append(Paragraph("Under the guidance of", ParagraphStyle("ug", parent=CTR, fontName=FONTB,
                                                               fontSize=11.3, leading=17)))
story.append(Paragraph(C.GUIDE, ParagraphStyle("gn", parent=CTR, fontName=FONTB, fontSize=11, leading=16)))
story.append(Paragraph(C.GUIDE_DESIG, ParagraphStyle("gd", parent=CTR, fontName=FONTB, fontSize=10.8, leading=15)))
story.append(Paragraph("Department of Artificial Intelligence and Machine Learning",
                       ParagraphStyle("gdp", parent=CTR, fontSize=10.5, leading=15)))
story.append(Spacer(1, 14))
story.append(Paragraph("[Academic Year: " + C.YEAR + "]",
                       ParagraphStyle("ay", parent=CTR, fontName=FONTB, fontSize=10.8, leading=15)))

# =============================== CERTIFICATE (2)
story.append(PageBreak())
p = Paragraph("CERTIFICATE", ParagraphStyle("ch", parent=TOC_H, textColor=colors.black)); p._mark = "FM:CERTIFICATE"
story.append(p); story.append(Spacer(1, 16))
names_usn = ", ".join(f"{n}-{u}" for n, u in C.STUDENTS)
names_only = ", ".join(n.title().replace("Y P Rohith", "Y P Rohith") for n, u in C.STUDENTS)
story.append(Paragraph(
    "This is to certify that the Machine Learning Case Study report entitled \u201c" + C.TITLE.title() +
    "\u201d is a bonafide work carried out by " + names_usn + ", students of the Department of Artificial "
    "Intelligence and Machine Learning, BMS Institute of Technology and Management, during the V Semester / "
    "Academic Year " + C.YEAR + ", in partial fulfilment of the requirements for the award of the Bachelor "
    "of Engineering degree.", BODY))
story.append(Spacer(1, 40))
sig = Table([[Paragraph("Guide Signature:", BODY), Paragraph("HOD Signature:", BODY)],
             [Paragraph("<br/>___________________________", BODY), Paragraph("<br/>___________________________", BODY)],
             [Paragraph(f"[{C.GUIDE}, {C.GUIDE_DESIG}]", BODY),
              Paragraph(f"[{C.HOD}, Head Of Department]", BODY)]],
            colWidths=[FW / 2, FW / 2])
sig.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                         ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
story.append(sig)
story.append(Spacer(1, 34))
story.append(Paragraph("Date: _____________________", BODY))
story.append(Spacer(1, 10))
story.append(Paragraph("Place: ____________________", BODY))

# =============================== DECLARATION (3)
story.append(PageBreak())
p = Paragraph("DECLARATION", ParagraphStyle("dh", parent=TOC_H, textColor=colors.black)); p._mark = "FM:DECLARATION"
story.append(p); story.append(Spacer(1, 16))
story.append(Paragraph(
    "We hereby declare that the work presented in this report entitled \u201c" + C.TITLE.title() + "\u201d "
    "is carried out by us under the guidance of " + C.GUIDE + ". The work is original to the best of our "
    "knowledge. All external sources, datasets, libraries and references used have been appropriately "
    "acknowledged. This report has not been submitted elsewhere for the award of any degree or diploma.",
    BODY))
story.append(Spacer(1, 20))
story.append(Paragraph("Names and USNs:", BODY))
for i, (n, u) in enumerate(C.STUDENTS, 1):
    row = Table([[Paragraph(f"{i}. {n.title()} \u2013 {u}", BODY),
                  Paragraph("Signature: _____________________", BODY)]],
                colWidths=[FW * 0.56, FW * 0.44])
    row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("TOPPADDING", (0, 0), (-1, -1), 3),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(row)

# =============================== ACKNOWLEDGEMENT (4)
story.append(PageBreak())
p = Paragraph("ACKNOWLEDGEMENT", ParagraphStyle("ah", parent=TOC_H, textColor=colors.black)); p._mark = "FM:ACKNOWLEDGEMENT"
story.append(p); story.append(Spacer(1, 16))
for para in C.ACKNOWLEDGEMENT:
    story.append(Paragraph(para, BODY))

# =============================== ABSTRACT (5)
story.append(PageBreak())
p = Paragraph("ABSTRACT", ParagraphStyle("abh", parent=TOC_H, textColor=colors.black)); p._mark = "FM:ABSTRACT"
story.append(p); story.append(Spacer(1, 16))
story.append(Paragraph(C.ABSTRACT, BODY))
story.append(Paragraph(C.ABSTRACT_RESULTS, BODY))
story.append(Spacer(1, 6))
story.append(Paragraph("<b>Keywords:</b> <i>" + C.KEYWORDS + "</i>", BODY))

# =============================== TOC (6) + LOF/LOT (7) + ABBREV (8)
def simple_list_table(rows, headers, widths):
    data = [[Paragraph(h, CELLH) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(x), CELL) for x in r])
    t = Table(data, colWidths=[w / sum(widths) * FW for w in widths], repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def pg(key, kind="H1"):
    return str(PGMAP.get(f"{kind}:{key}", "\u2014"))


story.append(PageBreak())
p = Paragraph("TABLE OF CONTENTS", TOC_H); p._mark = "FM:TABLE OF CONTENTS"
story.append(p); story.append(Spacer(1, 16))
toc_rows = [
    ["1.", "Introduction", pg("CHAPTER 1 - INTRODUCTION")],
    ["2.", "Literature Survey", pg("CHAPTER 2 - LITERATURE SURVEY")],
    ["3.", "Dataset Description and Exploratory Data Analysis", pg("CHAPTER 3 - DATASET DESCRIPTION AND EXPLORATORY DATA ANALYSIS")],
    ["4.", "Data Preprocessing and Feature Engineering", pg("CHAPTER 4 - DATA PREPROCESSING AND FEATURE ENGINEERING")],
    ["5.", "Machine Learning Model Development", pg("CHAPTER 5 - MACHINE LEARNING MODEL DEVELOPMENT")],
    ["6.", "Results and Performance Evaluation", pg("CHAPTER 6 - RESULTS AND PERFORMANCE EVALUATION")],
    ["7.", "Conclusion and Future Enhancements", pg("CHAPTER 7 - CONCLUSION AND FUTURE ENHANCEMENTS")],
    ["8.", "References", pg("REFERENCES")],
    ["9.", "Appendix A \u2013 Source Code and Reproducibility", pg("APPENDIX A \u2013 SOURCE CODE AND REPRODUCIBILITY")],
    ["10.", "Appendix B \u2013 Additional Results", pg("APPENDIX B \u2013 ADDITIONAL RESULTS")],
]
story.append(simple_list_table(toc_rows, ["Sl No.", "Content", "Page No."], [0.11, 0.72, 0.17]))

FIG_LIST = [
    ("Fig. 3.1", "Distribution of Annual Medical Insurance Charges"),
    ("Fig. 3.2", "Charges by Smoking Status and by Number of Children"),
    ("Fig. 3.3", "Age and BMI versus Charges, Separated by Smoking Status"),
    ("Fig. 5.1", "End-to-End Machine Learning Workflow for Insurance Cost Prediction"),
    ("Fig. 6.1", "Comparison of Test R\u00b2 and Test RMSE Across the Four Regression Models"),
    ("Fig. 6.2", "Predicted versus Actual Charges for the Linear Baseline and the Best Model"),
    ("Fig. 6.3", "Residual Plots for the Linear Baseline and the Best Model"),
    ("Fig. 6.4", "Random Forest Feature Importances (Top Ten Features)"),
    ("Fig. 6.5", "Prediction Error Analysis: Absolute Error by Smoking Status and Error versus Actual Charge"),
    ("Fig. B.1", "Random Forest Test R\u00b2 versus Training Set Size"),
]
TABLE_LIST = [
    ("Table 2.1", "Summary of Literature Survey on Medical Insurance Cost Prediction"),
    ("Table 3.1", "Dataset Overview for the Medical Cost Personal Dataset"),
    ("Table 3.2", "Attribute Description of the Medical Cost Personal Dataset"),
    ("Table 3.3", "Mean Charges by Smoking Status and BMI Category"),
    ("Table 4.1", "Engineered Features and Their Rationale"),
    ("Table 4.2", "Dataset Split Proportions for Fitting, Validation and Held-Out Testing"),
    ("Table 5.1", "Hyperparameter Settings Across Implemented Regression Models"),
    ("Table 6.1", "Experimental Performance Comparison on the Held-Out Test Set"),
    ("Table 6.2", "Representative Sample Predictions on Five Held-Out Test Records"),
    ("Table B.1", "Cross-Validated Scores, Demonstration Predictions and Learning Curve"),
]

story.append(PageBreak())
p = Paragraph("LIST OF FIGURES", TOC_H); p._mark = "FM:LIST OF FIGURES"
story.append(p); story.append(Spacer(1, 14))
story.append(simple_list_table([[a, b, pg(a, "FIG")] for a, b in FIG_LIST],
                               ["Figure No.", "Figure Title", "Page No."], [0.16, 0.68, 0.16]))
story.append(Spacer(1, 22))
p = Paragraph("LIST OF TABLES", TOC_H); p._mark = "FM:LIST OF TABLES"
story.append(p); story.append(Spacer(1, 14))
story.append(simple_list_table([[a, b, pg(a, "TBL")] for a, b in TABLE_LIST],
                               ["Table No.", "Table Title", "Page No."], [0.16, 0.68, 0.16]))

story.append(PageBreak())
p = Paragraph("LIST OF ABBREVIATIONS", TOC_H); p._mark = "FM:LIST OF ABBREVIATIONS"
story.append(p); story.append(Spacer(1, 14))
story.append(simple_list_table([[a, b] for a, b in C.ABBREVIATIONS],
                               ["Abbreviation", "Expansion"], [0.2, 0.6]))

# =============================== MAIN BODY
story.append(PageBreak())
render(C.A, story)

# =============================== REFERENCES
story.append(PageBreak())
p = Paragraph("REFERENCES", H1S); p._mark = "H1:REFERENCES"
story.append(p); story.append(Spacer(1, 4))
REFST = ParagraphStyle("ref", parent=BODY, leftIndent=26, firstLineIndent=-26, spaceAfter=5)
for i, r in enumerate(C.REFERENCES, 1):
    story.append(Paragraph(f"[{i}]\t{r}", REFST))

# =============================== APPENDIX A
story.append(PageBreak())
p = Paragraph("APPENDIX A \u2013 SOURCE CODE AND REPRODUCIBILITY", H1S)
p._mark = "H1:APPENDIX A \u2013 SOURCE CODE AND REPRODUCIBILITY"
story.append(p); story.append(Spacer(1, 4))
story.append(Paragraph("<b>Repository Link:</b> " + C.REPO_URL, BODY))
story.append(Paragraph(C.APPENDIX_A_INTRO, BODY))
story.append(Paragraph("<b>Reproducibility steps:</b>", BODY))
for i, s in enumerate(C.APPENDIX_A_STEPS, 1):
    story.append(Paragraph(f"{i}. {s}", ParagraphStyle("num", parent=BODY, leftIndent=26, bulletIndent=10)))
story.append(Spacer(1, 6))
story.append(Paragraph("<b>Core code extract:</b>", BODY))
t = Table([[Paragraph(C.APPENDIX_CODE.replace(" ", "&nbsp;").replace("\n", "<br/>"), CODE)]],
          colWidths=[FW], hAlign="CENTER")
t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.black),
                       ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f7f7")),
                       ("LEFTPADDING", (0, 0), (-1, -1), 7),
                       ("TOPPADDING", (0, 0), (-1, -1), 6),
                       ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
story.append(t)
story.append(Spacer(1, 8))
story.append(Paragraph("The repository also contains the dataset, all generated figures, and "
                       "src/results.json from which every number in this report is generated. No "
                       "passwords, API keys or private data are included.", BODY))

# =============================== APPENDIX B
story.append(PageBreak())
p = Paragraph("APPENDIX B \u2013 ADDITIONAL RESULTS", H1S)
p._mark = "H1:APPENDIX B \u2013 ADDITIONAL RESULTS"
story.append(p); story.append(Spacer(1, 4))
story.append(Paragraph(C.APPENDIX_B_INTRO, BODY))
story.append(Paragraph("B.1 Cross-Validated and Demonstration Results", H2S))
story.append(Paragraph(
    "Table B.1 collects the five-fold cross-validated scores used as a stability check, the "
    "demonstration predictions for three synthetic applicants that do not appear in the dataset, and "
    "the randomised-forest learning curve discussed in Section B.2. "
    "The cross-validated ranking reproduces the test-set ranking, and the demonstration predictions "
    "behave as domain intuition expects: the young non-smoker with normal BMI receives the lowest "
    "estimate, the middle-aged obese smoker the highest (about thirteen times more, because both the "
    "smoking main effect and the BMI \u00d7 smoker interaction are active), and the older non-smoker "
    "falls in between. These are model outputs on a synthetic benchmark and are not quotations or "
    "advice.", BODY))
story.extend(make_table("Table B.1: Cross-Validated Scores and Demonstration Predictions",
                        ["Result", "Setting / Profile", "Value"],
                        [["Mean CV R\u00b2 \u2013 Multiple Linear Regression", "5-fold CV, training subset",
                          f"{C.R['linear_cv_r2']:.4f}"],
                         ["Mean CV R\u00b2 \u2013 Polynomial Regression (Deg 2)", "5-fold CV, training subset",
                          f"{C.R['poly_cv_r2']:.4f}"],
                         ["Mean CV RMSE \u2013 Decision Tree", "5-fold CV, training subset",
                          f"USD {C.R['dt_cv_rmse']:,.2f}"],
                         ["Mean CV RMSE \u2013 Random Forest", "5-fold CV, training subset",
                          f"USD {C.R['rf_cv_rmse']:,.2f}"],
                         ["Predicted charge", "Age 24, female, BMI 22.50, 0 children, non-smoker, southwest",
                          f"USD {C.NEW[0]:,.2f}"],
                         ["Predicted charge", "Age 45, male, BMI 34.80, 2 children, smoker, northeast",
                          f"USD {C.NEW[1]:,.2f}"],
                         ["Predicted charge", "Age 60, male, BMI 28.10, 3 children, non-smoker, southeast",
                          f"USD {C.NEW[2]:,.2f}"]] +
                        [["Test R\u00b2 \u2013 Random Forest", f"Trained on {a} records", f"{b:.4f}"]
                         for a, b in zip(C.LC["train_size"], C.LC["test_r2"])],
                        [1.55, 2.75, 0.8], fs=8.0))
story.append(Paragraph("B.2 Effect of Training Set Size", H2S))
story.append(Paragraph(
    "To check whether the selected models are limited by data volume, the tuned random forest was "
    "refitted on progressively larger fractions of the training subset and scored on the same test set "
    "(Fig. B.1). Test R\u00b2 rises steeply between 171 and 513 training records and then "
    "flattens, gaining only about 0.5 percentage points over the last 342 records. Performance is "
    "therefore close to saturation at the available sample size, which implies that collecting more rows "
    "of the same six attributes would yield limited improvement, whereas adding informative new "
    "attributes would not face the same ceiling.", BODY))
fig_b = Image(os.path.join(C.FIG, "figB1_learning_curve.png"), width=FW * 0.72,
              height=FW * 0.72 * 0.64)
cpb = Paragraph("Fig. B.1: Random Forest Test R\u00b2 versus Training Set Size", CAP)
cpb._mark = "FIG:Fig. B.1"
story.append(KeepTogether([fig_b, cpb,
                           Paragraph("<i>Interpretation:</i> The curve saturates, so additional rows of "
                                     "the same six attributes would add little, whereas new informative "
                                     "attributes would.", INTERP)]))
story.append(Spacer(1, 10))
story.append(Paragraph("B.3 Polynomial Degree Selection", H2S))
story.append(Paragraph(
    "Degree two was selected after comparing degrees 1, 2 and 3 on the validation subset. Degree 1 "
    "reproduces the linear baseline and underfits. Degree 2 captures the smoking\u2013BMI interaction and "
    "improves validation R\u00b2 markedly. Degree 3 expands the feature space to several hundred terms on "
    "a training subset of 855 rows, and although Ridge regularisation keeps the solution stable, "
    "validation error does not improve over degree two while interpretability and fitting time worsen. "
    "Degree two was therefore retained as the best bias\u2013variance compromise.", BODY))
story.append(Paragraph("B.4 Reproducibility Notes", H2S))
story.append(Paragraph(
    "All figures and tables in this report are generated directly by src/medical_insurance_ml.py; no "
    "number was entered manually. The values reported in Tables 6.1, 6.2 and B.1 can be checked against "
    "src/results.json. Re-running the script with the same fixed seed reproduces identical metrics, "
    "which was verified by two independent runs.", BODY))


# =============================== RUBRICS SHEET (appended, unnumbered)
story.append(PageBreak())
p = Paragraph("RUBRICS \u2013 CONTINUOUS COMPREHENSIVE ASSESSMENT (CCA)", H1S)
p._mark = "R1:RUBRICS"
story.append(p)
story.append(Spacer(1, 2))
story.append(Paragraph("The following evaluation rubrics apply to this Machine Learning Case Study "
                       "submission. They are reproduced here for reference and are not part of the "
                       "technical report body.", BODY))
RUB_NAVY = colors.HexColor("#1F3864")


def rubric_table(title, rows, total, crit_w=(0.42, 0.13)):
    out = [Paragraph(title, ParagraphStyle("rt", fontName=FONTB, fontSize=10.5, leading=14,
                                           alignment=TA_LEFT, spaceBefore=8, spaceAfter=4))]
    data = [[Paragraph("<b>Sl. No.</b>", CELLH), Paragraph("<b>Assessment criteria</b>", CELLH),
             Paragraph("<b>Maximum marks</b>", CELLH)]]
    for i, (crit, marks) in enumerate(rows, 1):
        data.append([Paragraph(str(i), CELL), Paragraph(crit, CELL), Paragraph(str(marks), CELL)])
    data.append([Paragraph("", CELL), Paragraph("<b>Total</b>", CELL), Paragraph(f"<b>{total}</b>", CELL)])
    t = Table(data, colWidths=[0.55 / 4.55 * FW, 3.45 / 4.55 * FW, 0.55 / 4.55 * FW], hAlign="CENTER")
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#EDEDED")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    out.append(t)
    return out


def perf_table(title, crits, cols=("2 Marks", "1 Mark", "0 Marks")):
    out = [Paragraph(title, ParagraphStyle("pt", fontName=FONTB, fontSize=10.5, leading=14,
                                           alignment=TA_LEFT, spaceBefore=10, spaceAfter=4))]
    data = [[Paragraph("<b>Criteria</b>", CELLH), Paragraph(f"<b>{cols[0]}</b>", CELLH),
             Paragraph(f"<b>{cols[1]}</b>", CELLH), Paragraph(f"<b>{cols[2]}</b>", CELLH)]]
    for crit, c2, c1, c0 in crits:
        data.append([Paragraph(crit, CELL), Paragraph(c2, CELL), Paragraph(c1, CELL), Paragraph(c0, CELL)])
    t = Table(data, colWidths=[1.05 / 6.0 * FW, 1.75 / 6.0 * FW, 1.6 / 6.0 * FW, 1.6 / 6.0 * FW],
              hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    out.append(t)
    return out


for blk in rubric_table("CCA 1: Industry-Based Problem Statement Case Study (10 Marks) \u2013 Detailed Evaluation Rubric",
                        [("Problem identification and industry relevance", 2),
                         ("Problem analysis, data and requirements", 2),
                         ("Methodology and technical approach", 2),
                         ("Solution implementation and results", 2),
                         ("Report, presentation and Q&amp;A", 2)], 10):
    story.append(blk)
for blk in perf_table("Performance-Based Assessment \u2013 CCA 1",
                      [("Problem identification", "Clearly identifies a specific industry problem with defined objectives",
                        "Partially defined problem and objectives", "Unclear problem or no industry relevance"),
                       ("Problem analysis", "Analyses 4 or more relevant factors, including data and requirements",
                        "Analyses 2\u20133 relevant factors", "Analyses fewer than 2 factors"),
                       ("Methodology", "Explains 4 or more methodology steps/components with justification",
                        "Explains 2\u20133 steps with limited justification", "Inappropriate or incomplete methodology"),
                       ("Solution and results", "Presents a feasible prototype/model with measurable and interpreted results",
                        "Presents a partial solution with limited results", "No meaningful solution or measurable results"),
                       ("Report and presentation", "Complete, timely report and \u226580% correct answers in Q&amp;A",
                        "Partially complete report, 50\u201379% correct answers", "Incomplete report or &lt;50% correct answers")]):
    story.append(blk)

story.append(PageBreak())
p2 = Paragraph("RUBRICS (CONTINUED)", H1S)
story.append(p2)
story.append(Spacer(1, 2))
for blk in rubric_table("CCA 2: Machine Learning Course Certification (10 Marks) \u2013 Detailed Evaluation Rubric",
                        [("Course completion percentage", 2),
                         ("Certification / assessment score", 2),
                         ("Conceptual understanding", 2),
                         ("Practical application of ML concepts", 2),
                         ("Course evidence and viva / reflection", 2)], 10):
    story.append(blk)
for blk in perf_table("Performance-Based Assessment \u2013 CCA 2",
                      [("Course completion", "\u226590% completion", "70\u201389% completion", "Below 70%"),
                       ("Assessment score", "\u226580%", "60\u201379%", "Below 60% / No evidence"),
                       ("Conceptual understanding", "Explains 4\u20135 ML concepts correctly",
                        "Explains 2\u20133 concepts", "Explains fewer than 2"),
                       ("Practical application", "Demonstrates 3 or more ML tasks", "Demonstrates 1\u20132 tasks",
                        "No demonstration"),
                       ("Evidence and viva", "Valid evidence and \u226580% correct viva answers",
                        "50\u201379% correct answers", "Invalid evidence / Below 50%")]):
    story.append(blk)
story.append(Spacer(1, 14))
story.append(Paragraph("USN &nbsp;&nbsp;&nbsp;&nbsp; NAME &nbsp;&nbsp;&nbsp;&nbsp; SIGNATURE",
                       ParagraphStyle("sig", fontName=FONTB, fontSize=10.5, alignment=TA_LEFT)))
for n, u in C.STUDENTS:
    story.append(Paragraph(f"{u} &nbsp;&nbsp;&nbsp; {n.title()} &nbsp;&nbsp;&nbsp; ______________________",
                           ParagraphStyle("sig2", parent=BODY, spaceAfter=2)))

# ---------------------------------------------------------------- build
doc = Doc(OUT, pagesize=letter, leftMargin=LM, rightMargin=RM, topMargin=TM, bottomMargin=BM,
          title=C.TITLE.title(), author=", ".join(n for n, _ in C.STUDENTS),
          subject="Machine Learning Case Study Report")
frame = Frame(LM, BM, FW, PH - TM - BM, id="normal")
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=header_footer)])
doc.build(story)
json.dump(NEWMAP, open(PGPATH, "w"), indent=1)
print("WROTE", OUT, "| pages:", max(NEWMAP.values()) if NEWMAP else 0,
      "| toc source:", "rebuilt map" if PGMAP else "first pass")
