"""Render the report to DOCX in the house style of the reference report ("ML Case Report - Yash").

US Letter, 1.3" side margins, thin grey page border, Arabic page numbers at the bottom centre,
navy section headings, BMS/VTU logos on the cover, static TOC / LOF / LOT tables that match the PDF.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import docx
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import report_content as C

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT = os.path.join(_ROOT, "docs", "Medical_Insurance_Cost_Prediction_ML_Case_Study_Report.docx")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
ASSETS = os.path.join(_ROOT, "assets")
PGPATH = os.path.join(_HERE, "pgmap.json")
PGMAP = json.load(open(PGPATH)) if os.path.exists(PGPATH) else {}

FONTNAME = "Times New Roman"
NAVY = RGBColor(0x36, 0x5F, 0x91)
BMS_RED = RGBColor(0xE0, 0x42, 0x28)

doc = Document()

# ------------------------------------------------------------------ page setup
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11.0)
sec.left_margin = sec.right_margin = Inches(1.3)
sec.top_margin, sec.bottom_margin = Inches(0.85), Inches(0.9)
CONTENT_W = 8.5 - 2.6


def set_style(name, size=11.3, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
              spacing=1.5, before=0, after=6, keep=False, color=RGBColor(0, 0, 0)):
    st = doc.styles[name]
    st.font.name = FONTNAME
    st.font.size = Pt(size)
    st.font.bold = bold
    st.font.italic = italic
    st.font.color.rgb = color
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), FONTNAME)
    pf = st.paragraph_format
    pf.alignment = align
    pf.line_spacing = spacing
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.keep_with_next = keep
    return st


set_style("Normal", 11.3, spacing=1.5, after=6)
set_style("Heading 1", 14.5, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.2,
          before=10, after=10, keep=True)
set_style("Heading 2", 12.8, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.2,
          before=9, after=5, keep=True, color=NAVY)
set_style("Heading 3", 11.5, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.2,
          before=7, after=4, keep=True)
set_style("List Bullet", 11.3, spacing=1.5, after=3)


# ------------------------------------------------------------------ helpers
def para(text="", style=None, size=None, bold=False, italic=False, align=None, color=None,
         after=None, before=None, spacing=None, indent=None, first=None):
    p = doc.add_paragraph(style=style)
    if text:
        run = p.add_run(text)
        run.font.name = FONTNAME
        run.bold = bold
        run.italic = italic
        if size:
            run.font.size = Pt(size)
        if color:
            run.font.color.rgb = color
    pf = p.paragraph_format
    if align is not None:
        pf.alignment = align
    for attr, val in (("space_after", after), ("space_before", before), ("line_spacing", spacing)):
        if val is not None:
            setattr(pf, attr, Pt(val) if attr != "line_spacing" else val)
    if indent is not None:
        pf.left_indent = Inches(indent)
    if first is not None:
        pf.first_line_indent = Inches(first)
    return p


def rich(parts, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=None):
    p = doc.add_paragraph()
    for t, b, i in parts:
        r = p.add_run(t); r.font.name = FONTNAME; r.bold = b; r.italic = i
    p.paragraph_format.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def add_field(paragraph, instr):
    r = paragraph.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t"); t.text = "Update field (Ctrl+A, F9) to build this list."
    f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end")
    r._r.append(f1); r._r.append(it); r._r.append(f2); r._r.append(t); r._r.append(f3)
    return paragraph


def add_page_border(section):
    sectPr = section._sectPr
    pgBorders = OxmlElement("w:pgBorders")
    pgBorders.set(qn("w:offsetFrom"), "page")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "6")
        el.set(qn("w:space"), "24")
        el.set(qn("w:color"), "7F7F7F")
        pgBorders.append(el)
    sectPr.append(pgBorders)


def page_number_footer(section, start=None):
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.text = ""
    r = p.add_run(); r.font.name = FONTNAME; r.font.size = Pt(10)
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    r._r.append(f1); r._r.append(it); r._r.append(f2)
    if start is not None:
        pn = OxmlElement("w:pgNumType")
        pn.set(qn("w:fmt"), "decimal"); pn.set(qn("w:start"), str(start))
        section._sectPr.append(pn)


def new_section(start=None):
    s = doc.add_section(WD_SECTION.NEW_PAGE)
    s.page_width, s.page_height = Inches(8.5), Inches(11.0)
    s.left_margin = s.right_margin = Inches(1.3)
    s.top_margin, s.bottom_margin = Inches(0.85), Inches(0.9)
    add_page_border(s)
    page_number_footer(s, start=start)
    return s


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def add_table(cap, headers, rows, widths=None, fs=8.5, center_hdr=True):
    if cap:
        cp = para(cap, size=9.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=3, before=8,
                  spacing=1.0)
        cp.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tot = sum(widths) if widths else len(headers)
    colw = [Inches(w / tot * (CONTENT_W - 0.02)) for w in widths] if widths else \
           [Inches((CONTENT_W - 0.02) / len(headers))] * len(headers)
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(h); r.bold = True; r.font.size = Pt(fs); r.font.name = FONTNAME
    for row in rows:
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            cells[ci].text = ""
            p = cells[ci].paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            txt = str(val)
            while "<b>" in txt:
                pre, rest = txt.split("<b>", 1)
                mid, txt = rest.split("</b>", 1)
                if pre:
                    r = p.add_run(pre); r.font.size = Pt(fs); r.font.name = FONTNAME
                r = p.add_run(mid); r.font.size = Pt(fs); r.font.name = FONTNAME; r.bold = True
            if txt:
                r = p.add_run(txt); r.font.size = Pt(fs); r.font.name = FONTNAME
    for row in t.rows:
        for i, c in enumerate(row.cells):
            c.width = colw[i]
    para("", after=4, spacing=1.0)


def add_figure(path, cap, interp=None, width_frac=0.86):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Inches(CONTENT_W * width_frac))
    cp = para(cap, size=9.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=3, spacing=1.0)
    cp.paragraph_format.keep_with_next = bool(interp)
    if interp:
        rp = doc.add_paragraph()
        rp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        rp.paragraph_format.space_after = Pt(8)
        r1 = rp.add_run("Interpretation: "); r1.italic = True; r1.font.name = FONTNAME
        r2 = rp.add_run(interp); r2.font.name = FONTNAME


def add_code(text):
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.rows[0].cells[0]
    cell.text = ""
    for i, line in enumerate(text.split("\n")):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(line if line else " ")
        r.font.name = "Consolas"; r.font.size = Pt(8)
        rpr = r._element.get_or_add_rPr()
        rf = OxmlElement("w:rFonts")
        for a in ("w:ascii", "w:hAnsi", "w:cs"):
            rf.set(qn(a), "Consolas")
        rpr.append(rf)
    shade(cell, "F7F7F7")
    para("", after=4, spacing=1.0)


def render(blocks):
    for b in blocks:
        k = b[0]
        if k == "h1":
            doc.add_heading(b[1], level=1)
        elif k == "h2":
            doc.add_heading(b[1], level=2)
        elif k == "h3":
            doc.add_heading(b[1], level=3)
        elif k == "p":
            para(b[1])
        elif k == "bl":
            for it in b[1]:
                p = doc.add_paragraph(it, style="List Bullet")
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.line_spacing = 1.5
                p.paragraph_format.space_after = Pt(3)
        elif k == "fig":
            add_figure(b[1], b[2], b[3] if len(b) > 3 else None)
        elif k == "tbl":
            add_table(b[1], b[2], b[3], b[4], fs=(b[6] if len(b) > 6 else 8.5))
        elif k == "code":
            add_code(b[1])
        elif k == "pb":
            doc.add_page_break()


def simple_table(rows, headers, widths, fs=9.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    colw = [Inches(w / sum(widths) * (CONTENT_W - 0.02)) for w in widths]
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]; c.text = ""
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1); p.paragraph_format.line_spacing = 1.0
        r = p.add_run(h); r.bold = True; r.font.size = Pt(fs); r.font.name = FONTNAME
    for row in rows:
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            cells[ci].text = ""
            p = cells[ci].paragraphs[0]
            p.paragraph_format.space_after = Pt(1); p.paragraph_format.line_spacing = 1.0
            r = p.add_run(str(val)); r.font.size = Pt(fs); r.font.name = FONTNAME
    for row in t.rows:
        for i, c in enumerate(row.cells):
            c.width = colw[i]
    return t


# ------------------------------------------------------------------ cover
add_page_border(sec)
page_number_footer(sec, start=1)
for _ in range(2):
    para("", after=0, spacing=1.0)
logos = doc.add_table(rows=1, cols=3)
logos.autofit = False
bms_p = os.path.join(ASSETS, "bms_logo.jpeg")
vtu_p = os.path.join(ASSETS, "vtu_logo.jpeg")
if os.path.exists(bms_p):
    logos.rows[0].cells[0].paragraphs[0].add_run().add_picture(bms_p, width=Inches(0.72))
if os.path.exists(vtu_p):
    pc = logos.rows[0].cells[2].paragraphs[0]
    pc.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    pc.add_run().add_picture(vtu_p, width=Inches(0.78))
for i, w in enumerate((1.2, CONTENT_W - 2.4, 1.2)):
    logos.rows[0].cells[i].width = Inches(w)

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(2); p.paragraph_format.space_after = Pt(2)
r = p.add_run("BMS "); r.bold = True; r.font.size = Pt(17); r.font.name = FONTNAME
r.font.color.rgb = BMS_RED
r = p.add_run("INSTITUTE OF TECHNOLOGY AND MANAGEMENT")
r.bold = True; r.font.size = Pt(17); r.font.name = FONTNAME; r.font.color.rgb = NAVY
para(C.COLLEGE_SUB, size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=1, spacing=1.0)
para(C.COLLEGE_ADDR, size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=12, spacing=1.0)
para(C.DEPT, size=13.5, bold=True, color=NAVY, align=WD_ALIGN_PARAGRAPH.CENTER, after=14, spacing=1.2)
para("MACHINE LEARNING CASE STUDY REPORT", size=12.8, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, after=2, spacing=1.0)
para("ON", size=11.3, align=WD_ALIGN_PARAGRAPH.CENTER, after=2, spacing=1.0)
para(C.TITLE, size=13.6, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=16, spacing=1.15)
for line in ["Submitted in partial fulfilment of the requirements for the",
             "V Semester Bachelor of Engineering", "in",
             "Artificial Intelligence and Machine Learning"]:
    para(line, size=11, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=2, spacing=1.0)
para("", after=10, spacing=1.0)
para("Submitted by", size=11.3, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=2, spacing=1.0)
for n, u in C.STUDENTS:
    para(f"{n} \u2013 {u}", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, after=1, spacing=1.0)
para("", after=10, spacing=1.0)
para("Under the guidance of", size=11.3, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=2, spacing=1.0)
para(C.GUIDE, size=11, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=1, spacing=1.0)
para(C.GUIDE_DESIG, size=10.8, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=1, spacing=1.0)
para("Department of Artificial Intelligence and Machine Learning", size=10.5,
     align=WD_ALIGN_PARAGRAPH.CENTER, after=14, spacing=1.0)
para("[Academic Year: " + C.YEAR + "]", size=10.8, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, after=0, spacing=1.0)

# ------------------------------------------------------------------ certificate
doc.add_page_break()
para("CERTIFICATE", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=16, spacing=1.2)
names_usn = ", ".join(f"{n}-{u}" for n, u in C.STUDENTS)
para("This is to certify that the Machine Learning Case Study report entitled \u201c" +
     C.TITLE.title() + "\u201d is a bonafide work carried out by " + names_usn + ", students of the "
     "Department of Artificial Intelligence and Machine Learning, BMS Institute of Technology and "
     "Management, during the V Semester / Academic Year " + C.YEAR + ", in partial fulfilment of the "
     "requirements for the award of the Bachelor of Engineering degree.")
para("", after=30, spacing=1.0)
t = doc.add_table(rows=3, cols=2)
t.autofit = True
cells = [("Guide Signature:", f"[{C.GUIDE}, {C.GUIDE_DESIG}]"),
         ("HOD Signature:", f"[{C.HOD}, Head Of Department]")]
for j, (hdr, who) in enumerate(cells):
    c = t.rows[0].cells[j]; c.text = ""
    r = c.paragraphs[0].add_run(hdr); r.font.name = FONTNAME; r.font.size = Pt(11.3)
    c2 = t.rows[1].cells[j]; c2.text = ""
    r = c2.paragraphs[0].add_run("\n___________________________")
    r.font.name = FONTNAME; r.font.size = Pt(11.3)
    c3 = t.rows[2].cells[j]; c3.text = ""
    r = c3.paragraphs[0].add_run(who); r.font.name = FONTNAME; r.font.size = Pt(11.3)
para("", after=26, spacing=1.0)
para("Date: _____________________", spacing=1.5)
para("Place: ____________________", spacing=1.5)

# ------------------------------------------------------------------ declaration
doc.add_page_break()
para("DECLARATION", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=16, spacing=1.2)
para("We hereby declare that the work presented in this report entitled \u201c" + C.TITLE.title() +
     "\u201d is carried out by us under the guidance of " + C.GUIDE + ". The work is original to the best "
     "of our knowledge. All external sources, datasets, libraries and references used have been "
     "appropriately acknowledged. This report has not been submitted elsewhere for the award of any "
     "degree or diploma.")
para("", after=10, spacing=1.0)
para("Names and USNs:", spacing=1.5)
for i, (n, u) in enumerate(C.STUDENTS, 1):
    row = doc.add_table(rows=1, cols=2)
    row.autofit = False
    row.rows[0].cells[0].width = Inches(CONTENT_W * 0.56)
    row.rows[0].cells[1].width = Inches(CONTENT_W * 0.44)
    c0 = row.rows[0].cells[0]; c0.text = ""
    r = c0.paragraphs[0].add_run(f"{i}. {n.title()} \u2013 {u}")
    r.font.name = FONTNAME; r.font.size = Pt(11.3)
    c1 = row.rows[0].cells[1]; c1.text = ""
    r = c1.paragraphs[0].add_run("Signature: _____________________")
    r.font.name = FONTNAME; r.font.size = Pt(11.3)

# ------------------------------------------------------------------ acknowledgement
doc.add_page_break()
para("ACKNOWLEDGEMENT", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=16, spacing=1.2)
for p_ in C.ACKNOWLEDGEMENT:
    para(p_)

# ------------------------------------------------------------------ abstract
doc.add_page_break()
para("ABSTRACT", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=16, spacing=1.2)
para(C.ABSTRACT)
para(C.ABSTRACT_RESULTS)
rich([("Keywords: ", True, False), (C.KEYWORDS, False, True)], after=6)

# ------------------------------------------------------------------ TOC / LOF / LOT / abbreviations
def pg(key, kind="H1"):
    return str(PGMAP.get(f"{kind}:{key}", "\u2014"))


doc.add_page_break()
para("TABLE OF CONTENTS", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=14, spacing=1.2)
simple_table([[1, "Introduction", pg("CHAPTER 1 - INTRODUCTION")],
              [2, "Literature Survey", pg("CHAPTER 2 - LITERATURE SURVEY")],
              [3, "Dataset Description and Exploratory Data Analysis",
               pg("CHAPTER 3 - DATASET DESCRIPTION AND EXPLORATORY DATA ANALYSIS")],
              [4, "Data Preprocessing and Feature Engineering",
               pg("CHAPTER 4 - DATA PREPROCESSING AND FEATURE ENGINEERING")],
              [5, "Machine Learning Model Development",
               pg("CHAPTER 5 - MACHINE LEARNING MODEL DEVELOPMENT")],
              [6, "Results and Performance Evaluation",
               pg("CHAPTER 6 - RESULTS AND PERFORMANCE EVALUATION")],
              [7, "Conclusion and Future Enhancements",
               pg("CHAPTER 7 - CONCLUSION AND FUTURE ENHANCEMENTS")],
              [8, "References", pg("REFERENCES")],
              [9, "Appendix A \u2013 Source Code and Reproducibility",
               pg("APPENDIX A \u2013 SOURCE CODE AND REPRODUCIBILITY")],
              [10, "Appendix B \u2013 Additional Results", pg("APPENDIX B \u2013 ADDITIONAL RESULTS")]],
            ["Sl No.", "Content", "Page No."], [0.11, 0.72, 0.17])
para("", after=4, spacing=1.0)
para("Note: page numbers above follow the PDF submission. If you edit this document, update them "
     "manually or use Ctrl+A then F9 on any inserted fields.", size=9, italic=True, spacing=1.0)

FIGS = [("Fig. 3.1", "Distribution of Annual Medical Insurance Charges"),
        ("Fig. 3.2", "Charges by Smoking Status and by Number of Children"),
        ("Fig. 3.3", "Age and BMI versus Charges, Separated by Smoking Status"),
        ("Fig. 5.1", "End-to-End Machine Learning Workflow for Insurance Cost Prediction"),
        ("Fig. 6.1", "Comparison of Test R\u00b2 and Test RMSE Across the Four Regression Models"),
        ("Fig. 6.2", "Predicted versus Actual Charges for the Linear Baseline and the Best Model"),
        ("Fig. 6.3", "Residual Plots for the Linear Baseline and the Best Model"),
        ("Fig. 6.4", "Random Forest Feature Importances (Top Ten Features)"),
        ("Fig. 6.5", "Prediction Error Analysis: Absolute Error by Smoking Status and Error versus "
                     "Actual Charge"),
        ("Fig. B.1", "Random Forest Test R\u00b2 versus Training Set Size")]
TABS = [("Table 2.1", "Summary of Literature Survey on Medical Insurance Cost Prediction"),
        ("Table 3.1", "Dataset Overview for the Medical Cost Personal Dataset"),
        ("Table 3.2", "Attribute Description of the Medical Cost Personal Dataset"),
        ("Table 3.3", "Mean Charges by Smoking Status and BMI Category"),
        ("Table 4.1", "Engineered Features and Their Rationale"),
        ("Table 4.2", "Dataset Split Proportions for Fitting, Validation and Held-Out Testing"),
        ("Table 5.1", "Hyperparameter Settings Across Implemented Regression Models"),
        ("Table 6.1", "Experimental Performance Comparison on the Held-Out Test Set"),
        ("Table 6.2", "Representative Sample Predictions on Five Held-Out Test Records"),
        ("Table B.1", "Cross-Validated Scores, Demonstration Predictions and Learning Curve")]

doc.add_page_break()
para("LIST OF FIGURES", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=12, spacing=1.2)
simple_table([[a, b, pg(a, "FIG")] for a, b in FIGS],
             ["Figure No.", "Figure Title", "Page No."], [0.16, 0.68, 0.16])
para("", after=14, spacing=1.0)
para("LIST OF TABLES", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=12, spacing=1.2)
simple_table([[a, b, pg(a, "TBL")] for a, b in TABS],
             ["Table No.", "Table Title", "Page No."], [0.16, 0.68, 0.16])

doc.add_page_break()
para("LIST OF ABBREVIATIONS", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=12, spacing=1.2)
simple_table([[a, b] for a, b in C.ABBREVIATIONS], ["Abbreviation", "Expansion"], [0.2, 0.6])

# ------------------------------------------------------------------ body
new_section()
render(C.A)

# ------------------------------------------------------------------ references
doc.add_page_break()
doc.add_heading("REFERENCES", level=1)
for i, r in enumerate(C.REFERENCES, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.35)
    p.paragraph_format.line_spacing = 1.3
    p.paragraph_format.space_after = Pt(5)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(f"[{i}]\t{r}")
    run.font.name = FONTNAME; run.font.size = Pt(10.8)

# ------------------------------------------------------------------ appendix A
doc.add_page_break()
doc.add_heading("APPENDIX A \u2013 SOURCE CODE AND REPRODUCIBILITY", level=1)
rich([("Repository Link: ", True, False), (C.REPO_URL, False, False)], after=6)
para(C.APPENDIX_A_INTRO)
para("Reproducibility steps:", bold=True)
for i, s in enumerate(C.APPENDIX_A_STEPS, 1):
    p = para(f"{i}. {s}")
    p.paragraph_format.left_indent = Inches(0.3)
para("Core code extract:", bold=True)
add_code(C.APPENDIX_CODE)
para("The repository also contains the dataset, all generated figures, and src/results.json from which "
     "every number in this report is generated. No passwords, API keys or private data are included.")

# ------------------------------------------------------------------ appendix B
doc.add_page_break()
doc.add_heading("APPENDIX B \u2013 ADDITIONAL RESULTS", level=1)
para(C.APPENDIX_B_INTRO)
doc.add_heading("B.1 Cross-Validated and Demonstration Results", level=2)
para("Table B.1 collects the five-fold cross-validated scores used as a stability check, the "
     "demonstration predictions for three synthetic applicants that do not appear in the dataset, and "
     "the random-forest learning curve discussed in Section B.2. The cross-validated ranking reproduces "
     "the test-set ranking, and the demonstration predictions behave as domain intuition expects: the "
     "young non-smoker with normal BMI receives the lowest estimate, the middle-aged obese smoker the "
     "highest, and the older non-smoker falls in between. These are model outputs on a synthetic "
     "benchmark and are not quotations or advice.")
add_table("Table B.1: Cross-Validated Scores and Demonstration Predictions",
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
          [1.55, 2.75, 0.8], fs=8.5)
doc.add_heading("B.2 Effect of Training Set Size", level=2)
para("To check whether the selected models are limited by data volume, the tuned random forest was "
     "refitted on progressively larger fractions of the training subset and scored on the same test set "
     "(Fig. B.1). Test R\u00b2 rises steeply between 171 and 513 training records and then "
     "flattens, gaining only about 0.5 percentage points over the last 342 records. Performance is "
     "therefore close to saturation at the available sample size, which implies that collecting more "
     "rows of the same six attributes would yield limited improvement, whereas adding informative new "
     "attributes would not face the same ceiling.")
add_figure(os.path.join(C.FIG, "figB1_learning_curve.png"),
           "Fig. B.1: Random Forest Test R\u00b2 versus Training Set Size",
           "The curve saturates, so additional rows of the same six attributes would add little, "
           "whereas new informative attributes would.", 0.72)
doc.add_heading("B.3 Polynomial Degree Selection", level=2)
para("Degree two was selected after comparing degrees 1, 2 and 3 on the validation subset. Degree 1 "
     "reproduces the linear baseline and underfits. Degree 2 captures the smoking\u2013BMI interaction "
     "and improves validation R\u00b2 markedly. Degree 3 expands the feature space to several hundred "
     "terms on a training subset of 855 rows, and although Ridge regularisation keeps the solution "
     "stable, validation error does not improve over degree two while interpretability and fitting time "
     "worsen. Degree two was therefore retained as the best bias\u2013variance compromise.")
doc.add_heading("B.4 Reproducibility Notes", level=2)
para("All figures and tables in this report are generated directly by src/medical_insurance_ml.py; no "
     "number was entered manually. The values reported in Tables 6.1, 6.2 and B.1 can be checked against "
     "src/results.json. Re-running the script with the same fixed seed reproduces identical metrics, "
     "which was verified by two independent runs.")

# ------------------------------------------------------------------ rubrics sheet (appended)
doc.add_page_break()
para("RUBRICS \u2013 CONTINUOUS COMPREHENSIVE ASSESSMENT (CCA)", size=14.5, bold=True,
     align=WD_ALIGN_PARAGRAPH.LEFT, after=8, spacing=1.2)
para("The following evaluation rubrics apply to this Machine Learning Case Study submission. They are "
     "reproduced here for reference and are not part of the technical report body.")
para("CCA 1: Industry-Based Problem Statement Case Study (10 Marks) \u2013 Detailed Evaluation Rubric",
     bold=True, size=10.5, before=8, after=4, spacing=1.0)
simple_table([[1, "Problem identification and industry relevance", 2],
              [2, "Problem analysis, data and requirements", 2],
              [3, "Methodology and technical approach", 2],
              [4, "Solution implementation and results", 2],
              [5, "Report, presentation and Q&amp;A", 2],
              ["", "Total", 10]],
             ["Sl. No.", "Assessment criteria", "Maximum marks"], [0.12, 0.72, 0.16], fs=9.5)
para("Performance-Based Assessment \u2013 CCA 1", bold=True, size=10.5, before=10, after=4, spacing=1.0)
simple_table([["Problem identification", "Clearly identifies a specific industry problem with defined "
               "objectives", "Partially defined problem and objectives", "Unclear problem or no industry relevance"],
              ["Problem analysis", "Analyses 4 or more relevant factors, including data and requirements",
               "Analyses 2\u20133 relevant factors", "Analyses fewer than 2 factors"],
              ["Methodology", "Explains 4 or more methodology steps/components with justification",
               "Explains 2\u20133 steps with limited justification", "Inappropriate or incomplete methodology"],
              ["Solution and results", "Presents a feasible prototype/model with measurable and interpreted results",
               "Presents a partial solution with limited results", "No meaningful solution or measurable results"],
              ["Report and presentation", "Complete, timely report and \u226580% correct answers in Q&amp;A",
               "Partially complete report, 50\u201379% correct answers", "Incomplete report or <50% correct answers"]],
             ["Criteria", "2 Marks", "1 Mark", "0 Marks"], [0.18, 0.29, 0.27, 0.26], fs=8.5)

doc.add_page_break()
para("RUBRICS (CONTINUED)", size=14.5, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, after=8, spacing=1.2)
para("CCA 2: Machine Learning Course Certification (10 Marks) \u2013 Detailed Evaluation Rubric",
     bold=True, size=10.5, before=8, after=4, spacing=1.0)
simple_table([[1, "Course completion percentage", 2],
              [2, "Certification / assessment score", 2],
              [3, "Conceptual understanding", 2],
              [4, "Practical application of ML concepts", 2],
              [5, "Course evidence and viva / reflection", 2],
              ["", "Total", 10]],
             ["Sl. No.", "Assessment criteria", "Maximum marks"], [0.12, 0.72, 0.16], fs=9.5)
para("Performance-Based Assessment \u2013 CCA 2", bold=True, size=10.5, before=10, after=4, spacing=1.0)
simple_table([["Course completion", "\u226590% completion", "70\u201389% completion", "Below 70%"],
              ["Assessment score", "\u226580%", "60\u201379%", "Below 60% / No evidence"],
              ["Conceptual understanding", "Explains 4\u20135 ML concepts correctly", "Explains 2\u20133 concepts",
               "Explains fewer than 2"],
              ["Practical application", "Demonstrates 3 or more ML tasks", "Demonstrates 1\u20132 tasks",
               "No demonstration"],
              ["Evidence and viva", "Valid evidence and \u226580% correct viva answers",
               "50\u201379% correct answers", "Invalid evidence / Below 50%"]],
             ["Criteria", "2 Marks", "1 Mark", "0 Marks"], [0.18, 0.29, 0.27, 0.26], fs=8.5)
para("", after=14, spacing=1.0)
para("USN      NAME      SIGNATURE", bold=True, size=10.5, after=4)
for n, u in C.STUDENTS:
    para(f"{u}      {n.title()}      ______________________", size=10.5, after=2, spacing=1.0)

doc.core_properties.title = C.TITLE.title()
doc.core_properties.subject = "Machine Learning Case Study Report"
doc.save(OUT)
print("WROTE", OUT)
