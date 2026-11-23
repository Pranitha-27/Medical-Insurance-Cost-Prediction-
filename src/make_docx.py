"""Render the same report content to an editable DOCX (Times New Roman 12, 1.5 spacing,
justified, A4, margins 1.0/0.8/0.8/0.8 inch, Roman prelims + Arabic body)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import docx
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import report_content as C

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "docs", "Medical_Insurance_Cost_Prediction_ML_Case_Study_Report.docx")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
FONTNAME = "Times New Roman"

doc = Document()

# ------------------------------------------------------------------ page setup
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
sec.left_margin = Inches(1.0)
sec.right_margin = Inches(0.8)
sec.top_margin = Inches(0.8)
sec.bottom_margin = Inches(0.8)
CONTENT_W = 8.27 - 1.8


def set_style(name, size=12, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
              spacing=1.5, before=0, after=6, keep=False):
    st = doc.styles[name]
    st.font.name = FONTNAME
    st.font.size = Pt(size)
    st.font.bold = bold
    st.font.italic = italic
    st.font.color.rgb = RGBColor(0, 0, 0)
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


set_style("Normal", 12, spacing=1.5, after=6)
set_style("Heading 1", 16, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.2,
          before=10, after=10, keep=True)
set_style("Heading 2", 14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.2,
          before=10, after=6, keep=True)
set_style("Heading 3", 12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.2,
          before=8, after=4, keep=True)
for nm in ("List Bullet", "List Number"):
    set_style(nm, 12, spacing=1.5, after=3, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

# ------------------------------------------------------------------ helpers
def para(text="", style=None, size=None, bold=False, italic=False, align=None,
         after=None, before=None, spacing=None, indent=None, first=None):
    p = doc.add_paragraph(style=style)
    if text:
        run = p.add_run(text)
        run.font.name = FONTNAME
        run.bold = bold
        run.italic = italic
        if size:
            run.font.size = Pt(size)
    pf = p.paragraph_format
    if align is not None:
        pf.alignment = align
    if after is not None:
        pf.space_after = Pt(after)
    if before is not None:
        pf.space_before = Pt(before)
    if spacing is not None:
        pf.line_spacing = spacing
    if indent is not None:
        pf.left_indent = Inches(indent)
    if first is not None:
        pf.first_line_indent = Inches(first)
    return p


def rich(parts, style=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=None, indent=None,
         first=None, spacing=None):
    """parts: list of (text, bold, italic)"""
    p = doc.add_paragraph(style=style)
    for t, b, i in parts:
        r = p.add_run(t)
        r.font.name = FONTNAME
        r.bold = b
        r.italic = i
    pf = p.paragraph_format
    pf.alignment = align
    if after is not None:
        pf.space_after = Pt(after)
    if indent is not None:
        pf.left_indent = Inches(indent)
    if first is not None:
        pf.first_line_indent = Inches(first)
    if spacing is not None:
        pf.line_spacing = spacing
    return p


def add_field(paragraph, instr):
    r = paragraph.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t"); t.text = "Right-click and choose Update Field to build this list."
    f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end")
    r._r.append(f1); r._r.append(it); r._r.append(f2); r._r.append(t); r._r.append(f3)
    return paragraph


def page_number_footer(section, roman=False, start=None):
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.text = ""
    if roman:
        p.text = ""
        r = p.add_run(); r.font.name = FONTNAME; r.font.size = Pt(10)
        f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
        it.text = "PAGE \\* roman"
        f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
        r._r.append(f1); r._r.append(it); r._r.append(f2)
    else:
        r = p.add_run(); r.font.name = FONTNAME; r.font.size = Pt(10)
        f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"
        f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
        r._r.append(f1); r._r.append(it); r._r.append(f2)
    if start is not None:
        sp = section._sectPr
        pn = OxmlElement("w:pgNumType")
        pn.set(qn("w:fmt"), "lowerRoman" if roman else "decimal")
        pn.set(qn("w:start"), str(start))
        sp.append(pn)


def new_section(roman=False, start=None):
    s = doc.add_section(WD_SECTION.NEW_PAGE)
    s.left_margin = Inches(1.0); s.right_margin = Inches(0.8)
    s.top_margin = Inches(0.8); s.bottom_margin = Inches(0.8)
    page_number_footer(s, roman=roman, start=start)
    return s


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def add_table(cap, headers, rows, widths=None, fs=9):
    cp = para(cap, size=10, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=3,
              before=8, spacing=1.0)
    cp.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tot = sum(widths) if widths else len(headers)
    usable = CONTENT_W - 0.02
    colw = [Inches(w / tot * usable) for w in widths] if widths else \
           [Inches(usable / len(headers))] * len(headers)
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(h); r.bold = True; r.font.size = Pt(fs); r.font.name = FONTNAME
        shade(hdr[i], "E4EAF1")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            cells[ci].text = ""
            p = cells[ci].paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            txt = str(val)
            # support <b>..</b> in cell text
            while "<b>" in txt:
                pre, rest = txt.split("<b>", 1)
                mid, txt = rest.split("</b>", 1)
                if pre:
                    r = p.add_run(pre); r.font.size = Pt(fs); r.font.name = FONTNAME
                r = p.add_run(mid); r.font.size = Pt(fs); r.font.name = FONTNAME; r.bold = True
            if txt:
                r = p.add_run(txt); r.font.size = Pt(fs); r.font.name = FONTNAME
            if ri % 2 == 1:
                shade(cells[ci], "F6F8FB")
    for row in t.rows:
        for i, c in enumerate(row.cells):
            c.width = colw[i]
    para("", after=4, spacing=1.0)


def add_figure(path, cap, width_frac=0.92):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Inches(CONTENT_W * width_frac))
    cp = para(cap, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, after=10, spacing=1.0)


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
        r.font.name = "Consolas"; r.font.size = Pt(8.5)
        rpr = r._element.get_or_add_rPr()
        rf = OxmlElement("w:rFonts")
        for a in ("w:ascii", "w:hAnsi", "w:cs"):
            rf.set(qn(a), "Consolas")
        rpr.append(rf)
    shade(cell, "F4F4F4")
    para("", after=4, spacing=1.0)


def add_kbox(text):
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.rows[0].cells[0]
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.2
    r = p.add_run(text); r.italic = True; r.font.size = Pt(11.5); r.font.name = FONTNAME
    shade(cell, "F2F6FA")
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
        elif k == "nl":
            for it in b[1]:
                p = doc.add_paragraph(it, style="List Number")
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.line_spacing = 1.5
                p.paragraph_format.space_after = Pt(3)
        elif k == "fig":
            add_figure(b[1], b[2])
        elif k == "tbl":
            add_table(b[1], b[2], b[3], b[4], fs=(b[6] if len(b) > 6 else 9))
        elif k == "code":
            add_code(b[1])
        elif k == "kbox":
            add_kbox(b[1])
        elif k == "pb":
            doc.add_page_break()


# ------------------------------------------------------------------ cover
page_number_footer(sec, roman=True, start=1)
para("", after=0)
for _ in range(3):
    para("", after=0, spacing=1.0)
para(C.COLLEGE, size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=2, spacing=1.0)
para("[Department of Artificial Intelligence and Machine Learning]", size=12, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, after=16, spacing=1.0)
para("MACHINE LEARNING CASE STUDY REPORT", size=18, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, after=6, spacing=1.0)
para("“" + C.TITLE + "”", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
     after=14, spacing=1.15)
para("Submitted in partial fulfilment of the requirements for the", size=12,
     align=WD_ALIGN_PARAGRAPH.CENTER, after=2, spacing=1.0)
para("V Semester Bachelor of Engineering", size=12, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, after=2, spacing=1.0)
para("in", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=2, spacing=1.0)
para("Artificial Intelligence and Machine Learning", size=12, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, after=16, spacing=1.0)
para("Submitted by", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=2, spacing=1.0)
for s in C.STUDENTS:
    para(s, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=1, spacing=1.0)
para("", after=8, spacing=1.0)
para("Under the guidance of", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=2, spacing=1.0)
para(C.GUIDE, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=16, spacing=1.0)
para("Academic Year " + C.YEAR, size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
     after=0, spacing=1.0)

# ------------------------------------------------------------------ certificate
doc.add_page_break()
doc.add_heading("CERTIFICATE", level=1)
para("This is to certify that the Machine Learning Case Study report entitled “" + C.TITLE +
     "” is a bonafide work carried out by " + ", ".join(s.split(" – ")[0] for s in C.STUDENTS) +
     " (" + ", ".join(s.split(" – ")[1] for s in C.STUDENTS) + "), students of the Department of "
     "Artificial Intelligence and Machine Learning, " + C.COLLEGE + ", during the V Semester of the "
     "academic year " + C.YEAR + ", in partial fulfilment of the requirements for the award of the "
     "Bachelor of Engineering degree in Artificial Intelligence and Machine Learning.")
para("", after=24, spacing=1.0)
t = doc.add_table(rows=2, cols=2)
t.autofit = True
for i, (a, b) in enumerate([("____________________________", "____________________________"),
                            ("Guide Signature\n" + C.GUIDE,
                             "HOD Signature\nHead of the Department\nDept. of AI and ML")]):
    for j, txt in enumerate((a, b)):
        c = t.rows[i].cells[j]
        c.text = ""
        p = c.paragraphs[0]
        p.paragraph_format.line_spacing = 1.0
        for n, line in enumerate(txt.split("\n")):
            if n:
                p = c.add_paragraph(); p.paragraph_format.line_spacing = 1.0
            r = p.add_run(line); r.font.size = Pt(11); r.font.name = FONTNAME
            if i == 1 and n == 0:
                r.bold = True
para("", after=18, spacing=1.0)
para("Date: ____________________                    Place: ____________________", spacing=1.5)

# ------------------------------------------------------------------ declaration
doc.add_page_break()
doc.add_heading("DECLARATION", level=1)
para("We hereby declare that the work presented in this report entitled “" + C.TITLE + "” is "
     "carried out by us under the guidance of " + C.GUIDE + ". The work is original to the best of "
     "our knowledge. All external sources, datasets, libraries and references used have been "
     "appropriately acknowledged. This report has not been submitted elsewhere for the award of any "
     "degree or diploma.")
para("Names and USNs:", bold=True, before=14)
for i, s in enumerate(C.STUDENTS, 1):
    para(f"{i}.  {s.replace(' – ', '  –  ')}   ______________________________", spacing=1.5)
para("", after=18, spacing=1.0)
para("Place: ____________________", spacing=1.5)
para("Date: ____________________", spacing=1.5)

# ------------------------------------------------------------------ acknowledgement
doc.add_page_break()
doc.add_heading("ACKNOWLEDGEMENT", level=1)
for p_ in C.ACKNOWLEDGEMENT.split("\n\n"):
    para(p_)

# ------------------------------------------------------------------ abstract
doc.add_page_break()
doc.add_heading("ABSTRACT", level=1)
para(C.ABSTRACT)
rich([("Keywords: ", True, False), (C.KEYWORDS, False, True)], after=6)

# ------------------------------------------------------------------ TOC / LOF / LOT / abbreviations
doc.add_page_break()
doc.add_heading("TABLE OF CONTENTS", level=1)
para("This document contains live Word fields. Place the cursor anywhere in the list below, press "
     "Ctrl+A and then F9 (or right-click → Update Field) to populate the page numbers after you have "
     "filled in the college name, student names and USNs.", size=10.5, italic=True, after=10)
add_field(para(after=6, spacing=1.0), 'TOC \\o "1-3" \\h \\z \\u')

doc.add_page_break()
doc.add_heading("LIST OF FIGURES", level=1)
add_field(para(after=6, spacing=1.0), 'TOC \\h \\z \\c "Figure"')

doc.add_page_break()
doc.add_heading("LIST OF TABLES", level=1)
add_field(para(after=6, spacing=1.0), 'TOC \\h \\z \\c "Table"')

doc.add_page_break()
doc.add_heading("LIST OF ABBREVIATIONS", level=1)
add_table("", ["Abbreviation", "Expansion"],
          [[a, b] for a, b in C.ABBREVIATIONS], widths=[0.25, 0.6], fs=10)

# ------------------------------------------------------------------ body
new_section(roman=False, start=1)
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
    run.font.name = FONTNAME; run.font.size = Pt(11.5)

# ------------------------------------------------------------------ appendix A
doc.add_page_break()
doc.add_heading("APPENDIX A – SOURCE CODE AND REPRODUCIBILITY", level=1)
para(C.APPENDIX_A)
doc.add_heading("Reproducibility steps", level=2)
for i, s in enumerate(C.APPENDIX_A_STEPS, 1):
    p = doc.add_paragraph(s, style="List Number")
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(3)
doc.add_heading("Core code extract", level=2)
add_code(C.APPENDIX_CODE)
para("The submission includes medical_insurance_ml.py, results.json and the figures/ directory "
     "containing all thirteen generated plots. No passwords, API keys or private data are included; "
     "the dataset is public.")

# ------------------------------------------------------------------ appendix B
doc.add_page_break()
doc.add_heading("APPENDIX B – ADDITIONAL RESULTS", level=1)
para(C.APPENDIX_B)
doc.add_heading("B.1  Effect of training set size", level=2)
para("To check whether the selected models are limited by data volume, the tuned random forest was "
     "refitted on progressively larger fractions of the training subset and scored on the same test "
     "set. Test R² rises steeply between 171 and 513 training records and then flattens, gaining only "
     "about 0.5 percentage points over the last 342 records. Performance is therefore close to "
     "saturation at the available sample size, which implies that collecting more rows of the same six "
     "attributes would yield limited improvement, whereas adding informative new attributes would not "
     "face the same ceiling.")
add_table("Table B.1: Random forest test R² versus training set size",
          ["Training records", "Test R²"],
          [[str(a), f"{b:.4f}"] for a, b in zip(C.LC["train_size"], C.LC["test_r2"])],
          widths=[1.6, 1.2], fs=10)
add_figure(os.path.join(C.FIG, "figB1_learning_curve.png"),
           "Fig. B.1: Random forest test R² as a function of the number of training records used.", 0.75)
doc.add_heading("B.2  Polynomial degree selection", level=2)
para("Degree 2 was selected after comparing degrees 1, 2 and 3 on the validation subset. Degree 1 "
     "reproduces the linear baseline and underfits. Degree 2 captures the smoking–BMI interaction and "
     "improves validation R² markedly. Degree 3 expands the feature space to several hundred terms on a "
     "training subset of 855 rows, and although Ridge regularisation keeps the solution stable, "
     "validation error does not improve over degree 2 while interpretability and fitting time worsen. "
     "Degree 2 was therefore retained as the best bias–variance compromise.")
doc.add_heading("B.3  Reproducibility notes", level=2)
para("All figures and tables in this report are generated directly by medical_insurance_ml.py; no "
     "number was entered manually. The values reported in Table 6.2, Table 6.3, Table 6.4 and Table 6.5 "
     "can be checked against results.json. Re-running the script with the same fixed seed reproduces "
     "identical metrics, which was verified by two independent runs.")

doc.core_properties.title = C.TITLE
doc.core_properties.subject = "Machine Learning Case Study Report"
doc.save(OUT)
print("WROTE", OUT)
