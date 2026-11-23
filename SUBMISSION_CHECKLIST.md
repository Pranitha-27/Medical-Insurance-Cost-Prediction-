# Before you submit — checklist

Work through this list once. Everything below is a placeholder that must be replaced with your own details.

## 1. Report placeholders (5 minutes)

Open `docs/Medical_Insurance_Cost_Prediction_ML_Case_Study_Report.docx` and use **Ctrl+H → Replace**
(or just search for `[`), then rebuild the PDF — or edit the constants at the top of
`src/report_content.py` and re-run `python src/make_pdf.py` + `python src/make_docx.py`.

| Placeholder | Where | Replace with |
|---|---|---|
| `[COLLEGE NAME]` | cover, certificate | your college name |
| `[Department of Artificial Intelligence and Machine Learning]` | cover | your department name |
| `[Student Name – USN]` ×4 | cover, declaration, certificate | your names and USNs |
| `[Guide Name and Designation]` | cover, certificate, acknowledgement | your guide |
| Academic year `2025–26` | cover | your academic year (default is correct for V Sem 2025–26) |

Constants live at the top of `src/report_content.py`:

```python
COLLEGE = "[COLLEGE NAME]"
GUIDE   = "[Guide Name and Designation]"
STUDENTS = ["[Student Name – USN]", ...]
YEAR    = "2025–26"
ACCESS_DATE = "22 November 2025"   # dataset access date — update if you download on a different day
```

## 2. Signatures

Print the certificate and declaration pages, get the guide's and HOD's signatures, then scan/insert them
(or submit the signed hard copy as your department requires).

## 3. Word-only steps (if you submit the DOCX)

* Update the fields: **Ctrl+A → F9 → "Update entire table"** for the Table of Contents, List of Figures
  and List of Tables. The PDF already has correct page numbers.
* Check the TOC page numbers after any text edit.

## 4. GitHub upload

```bash
cd medical-insurance-cost-prediction
git init
git add .
git commit -m "ML case study: medical insurance cost prediction"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Then on GitHub: **About → description** (`Medical insurance cost prediction using regression models — ML case study`),
add topics `machine-learning`, `regression`, `scikit-learn`, `python`, `insurance`, `case-study`.

### Also update in the repo
* `README.md` → Authors table (names + USNs), guide name, and the clone URL in Quick start.
* `LICENSE` → replace `<Your Names>` with your names.
* Keep the `docs/` PDF so visitors can read the report without downloading Word.

## 5. Submission pack for the department

If they want a PDF + source code, submit:

```
Medical_Insurance_Cost_Prediction_ML_Case_Study_Report.pdf   (from docs/)
medical_insurance_ml.py                                       (from src/)
results.json                                                  (from src/)
figures/                                                      (13 PNGs)
```

Or simply give them the GitHub repo link.

## 6. Final sanity checks

- [ ] College name, student names, USNs, guide name appear correctly on the cover and certificate
- [ ] Certificate and declaration signed
- [ ] TOC / LOF / LOT page numbers match (rebuild after edits)
- [ ] Dataset URL, access date and licence stated in Section 3.1 (already done)
- [ ] All 13 figures are captioned and referenced in the text (already done)
- [ ] Report is 34 main pages — inside the 25–35 suggested range
- [ ] `python src/medical_insurance_ml.py` still reproduces R² = 0.9031 on a fresh machine
- [ ] No passwords, API keys or personal data anywhere in the repo
