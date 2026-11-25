# Before you submit — checklist

Work through this list once. Everything below is a placeholder that must be replaced with your own details.

## 0. Status

College, student names, USNs, guide, department and academic year are **already filled in** in
`src/report_content.py` and therefore in both `docs/` files. What remains is signatures, the
repository URL in Appendix A, and any wording your guide asks for.

## 1. Report placeholders (5 minutes)

Open `docs/Medical_Insurance_Cost_Prediction_ML_Case_Study_Report.docx` and use **Ctrl+H → Replace**
(or just search for `[`), then rebuild the PDF — or edit the constants at the top of
`src/report_content.py` and re-run `python src/make_pdf.py` + `python src/make_docx.py`.

| Placeholder | Where | Replace with |
|---|---|---|
| `https://github.com/<your-username>/medical-insurance-cost-prediction.git` | Appendix A (`REPO_URL`) | your actual repository URL |
| `Academic Year: 2026–2027` | cover | correct for V Sem of a 2024-admission batch — change only if the department says otherwise |

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
* Keep `FIGURE_GUIDE.md`, `EXECUTION_STEPS.md` and `VIVA_QA.md` — they are part of the submission.

## 5. Submission pack for the department

If they want a PDF + source code, submit:

```
Medical_Insurance_Cost_Prediction_ML_Case_Study_Report.pdf   (from docs/)
medical_insurance_ml.py                                       (from src/)
results.json                                                  (from src/)
figures/                                                      (10 PNGs)
```

Or simply give them the GitHub repo link.

## 6. Final sanity checks

- [ ] College name, student names, USNs, guide name appear correctly on the cover and certificate
- [ ] Certificate and declaration signed
- [ ] TOC / LOF / LOT page numbers match (rebuild after edits)
- [ ] Dataset URL, access date and licence stated in Section 3.1 (already done)
- [ ] All 10 figures and 10 tables are captioned and referenced in the text (already done)
- [ ] Report is 40 numbered pages **including** the CCA 1 / CCA 2 rubric pages appended at the end
      (check the page limit your department actually enforces — the rubric pages are numbered pages too and
      can be deleted from the DOCX if the limit is strict)
- [ ] The report's LOF / LOT lists show real page numbers, no `—` placeholders (already done)
- [ ] `python src/medical_insurance_ml.py` still reproduces R² = 0.9031 on a fresh machine
- [ ] No passwords, API keys or personal data anywhere in the repo

## 7. Extra files in this repository

| File | What it is |
|---|---|
| `FIGURE_GUIDE.md` | explanation of all 10 report figures plus the exact code block that draws each one |
| `EXECUTION_STEPS.md` | environment setup, run order, expected output, and a troubleshooting table |
| `VIVA_QA.md` | 40 basic-to-advanced viva questions with answers and a quick number sheet |
| `webapp/` | Flask demo that serves the trained model in a browser (run `python webapp/app.py`) |
