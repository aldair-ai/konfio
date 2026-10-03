# Project: Loan intent classification (Konfio DS exercise)

## Goal
Multi-label classification of free-text "use of proceeds" answers from SME credit applications (Mexico, Spanish). Deliverables: a CRISP-DM report (10 to 12 slides or a short doc) plus a clean, reproducible repo. Reviewers care about reasoning and documented decisions more than raw metric gains.

## Data
- File: data/raw/data_conf.xlsx, sheet Hoja1. Text column: motivos.
- 10 binary labels: crec, cred, equ, inic, inv, mkt, no, renta, sueldo, temp.
- The codebook omits inv; we assume inv = buy inventory or merchandise. State this assumption in the report.
- Codebook: cred pay debts, equ equipment, inic start a business, mkt marketing, no use not destined to working capital, renta rent, sueldo payroll, temp seasonal sales, crec growth without a specific plan.

## Known data facts (verified)
- 6,679 rows vs 6,726 stated in the brief.
- One cell contains about 22 records merged by embedded newlines (pattern: text,0,1,... per line). 6 rows have temp = NaN, mostly caused by this. Recover these records by parsing the lines; do not just drop them.
- 15 rows contain newlines, 9 look like gibberish (repeated chars), 1 text is non-string.
- About 42% of rows have mojibake (mercancÃ­a). Fix with ftfy.
- Every valid row has at least one label. Cardinality: 86% one label, 12.6% two, 1% three. Mean 1.15.
- Prevalence: inv 34.7%, equ 29.7%, renta 11.5%, crec 9.0%, no 6.6%, cred 6.4%, inic 5.7%, mkt 4.9%, sueldo 3.2%, temp 2.8% (189).
- no is near mutually exclusive (4 of 444 co-occur). Mostly personal consumption.
- Conflicting duplicates (annotation noise): canonical number is 58 groups (141 rows) on the clean data, grouped by the split's normalized key (ftfy, lowercase, collapsed whitespace); 124 duplicate groups in total, 289 rows. Use 58 everywhere. Exact string matching on the raw file gives 57 conflicting groups (and 159 extra copies), because it misses one pair that differs only by mojibake. The CV portion alone holds 52 groups (126 rows).
- temp is semantically mixed: only 21% mention season words; many describe financing the receivables cycle. 80 non-temp rows mention temporada.
- Texts are short: median 11 words, p95 35, max 57.
- All-zeros baseline Hamming loss is about 0.115. Company baseline: Hamming 0.06821, avg precision 67%, LASER + LogReg binary relevance.

## Non-negotiable rules
- Split by group: group key = normalized text (ftfy, lowercase, collapsed whitespace), so duplicates never cross splits. Use iterative stratification (iterstrat MultilabelStratifiedKFold) on top of groups.
- Fixed test set (about 15%), touched only once at the end. Model selection and threshold tuning use 5-fold CV on the remaining data.
- Any learned feature fed to a second-stage model must be produced out-of-fold. Never fit the encoder and the classifier on the same rows.
- Two preprocessing branches: lemmatized and normalized text for TF-IDF; only encoding-fixed, lightly cleaned text for transformers.
- Every experiment logs one row to reports/results.csv: experiment name, macro F1, micro F1, Hamming loss, samples F1, PR-AUC per label, temp F1, seed, timestamp.
- Seeds fixed (42). Configs in configs/base.yaml. No hardcoded paths.

## Repo layout
- src/intent/: data.py, preprocess.py, split.py, features.py, models.py, decode.py, evaluate.py
- notebooks/: 01_data_understanding, 02_preparation, 03_modeling, 04_evaluation, 05_error_analysis. Notebooks only call src functions and narrate.
- reports/figures/, reports/results.csv, reports/report.md
- Makefile targets: data, eda, train, evaluate.

## Style
- Python 3.11, type hints, short functions, docstrings explaining WHY decisions were made.
- Prefer sklearn, sentence-transformers, setfit, lightgbm, iterstrat, ftfy, spacy es_core_news_sm, cleanlab.
- Every decision in code that the reviewer might question gets a one-line comment with the reason.
- Plots: matplotlib, one message per figure, saved as PNG to reports/figures.
- No em dashes in any written text.
