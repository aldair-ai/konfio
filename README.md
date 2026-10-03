# Loan intent classification

Multi-label classification of free-text "use of proceeds" answers from SME credit
applications in Mexico (Spanish). Each answer gets one or more of 10 intent labels.
The project follows CRISP-DM; the full write-up lives in [reports/report.md](reports/report.md).

## Setup

Requires Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m spacy download es_core_news_sm
```

Place the raw file at `data/raw/data_conf.xlsx` (not versioned; confidential).

## Reproducing

```bash
make eda        # 01 data understanding
make data       # 02 preparation and splits
make train      # 03 model selection with 5-fold CV
make evaluate   # 04 test-set evaluation, 05 error analysis
```

All settings and paths are in `configs/base.yaml`. Seed is 42. Every experiment
appends one row to `reports/results.csv`.

## Repo layout

```
configs/base.yaml        settings and paths
src/intent/              data, preprocess, split, features, models, decode, evaluate
notebooks/               one notebook per CRISP-DM phase; they only call src/intent
reports/                 report.md, results.csv, figures/
```

## 1. Business understanding

_TODO: why intent matters for credit decisions, the 10 labels, success criteria
against the company baseline (Hamming 0.06821, avg precision 67%)._

Assumption: the codebook omits `inv`; we treat it as "buy inventory or merchandise".

## 2. Data understanding

_TODO: size, label prevalence and cardinality, co-occurrence, text length,
encoding issues, duplicates and conflicting labels, the ambiguity of `temp`._

## 3. Data preparation

_TODO: merged-record recovery, ftfy, two preprocessing branches, group-aware
iterative stratification, fixed test set._

## 4. Modeling

_TODO: baselines, candidate models, out-of-fold features, threshold tuning._

## 5. Evaluation

_TODO: test-set results vs baselines, per-label analysis, error analysis._

## 6. Deployment

_TODO: how the model would be served, monitored and retrained; known limits._
