# Each target executes the notebook for its CRISP-DM phase in place.
# Notebooks only call functions in src/intent, so the logic stays testable and the
# narrative stays next to the outputs reviewers read.

PYTHON ?= python
NBEXEC = $(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace

.PHONY: data eda train evaluate

data:      ## Load raw xlsx, recover merged records, clean, build splits
	$(PYTHON) -m intent.data
	$(PYTHON) -m intent.preprocess
	$(PYTHON) -m intent.split
	$(NBEXEC) notebooks/02_preparation.ipynb

eda:       ## Data understanding: prevalence, co-occurrence, text length, noise
	$(NBEXEC) notebooks/01_data_understanding.ipynb

train:     ## 5-fold CV model selection and threshold tuning (test set untouched)
	$(PYTHON) -m intent.models
	$(PYTHON) -m intent.models compare
	$(PYTHON) -m intent.models embeddings
	$(PYTHON) -m intent.finetune
	$(PYTHON) -m intent.models twostage
	$(PYTHON) -m intent.models final
	$(PYTHON) -m intent.models ablation
	$(NBEXEC) notebooks/03_modeling.ipynb

evaluate:  ## Single final evaluation on the held-out test set, then error analysis
	# Refuses to run twice: delete reports/test_results.csv only to reproduce from scratch.
	$(PYTHON) -m intent.models test
	$(NBEXEC) notebooks/04_evaluation.ipynb
	$(NBEXEC) notebooks/05_error_analysis.ipynb
