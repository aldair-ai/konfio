# Each target executes the notebook for its CRISP-DM phase in place.
# Notebooks only call functions in src/intent, so the logic stays testable and the
# narrative stays next to the outputs reviewers read.

PYTHON ?= python
NBEXEC = $(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace

.PHONY: data eda train evaluate serve demo

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

# Must match paths.serving_dir in configs/base.yaml (comment on its own line: make keeps
# whitespace before an inline # as part of the value).
SERVING_DIR := models/e9b

serve: $(SERVING_DIR)/manifest.json  ## Serve the frozen selected model: POST http://127.0.0.1:8000/predict
	$(PYTHON) -m uvicorn intent.serve:app --host 127.0.0.1 --port 8000

# Built once (needs a CUDA GPU): stage 1 rebuilt from selected_model, B2 refit, frozen thresholds.
$(SERVING_DIR)/manifest.json:
	$(PYTHON) -m intent.models package

demo: $(SERVING_DIR)/manifest.json  ## Interview demo: API on :8000 + Streamlit on http://localhost:8501, offline, CPU
	$(PYTHON) app/run_demo.py
