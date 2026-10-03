# Traceability: model selection and the test evaluation

All times are 2026-10-02, local time (UTC-5). `results.csv` stores UTC; converted here.

## Statements

1. **The model (E9b) was selected on CV at about 15:12**: after the last selection comparison was written (15:11:54) and before the ablation was regenerated from the config's selection block (15:13:01). No test data was involved.
2. **The test set was evaluated once, at 15:37**, with E9b and B2 (reference) only.
3. **The `model-selection-frozen` tag (and the only commit, `f4745f7`) were created at 19:36, after the test run.** They record the state at that time. They are not proof of ordering: the repository had no history before that commit.
4. **Config hash `1f3cf3b6c783` links the test run to a config that already contained the selection.** The hash was computed and written into `reports/test_results.csv` at 15:37. `reports/config_at_test_time.yaml` is a byte-exact reconstruction of `configs/base.yaml` at that moment; it contains `selection: experiment: E9b_blend_B2_E4, blend_weight: 0.5` and hashes to the same value.
5. **The frozen `selected_model` pipeline reproduces the evaluated test predictions.** Predicted positives per label, derived from the saved precision, recall and support (no test label read), match the evaluated run exactly: 1,270 for E9b, 1,330 for B2.

## Evidence

| Time (local) | Event | Source |
|---|---|---|
| 15:11:10 | E9b CV result logged (macro F1 0.7152) | `results.csv` row, timestamp 20:11:10 UTC; `oof_E9b_blend_B2_E4.parquet` mtime |
| 15:11:13 | Paired bootstrap of E8, E9, E9b and E4 vs E7 written | `reports/paired_bootstrap_final.csv` mtime |
| 15:11:54 | Selection comparisons (E9b vs E9, E4, B2) written | `reports/paired_bootstrap_selection.csv` mtime |
| 15:13:01 | Ablation regenerated, reading the selected model from the config's `selection` block | `reports/ablation.md` mtime (`ablation.png` 15:13:02) |
| 15:37:18 to 15:37:20 | Single test evaluation: per-label, paired and summary results written, then logged | `reports/test_*.csv`, `test_pr_curves.png` mtimes; `results.csv` test rows, 20:37:20 UTC |
| 19:33:08 | Test-time config reconstructed, hash verified | `reports/config_at_test_time.yaml` mtime |
| 19:36:15 | First commit `f4745f7` and annotated tag `model-selection-frozen` | `git log`, `git for-each-ref refs/tags` |

## Strength of the evidence

- **File modification times and log timestamps** are local and could in principle be edited. They are consistent with each other and are recorded here as observed on 2026-10-02.
- **Git** proves nothing about the order of selection and test: the first commit came after both.
- **The config hash is the strongest link.** It was written by the test run itself, and only a config containing the E9b selection reproduces it. This shows the selection was in place when the test ran; it does not bound how long before.

## How to verify

```bash
# 1. The test-time config hashes to the value recorded by the test run
python -c "import hashlib; print(hashlib.sha1(open('reports/config_at_test_time.yaml','rb').read() + open('configs/codebook.yaml','rb').read()).hexdigest()[:12])"
#    -> 1f3cf3b6c783, the config_sha column of reports/test_results.csv

# 2. That config already names the selected model
grep -n "experiment: E9b_blend_B2_E4" reports/config_at_test_time.yaml
```

`.gitattributes` marks both hashed files `-text` so that checkouts never change their bytes (line endings).
