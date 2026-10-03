"""Decode rules and evaluation helpers on tiny synthetic inputs."""

import numpy as np
import pytest

from intent.decode import DecodeRules, decode, tune_thresholds
from intent.evaluate import bootstrap_ci, log_result, summary_metrics

LABELS = ["inv", "no", "temp"]
T = np.array([0.5, 0.5, 0.5])


def test_at_least_one_assigns_argmax_only_when_enabled() -> None:
    score = np.array([[0.2, 0.1, 0.4]])
    assert decode(score, T, LABELS, DecodeRules(False, False)).sum() == 0
    assert decode(score, T, LABELS, DecodeRules(True, False)).tolist() == [[0, 0, 1]]


def test_no_exclusive_suppresses_others_only_when_no_is_top() -> None:
    no_top = np.array([[0.7, 0.9, 0.6]])
    no_second = np.array([[0.95, 0.9, 0.1]])
    on, off = DecodeRules(False, True), DecodeRules(False, False)
    assert decode(no_top, T, LABELS, on).tolist() == [[0, 1, 0]]
    assert decode(no_top, T, LABELS, off).tolist() == [[1, 1, 1]]
    assert decode(no_second, T, LABELS, on).tolist() == [[1, 1, 0]]


def test_tune_thresholds_separates_perfectly_ranked_label() -> None:
    y = np.array([[1], [1], [0], [0]])
    s = np.array([[0.9], [0.4], [0.3], [0.1]])
    th = tune_thresholds(y, s)
    assert 0.3 < th[0] <= 0.4


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y = rng.random((400, 3)) < 0.3
    p = np.where(rng.random((400, 3)) < 0.8, y, ~y)
    ci = bootstrap_ci(y, p, np.arange(400), focus_idx=2, n_resamples=300)
    macro = summary_metrics(y, p)["macro_f1"]
    assert ci["macro_f1_ci_low"] <= macro <= ci["macro_f1_ci_high"]


def test_log_result_rejects_mismatched_header(tmp_path) -> None:
    path = tmp_path / "results.csv"
    path.write_text("experiment,old_column\n", encoding="utf-8")
    cfg = {"seed": 42, "data": {"labels": LABELS}, "paths": {"results_csv": str(path)}}
    with pytest.raises(ValueError):
        log_result({"experiment": "x"}, cfg)


def test_paired_bootstrap_zero_for_identical_and_positive_for_better() -> None:
    from intent.evaluate import paired_bootstrap

    rng = np.random.default_rng(1)
    y = rng.random((300, 3)) < 0.3
    noisy = np.where(rng.random((300, 3)) < 0.7, y, ~y)
    same = paired_bootstrap(y, noisy, noisy, np.arange(300), "macro_f1", n=200)
    assert same["delta"] == 0 and same["ci_low"] == same["ci_high"] == 0
    better = paired_bootstrap(y, noisy, y, np.arange(300), "hamming_loss", n=200)
    assert better["delta"] > 0 and better["ci_low"] > 0


def test_margin_fallback_prefers_label_closest_to_its_threshold() -> None:
    score = np.array([[0.45, 0.1, 0.18]])
    thresholds = np.array([0.6, 0.5, 0.2])  # temp is 0.02 short, inv is 0.15 short
    rules = DecodeRules(at_least_one=True, no_exclusive=False, fallback="margin")
    assert decode(score, thresholds, LABELS, rules).tolist() == [[0, 0, 1]]
    assert decode(score, thresholds, LABELS, DecodeRules(True, False)).tolist() == [[1, 0, 0]]
