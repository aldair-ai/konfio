"""Model definitions behind a common fit / predict_proba interface, and the CV runner.

Responsibilities:
- Baselines B0 (all zeros) and B1 (most frequent label always) to anchor results.
- TF-IDF + logistic regression as binary relevance (B2) and as a classifier chain (B3).
- Out-of-fold probabilities over the 5 CV folds; the test set is never touched here.
- Every model outputs per-label probabilities; turning them into labels is
  decode.py's job.

Why probabilities only: separating scoring from decoding lets us tune thresholds
and label constraints without retraining.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import copy
import hashlib
import json

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from intent.data import load_config, resolve
from intent.decode import DecodeRules, cross_fit_thresholds, decode, tune_thresholds
from intent.evaluate import (
    ablation_markdown,
    ablation_table,
    bootstrap_metrics,
    evaluate,
    load_results,
    paired_bootstrap,
    plot_ablation,
    plot_pr_curves,
    summary_metrics,
)
from intent.features import codebook_similarity, embed, tfidf_union
from intent.finetune import load_finetune_features, train_final_model
from intent.split import TEST_FOLD, load_split_frame

ModelFactory = Callable[[], BaseEstimator]


def _logreg(cfg: dict[str, Any], C: float | None = None) -> LogisticRegression:
    """Shared classifier; C overrides the config only for explicitly labeled diagnostics."""
    p = cfg["models"]["logreg"]
    return LogisticRegression(
        C=p["C"] if C is None else C, class_weight=p["class_weight"], max_iter=p["max_iter"],
        solver="liblinear",  # fast and stable on sparse, high-dimensional TF-IDF
        random_state=cfg["seed"],
    )


def tfidf_lr_br(cfg: dict[str, Any]) -> ModelFactory:
    """B2: one independent logistic regression per label (binary relevance)."""
    return lambda: Pipeline([
        ("tfidf", tfidf_union(cfg["models"]["tfidf"])),
        # no n_jobs: joblib memmaps the sparse matrix read-only and liblinear sorts it in place
        ("clf", OneVsRestClassifier(_logreg(cfg))),
    ])


def tfidf_lr_chain(cfg: dict[str, Any], order: list[int]) -> ModelFactory:
    """B3: classifier chain, each label sees the previous labels' probabilities.

    Order goes from most to least frequent: the well-estimated frequent labels feed
    context to the rare ones, instead of noisy rare predictions feeding everything.
    Probabilities (not hard 0/1) are chained so uncertainty propagates.
    """
    return lambda: Pipeline([
        ("tfidf", tfidf_union(cfg["models"]["tfidf"])),
        ("clf", ClassifierChain(_logreg(cfg), order=order, chain_method="predict_proba",
                                random_state=cfg["seed"])),
    ])


def embedding_lr(cfg: dict[str, Any], C: float | None = None) -> ModelFactory:
    """E1 to E3: same logistic regression as B2 on dense features, so only the representation changes.

    Standardized first: the 10 codebook similarities sit in a narrow band (~0.7 to 0.9)
    while embedding dimensions are ~0.03 in scale; without scaling the shared L2
    penalty treats them unevenly. The scaler is fit inside each fold with the model.
    """
    return lambda: Pipeline([("scale", StandardScaler()), ("clf", OneVsRestClassifier(_logreg(cfg, C)))])


def tfidf_codebook_lr(cfg: dict[str, Any], sim_cols: list[str]) -> ModelFactory:
    """B2_codebook: B2's TF-IDF hstacked with the 10 codebook similarities, same LR.

    Input is a DataFrame (text_tfidf + similarity columns). The similarities are
    standardized inside each fold; sparse_threshold=1 keeps the stacked matrix sparse.
    Standardized similarities have unit variance while TF-IDF entries are small, so
    under the shared L2 penalty the 10 dense features are cheap to use; that is
    intended, since they are few and carry prior knowledge.
    """
    return lambda: Pipeline([
        ("features", ColumnTransformer([
            ("tfidf", tfidf_union(cfg["models"]["tfidf"]), "text_tfidf"),
            ("codebook", StandardScaler(), sim_cols),
        ], sparse_threshold=1.0)),
        ("clf", OneVsRestClassifier(_logreg(cfg))),
    ])


def lgbm_br(cfg: dict[str, Any]) -> ModelFactory:
    """E7: one LightGBM per label on the hybrid features.

    Small trees (num_leaves 15, min_child_samples 20) because rare labels have only
    ~150 positives in the training folds. deterministic + force_row_wise make the
    multithreaded histogram build reproducible under the fixed seed.
    """
    params = cfg["models"]["lgbm"]
    return lambda: OneVsRestClassifier(LGBMClassifier(
        **params, random_state=cfg["seed"], deterministic=True, force_row_wise=True, verbose=-1,
    ))


def oof_proba(
    make_model: ModelFactory, X: pd.Series | np.ndarray, y: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    """Out-of-fold probabilities: each row is scored by a model that never saw it."""
    out = np.zeros(y.shape, dtype=float)
    for k in np.unique(folds):
        held = folds == k
        model = make_model().fit(X[~held], y[~held])
        out[held] = model.predict_proba(X[held])
    return out


def most_frequent_oof(y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """B1: predict the training folds' most frequent label for every row.

    Scores are the training prevalences, so PR-AUC equals prevalence (no ranking),
    which is the right floor for a model that ignores the text.
    """
    pred, score = np.zeros_like(y), np.zeros(y.shape, dtype=float)
    for k in np.unique(folds):
        held = folds == k
        prevalence = y[~held].mean(axis=0)
        pred[held, prevalence.argmax()] = 1
        score[held] = prevalence
    return pred, score


def save_oof(name: str, keys: pd.DataFrame, scores: np.ndarray, cfg: dict[str, Any]) -> None:
    """Persist OOF probabilities for error analysis and any later stacking stage."""
    out = keys.reset_index(drop=True).copy()
    out[[f"p_{lab}" for lab in cfg["data"]["labels"]]] = scores
    out.to_parquet(resolve(cfg["paths"]["processed_dir"]) / f"oof_{name}.parquet", index=False)


def run_baselines(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run B0 to B3 on the CV folds, log each, and return the summary rows."""
    cfg = cfg or load_config()
    labels = cfg["data"]["labels"]
    df = load_split_frame(cfg)
    cv = df[df["fold"] != TEST_FOLD].reset_index(drop=True)
    y = cv[labels].to_numpy()
    folds, groups, texts = cv["fold"].to_numpy(), cv["group_key"].to_numpy(), cv["text_tfidf"]
    keys = cv[["source_row", "record_pos", "fold"]]
    rows = []

    def log(name: str, pred: np.ndarray, score: np.ndarray) -> None:
        row, _ = evaluate(name, y, pred, score, groups, cfg)
        rows.append(row)

    zeros = np.zeros(y.shape, dtype=float)
    log("B0_all_zeros", zeros.astype(int), zeros)
    log("B1_most_frequent", *most_frequent_oof(y, folds))

    order = list(np.argsort(-y.mean(axis=0)))
    for name, factory in [("B2_tfidf_lr_br", tfidf_lr_br(cfg)),
                          ("B3_tfidf_lr_chain", tfidf_lr_chain(cfg, order))]:
        score = oof_proba(factory, texts, y, folds)
        save_oof(name, keys, score, cfg)
        thresholds = cross_fit_thresholds(y, score, folds)
        log(name, decode(score, thresholds, labels), score)
        if name.startswith("B2"):
            # Decode ablation on the same scores: isolates what tuning and each rule add.
            fixed = np.full(len(labels), 0.5)
            log(f"{name}__thr0.5_norules", decode(score, fixed, labels, DecodeRules(False, False)), score)
            log(f"{name}__tuned_norules", decode(score, thresholds, labels, DecodeRules(False, False)), score)
            log(f"{name}__tuned_atleastone", decode(score, thresholds, labels, DecodeRules(True, False)), score)
            log(f"{name}__tuned_noexclusive", decode(score, thresholds, labels, DecodeRules(False, True)), score)
    return pd.DataFrame(rows)


def load_oof(name: str, cv: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    """Saved OOF probabilities aligned to cv's row order (join on the row key, not position)."""
    oof = pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / f"oof_{name}.parquet")
    aligned = cv[["source_row", "record_pos"]].merge(oof, on=["source_row", "record_pos"], how="left")
    scores = aligned[[f"p_{lab}" for lab in cfg["data"]["labels"]]].to_numpy()
    if np.isnan(scores).any():
        raise ValueError(f"oof_{name}.parquet does not cover the current CV rows: rerun run_baselines")
    return scores


def run_decode_comparisons(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Log B2_margin and run paired bootstraps on the saved OOF scores (no retraining).

    Comparisons: B3 vs B2 (model), tuned + both rules vs fixed 0.5 (decoding), and
    margin vs argmax fallback. Writes reports/paired_bootstrap.csv.
    """
    cfg = cfg or load_config()
    labels = cfg["data"]["labels"]
    cv = load_split_frame(cfg).query(f"fold != {TEST_FOLD}").reset_index(drop=True)
    y, folds, groups = cv[labels].to_numpy(), cv["fold"].to_numpy(), cv["group_key"].to_numpy()

    def tuned(scores: np.ndarray, rules: DecodeRules = DecodeRules()) -> np.ndarray:
        return decode(scores, cross_fit_thresholds(y, scores, folds), labels, rules)

    s2, s3 = load_oof("B2_tfidf_lr_br", cv, cfg), load_oof("B3_tfidf_lr_chain", cv, cfg)
    preds = {
        "B2": tuned(s2),
        "B3": tuned(s3),
        "B2_fixed0.5": decode(s2, np.full(len(labels), 0.5), labels, DecodeRules(False, False)),
        "B2_margin": tuned(s2, DecodeRules(fallback="margin")),
    }
    evaluate("B2_margin", y, preds["B2_margin"], s2, groups, cfg)

    pairs = [("B2", "B3"), ("B2_fixed0.5", "B2"), ("B2", "B2_margin")]
    metrics = ["macro_f1", "temp_f1", "samples_f1", "hamming_loss"]
    focus_idx = labels.index(cfg["evaluation"]["focus_label"])
    rows = [
        {"a": a, "b": b, **paired_bootstrap(y, preds[a], preds[b], groups, m, 2000, focus_idx, cfg["seed"])}
        for a, b in pairs for m in metrics
    ]
    table = pd.DataFrame(rows)
    table.to_csv(resolve(cfg["paths"]["reports_dir"]) / "paired_bootstrap.csv", index=False)
    return table


def run_embedding_experiments(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E1 to E3 for each candidate encoder, logged; then paired comparisons.

    E1: embeddings + LR, fixed 0.5. E2: E1 + 10 codebook similarities, fixed 0.5.
    E3: E2 scores with cross-fitted thresholds and both decoding rules (same decoder
    as the B2 headline row, so E3 vs B2 compares representations only).
    Returns (summary rows, paired bootstrap table).
    """
    cfg = cfg or load_config()
    labels = cfg["data"]["labels"]
    cv = load_split_frame(cfg).query(f"fold != {TEST_FOLD}").reset_index(drop=True)
    y, folds, groups = cv[labels].to_numpy(), cv["fold"].to_numpy(), cv["group_key"].to_numpy()
    keys = cv[["source_row", "record_pos", "fold"]]
    texts = cv["text_transformer"].tolist()
    fixed, raw = np.full(len(labels), 0.5), DecodeRules(False, False)
    rows, e3_preds = [], {}

    for enc in cfg["embeddings"]["encoders"]:
        emb = embed(texts, enc, cfg)
        sims = codebook_similarity(emb, enc, cfg)
        s1 = oof_proba(embedding_lr(cfg), emb, y, folds)
        rows.append(evaluate(f"E1_{enc}", y, decode(s1, fixed, labels, raw), s1, groups, cfg)[0])
        s2 = oof_proba(embedding_lr(cfg), np.hstack([emb, sims]), y, folds)
        save_oof(f"E2_{enc}", keys, s2, cfg)
        rows.append(evaluate(f"E2_{enc}", y, decode(s2, fixed, labels, raw), s2, groups, cfg)[0])
        e3_preds[enc] = decode(s2, cross_fit_thresholds(y, s2, folds), labels)
        rows.append(evaluate(f"E3_{enc}", y, e3_preds[enc], s2, groups, cfg)[0])

    s_b2 = load_oof("B2_tfidf_lr_br", cv, cfg)
    e3_preds["B2"] = decode(s_b2, cross_fit_thresholds(y, s_b2, folds), labels)
    a_enc, b_enc = list(cfg["embeddings"]["encoders"])[:2]
    focus_idx = labels.index(cfg["evaluation"]["focus_label"])
    pairs = [(f"E3_{a_enc}", f"E3_{b_enc}"), ("B2", f"E3_{a_enc}"), ("B2", f"E3_{b_enc}")]
    preds = {("B2" if k == "B2" else f"E3_{k}"): v for k, v in e3_preds.items()}
    comparisons = pd.DataFrame([
        {"a": a, "b": b, **paired_bootstrap(y, preds[a], preds[b], groups, m, 2000, focus_idx, cfg["seed"])}
        for a, b in pairs for m in ["macro_f1", "temp_f1"]
    ])
    comparisons.to_csv(resolve(cfg["paths"]["reports_dir"]) / "paired_bootstrap_embeddings.csv", index=False)
    return pd.DataFrame(rows), comparisons


def _cv_frame(cfg: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """CV rows (test excluded) with labels, grouped folds and group keys."""
    cv = load_split_frame(cfg).query(f"fold != {TEST_FOLD}").reset_index(drop=True)
    return cv, cv[cfg["data"]["labels"]].to_numpy(), cv["fold"].to_numpy(), cv["group_key"].to_numpy()


def _paired_table(
    y: np.ndarray, preds: dict[str, np.ndarray], pairs: list[tuple[str, str]],
    groups: np.ndarray, cfg: dict[str, Any], metrics: list[str],
) -> pd.DataFrame:
    focus_idx = cfg["data"]["labels"].index(cfg["evaluation"]["focus_label"])
    return pd.DataFrame([
        {"a": a, "b": b, **paired_bootstrap(y, preds[a], preds[b], groups, m, 2000, focus_idx, cfg["seed"])}
        for a, b in pairs for m in metrics
    ])


def run_b2_codebook(cfg: dict[str, Any] | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    """B2 + codebook similarities from the selected encoder; logged, plus paired bootstrap vs B2."""
    cfg = cfg or load_config()
    labels, enc = cfg["data"]["labels"], cfg["embeddings"]["selected"]
    cv, y, folds, groups = _cv_frame(cfg)
    sims = codebook_similarity(embed(cv["text_transformer"].tolist(), enc, cfg), enc, cfg)
    sim_cols = [f"cb_{lab}" for lab in labels]
    X = pd.concat([cv[["text_tfidf"]], pd.DataFrame(sims, columns=sim_cols)], axis=1)

    score = oof_proba(tfidf_codebook_lr(cfg, sim_cols), X, y, folds)
    save_oof("B2_codebook", cv[["source_row", "record_pos", "fold"]], score, cfg)
    preds = {"B2_codebook": decode(score, cross_fit_thresholds(y, score, folds), labels)}
    row, _ = evaluate("B2_codebook", y, preds["B2_codebook"], score, groups, cfg)

    s_b2 = load_oof("B2_tfidf_lr_br", cv, cfg)
    preds["B2"] = decode(s_b2, cross_fit_thresholds(y, s_b2, folds), labels)
    table = _paired_table(y, preds, [("B2", "B2_codebook")], groups, cfg,
                          ["macro_f1", "temp_f1", "samples_f1", "hamming_loss"])
    table.to_csv(resolve(cfg["paths"]["reports_dir"]) / "paired_bootstrap_codebook.csv", index=False)
    return row, table


def run_leakage_check(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Quantify duplicate leakage: B2 on a row-level split (groups ignored) vs the grouped split.

    The row-level folds exist only inside this function and are never saved or reused.
    Besides overall metrics, we score the rows whose duplicate sits in another fold
    under the row split: that is where leakage acts, so the effect is visible there
    even when it is diluted in the overall numbers.
    """
    cfg = cfg or load_config()
    labels = cfg["data"]["labels"]
    focus_idx = labels.index(cfg["evaluation"]["focus_label"])
    cv, y, grouped_folds, groups = _cv_frame(cfg)

    row_folds = np.empty(len(y), dtype=int)
    kfold = MultilabelStratifiedKFold(
        n_splits=cfg["split"]["n_folds"], shuffle=True, random_state=cfg["seed"]
    )
    for k, (_, idx) in enumerate(kfold.split(np.zeros((len(y), 1)), y)):
        row_folds[idx] = k

    s_row = oof_proba(tfidf_lr_br(cfg), cv["text_tfidf"], y, row_folds)
    p_row = decode(s_row, cross_fit_thresholds(y, s_row, row_folds), labels)
    evaluate("B2_random_split", y, p_row, s_row, groups, cfg,
             eval_set="cv_oof_rowsplit", note="leakage_check")
    s_grp = load_oof("B2_tfidf_lr_br", cv, cfg)
    p_grp = decode(s_grp, cross_fit_thresholds(y, s_grp, grouped_folds), labels)

    spans = pd.Series(row_folds).groupby(groups).transform("nunique").to_numpy() > 1
    out = []
    for split, pred in [("grouped (B2)", p_grp), ("row_random (B2_random_split)", p_row)]:
        m = summary_metrics(y.astype(bool), pred.astype(bool))
        leaked = summary_metrics(y[spans].astype(bool), pred[spans].astype(bool))
        out.append({
            "split": split, "macro_f1": m["macro_f1"], "hamming_loss": m["hamming_loss"],
            "samples_f1": m["samples_f1"],
            "temp_f1": f1_score(y[:, focus_idx], pred[:, focus_idx], zero_division=0),
            "rows_with_duplicate_in_other_fold": int(spans.sum()),
            "samples_f1_on_those_rows": leaked["samples_f1"],
        })
    table = pd.DataFrame(out)
    table.to_csv(resolve(cfg["paths"]["reports_dir"]) / "leakage_check.csv", index=False)
    return table


def run_c_diagnostic(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """E1 with the selected encoder at several C values, logged with note="diagnostic".

    Diagnostic only: it checks whether E1's gap to B2 is a regularization artifact.
    No model choice is made from these rows, so it is not tuning on the CV folds.
    """
    cfg = cfg or load_config()
    labels, enc = cfg["data"]["labels"], cfg["embeddings"]["selected"]
    cv, y, folds, groups = _cv_frame(cfg)
    emb = embed(cv["text_transformer"].tolist(), enc, cfg)
    fixed, raw = np.full(len(labels), 0.5), DecodeRules(False, False)
    rows = []
    for C in [0.01, 0.1, 1.0]:
        score = oof_proba(embedding_lr(cfg, C), emb, y, folds)
        pred = decode(score, fixed, labels, raw)
        rows.append(evaluate(f"E1_{enc}_C{C}", y, pred, score, groups, cfg, note="diagnostic")[0])
    return pd.DataFrame(rows)


# B2_codebook did not beat B2 (macro F1 delta -0.0045, CI [-0.015, +0.005]), so B2 stays the reference.
REFERENCE = "B2_tfidf_lr_br"


def run_two_stage(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 2 on out-of-fold stage-1 features: E4 to E7, logged, plus paired bootstrap vs B2.

    E4: fine-tuned head probabilities alone (the neural model as reference).
    E5: fine-tuned pooled embedding + LR (scaled inside each fold).
    E6: [fine-tuned embedding, B2 OOF probabilities, frozen-e5 codebook similarities] + LR.
    E7: the E6 features + LightGBM per label.
    All use cross-fitted thresholds and both decoding rules, like the B2 reference row.
    Every stage-1 input is out-of-fold; see finetune.py for the remaining limitation.
    """
    cfg = cfg or load_config()
    labels, enc = cfg["data"]["labels"], cfg["embeddings"]["selected"]
    cv, y, folds, groups = _cv_frame(cfg)
    keys = cv[["source_row", "record_pos", "fold"]]

    ft_emb, ft_probs = load_finetune_features(cv, cfg, split="cv")
    b2 = load_oof(REFERENCE, cv, cfg)
    sims = codebook_similarity(embed(cv["text_transformer"].tolist(), enc, cfg), enc, cfg)
    hybrid = np.hstack([ft_emb, b2, sims])

    def tuned(scores: np.ndarray) -> np.ndarray:
        return decode(scores, cross_fit_thresholds(y, scores, folds), labels)

    scores = {"E4_ft_head": ft_probs}
    scores["E5_ft_emb_lr"] = oof_proba(embedding_lr(cfg), ft_emb, y, folds)
    scores["E6_hybrid_lr"] = oof_proba(embedding_lr(cfg), hybrid, y, folds)
    scores["E7_hybrid_lgbm"] = oof_proba(lgbm_br(cfg), hybrid, y, folds)

    rows, preds = [], {"B2": tuned(b2)}
    for name, score in scores.items():
        save_oof(name, keys, score, cfg)
        preds[name] = tuned(score)
        rows.append(evaluate(name, y, preds[name], score, groups, cfg)[0])

    table = _paired_table(y, preds, [("B2", name) for name in scores], groups, cfg, ["macro_f1", "temp_f1"])
    table.to_csv(resolve(cfg["paths"]["reports_dir"]) / "paired_bootstrap_two_stage.csv", index=False)
    return pd.DataFrame(rows), table


BEST_STAGE2 = "E7_hybrid_lgbm"  # highest CV macro F1 among stage-2 classifiers (0.7021)


def _oof_binary(make_model: ModelFactory, X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold positive-class probability for a single binary target."""
    out = np.zeros(len(y), dtype=float)
    for k in np.unique(folds):
        held = folds == k
        out[held] = make_model().fit(X[~held], y[~held]).predict_proba(X[held])[:, 1]
    return out


def hierarchical_oof(
    X: np.ndarray, y: np.ndarray, folds: np.ndarray, labels: list[str], cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """E8: head 1 routes no vs business; head 2 (E7's LightGBM) labels business rows only.

    Head 1: scaled LR on the same hybrid features, threshold cross-fitted for the
    no label. Head 2: trained on business rows (no == 0) of the training folds, so it
    never learns from personal-consumption text. Its thresholds are cross-fitted on
    the rows head 1 routes to business, the exact population it sees in use.
    Returns (pred, score, info); score is the hierarchical probability
    p(no) = h1 and p(label) = (1 - h1) * h2, used for PR-AUC.
    """
    no_idx = labels.index("no")
    biz = [j for j in range(len(labels)) if j != no_idx]
    biz_labels = [labels[j] for j in biz]
    is_no = y[:, no_idx]

    head1 = lambda: Pipeline([("scale", StandardScaler()), ("clf", _logreg(cfg))])  # noqa: E731
    h1 = _oof_binary(head1, X, is_no, folds)
    t1 = cross_fit_thresholds(is_no[:, None], h1[:, None], folds)[:, 0]
    routed_no = h1 >= t1

    h2 = np.zeros((len(y), len(biz)), dtype=float)
    t2 = np.zeros_like(h2)
    for k in np.unique(folds):
        held, train = folds == k, (folds != k) & (is_no == 0)
        model = lgbm_br(cfg)().fit(X[train], y[train][:, biz])
        h2[held] = model.predict_proba(X[held])
    for k in np.unique(folds):
        held, tune = folds == k, (folds != k) & ~routed_no
        t2[held] = tune_thresholds(y[tune][:, biz], h2[tune])

    pred = np.zeros_like(y)
    pred[routed_no, no_idx] = 1
    business = ~routed_no
    # no_exclusive is moot here: no is not among head 2's labels.
    pred[np.ix_(business, biz)] = decode(h2[business], t2[business], biz_labels, DecodeRules(no_exclusive=False))
    score = np.zeros(y.shape, dtype=float)
    score[:, no_idx] = h1
    score[:, biz] = (1 - h1)[:, None] * h2
    return pred, score, {"routed_no_share": float(routed_no.mean())}


def cross_fit_blend(
    y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, folds: np.ndarray, labels: list[str],
    grid: np.ndarray = np.round(np.linspace(0, 1, 21), 2), rules: DecodeRules = DecodeRules(),
) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    """Blend w * s_a + (1 - w) * s_b with w cross-fitted like the thresholds.

    For held-out fold k, w is the grid value maximizing macro F1 on the other folds
    (after tuning thresholds on those same folds), then fold k is decoded with w and
    the other folds' thresholds. One global w, not one per label: per-label weights
    on ~150 positives would overfit the rare labels. Ties go to the lowest w (more
    weight on s_b), since max keeps the first grid value. Returns (blended scores,
    row-wise thresholds, w per fold).
    """
    blended = np.zeros_like(s_a, dtype=float)
    thresholds = np.zeros_like(s_a, dtype=float)
    weights: dict[int, float] = {}
    for k in np.unique(folds):
        held, other = folds == k, folds != k

        def macro_on_other(w: float) -> float:
            mix = w * s_a[other] + (1 - w) * s_b[other]
            pred = decode(mix, tune_thresholds(y[other], mix), labels, rules)
            return f1_score(y[other], pred, average="macro", zero_division=0)

        w = float(max(grid, key=macro_on_other))
        weights[int(k)] = w
        blended[held] = w * s_a[held] + (1 - w) * s_b[held]
        thresholds[held] = tune_thresholds(y[other], w * s_a[other] + (1 - w) * s_b[other])
    return blended, thresholds, weights


def run_hierarchy_and_blends(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """E8 (hierarchical), E9 (blend B2 + E7, as specified), E9b (blend B2 + E4); paired vs E7.

    E9b is the clean stacking-vs-blending test: E7 stacks B2 probabilities with the
    fine-tuned features, E9b linearly blends B2 with the fine-tuned model's own
    probabilities. E9 blends B2 into a model that already consumes B2.
    """
    cfg = cfg or load_config()
    labels = cfg["data"]["labels"]
    no_idx = labels.index("no")
    cv, y, folds, groups = _cv_frame(cfg)
    keys = cv[["source_row", "record_pos", "fold"]]
    enc = cfg["embeddings"]["selected"]

    ft_emb, _ = load_finetune_features(cv, cfg, split="cv")
    b2 = load_oof(REFERENCE, cv, cfg)
    sims = codebook_similarity(embed(cv["text_transformer"].tolist(), enc, cfg), enc, cfg)
    hybrid = np.hstack([ft_emb, b2, sims])
    e7, e4 = load_oof(BEST_STAGE2, cv, cfg), load_oof("E4_ft_head", cv, cfg)

    def tuned(scores: np.ndarray) -> np.ndarray:
        return decode(scores, cross_fit_thresholds(y, scores, folds), labels)

    preds = {BEST_STAGE2: tuned(e7), "E4_ft_head": tuned(e4)}
    tables: dict[str, pd.DataFrame] = {}
    _, tables[BEST_STAGE2] = evaluate(BEST_STAGE2, y, preds[BEST_STAGE2], e7, groups, cfg, log=False)

    pred8, score8, info8 = hierarchical_oof(hybrid, y, folds, labels, cfg)
    save_oof("E8_hierarchical", keys, score8, cfg)
    preds["E8_hierarchical"] = pred8
    _, tables["E8_hierarchical"] = evaluate("E8_hierarchical", y, pred8, score8, groups, cfg)

    blend_weights = {}
    for name, other in [("E9_blend_B2_E7", e7), ("E9b_blend_B2_E4", e4)]:
        blended, thr, blend_weights[name] = cross_fit_blend(y, other, b2, folds, labels)
        save_oof(name, keys, blended, cfg)
        preds[name] = decode(blended, thr, labels)
        _, tables[name] = evaluate(name, y, preds[name], blended, groups, cfg)

    no_stats = pd.DataFrame({
        name: {"macro_f1": f1_score(y, preds[name], average="macro", zero_division=0),
               "no_precision": t.loc["no", "precision"], "no_recall": t.loc["no", "recall"],
               "no_f1": t.loc["no", "f1"]}
        for name, t in tables.items() if name in (BEST_STAGE2, "E8_hierarchical")
    }).T
    pairs = [(BEST_STAGE2, n) for n in ["E4_ft_head", "E8_hierarchical", "E9_blend_B2_E7", "E9b_blend_B2_E4"]]
    paired = _paired_table(y, preds, pairs, groups, cfg, ["macro_f1", "temp_f1"])
    paired.to_csv(resolve(cfg["paths"]["reports_dir"]) / "paired_bootstrap_final.csv", index=False)
    return {"no_stats": no_stats, "paired": paired, "blend_weights": blend_weights,
            "routed_no_share": info8["routed_no_share"], "true_no_share": float(y[:, no_idx].mean())}


COMPARABLE = "Comparable: grouped 5-fold CV"
ABLATION_SPEC: list[tuple[str, str, str]] = [
    ("Baselines", "B0_all_zeros", "Predict nothing"),
    ("Baselines", "B1_most_frequent", "Always the most frequent label (inv)"),
    ("TF-IDF + logistic regression", "B2_tfidf_lr_br__thr0.5_norules", "B2 scores, fixed 0.5, no rules"),
    ("TF-IDF + logistic regression", "B2_tfidf_lr_br__tuned_norules", "+ cross-fitted thresholds"),
    ("TF-IDF + logistic regression", "B2_tfidf_lr_br__tuned_atleastone", "+ at-least-one rule"),
    ("TF-IDF + logistic regression", "B2_tfidf_lr_br__tuned_noexclusive", "+ no-exclusive rule (no at-least-one)"),
    ("TF-IDF + logistic regression", "B2_tfidf_lr_br", "B2: thresholds + both rules (reference)"),
    ("TF-IDF + logistic regression", "B2_margin", "B2, at-least-one picks the largest margin"),
    ("TF-IDF + logistic regression", "B3_tfidf_lr_chain", "Classifier chain instead of binary relevance"),
    ("TF-IDF + logistic regression", "B2_codebook", "B2 + 10 codebook similarities"),
    ("Frozen sentence embeddings", "E1_e5", "e5 embeddings + LR, fixed 0.5"),
    ("Frozen sentence embeddings", "E2_e5", "+ codebook similarities"),
    ("Frozen sentence embeddings", "E3_e5", "+ thresholds + rules"),
    ("Frozen sentence embeddings", "E1_mpnet", "mpnet embeddings + LR, fixed 0.5"),
    ("Frozen sentence embeddings", "E2_mpnet", "+ codebook similarities"),
    ("Frozen sentence embeddings", "E3_mpnet", "+ thresholds + rules"),
    ("Fine-tuned e5 (two-stage)", "E4_ft_head", "Fine-tuned head probabilities"),
    ("Fine-tuned e5 (two-stage)", "E5_ft_emb_lr", "Fine-tuned embedding + LR"),
    ("Fine-tuned e5 (two-stage)", "E6_hybrid_lr", "Embedding + B2 probs + codebook, LR"),
    ("Fine-tuned e5 (two-stage)", "E7_hybrid_lgbm", "Same hybrid features, LightGBM"),
    ("Structure and blending", "E8_hierarchical", "no vs business, then E7 on business rows"),
    ("Structure and blending", "E9_blend_B2_E7", "Blend B2 + E7, weight cross-fitted"),
    ("Structure and blending", "E9b_blend_B2_E4", "Blend B2 + E4, weight cross-fitted"),
    ("Not comparable: diagnostics", "B2_random_split", "B2 on a row-level split (duplicate leakage)"),
    ("Not comparable: diagnostics", "E1_e5_C0.01", "E1_e5 with C = 0.01"),
    ("Not comparable: diagnostics", "E1_e5_C0.1", "E1_e5 with C = 0.1"),
]


def build_ablation(selected: str, title: str, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Write reports/ablation.md and reports/figures/ablation.png from results.csv."""
    cfg = cfg or load_config()
    table = ablation_table(ABLATION_SPEC, cfg)
    header = (
        "All numbers are pooled out-of-fold predictions on the 85% CV portion (5 grouped folds); "
        "the test set is untouched. 95% CIs resample groups (normalized text). "
        "Diagnostic rows use a different split or a non-default C and are never used for selection.\n\n"
    )
    md_path = resolve(cfg["paths"]["reports_dir"]) / "ablation.md"
    md_path.write_text("# Ablation (CV)\n\n" + header + ablation_markdown(table, selected), encoding="utf-8")
    subtitle = ("Macro F1, pooled out-of-fold over 5 grouped CV folds, with 95% group-bootstrap CIs. "
                "Test set untouched.")
    plot_ablation(table, selected, REFERENCE, title, subtitle,
                  resolve(cfg["paths"]["figures_dir"]) / "ablation.png")
    return table


COMPANY_BASELINE = {"hamming_loss": 0.06821, "average_precision": 0.67}  # from the brief (LASER + LogReg)


def _config_digest(cfg: dict[str, Any]) -> str:
    """Hash of every config file, recorded with the test results as proof of what was frozen."""
    blob = b"".join(resolve(f).read_bytes() for f in ["configs/base.yaml", cfg["embeddings"]["codebook"]])
    return hashlib.sha1(blob).hexdigest()[:12]


SELECTED_PIPELINES = {"E9b": "E9b_blend_B2_E4"}  # selected_model names with an implemented final pipeline


def final_config(cfg: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """(config view, selected_model): the final pipeline's settings, taken only from selected_model.

    The view overlays selected_model onto the sections the rest of the code reads
    (seed, split, fine-tune, TF-IDF, logistic regression, spaCy model). If any overlaid
    value differs from the top-level section, the cached artifacts (folds, lemma cache,
    fine-tune features, OOF scores) were built with other settings, so this refuses
    instead of silently mixing them. Only names in SELECTED_PIPELINES can run, so the
    final evaluation cannot pick any other configuration.
    """
    cfg = cfg or load_config()
    sm = cfg["selected_model"]
    if SELECTED_PIPELINES.get(sm["name"]) != sm["experiment"]:
        raise ValueError(f"no final pipeline for selected_model {sm['name']!r} / {sm['experiment']!r}")
    if sm["tfidf_lr"]["logreg"].get("solver") != "liblinear":
        raise ValueError("models._logreg is fixed to liblinear")
    if sm["thresholds"]["method"] != "per_label_f1_max":
        raise ValueError("only per_label_f1_max thresholds are implemented")

    view = copy.deepcopy(cfg)
    overlays = {
        ("seed",): sm["seed"],
        ("split", "test_size"): sm["split"]["test_size"],
        ("split", "n_folds"): sm["split"]["n_folds"],
        ("finetune",): {**sm["finetuned_head"], "log_csv": cfg["finetune"]["log_csv"]},
        ("models", "tfidf"): sm["tfidf_lr"]["tfidf"],
        ("models", "logreg"): {k: v for k, v in sm["tfidf_lr"]["logreg"].items() if k != "solver"},
        ("preprocess", "tfidf_branch", "spacy_model"): sm["tfidf_lr"]["spacy_model"],
    }
    mismatched = []
    for path, value in overlays.items():
        node = view
        for key in path[:-1]:
            node = node[key]
        if node[path[-1]] != value:
            mismatched.append(".".join(path))
        node[path[-1]] = value
    if mismatched:
        raise ValueError(f"selected_model disagrees with the settings behind the cached artifacts: {mismatched}")
    return view, sm


def selected_rules(sm: dict[str, Any]) -> DecodeRules:
    d = sm["decoding"]
    return DecodeRules(at_least_one=d["at_least_one"], no_exclusive=d["no_exclusive"], fallback=d["fallback"])


def selected_weight_grid(sm: dict[str, Any]) -> np.ndarray:
    start, stop, step = sm["blend"]["weight_grid"]
    return np.round(np.arange(start, stop + step / 2, step), 2)


def frozen_thresholds(cfg: dict[str, Any] | None = None) -> dict[str, float]:
    """Deployment thresholds of the selected model, from CV out-of-fold scores only.

    Reads no test label; used to check that the frozen section reproduces exactly the
    thresholds applied in the test evaluation.
    """
    cfg, sm = final_config(cfg)
    w = float(sm["blend"]["weight_on_finetuned_head"])
    cv, y, _, _ = _cv_frame(cfg)
    mix = w * load_oof("E4_ft_head", cv, cfg) + (1 - w) * load_oof(REFERENCE, cv, cfg)
    return dict(zip(cfg["data"]["labels"], tune_thresholds(y, mix)))


def run_test_evaluation(cfg: dict[str, Any] | None = None, force: bool = False) -> dict[str, Any]:
    """The single test-set evaluation of the selected configuration, with B2 as reference.

    Protocol, fixed before any test label is read:
    - Stage 1: the cached "test" fine-tune (trained on all non-test rows, minus its
      10% early-stopping slice) supplies the head probabilities for the test rows.
    - B2: refit on all non-test rows.
    - Selected (E9b): w * head + (1 - w) * B2, every setting from selected_model via
      final_config (which refuses any other configuration).
    - Thresholds: tuned on CV out-of-fold scores only (blend at the same w for E9b,
      B2's own OOF for B2), never on test. Decoding rules from selected_model.
    - Only these two systems touch the test set; nothing is chosen from test results.
    Everything is computed and written before anything is printed; a second run is
    refused unless force=True, so the test set cannot be re-queried by accident.
    """
    cfg = cfg or load_config()
    reports = resolve(cfg["paths"]["reports_dir"])
    out_csv = reports / "test_results.csv"
    if out_csv.exists() and not force:
        raise RuntimeError(f"{out_csv} exists: the test set was already evaluated once")
    cfg, sm = final_config(cfg)
    labels, selected, rules = cfg["data"]["labels"], sm["experiment"], selected_rules(sm)
    w = float(sm["blend"]["weight_on_finetuned_head"])
    focus_idx = labels.index(cfg["evaluation"]["focus_label"])

    frame = load_split_frame(cfg)
    cv = frame[frame["fold"] != TEST_FOLD].reset_index(drop=True)
    test = frame[frame["fold"] == TEST_FOLD].reset_index(drop=True)
    y_cv, y_te, groups_te = cv[labels].to_numpy(), test[labels].to_numpy(), test["group_key"].to_numpy()

    # Fit on all non-test data; thresholds from CV out-of-fold scores only.
    b2_te = tfidf_lr_br(cfg)().fit(cv["text_tfidf"], y_cv).predict_proba(test["text_tfidf"])
    _, head_te = load_finetune_features(test, cfg, split="test")
    b2_oof, head_oof = load_oof(REFERENCE, cv, cfg), load_oof("E4_ft_head", cv, cfg)
    systems = {
        selected: (w * head_te + (1 - w) * b2_te, tune_thresholds(y_cv, w * head_oof + (1 - w) * b2_oof)),
        REFERENCE: (b2_te, tune_thresholds(y_cv, b2_oof)),
    }
    preds = {name: decode(score, thr, labels, rules) for name, (score, thr) in systems.items()}

    boot = cfg["evaluation"]["bootstrap"]
    long_rows, per_label = [], []
    for name, (score, _) in systems.items():
        for metric, (pt, lo, hi) in bootstrap_metrics(
            y_te, preds[name], score, groups_te, focus_idx, boot["n_resamples"], boot["ci"], cfg["seed"]
        ).items():
            long_rows.append({"model": name, "metric": metric, "value": pt, "ci_low": lo, "ci_high": hi, "note": ""})
        _, table = evaluate(name, y_te, preds[name], score, groups_te, cfg, eval_set="test", log=False)
        per_label.append(table.add_prefix(f"{name}__"))

    leak = pd.read_csv(reports / "leakage_check.csv")
    grouped, rowsplit = leak.iloc[0], leak.iloc[1]
    leak_note = (f"likely row-level split; our leakage check (B2, CV) shows a row split lowers Hamming by "
                 f"{grouped.hamming_loss - rowsplit.hamming_loss:.4f} and raises macro F1 by "
                 f"{rowsplit.macro_f1 - grouped.macro_f1:.4f}; their test set and row count also differ")
    long_rows += [
        {"model": "company_baseline_reported", "metric": "hamming_loss", "value": COMPANY_BASELINE["hamming_loss"],
         "ci_low": np.nan, "ci_high": np.nan, "note": leak_note},
        {"model": "company_baseline_reported", "metric": "average_precision_ambiguous",
         "value": COMPANY_BASELINE["average_precision"], "ci_low": np.nan, "ci_high": np.nan,
         "note": "definition not given: compare with macro_precision, micro_precision and mean_pr_auc"},
    ]
    results = pd.DataFrame(long_rows)
    results.insert(0, "eval_set", "test")
    results["n_test_rows"] = len(test)
    results["config_sha"] = _config_digest(cfg)

    paired = _paired_table(
        y_te, {"B2": preds[REFERENCE], selected: preds[selected]},
        [("B2", selected)], groups_te, cfg, ["macro_f1", "temp_f1", "samples_f1", "hamming_loss"],
    )
    per_label_df = pd.concat(per_label, axis=1)

    # Write every artifact first, log last, print nothing until the caller does.
    per_label_df.to_csv(reports / "test_per_label.csv")
    paired.to_csv(reports / "test_paired_vs_B2.csv", index=False)
    plot_pr_curves(
        y_te, {"E9b blend (selected)": systems[selected][0], "B2 TF-IDF + LR": b2_te}, labels,
        "Test set: precision-recall per label, selected blend vs TF-IDF baseline",
        f"Single evaluation on the held-out test set ({len(test)} rows). AP shown as selected / B2.",
        resolve(cfg["paths"]["figures_dir"]) / "test_pr_curves.png",
    )
    results.to_csv(out_csv, index=False)
    for name, (score, _) in systems.items():
        evaluate(name, y_te, preds[name], score, groups_te, cfg, eval_set="test", note="final test")
    return {"results": results, "per_label": per_label_df, "paired": paired,
            "thresholds": {n: dict(zip(labels, np.round(t, 4))) for n, (_, t) in systems.items()}}


def package_selected_model(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write the serving artifacts of the frozen selected model to paths.serving_dir.

    - trainable.safetensors + tokenizer/: stage 1 rebuilt with the "test" job's rows and
      settings; frozen weights are reloaded from the pretrained base model.
    - b2.joblib: B2 refit on all non-test rows, as in the test evaluation.
    - manifest.json: blend weight, decoding rules, and the frozen thresholds for E9b and
      for B2 (the fallback model), all from CV out-of-fold scores, never from test.
    Returns the manifest.
    """
    import joblib

    cfg, sm = final_config(cfg)
    labels = cfg["data"]["labels"]
    out = resolve(cfg["paths"]["serving_dir"])
    info = train_final_model(cfg, out)

    cv, y, _, _ = _cv_frame(cfg)
    joblib.dump(tfidf_lr_br(cfg)().fit(cv["text_tfidf"], y), out / "b2.joblib")
    b2_thresholds = tune_thresholds(y, load_oof(REFERENCE, cv, cfg))
    manifest = {
        "name": sm["name"], "experiment": sm["experiment"], "labels": labels,
        "blend_weight_on_finetuned_head": float(sm["blend"]["weight_on_finetuned_head"]),
        "decoding": sm["decoding"],
        "thresholds": {k: float(v) for k, v in frozen_thresholds(cfg).items()},
        "fallback_b2_thresholds": dict(zip(labels, map(float, b2_thresholds))),
        "finetuned_head": {k: sm["finetuned_head"][k] for k in ("model", "prefix", "max_length", "dropout")},
        "stage1_training": {k: (v if not isinstance(v, float) else round(v, 5)) for k, v in info.items()},
        "config_sha": _config_digest(cfg),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def error_analysis_data(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """CV rows with given labels and OOF scores and predictions for E9b (selected), E4 and B2.

    E9b is reproduced exactly as evaluated in CV: blend weight and thresholds
    cross-fitted per fold, both decoding rules. Test rows are excluded.
    """
    cfg, sm = final_config(cfg)
    labels, rules = cfg["data"]["labels"], selected_rules(sm)
    cv, y, folds, groups = _cv_frame(cfg)
    b2, e4 = load_oof(REFERENCE, cv, cfg), load_oof("E4_ft_head", cv, cfg)
    s9, thr9, _ = cross_fit_blend(y, e4, b2, folds, labels, selected_weight_grid(sm), rules)
    score = {"E9b": s9, "E4": e4, "B2": b2}
    pred = {"E9b": decode(s9, thr9, labels, rules),
            "E4": decode(e4, cross_fit_thresholds(y, e4, folds), labels),
            "B2": decode(b2, cross_fit_thresholds(y, b2, folds), labels)}
    rows = cv[["source_row", "record_pos", "fold", "group_key", "text_transformer", "text_tfidf"]]
    return {"rows": rows.rename(columns={"text_transformer": "text"}), "y": y, "labels": labels,
            "score": score, "pred": pred, "groups": groups}


def two_stage_table(rows: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """One compact row per experiment: point metrics, CIs, and paired deltas vs B2."""
    def ci(lo: float, hi: float) -> str:
        return f"[{lo:+.3f}, {hi:+.3f}]"

    out = rows.set_index("experiment")
    table = pd.DataFrame({
        "macro_f1": out["macro_f1"].round(4),
        "hamming": out["hamming_loss"].round(4),
        "samples_f1": out["samples_f1"].round(4),
        "temp_f1": out["temp_f1"].round(4),
        "mean_pr_auc": out.filter(like="pr_auc_").mean(axis=1).round(4),
    })
    for metric, short in [("macro_f1", "macro"), ("temp_f1", "temp")]:
        p = paired[paired["metric"] == metric].set_index("b")
        table[f"d_{short}_vs_B2"] = p["delta"].round(4)
        table[f"d_{short}_ci"] = [ci(p.loc[e, "ci_low"], p.loc[e, "ci_high"]) for e in table.index]
    return table


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    if "embeddings" in sys.argv[1:]:
        summary, comp = run_embedding_experiments()
        cols = ["macro_f1", "micro_f1", "hamming_loss", "samples_f1", "temp_f1"]
        summary["mean_pr_auc"] = summary.filter(like="pr_auc_").mean(axis=1)
        print(summary.set_index("experiment")[cols + ["mean_pr_auc"]].round(4).to_string())
        comp["comparison"] = comp["b"] + " vs " + comp["a"]
        print()
        print(comp.set_index(["comparison", "metric"])[
            ["delta", "ci_low", "ci_high", "p_b_not_better"]].round(4).to_string())
        sys.exit()
    cols = ["macro_f1", "micro_f1", "hamming_loss", "samples_f1", "temp_f1"]
    if "codebook" in sys.argv[1:]:
        row, comp = run_b2_codebook()
        print(pd.DataFrame([row]).set_index("experiment")[cols].round(4).to_string())
        print(comp[["a", "b", "metric", "delta", "ci_low", "ci_high", "p_b_not_better"]].round(4).to_string())
    if "leakage" in sys.argv[1:]:
        print(run_leakage_check().round(4).to_string())
    if "cdiag" in sys.argv[1:]:
        print(run_c_diagnostic().set_index("experiment")[cols].round(4).to_string())
    if "twostage" in sys.argv[1:]:
        print(two_stage_table(*run_two_stage()).to_string())
    if "final" in sys.argv[1:]:
        out = run_hierarchy_and_blends()
        print(out["no_stats"].round(4).to_string())
        print(f"routed to no: {out['routed_no_share']:.4f} (true share {out['true_no_share']:.4f})")
        print("blend weights on the stage-2 model, per fold:", out["blend_weights"])
        print(out["paired"][["a", "b", "metric", "score_a", "score_b", "delta", "ci_low", "ci_high",
                             "p_b_not_better"]].round(4).to_string())
    if "ablation" in sys.argv[1:]:
        config = load_config()
        selected = final_config(config)[1]["experiment"]
        res = load_results(config)
        title = (f"Blending TF-IDF with fine-tuned e5 gives the best CV macro F1: "
                 f"{res.loc[selected, 'macro_f1']:.3f} vs {res.loc[REFERENCE, 'macro_f1']:.3f}")
        print(f"{len(build_ablation(selected, title, config))} rows -> reports/ablation.md, reports/figures/ablation.png")
    if "package" in sys.argv[1:]:
        m = package_selected_model()
        print(json.dumps({k: m[k] for k in ("name", "experiment", "stage1_training", "config_sha")}, indent=2))
    if "test" in sys.argv[1:]:
        out = run_test_evaluation()
        res = out["results"]
        print(res.pivot_table(index="metric", columns="model", values="value").round(4).to_string())
        print()
        print(res[["model", "metric", "value", "ci_low", "ci_high"]].round(4).to_string())
        print()
        print(out["per_label"].round(3).to_string())
        print()
        print(out["paired"][["metric", "score_a", "score_b", "delta", "ci_low", "ci_high",
                             "p_b_not_better"]].round(4).to_string())
        print()
        print(json.dumps(out["thresholds"], default=float))
    if {"codebook", "leakage", "cdiag", "twostage", "final", "ablation", "test", "package"} & set(sys.argv[1:]):
        sys.exit()
    if "compare" in sys.argv[1:]:
        table = run_decode_comparisons()
        table["comparison"] = table["b"] + " vs " + table["a"]
        cols = ["score_a", "score_b", "delta", "ci_low", "ci_high", "p_b_not_better"]
        print(table.set_index(["comparison", "metric"])[cols].round(4).to_string())
        sys.exit()
    results = run_baselines()
    shown = ["experiment", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "micro_f1",
             "hamming_loss", "samples_f1", "temp_f1", "temp_f1_ci_low", "temp_f1_ci_high"]
    print(results[shown].set_index("experiment").round(4).to_string())
    pr_auc = results.set_index("experiment").filter(like="pr_auc_")
    print("\nPR-AUC per label\n" + pr_auc.round(3).rename(columns=lambda c: c[7:]).to_string())
