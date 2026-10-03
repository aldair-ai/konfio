"""Turn per-label probabilities into final label sets.

Responsibilities:
- Per-label thresholds that maximize each label's F1 on out-of-fold predictions,
  never on the test set.
- Two structural rules from the data, each toggleable for ablation:
  (a) at_least_one: every application has at least one label, so if none passes
      its threshold we assign the argmax.
  (b) no_exclusive: no is near mutually exclusive (4 of 444 co-occur), so when it
      is the top label and passes its threshold, the other labels are suppressed.

Why a dedicated step: a single 0.5 cutoff misplaces the operating point for rare
labels such as temp (2.8%) and sueldo (3.2%); decoding is where macro F1 is won or lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import precision_recall_curve

DEFAULT_THRESHOLD = 0.5  # used only when a label has no positives to tune on


@dataclass(frozen=True)
class DecodeRules:
    """Switches for the structural rules, so each one's effect can be measured alone."""

    at_least_one: bool = True
    no_exclusive: bool = True
    # Which label the at_least_one fallback picks. "argmax": highest raw score, which
    # favors frequent labels. "margin": largest score minus own threshold, so a rare
    # label sitting just under its low cutoff can win.
    fallback: Literal["argmax", "margin"] = "argmax"


def tune_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Per-label threshold maximizing F1, read off the exact precision-recall curve.

    The PR curve enumerates every distinct score, so this is the exact optimum
    rather than a grid approximation.
    """
    thresholds = np.full(y_true.shape[1], DEFAULT_THRESHOLD)
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() == 0:
            continue
        precision, recall, cut = precision_recall_curve(y_true[:, j], y_score[:, j])
        denom = precision[:-1] + recall[:-1]  # last PR point has no threshold
        f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom > 0)
        thresholds[j] = cut[np.argmax(f1)]
    return thresholds


def cross_fit_thresholds(y_true: np.ndarray, y_score: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Row-wise thresholds where fold k uses thresholds tuned on the other folds only.

    Tuning on the same pooled OOF scores we then evaluate would let each row's label
    influence its own cutoff, which inflates F1 most for rare labels. This keeps the
    CV estimate honest; the final model uses tune_thresholds on all OOF rows.
    """
    out = np.empty_like(y_score, dtype=float)
    for k in np.unique(folds):
        held = folds == k
        out[held] = tune_thresholds(y_true[~held], y_score[~held])
    return out


def decode(
    y_score: np.ndarray,
    thresholds: np.ndarray,
    labels: list[str],
    rules: DecodeRules = DecodeRules(),
) -> np.ndarray:
    """Apply thresholds (per label, or per row and label) and then the structural rules.

    no_exclusive runs before at_least_one; the order does not matter for correctness,
    since suppressing others always leaves no itself as a positive.
    """
    y_pred = y_score >= thresholds
    top = y_score.argmax(axis=1)
    rows = np.arange(len(y_score))

    if rules.no_exclusive and "no" in labels:
        no_idx = labels.index("no")
        no_wins = (top == no_idx) & y_pred[:, no_idx]
        y_pred[no_wins] = False
        y_pred[no_wins, no_idx] = True

    if rules.at_least_one:
        empty = ~y_pred.any(axis=1)
        pick = top if rules.fallback == "argmax" else (y_score - thresholds).argmax(axis=1)
        y_pred[rows[empty], pick[empty]] = True

    return y_pred.astype(np.int8)
