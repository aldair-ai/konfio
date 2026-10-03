"""Leak-free train/test split and cross-validation folds.

Responsibilities:
- Collapse rows to groups by normalized text so duplicates (including the ones with
  conflicting labels) stay together. Otherwise CV would reward memorization.
- Apply iterative stratification (iterstrat) at group level so rare labels such as
  temp and sueldo appear in every fold and in the test set.
- Carve a fixed ~15% test set once and 5 CV folds on the rest; persist both to
  data/processed/folds.parquet.

Why persisted assignments: the test set is touched only once at the end; storing
them makes that auditable and identical across runs (seed 42).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

from intent.data import load_config, resolve
from intent.preprocess import add_text_columns, normalize_key

TEST_FOLD = -1  # fold id for held-out rows, so one int column encodes the whole split
ROW_KEY = ["source_row", "record_pos"]  # unique row id from data.py, survives re-sorting


def group_labels(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """One label row per group, as the union (max) of its members' labels.

    Union rather than majority: conflicting duplicates still count as positives for
    the rare label, so stratification places them deliberately instead of by chance.
    """
    return df.groupby("group_key")[labels].max()


def _stratify(groups: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """Assign each group a fold id: TEST_FOLD for the hold-out, 0..k-1 for CV."""
    seed, split_cfg = cfg["seed"], cfg["split"]
    y = groups.to_numpy()
    x = np.zeros((len(y), 1))  # iterstrat only uses y; x is a placeholder for the sklearn API

    holdout = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=split_cfg["test_size"], random_state=seed
    )
    cv_idx, test_idx = next(holdout.split(x, y))

    fold = np.full(len(y), TEST_FOLD)
    kfold = MultilabelStratifiedKFold(n_splits=split_cfg["n_folds"], shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(kfold.split(x[cv_idx], y[cv_idx])):
        fold[cv_idx[val_idx]] = k
    return pd.Series(fold, index=groups.index, name="fold")


def assign_folds(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return one row per input row with its group_key, split and fold."""
    cfg = cfg or load_config()
    keyed = df.assign(group_key=df[cfg["data"]["text_col"]].map(normalize_key))
    fold_of_group = _stratify(group_labels(keyed, cfg["data"]["labels"]), cfg)
    out = keyed[ROW_KEY + ["group_key"]].assign(fold=keyed["group_key"].map(fold_of_group))
    out["split"] = np.where(out["fold"] == TEST_FOLD, "test", "cv")
    return out


def fold_label_counts(df: pd.DataFrame, folds: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Positives per label per fold (test as its own row), plus row counts."""
    merged = df.merge(folds, on=ROW_KEY)
    counts = merged.groupby("fold")[labels].sum()
    counts.insert(0, "rows", merged.groupby("fold").size())
    counts.index = counts.index.map(lambda f: "test" if f == TEST_FOLD else f"fold_{f}")
    return counts


def load_split_frame(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Clean rows joined with their fold and both text branches; the entry point for modeling.

    Reads the persisted folds instead of recomputing them, so every experiment sees
    the exact assignment that was saved and tested.
    """
    cfg = cfg or load_config()
    df = pd.read_parquet(resolve(cfg["paths"]["interim_dir"]) / cfg["data"]["clean_file"])
    folds = pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / cfg["split"]["folds_file"])
    merged = df.merge(folds, on=ROW_KEY, how="inner", validate="one_to_one")
    if len(merged) != len(df):
        raise ValueError("folds.parquet is stale: rerun intent.split")
    return add_text_columns(merged.drop(columns="group_key"), cfg)


def build_folds(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read clean.parquet, assign folds, write folds.parquet; return (clean, folds)."""
    cfg = cfg or load_config()
    df = pd.read_parquet(resolve(cfg["paths"]["interim_dir"]) / cfg["data"]["clean_file"])
    folds = assign_folds(df, cfg)
    out = resolve(cfg["paths"]["processed_dir"]) / cfg["split"]["folds_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(out, index=False)
    return df, folds


if __name__ == "__main__":
    config = load_config()
    labels = config["data"]["labels"]
    clean_df, fold_df = build_folds(config)
    counts = fold_label_counts(clean_df, fold_df, labels)
    print(f"groups: {fold_df['group_key'].nunique()}  rows: {len(fold_df)}  "
          f"test share: {(fold_df['split'] == 'test').mean():.3f}\n")
    print(counts.to_string())
    temp_cv = counts.drop(index="test")["temp"]
    print(f"\ntemp positives per CV fold: min {temp_cv.min()}, max {temp_cv.max()}; "
          f"test: {counts.loc['test', 'temp']}")
