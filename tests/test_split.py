"""Leakage and balance checks on the persisted split."""

import pandas as pd
import pytest

from intent.data import clean, load_config, resolve
from intent.split import ROW_KEY, TEST_FOLD, assign_folds, fold_label_counts

CFG = load_config()
LABELS = CFG["data"]["labels"]


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not resolve(CFG["paths"]["raw_data"]).exists():
        pytest.skip("raw data not present (confidential, not versioned)")
    df, _, _ = clean(CFG)
    return df, assign_folds(df, CFG)


def test_no_group_in_two_splits(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """A group with more than one fold id would leak duplicates across test or CV folds."""
    _, folds = data
    folds_per_group = folds.groupby("group_key")["fold"].nunique()
    assert (folds_per_group == 1).all(), folds_per_group[folds_per_group > 1].head()
    test_groups = set(folds.loc[folds["fold"] == TEST_FOLD, "group_key"])
    cv_groups = set(folds.loc[folds["fold"] != TEST_FOLD, "group_key"])
    assert test_groups.isdisjoint(cv_groups)


def test_every_row_assigned_once(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df, folds = data
    assert len(folds) == len(df)
    assert not folds.duplicated(ROW_KEY).any()
    assert set(folds["fold"]) == {TEST_FOLD, *range(CFG["split"]["n_folds"])}


def test_test_share_near_target(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, folds = data
    assert abs((folds["fold"] == TEST_FOLD).mean() - CFG["split"]["test_size"]) < 0.02


def test_every_label_present_in_every_split(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df, folds = data
    counts = fold_label_counts(df, folds, LABELS)
    print("\n" + counts.to_string())  # visible with pytest -s
    assert (counts[LABELS] > 0).all().all()


def test_saved_folds_match_fresh_run(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Guards determinism (seed 42) and catches a stale folds.parquet."""
    path = resolve(CFG["paths"]["processed_dir"]) / CFG["split"]["folds_file"]
    if not path.exists():
        pytest.skip("folds.parquet not built yet (make data)")
    _, folds = data
    saved = pd.read_parquet(path)
    pd.testing.assert_frame_equal(
        saved.sort_values(ROW_KEY, ignore_index=True), folds.sort_values(ROW_KEY, ignore_index=True)
    )
