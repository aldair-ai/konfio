"""Invariants the clean dataset must satisfy before any split or model sees it."""

import pandas as pd
import pytest

from intent.data import MOJIBAKE, clean, load_config, parse_merged_cell, resolve

CFG = load_config()
LABELS = CFG["data"]["labels"]
TEXT = CFG["data"]["text_col"]


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    if not resolve(CFG["paths"]["raw_data"]).exists():
        pytest.skip("raw data not present (confidential, not versioned)")
    df, _, _ = clean(CFG)
    return df


def test_no_nan_labels(clean_df: pd.DataFrame) -> None:
    assert not clean_df[LABELS].isna().any().any()


def test_every_row_has_a_label(clean_df: pd.DataFrame) -> None:
    assert (clean_df[LABELS].sum(axis=1) >= 1).all()


def test_no_mojibake(clean_df: pd.DataFrame) -> None:
    bad = clean_df[clean_df[TEXT].str.contains(MOJIBAKE)]
    assert bad.empty, bad[TEXT].head().tolist()


def test_parse_merged_cell_handles_split_first_record_and_tail() -> None:
    """Mirrors the real defect: first labels on their own line, unlabeled quoted tail."""
    cell = 'Capital de trabajo\n,0,0,0,0,0,0,0,1,0,0\nCompra de equipo,0,0,1,0,0,0,0,0,0,0\nFin"'
    outer = [0.0] * 9 + [float("nan")]
    records = parse_merged_cell(cell, outer)
    assert [r[0] for r in records] == ["Capital de trabajo", "Compra de equipo", "Fin"]
    assert records[0][1][7] == 1.0  # renta
    assert records[1][1][2] == 1.0  # equ
    assert records[2][1] == outer
