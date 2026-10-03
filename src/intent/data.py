"""Load the raw spreadsheet and repair structural defects before any text cleaning.

Responsibilities:
- Read data/raw/data_conf.xlsx (sheet Hoja1) using paths from configs/base.yaml.
- Recover the records merged into one cell by embedded newlines (each line follows
  the pattern text,0,1,...). We parse them back into rows instead of dropping them,
  because temp is already the rarest label (2.8%) and every labeled row counts.
- Fix mojibake with ftfy, then drop gibberish and non-string texts. Every exclusion
  is counted and saved with its reason so it is visible in the report.
- Validate labels: no missing values, at least one positive label per row.

Why a separate module: structural repair must happen before text normalization,
otherwise the merged-record pattern is destroyed by whitespace collapsing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import ftfy
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"

# A line inside a merged cell: free text, then exactly 10 binary labels at the end.
# Text may be empty: the first record's labels sit alone on the following line.
EMBEDDED_RECORD = re.compile(r"^(?P<text>.*?),(?P<labels>[01](?:,[01]){9})\s*$")

# UTF-8 bytes decoded as cp1252/latin-1 always produce one of these sequences.
# None of them occurs in legitimate Spanish text, so any hit means unrepaired mojibake.
MOJIBAKE = re.compile("Ã|Â|â€|�")

VOWELS = set("aeiouáéíóúü")


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read the YAML config. All paths in it are relative to the repo root."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(relative: str) -> Path:
    """Anchor a config path at the repo root so code works from any cwd (notebooks, make)."""
    return REPO_ROOT / relative


def load_raw(cfg: dict[str, Any]) -> pd.DataFrame:
    """Read the raw sheet and tag each row with its original position for traceability."""
    df = pd.read_excel(resolve(cfg["paths"]["raw_data"]), sheet_name=cfg["data"]["sheet"])
    # object dtype: the column mixes str and a stray int (-1); Arrow strings would coerce it.
    df[cfg["data"]["text_col"]] = df[cfg["data"]["text_col"]].astype(object)
    df.insert(0, "source_row", df.index)
    return df


def parse_merged_cell(text: str, outer_labels: list[float]) -> list[tuple[str, list[float]]]:
    """Split a cell holding several CSV records into (text, labels) pairs.

    Lines without trailing labels are buffered and prepended to the next labeled
    line, which covers the first record whose label tail sits on its own line.
    Text left in the buffer at the end is the last record of the broken quoted field;
    its labels are the ones the spreadsheet assigned to the whole cell.
    """
    records: list[tuple[str, list[float]]] = []
    buffer: list[str] = []
    for line in text.split("\n"):
        match = EMBEDDED_RECORD.match(line)
        if match is None:
            buffer.append(line)
            continue
        labels = [float(v) for v in match["labels"].split(",")]
        records.append((_join(buffer + [match["text"]]), labels))
        buffer = []
    tail = _join(buffer)
    if tail:
        records.append((tail, list(outer_labels)))
    return records


def _join(parts: list[str]) -> str:
    """Join fragments and drop the stray CSV quote left by the broken quoted field."""
    return " ".join(p.strip() for p in parts if p.strip()).strip('"').strip()


def is_merged_cell(value: Any) -> bool:
    """A cell is merged when at least one of its lines ends with 10 binary labels."""
    return isinstance(value, str) and any(
        EMBEDDED_RECORD.match(line) for line in value.split("\n")
    )


def expand_merged_cells(
    df: pd.DataFrame, text_col: str, labels: list[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace each merged cell with one row per recovered record.

    The first record keeps the original source_row (record_pos 0), so the original
    row survives in place and the extra records are traceable to the same cell.
    """
    merged_mask = df[text_col].map(is_merged_cell)
    new_rows: list[dict[str, Any]] = []
    n_tail = 0
    for _, row in df[merged_mask].iterrows():
        outer = row[labels].tolist()
        records = parse_merged_cell(row[text_col], outer)
        n_labeled = sum(bool(EMBEDDED_RECORD.match(ln)) for ln in row[text_col].split("\n"))
        n_tail += len(records) - n_labeled
        for pos, (text, values) in enumerate(records):
            new_rows.append(
                {"source_row": row["source_row"], "record_pos": pos, "from_merged_cell": True,
                 text_col: text, **dict(zip(labels, values))}
            )
    kept = df[~merged_mask].assign(record_pos=0, from_merged_cell=False)
    out = pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.sort_values(["source_row", "record_pos"], ignore_index=True)
    stats = {
        "merged_cells": int(merged_mask.sum()),
        "records_from_merged_cells": len(new_rows),
        "rows_added_by_recovery": len(new_rows) - int(merged_mask.sum()),
        "merged_tail_records_with_cell_labels": n_tail,
    }
    return out[["source_row", "record_pos", "from_merged_cell", text_col, *labels]], stats


def join_newlines(series: pd.Series) -> tuple[pd.Series, int]:
    """Replace line breaks inside free-text answers with spaces.

    These are applicants wrapping long answers, not separate records, so the
    lines belong to one sentence.
    """
    has_newline = series.map(lambda t: isinstance(t, str) and "\n" in t)
    fixed = series.where(~has_newline, series[has_newline].map(lambda t: " ".join(t.split("\n")).strip()))
    return fixed, int(has_newline.sum())


def fix_encoding(series: pd.Series) -> tuple[pd.Series, int]:
    """Repair mojibake (e.g. mercancÃ­a -> mercancía) with ftfy; count rows changed."""
    fixed = series.map(ftfy.fix_text)
    return fixed, int((fixed != series).sum())


def gibberish_reason(text: str, rules: dict[str, Any]) -> str | None:
    """Return why a text carries no usable intent, or None if it looks like language.

    Runs of repeated characters are removed first, so a real answer padded with
    "Mmmmm" or "....." is kept, while "GGGGGG" collapses to nothing.
    """
    stripped = text.strip()
    if len(stripped) < rules["min_chars"]:
        return "too_short"
    n = rules["min_run"] - 1
    collapsed = re.sub(rf"(.)\1{{{n},}}", "", stripped.lower())
    letters = [c for c in collapsed if c.isalpha()]
    if len(letters) < rules["min_letters"]:
        return "repeated_chars"
    if sum(c in VOWELS for c in letters) / len(letters) < rules["min_vowel_ratio"]:
        return "keyboard_mash"
    return None


def clean(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run every repair and filter step; return (clean, dropped, data_quality).

    Order matters: merged cells are parsed before newlines are joined (they share
    the newline signal), and ftfy runs before the gibberish check so that mojibake
    bytes are not counted as letters.
    """
    text_col, labels = cfg["data"]["text_col"], cfg["data"]["labels"]
    quality: dict[str, Any] = {}
    dropped: list[pd.DataFrame] = []

    df = load_raw(cfg)
    quality["raw_rows"] = len(df)

    df, merge_stats = expand_merged_cells(df, text_col, labels)
    quality.update(merge_stats)

    df[text_col], quality["newline_rows_joined"] = join_newlines(df[text_col])

    non_string = ~df[text_col].map(lambda t: isinstance(t, str))
    dropped.append(df[non_string].assign(drop_reason="non_string"))
    df = df[~non_string].copy()
    quality["non_string_dropped"] = int(non_string.sum())

    quality["mojibake_rows_before_ftfy"] = int(df[text_col].str.contains(MOJIBAKE).sum())
    df[text_col], quality["ftfy_rows_changed"] = fix_encoding(df[text_col])
    quality["mojibake_rows_after_ftfy"] = int(df[text_col].str.contains(MOJIBAKE).sum())

    reasons = df[text_col].map(lambda t: gibberish_reason(t, cfg["data"]["gibberish"]))
    dropped.append(df[reasons.notna()].assign(drop_reason="gibberish_" + reasons[reasons.notna()]))
    for reason, count in reasons.value_counts().items():
        quality[f"gibberish_{reason}_dropped"] = int(count)
    df = df[reasons.isna()].copy()

    # Rows with a NaN label are dropped, not imputed: the ones observed also carry
    # implausible labels (e.g. "compra de activos fijos" tagged inic), so the NaN marks
    # a corrupted record rather than a single missing value.
    missing = df[labels].isna()
    quality["missing_label_by_column"] = {k: int(v) for k, v in missing.sum().items() if v}
    quality["missing_label_rows_dropped"] = int(missing.any(axis=1).sum())
    dropped.append(df[missing.any(axis=1)].assign(drop_reason="missing_label"))
    df = df[~missing.any(axis=1)].copy()

    no_label = df[labels].sum(axis=1) == 0
    quality["no_positive_label_dropped"] = int(no_label.sum())
    dropped.append(df[no_label].assign(drop_reason="no_positive_label"))
    df = df[~no_label].copy()

    df[labels] = df[labels].astype("int8")
    df = df.reset_index(drop=True)
    quality["final_rows"] = len(df)
    quality["total_dropped"] = quality["raw_rows"] + quality["rows_added_by_recovery"] - len(df)

    dropped_df = pd.concat(dropped, ignore_index=True)
    dropped_df[text_col] = dropped_df[text_col].astype(str)  # parquet needs one type per column
    return df, dropped_df, quality


def build_interim(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the raw data and write clean.parquet and dropped.parquet to the interim dir."""
    cfg = cfg or load_config()
    df, dropped, quality = clean(cfg)
    out_dir = resolve(cfg["paths"]["interim_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / cfg["data"]["clean_file"], index=False)
    dropped.to_parquet(out_dir / cfg["data"]["dropped_file"], index=False)
    return df, quality


if __name__ == "__main__":
    import json

    clean_df, data_quality = build_interim()
    print(json.dumps(data_quality, indent=2))
    print(f"Final row count: {len(clean_df)}")
