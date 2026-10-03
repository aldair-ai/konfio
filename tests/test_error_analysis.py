"""Error-analysis helpers on tiny hand-built inputs."""

import numpy as np
import pandas as pd

from intent.evaluate import categorize_errors, confused_pairs, conflicting_duplicate_mask, row_f1

LABELS = ["crec", "cred", "equ", "inic", "inv", "mkt", "no", "renta", "sueldo", "temp"]


def onehot(*names: str) -> list[int]:
    return [int(lab in names) for lab in LABELS]


def test_row_f1_is_set_overlap() -> None:
    y = np.array([onehot("inv", "equ"), onehot("no")], dtype=bool)
    p = np.array([onehot("inv"), onehot("no")], dtype=bool)
    assert np.allclose(row_f1(y, p), [2 / 3, 1.0])


def test_confused_pairs_counts_missed_then_predicted_instead() -> None:
    y = np.array([onehot("equ"), onehot("equ"), onehot("inv")])
    p = np.array([onehot("inv"), onehot("inv"), onehot("inv")])
    top = confused_pairs(y, p, LABELS).iloc[0]
    assert (top["missed (given)"], top["predicted instead"], top["rows"]) == ("equ", "inv", 2)
    assert top["share_of_missed_label"] == 1.0


def test_conflicting_duplicates_need_same_text_and_different_labels() -> None:
    keys = np.array(["a", "a", "b", "b", "c"])
    y = np.array([onehot("inv"), onehot("equ"), onehot("no"), onehot("no"), onehot("inv")])
    assert conflicting_duplicate_mask(keys, y).tolist() == [True, True, False, False, False]


def test_primary_category_follows_priority_and_correct_rows_are_excluded() -> None:
    texts = pd.Series(["capital de trabajo",  # vague (generic words only)
                       "compra de mercancia y pago de nomina del personal administrativo",  # multi-intent
                       "compra de equipo para la temporada alta de fin de ano en la tienda",  # temp
                       "compra de equipo de computo para la oficina principal del negocio"])  # correct
    tfidf = pd.Series(["capital trabajo", "compra mercancia pago nomina personal administrativo",
                       "compra equipo temporada alta fin ano tienda", "compra equipo computo oficina principal negocio"])
    y = np.array([onehot("crec"), onehot("inv", "sueldo"), onehot("temp"), onehot("equ")])
    p = np.array([onehot("inv"), onehot("inv"), onehot("equ"), onehot("equ")])
    issues = np.array([True, True, True, False])
    out = categorize_errors(texts, tfidf, y, p, issues, LABELS)
    assert out["primary"].tolist() == ["very short or vague", "multiple intents", "temp ambiguity"]
    assert out.loc[2, "temp_reading"] == "season"
