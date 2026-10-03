"""Codebook coverage and embedding cache behavior (encoder stubbed, no model download)."""

import numpy as np

from intent import features
from intent.data import load_config

CFG = load_config()


def test_codebook_has_3_to_5_descriptions_per_label() -> None:
    book = features.load_codebook(CFG)
    for label in CFG["data"]["labels"]:
        assert 3 <= len(book[label]) <= 5, label


def test_temp_descriptions_cover_season_and_receivables_cycle() -> None:
    temp = " ".join(features.load_codebook(CFG)["temp"]).lower()
    assert "temporada" in temp
    assert "cobr" in temp or "factura" in temp


def test_embed_cache_encodes_each_text_once(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_encode(texts, spec, batch_size):
        calls.append(list(texts))
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)

    monkeypatch.setattr(features, "_encode", fake_encode)
    cfg = {**CFG, "paths": {**CFG["paths"], "processed_dir": str(tmp_path)}}
    first = features.embed(["a", "bb", "a"], "e5", cfg)
    second = features.embed(["bb", "ccc"], "e5", cfg)
    assert calls == [["a", "bb"], ["ccc"]]  # duplicates and cached texts are not re-encoded
    assert first[0].tolist() == first[2].tolist() == [1.0, 1.0]
    assert second[0].tolist() == first[1].tolist()
