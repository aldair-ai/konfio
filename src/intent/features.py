"""Feature construction for both model families.

Responsibilities:
- Sparse features: word and character n-gram TF-IDF on the lemmatized branch.
  Character n-grams help with the typos and spelling variants common in free text.
- Dense features: sentence embeddings from pretrained multilingual encoders on the
  lightly cleaned branch.
- Any learned representation passed to a second-stage model is produced
  out-of-fold, so the encoder never sees the rows the classifier is scored on.

Why out-of-fold: fitting encoder and classifier on the same rows inflates CV
scores and hides overfitting, which would invalidate model selection.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from intent.data import load_config, resolve


def tfidf_union(params: dict[str, Any]) -> FeatureUnion:
    """Word 1-2 grams plus char_wb 3-5 grams, each L2-normalized separately.

    char_wb (not char) keeps n-grams inside word boundaries, so they capture
    misspellings (mercansia, makinaria) and lemma errors (fertilizant) without
    spanning unrelated neighboring words. Returned unfitted: it must be fit inside
    each CV fold, never on the full data.
    """
    common = {"min_df": params["min_df"], "sublinear_tf": params["sublinear_tf"]}
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=tuple(params["word_ngrams"]), **common)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(params["char_ngrams"]), **common)),
    ])


@lru_cache(maxsize=2)
def _encoder(model_name: str):
    """Load a sentence-transformers model once per process (imported lazily: torch is slow to load)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def _encode(texts: Sequence[str], spec: dict[str, Any], batch_size: int) -> np.ndarray:
    """L2-normalized embeddings, so a dot product is a cosine similarity."""
    prefixed = [spec["prefix"] + t for t in texts]
    return _encoder(spec["model"]).encode(
        prefixed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)


def _embedding_cache(encoder: str, cfg: dict[str, Any]) -> Path:
    """Cache name hashes model id and prefix: changing either never reuses stale vectors."""
    spec = cfg["embeddings"]["encoders"][encoder]
    digest = hashlib.sha1(f"{spec['model']}|{spec['prefix']}".encode()).hexdigest()[:10]
    return resolve(cfg["paths"]["processed_dir"]) / f"emb_{encoder}_{digest}.npz"


def embed(texts: Sequence[str], encoder: str, cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Frozen sentence embeddings for texts, cached per encoder in data/processed.

    Frozen means no weight is fit on our data, so these features need no out-of-fold
    treatment; only the downstream classifier is fit inside folds.
    """
    cfg = cfg or load_config()
    path = _embedding_cache(encoder, cfg)
    cached: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            cached = dict(zip(z["texts"].tolist(), z["vectors"]))
    missing = list(dict.fromkeys(t for t in texts if t not in cached))
    if missing:
        spec = cfg["embeddings"]["encoders"][encoder]
        cached.update(zip(missing, _encode(missing, spec, cfg["embeddings"]["batch_size"])))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, texts=np.array(list(cached)), vectors=np.stack(list(cached.values())))
    return np.stack([cached[t] for t in texts])


def load_codebook(cfg: dict[str, Any]) -> dict[str, list[str]]:
    with open(resolve(cfg["embeddings"]["codebook"]), encoding="utf-8") as f:
        book = yaml.safe_load(f)
    missing = set(cfg["data"]["labels"]) - set(book)
    if missing:
        raise ValueError(f"codebook has no descriptions for {sorted(missing)}")
    return book


def codebook_similarity(
    text_emb: np.ndarray, encoder: str, cfg: dict[str, Any] | None = None
) -> np.ndarray:
    """Max cosine similarity between each text and each label's descriptions (n x 10).

    Max rather than mean: a label like temp has several distinct meanings (season vs
    receivables cycle), and a text only needs to match one of them. Descriptions are
    embedded with the same encoder and prefix as the texts so the spaces match.
    """
    cfg = cfg or load_config()
    book = load_codebook(cfg)
    spec = cfg["embeddings"]["encoders"][encoder]
    out = np.empty((len(text_emb), len(cfg["data"]["labels"])), dtype=np.float32)
    for j, label in enumerate(cfg["data"]["labels"]):
        desc = _encode(book[label], spec, cfg["embeddings"]["batch_size"])
        out[:, j] = (text_emb @ desc.T).max(axis=1)
    return out
