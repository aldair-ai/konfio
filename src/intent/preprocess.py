"""Text normalization, split into two branches with different goals.

Responsibilities:
- Group key: ftfy + lowercase + collapsed whitespace. Used by split.py so that
  duplicate texts never land in different splits.
- Transformer branch: only encoding fix and whitespace cleanup. Pretrained encoders
  were trained on natural text with accents and case, so lemmatizing or stripping
  accents would push inputs off-distribution.
- TF-IDF branch: lowercase, spaCy es_core_news_sm lemmatization, accent stripping,
  no punctuation or digits, custom stopwords. Sparse models need surface variants
  (deudas, deuda, DEUDA) collapsed so they share one weight.

Why two branches: one normalization cannot serve both model families well.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import ftfy
import pandas as pd
import spacy

from intent.data import load_config, resolve

# Negations flip intent ("no es para el negocio" is the core signal for label no),
# and spaCy's default Spanish list removes all of them.
NEGATIONS = frozenset({"no", "sin", "ni", "nunca", "tampoco", "nada"})

# Hand-curated function words, written as lemmas without accents because the filter
# runs after lemmatization and accent stripping. spaCy's default list (521 words) is
# not used: it also drops domain signals such as nuevo, propio, mas, mayor, mejor
# (inic and crec cues) and the negations above.
STOPWORDS = frozenset({
    # articles and determiners (spaCy lemmatizes la/las/los to el, una to uno)
    "el", "lo", "uno", "unos", "este", "ese", "aquel", "esto", "eso", "cual", "cuyo",
    # prepositions and contractions ("sin" is kept as a negation)
    "a", "al", "de", "del", "en", "con", "por", "para", "entre", "hasta", "desde",
    "sobre", "hacia", "ante", "tras", "segun", "mediante", "durante",
    # conjunctions and relatives
    "y", "e", "o", "u", "que", "pero", "porque", "pues", "como", "cuando", "donde",
    "si", "aunque", "mientras",
    # pronouns and possessives
    "yo", "tu", "ella",  # "el" (from él) already listed above "nosotros", "ellos", "me", "te", "se", "le", "les", "nos",
    "mi", "su", "nuestro", "vuestro", "mio", "suyo",
    # auxiliaries and copulas: carry tense, not intent
    "ser", "estar", "haber",
    # fillers
    "muy", "ya", "asi", "tambien", "etc",
})

TFIDF_CACHE_VERSION = 1  # bump when clean_for_tfidf logic changes, invalidates the cache

_WS = re.compile(r"\s+")
_LONG_RUN = re.compile(r"(.)\1{2,}")  # Spanish never triples a letter: "creeer", "Mmmmm"
_LETTERS = re.compile(r"[a-zñ]+")


def normalize_key(text: str) -> str:
    """Group key for duplicate detection: same answer modulo encoding, case and spacing."""
    return _WS.sub(" ", ftfy.fix_text(text).lower()).strip()


def clean_for_transformer(text: str) -> str:
    """Encoding fix and whitespace collapse only; accents, case and punctuation stay."""
    return _WS.sub(" ", ftfy.fix_text(text)).strip()


def strip_accents(text: str) -> str:
    """Remove diacritics but keep ñ, which is a separate letter (año vs ano)."""
    text = text.replace("ñ", "\0")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.replace("\0", "ñ")


@lru_cache(maxsize=1)
def _nlp(model: str) -> spacy.language.Language:
    """Load spaCy once. parser and ner are unused; the lemmatizer only needs POS."""
    return spacy.load(model, disable=["parser", "ner"])


def _pre_lemma(text: str) -> str:
    """Prepare text for spaCy.

    Lowercase before tagging because all-caps answers (common here) get tagged as
    proper nouns and are not lemmatized. Accents stay until after lemmatization,
    since the lemmatizer's lookup tables are accented.
    """
    text = clean_for_transformer(text).lower()
    return _LONG_RUN.sub(r"\1\1", text)


def _post_lemma(lemmas: list[str]) -> str:
    """Strip accents, keep letter runs only (drops digits and punctuation), filter stopwords."""
    tokens: list[str] = []
    for lemma in lemmas:
        for tok in _LETTERS.findall(strip_accents(lemma.lower())):
            # single letters are leftovers like the X in "rayos X", never intent cues
            if tok in NEGATIONS or (len(tok) > 1 and tok not in STOPWORDS):
                tokens.append(tok)
    return " ".join(tokens)


def _lemmatize(texts: Sequence[str], model: str) -> list[str]:
    nlp = _nlp(model)
    docs = nlp.pipe((_pre_lemma(t) for t in texts), batch_size=256)
    return [_post_lemma([tok.lemma_ for tok in doc]) for doc in docs]


def _cache_path(cfg: dict[str, Any]) -> Path:
    """Cache file name encodes everything that changes the output, so stale caches are never read."""
    model = cfg["preprocess"]["tfidf_branch"]["spacy_model"]
    fingerprint = repr((
        TFIDF_CACHE_VERSION, spacy.__version__, _nlp(model).meta["version"],
        sorted(STOPWORDS), sorted(NEGATIONS),
    ))
    digest = hashlib.sha1(fingerprint.encode()).hexdigest()[:10]
    return resolve(cfg["paths"]["processed_dir"]) / f"tfidf_lemmas_{digest}.parquet"


def clean_for_tfidf(
    texts: Sequence[str], cfg: dict[str, Any] | None = None, use_cache: bool = True
) -> list[str]:
    """Lemmatized, accent-free, stopword-filtered text for sparse models.

    Takes a batch, not one string, because spaCy's nlp.pipe is much faster in batch
    and the cache lookup is a single join. Lemmatizing the full set takes about a
    minute; the parquet cache in data/processed makes reruns instant.
    """
    cfg = cfg or load_config()
    model = cfg["preprocess"]["tfidf_branch"]["spacy_model"]
    if not use_cache:
        return _lemmatize(texts, model)

    path = _cache_path(cfg)
    cache = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["text", "tfidf"])
    known = dict(zip(cache["text"], cache["tfidf"]))
    missing = list(dict.fromkeys(t for t in texts if t not in known))
    if missing:
        known.update(zip(missing, _lemmatize(missing, model)))
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"text": list(known), "tfidf": list(known.values())}).to_parquet(path, index=False)
    return [known[t] for t in texts]


def add_text_columns(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Attach group_key and both model inputs to a clean DataFrame."""
    cfg = cfg or load_config()
    raw = df[cfg["data"]["text_col"]]
    return df.assign(
        group_key=raw.map(normalize_key),
        text_transformer=raw.map(clean_for_transformer),
        text_tfidf=clean_for_tfidf(raw.tolist(), cfg),
    )


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    config = load_config()
    clean_df = pd.read_parquet(resolve(config["paths"]["interim_dir"]) / config["data"]["clean_file"])
    out = add_text_columns(clean_df, config)
    print(f"cache: {_cache_path(config)}")
    print(f"empty tfidf texts: {(out['text_tfidf'] == '').sum()}")
    for _, row in out.sample(8, random_state=config["seed"]).iterrows():
        print(f"\n  raw:   {row[config['data']['text_col']]}\n  tfidf: {row['text_tfidf']}")
