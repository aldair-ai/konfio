"""Behavior of the two preprocessing branches on hand-picked cases."""

from intent.preprocess import (
    NEGATIONS,
    STOPWORDS,
    clean_for_tfidf,
    clean_for_transformer,
    normalize_key,
)

DOMAIN_WORDS = {
    "pago", "deuda", "credito", "equipo", "maquinaria", "inventario", "mercancia", "renta",
    "local", "nomina", "sueldo", "temporada", "nuevo", "propio", "mas", "mayor", "capital",
    "trabajo", "negocio", "personal", "casa", "publicidad",
}


def test_stopwords_keep_domain_words_and_negations() -> None:
    assert not (DOMAIN_WORDS | NEGATIONS) & STOPWORDS


def test_transformer_branch_keeps_accents_and_case() -> None:
    assert clean_for_transformer("  Compra de  MERCANCÃ­a\n") == "Compra de MERCANCía"


def test_group_key_ignores_case_and_spacing() -> None:
    assert normalize_key("Pago  de Deudas ") == normalize_key("pago de deudas")


def test_tfidf_branch() -> None:
    out = clean_for_tfidf(
        ["No es para el negocio, sin fines comerciales", "Compra de 3 máquinas y año 2024!!!",
         "PAGAR DEUDAS"],
        use_cache=False,
    )
    assert out[0].split()[:1] == ["no"] and "sin" in out[0]
    assert "maquina" in out[1] and "año" in out[1]  # accents stripped, ñ kept
    assert not any(c.isdigit() or c in "!,." for c in "".join(out))
    assert out[2] == "pagar deuda"  # all-caps input still lemmatized
