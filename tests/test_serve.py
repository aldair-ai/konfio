"""POST /predict and /predict_batch: contract and review rule (stub), and the real frozen model (if packaged)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from intent.data import load_config, resolve
from intent.decode import DecodeRules
from intent.serve import MAX_BATCH, SelectedModel, create_app

CFG = load_config()
LABELS = CFG["data"]["labels"]
EXAMPLES = [
    "Compra de mercancía para surtir mi tienda de abarrotes",
    "Pagar la nómina de mis empleados en diciembre",
    "Quiero liquidar las deudas de mis tarjetas de crédito",
]


def _stub(proba: np.ndarray) -> SelectedModel:
    """A SelectedModel without artifacts: fixed thresholds 0.5, fixed probabilities."""
    model = SelectedModel.__new__(SelectedModel)
    model.labels, model.name, model.margin = LABELS, "stub", 0.1
    model.thresholds = model.b2_thresholds = np.full(len(LABELS), 0.5)
    model.rules = DecodeRules()
    model.review_idx = [LABELS.index("cred"), LABELS.index("no")]
    model.predict = lambda texts: model.decode_rows(np.tile(proba, (len(texts), 1)), model.thresholds)
    model.predict_b2 = model.predict
    model.b2_top_terms = lambda text, labels, k=5: {lab: [] for lab in labels}
    return model


def _proba(**values: float) -> np.ndarray:
    p = np.full(len(LABELS), 0.05)
    for lab, v in values.items():
        p[LABELS.index(lab)] = v
    return p


@pytest.mark.parametrize("proba, labels, reasons", [
    (_proba(inv=0.95), ["inv"], []),                                            # confident, not a risk label
    (_proba(inv=0.95, equ=0.55), ["equ", "inv"], ["close to threshold: equ"]),  # equ within 0.1 of threshold
    (_proba(cred=0.95), ["cred"], ["risk label: cred"]),                        # risk label: always reviewed
])
def test_contract_and_review_rule_with_stub(proba, labels, reasons) -> None:
    with TestClient(create_app(loader=lambda: _stub(proba))) as client:
        body = client.post("/predict", json={"text": "texto"}).json()
    # original contract, unchanged
    assert body["labels"] == labels and body["needs_review"] is bool(reasons)
    assert set(body["probabilities"]) == set(LABELS)
    # explanation fields
    assert body["review_reasons"] == reasons
    assert set(body["thresholds"]) == set(LABELS) and body["latency_ms"] >= 0
    assert body["b2"]["labels"] == labels and set(body["b2"]["top_terms"]) == set(labels)


def test_empty_text_is_rejected() -> None:
    with TestClient(create_app(loader=lambda: _stub(_proba(inv=0.9)))) as client:
        assert client.post("/predict", json={"text": ""}).status_code == 422


def test_batch_endpoint_and_its_limit() -> None:
    with TestClient(create_app(loader=lambda: _stub(_proba(cred=0.9)))) as client:
        ok = client.post("/predict_batch", json={"texts": ["a", "b", "c"]}).json()
        assert len(ok["predictions"]) == 3 and all(p["needs_review"] for p in ok["predictions"])
        assert client.post("/predict_batch", json={"texts": ["x"] * (MAX_BATCH + 1)}).status_code == 422
        assert client.post("/predict_batch", json={"texts": ["a", ""]}).status_code == 422


@pytest.mark.skipif(not (resolve(CFG["paths"]["serving_dir"]) / "manifest.json").exists(),
                    reason="serving artifacts not built (python -m intent.models package)")
def test_frozen_model_on_three_examples() -> None:
    with TestClient(create_app()) as client:
        bodies = [client.post("/predict", json={"text": t}).json() for t in EXAMPLES]
        batch = client.post("/predict_batch", json={"texts": EXAMPLES}).json()["predictions"]
    for body, item in zip(bodies, batch):
        assert body["labels"] and set(body["labels"]) <= set(LABELS)
        assert all(0.0 <= p <= 1.0 for p in body["probabilities"].values())
        if {"no", "cred"} & set(body["labels"]):
            assert body["needs_review"]
        assert item["labels"] == body["labels"]  # batch and single paths agree
        for lab in body["b2"]["labels"]:  # B2 explains every label it predicts
            assert body["b2"]["top_terms"][lab]
    assert "inv" in bodies[0]["labels"]
    assert "sueldo" in bodies[1]["labels"]
    assert "cred" in bodies[2]["labels"] and bodies[2]["needs_review"]
    assert any(r.startswith("risk label") for r in bodies[2]["review_reasons"])
