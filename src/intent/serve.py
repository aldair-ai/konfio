"""HTTP inference for the frozen selected model (E9b).

POST /predict with {"text": "..."} returns the predicted labels, the per-label blended
probabilities and needs_review (the original contract), plus optional explanation fields
for the demo: review reasons, thresholds, latency, and B2's own prediction with its top
contributing terms. POST /predict_batch scores up to serving.max_batch texts at once.
The model is loaded once at startup from the artifacts written by
`python -m intent.models package` and runs on CPU, offline.

Why reuse the training code paths: clean_for_transformer, clean_for_tfidf, the blend
weight, the frozen thresholds and the decoding rules are exactly those of the evaluated
pipeline, so a served prediction means the same thing as a reported one.

needs_review routes a request to a human when the decision is fragile (some probability
within review_margin of its threshold) or when a risk label (no, cred) is predicted:
personal use and refinancing change the credit decision, so they are confirmed by a person.
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from intent.data import load_config, resolve
from intent.decode import DecodeRules, decode
from intent.finetune import TRAINABLE_FILE, E5Classifier, load_trainable
from intent.preprocess import clean_for_tfidf, clean_for_transformer


class SelectedModel:
    """Frozen E9b on CPU: w * fine-tuned e5 head + (1 - w) * B2, thresholds and rules from the manifest.

    B2 alone (with its own frozen thresholds) is exposed as predict_b2: the fallback when
    latency or compute cost matters more than the macro F1 gap.
    """

    def __init__(self, cfg: dict[str, Any] | None = None, device: str = "cpu") -> None:
        import joblib
        from transformers import AutoTokenizer

        from intent.models import final_config

        cfg, sm = final_config(cfg)
        art = resolve(cfg["paths"]["serving_dir"])
        manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
        if manifest["experiment"] != sm["experiment"]:
            raise ValueError("serving artifacts were built for a different selected model; re-run package")
        self.cfg, self.labels, self.device, self.name = cfg, manifest["labels"], device, manifest["name"]
        self.w = float(manifest["blend_weight_on_finetuned_head"])
        self.thresholds = np.array([manifest["thresholds"][lab] for lab in self.labels])
        self.b2_thresholds = np.array([manifest["fallback_b2_thresholds"][lab] for lab in self.labels])
        d = manifest["decoding"]
        self.rules = DecodeRules(at_least_one=d["at_least_one"], no_exclusive=d["no_exclusive"], fallback=d["fallback"])
        head = manifest["finetuned_head"]
        self.prefix, self.max_length = head["prefix"], head["max_length"]
        self.margin = float(cfg["serving"]["review_margin"])
        self.review_idx = [self.labels.index(lab) for lab in cfg["serving"]["review_labels"]]

        self.b2 = joblib.load(art / "b2.joblib")
        self.tok = AutoTokenizer.from_pretrained(art / "tokenizer")
        # pretrained base + the fine-tuned (trainable) tensors; frozen weights come from the base
        self.model = E5Classifier(head["model"], len(self.labels), head["dropout"], local_files_only=True)
        load_trainable(self.model, art / TRAINABLE_FILE)
        self.model.to(device).eval()  # eval: dropout off
        self.b2_features = self.b2.named_steps["tfidf"].get_feature_names_out()

    def thresholds_dict(self, b2: bool = False) -> dict[str, float]:
        values = self.b2_thresholds if b2 else self.thresholds
        return {lab: round(float(t), 4) for lab, t in zip(self.labels, values)}

    def b2_proba(self, texts: Sequence[str]) -> np.ndarray:
        # use_cache=False: serving must not write to the training lemma cache
        return self.b2.predict_proba(clean_for_tfidf(list(texts), self.cfg, use_cache=False))

    @torch.no_grad()
    def head_proba(self, texts: Sequence[str], chunk: int = 64) -> np.ndarray:
        """Sigmoid head probabilities, in chunks so large batches keep memory bounded."""
        out = []
        for i in range(0, len(texts), chunk):
            enc = self.tok([self.prefix + clean_for_transformer(t) for t in texts[i:i + chunk]], truncation=True,
                           max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            _, logits = self.model(enc["input_ids"], enc["attention_mask"])
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.vstack(out)

    def predict(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        proba = self.w * self.head_proba(texts) + (1 - self.w) * self.b2_proba(texts)
        return self.decode_rows(proba, self.thresholds)

    def predict_b2(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        return self.decode_rows(self.b2_proba(texts), self.b2_thresholds)

    def decode_rows(self, proba: np.ndarray, thresholds: np.ndarray) -> list[dict[str, Any]]:
        """Labels by the frozen rules, plus the review flag and its reasons for each row."""
        pred = decode(proba, thresholds, self.labels, self.rules).astype(bool)
        near = np.abs(proba - thresholds) <= self.margin
        rows = []
        for row, probs, close in zip(pred, proba, near):
            reasons = []
            if close.any():
                reasons.append("close to threshold: " + ", ".join(np.array(self.labels)[close]))
            risky = [self.labels[i] for i in self.review_idx if row[i]]
            if risky:
                reasons.append("risk label: " + ", ".join(risky))
            rows.append({
                "labels": [lab for lab, on in zip(self.labels, row) if on],
                "probabilities": {lab: round(float(p), 4) for lab, p in zip(self.labels, probs)},
                "needs_review": bool(reasons),
                "review_reasons": reasons,
            })
        return rows

    def b2_top_terms(self, text: str, labels: Sequence[str], k: int = 5) -> dict[str, list[dict[str, Any]]]:
        """Top k positive contributions (coefficient x TF-IDF value) to each label's B2 score.

        Linear model, so the contributions plus the intercept are exactly the logit:
        this is the model's own reasoning, not a post-hoc approximation. Terms are lemmas
        without accents (word) or character 3-5 grams (char), as B2 sees them.
        """
        x = self.b2.named_steps["tfidf"].transform(clean_for_tfidf([text], self.cfg, use_cache=False)).tocsr()
        estimators = self.b2.named_steps["clf"].estimators_
        out = {}
        for lab in labels:
            contrib = x.multiply(estimators[self.labels.index(lab)].coef_[0]).tocsr()
            order = np.argsort(-contrib.data)[:k]
            terms = []
            for i in order:
                if contrib.data[i] <= 0:
                    break
                kind, term = self.b2_features[contrib.indices[i]].split("__", 1)
                terms.append({"term": term, "kind": kind, "contribution": round(float(contrib.data[i]), 3)})
            out[lab] = terms
        return out


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, description="Free-text use of proceeds, in Spanish")


MAX_BATCH = load_config()["serving"]["max_batch"]


class TermContribution(BaseModel):
    term: str
    kind: str  # "word" (lemma) or "char" (character n-gram)
    contribution: float


class B2Result(BaseModel):
    labels: list[str]
    probabilities: dict[str, float]
    thresholds: dict[str, float]
    needs_review: bool
    review_reasons: list[str]
    latency_ms: float
    top_terms: dict[str, list[TermContribution]]


class PredictResponse(BaseModel):
    # Original contract: unchanged and always present.
    labels: list[str]
    probabilities: dict[str, float]
    needs_review: bool
    # Explanation fields (added for the demo); optional, so old clients are unaffected.
    model: str | None = None
    review_reasons: list[str] = []
    thresholds: dict[str, float] = {}
    latency_ms: float | None = None
    b2: B2Result | None = None


class BatchRequest(BaseModel):
    texts: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1, max_length=MAX_BATCH)


class BatchItem(BaseModel):
    labels: list[str]
    probabilities: dict[str, float]
    needs_review: bool
    review_reasons: list[str]


class BatchResponse(BaseModel):
    predictions: list[BatchItem]
    latency_ms: float


def create_app(loader: Callable[[], Any] = SelectedModel) -> FastAPI:
    """App factory; loader runs once at startup (tests can pass a stub)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = loader()
        # warm-up: spaCy and the allocator load lazily; pay that at startup, not on the first request
        app.state.model.predict(["calentamiento"])
        yield

    app = FastAPI(title="Loan intent classifier (E9b)", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        return {"status": "ok", "model": str(request.app.state.model.name)}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, request: Request) -> PredictResponse:
        model = request.app.state.model
        start = time.perf_counter()
        main = model.predict([req.text])[0]
        e9b_ms = (time.perf_counter() - start) * 1000  # model time only, as in reports/latency.csv
        start = time.perf_counter()
        b2 = model.predict_b2([req.text])[0]
        b2_ms = (time.perf_counter() - start) * 1000
        return PredictResponse(
            **main, model=model.name, thresholds=model.thresholds_dict(), latency_ms=round(e9b_ms, 1),
            b2=B2Result(**b2, thresholds=model.thresholds_dict(b2=True), latency_ms=round(b2_ms, 1),
                        top_terms=model.b2_top_terms(req.text, b2["labels"])),
        )

    @app.post("/predict_batch", response_model=BatchResponse)
    def predict_batch(req: BatchRequest, request: Request) -> BatchResponse:
        start = time.perf_counter()
        rows = request.app.state.model.predict(req.texts)
        return BatchResponse(predictions=rows, latency_ms=round((time.perf_counter() - start) * 1000, 1))

    return app


app = create_app()


def measure_latency(model: SelectedModel, texts: Sequence[str], warmup: int) -> pd.DataFrame:
    """Per-request CPU latency (one text per call, as the API serves it) for E9b and B2.

    The first `warmup` calls are excluded: they pay one-off costs (spaCy load, allocator).
    """
    rows = []
    for name, fn in [("E9b (selected)", model.predict), ("B2 (fallback)", model.predict_b2)]:
        for t in texts[:warmup]:
            fn([t])
        times = []
        for t in texts:
            start = time.perf_counter()
            fn([t])
            times.append((time.perf_counter() - start) * 1000)
        rows.append({"model": name, "requests": len(times), "p50_ms": np.percentile(times, 50),
                     "p95_ms": np.percentile(times, 95), "mean_ms": float(np.mean(times))})
    out = pd.DataFrame(rows).round(1)
    out["device"] = "cpu"
    out["cpu"] = platform.processor()
    out["logical_cores"] = os.cpu_count()
    out["torch_threads"] = torch.get_num_threads()
    out["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


if __name__ == "__main__":
    import sys

    if "latency" in sys.argv[1:]:
        from intent.split import TEST_FOLD, load_split_frame

        config = load_config()
        frame = load_split_frame(config)
        cv = frame[frame["fold"] != TEST_FOLD]  # CV texts only: the test set stays untouched
        n = config["serving"]["latency_requests"]
        texts = cv[config["data"]["text_col"]].sample(n=n, random_state=config["seed"]).tolist()
        result = measure_latency(SelectedModel(config), texts, config["serving"]["latency_warmup"])
        result.to_csv(resolve(config["paths"]["latency_csv"]), index=False)
        print(result.to_string(index=False))
