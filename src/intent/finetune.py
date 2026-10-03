"""Stage 1 of the two-stage model: fine-tune multilingual-e5-base as a feature extractor.

Architecture: e5-base encoder ("query: " prefix), attention-masked mean pooling,
dropout 0.1, linear head with 10 sigmoid outputs, BCEWithLogitsLoss with
pos_weight = neg/pos per label (capped at 10). Outputs per row: the 768-dim pooled
embedding and the 10 head probabilities.

Why out-of-fold: stage 2 trains a classifier on these features. If an encoder had
been fine-tuned on a row's label, that row's embedding would encode its own answer,
the stage-2 classifier would learn to trust the features far more than they deserve,
and CV would report a score the model cannot reach on new data. So for each CV fold
k we fine-tune on the other four folds and only write features for fold k. Early
stopping uses an inner 10% group-aware slice of those four folds, never fold k.
For the test set we fine-tune once on all non-test rows.

Known limitation: when stage 2 is evaluated on fold j, its training rows (folds
i != j) carry features from encoders trained on folds != i, which include fold j.
Fold j's labels therefore reach stage 2 indirectly, through the encoders that
produced the training rows' features. Full nested CV removes this but costs about
25 fine-tunes instead of 5; we judged that not worth it. The untouched test set,
whose features come from an encoder that never saw a test label, is the clean check.

GPU runs are seeded (42) but not bit-exact: cuDNN and atomic adds are nondeterministic.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from intent.data import load_config, resolve
from intent.split import ROW_KEY, TEST_FOLD, group_labels, load_split_frame


class E5Classifier(nn.Module):
    """Encoder + mean pooling + dropout + linear head; forward returns (pooled, logits)."""

    def __init__(self, model_name: str, n_labels: int, dropout: float) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, n_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = mean_pool(hidden, attention_mask)
        return pooled, self.head(self.dropout(pooled))


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average over real tokens only; e5 was pretrained with exactly this pooling."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)


def pos_weight(y: np.ndarray, cap: float) -> torch.Tensor:
    """neg/pos per label, capped so temp (~35:1) does not dominate the gradient."""
    pos = y.sum(axis=0)
    return torch.tensor(np.minimum((len(y) - pos) / np.maximum(pos, 1), cap), dtype=torch.float32)


def inner_split(frame: pd.DataFrame, labels: list[str], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Group-aware, label-stratified early-stopping slice of the training rows.

    Same rule as the main split (groups by normalized text, iterative
    stratification), so duplicates cannot sit on both sides of early stopping.
    Returns positional indices (train, val) into frame.
    """
    groups = group_labels(frame, labels)
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=cfg["finetune"]["inner_val_size"], random_state=cfg["seed"]
    )
    _, g_val = next(splitter.split(np.zeros((len(groups), 1)), groups.to_numpy()))
    is_val = frame["group_key"].isin(set(groups.index[g_val])).to_numpy()
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def _loader(texts: list[str], y: np.ndarray | None, tok, ft: dict[str, Any], train: bool, seed: int) -> DataLoader:
    """Tokenize per batch so padding is per batch (dynamic), not to max_length."""
    prefixed = [ft["prefix"] + t for t in texts]

    def collate(idx: list[int]) -> dict[str, torch.Tensor]:
        enc = tok([prefixed[i] for i in idx], truncation=True, max_length=ft["max_length"],
                  padding=True, return_tensors="pt")
        if y is not None:
            enc["labels"] = torch.tensor(y[idx], dtype=torch.float32)
        return enc

    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        list(range(len(texts))), batch_size=ft["batch_size"] if train else ft["eval_batch_size"],
        shuffle=train, generator=gen, collate_fn=collate,
        num_workers=0,  # Windows spawns workers by re-importing; tokenization is cheap anyway
    )


@torch.no_grad()
def _predict(model: E5Classifier, loader: DataLoader, fp16: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pooled_all, probs_all = [], []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            pooled, logits = model(batch["input_ids"].cuda(), batch["attention_mask"].cuda())
        pooled_all.append(pooled.float().cpu().numpy())
        probs_all.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(pooled_all), np.concatenate(probs_all)


def _macro_ap(y: np.ndarray, p: np.ndarray) -> float:
    """Mean PR-AUC over labels with positives: threshold-free early-stopping criterion.

    Chosen over validation loss because the loss is reweighted by pos_weight and
    shifts with it, while PR-AUC measures the ranking stage 2 actually consumes.
    """
    cols = [j for j in range(y.shape[1]) if y[:, j].any()]
    return float(np.mean([average_precision_score(y[:, j], p[:, j]) for j in cols]))


def _optimizer(model: nn.Module, ft: dict[str, Any]) -> torch.optim.AdamW:
    """AdamW without decay on biases and LayerNorm weights (standard for transformers).

    fused=True: same update rule in one kernel, without the full-size temporary the
    default multi-tensor path allocates for the second-moment sqrt. Less memory and
    fewer launches; the default path also failed here with a CUDA "unknown error".
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if name.endswith("bias") or "LayerNorm" in name else decay).append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": ft["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=ft["lr"], fused=True,
    )


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Best-epoch copy of the trainable parameters, kept on the GPU.

    Frozen weights cannot change, so they are skipped. Staying on the device avoids
    a 1 GB device-to-host copy, which crashed this setup (torch 2.14.1+cu130, WDDM
    driver 591.74) with a Windows access violation after a few dozen optimizer steps.
    Dropout has no state, and LayerNorm statistics are parameters, so this restores fully.
    """
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def _restore(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    for name, value in snapshot.items():
        params[name].copy_(value)


def fine_tune(
    train_texts: list[str], train_y: np.ndarray, val_texts: list[str], val_y: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[E5Classifier, Any, dict[str, Any]]:
    """Fine-tune with early stopping on val macro PR-AUC; return the best-epoch model."""
    ft, seed = cfg["finetune"], cfg["seed"]
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(ft["model"])
    model = E5Classifier(ft["model"], train_y.shape[1], ft["dropout"]).cuda()
    if ft["freeze_word_embeddings"]:
        # The 250k-token table is 69% of the parameters, but ~5k short answers touch a
        # small slice of it. Training it does not fit the 8 GB card: AdamW state is ~6 GB
        # and the best-epoch snapshot must stay on the GPU (see _snapshot), which ran out
        # of memory. All transformer layers, position embeddings and the head still train.
        model.encoder.embeddings.word_embeddings.weight.requires_grad_(False)
    train_loader = _loader(train_texts, train_y, tok, ft, train=True, seed=seed)
    val_loader = _loader(val_texts, None, tok, ft, train=False, seed=seed)

    optimizer = _optimizer(model, ft)
    total_steps = len(train_loader) * ft["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(ft["warmup_ratio"] * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=ft["fp16"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_y, ft["pos_weight_cap"]).cuda())

    best_ap, best_state, best_epoch, stale, history = -1.0, None, 0, 0, []
    for epoch in range(1, ft["epochs"] + 1):
        model.train()
        for batch in train_loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=ft["fp16"]):
                _, logits = model(batch["input_ids"].cuda(), batch["attention_mask"].cuda())
            loss = loss_fn(logits.float(), batch["labels"].cuda())  # loss in fp32 for stability
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), ft["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        val_ap = _macro_ap(val_y, _predict(model, val_loader, ft["fp16"])[1])
        history.append(round(val_ap, 5))
        if val_ap > best_ap:
            best_ap, best_epoch, stale = val_ap, epoch, 0
            best_state = _snapshot(model)
        else:
            stale += 1
            if stale >= ft["patience"]:
                break

    _restore(model, best_state)
    return model, tok, {"best_epoch": best_epoch, "val_macro_ap": history, "best_val_macro_ap": best_ap}


def _fingerprint(cfg: dict[str, Any], frame: pd.DataFrame) -> str:
    """Hash of fine-tune config, labels, seed and fold assignment: any change invalidates the cache."""
    folds = frame.sort_values(ROW_KEY)[[*ROW_KEY, "fold"]].to_numpy().tobytes()
    payload = json.dumps([cfg["finetune"], cfg["data"]["labels"], cfg["seed"]], sort_keys=True).encode()
    return hashlib.sha1(payload + folds).hexdigest()[:10]


def cache_path(tag: str, digest: str, cfg: dict[str, Any]) -> Path:
    return resolve(cfg["paths"]["processed_dir"]) / f"ft_e5_{digest}_{tag}.npz"


def _log_run(row: dict[str, Any], cfg: dict[str, Any]) -> None:
    path = resolve(cfg["finetune"]["log_csv"])
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_split(
    frame: pd.DataFrame, train_mask: np.ndarray, eval_mask: np.ndarray, tag: str, digest: str,
    cfg: dict[str, Any],
) -> Path:
    """Fine-tune on rows in train_mask (minus inner val), write features for eval_mask rows."""
    if (train_mask & eval_mask).any():
        raise ValueError("train and eval rows overlap")
    labels = cfg["data"]["labels"]
    train_frame = frame[train_mask].reset_index(drop=True)
    eval_frame = frame[eval_mask].reset_index(drop=True)
    tr_idx, val_idx = inner_split(train_frame, labels, cfg)
    y = train_frame[labels].to_numpy()
    texts = train_frame["text_transformer"].tolist()

    torch.cuda.synchronize()
    start = time.perf_counter()
    model, tok, info = fine_tune(
        [texts[i] for i in tr_idx], y[tr_idx], [texts[i] for i in val_idx], y[val_idx], cfg
    )
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    pooled, probs = _predict(
        model, _loader(eval_frame["text_transformer"].tolist(), None, tok, cfg["finetune"], False, cfg["seed"]),
        cfg["finetune"]["fp16"],
    )
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - start

    path = cache_path(tag, digest, cfg)
    meta = {"tag": tag, "n_train": len(tr_idx), "n_inner_val": len(val_idx), "n_eval": len(eval_frame),
            "train_folds": sorted(int(f) for f in train_frame["fold"].unique()), **info,
            "train_seconds": round(train_seconds, 1), "gpu_seconds": round(gpu_seconds, 1),
            "device": torch.cuda.get_device_name(0)}
    np.savez(path, source_row=eval_frame["source_row"].to_numpy(), record_pos=eval_frame["record_pos"].to_numpy(),
             fold=eval_frame["fold"].to_numpy(), embedding=pooled.astype(np.float32),
             probs=probs.astype(np.float32), meta=json.dumps(meta))
    _log_run({**meta, "val_macro_ap": json.dumps(meta["val_macro_ap"]),
              "train_folds": json.dumps(meta["train_folds"]),
              "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}, cfg)
    print(f"[{tag}] best epoch {info['best_epoch']} val AP {info['val_macro_ap']} "
          f"GPU time {gpu_seconds:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    return path


def finetune_jobs(folds: np.ndarray, n_folds: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(train_mask, eval_mask) per fine-tune.

    foldk: train on CV folds != k, write features for fold k only. The eval mask is
    explicit rather than "not train", because the complement would also contain the
    test rows. test: train on every CV row, write features for test rows only.
    """
    cv = folds != TEST_FOLD
    jobs = {f"fold{k}": (cv & (folds != k), folds == k) for k in range(n_folds)}
    jobs["test"] = (cv, folds == TEST_FOLD)
    return jobs


def build_finetune_features(cfg: dict[str, Any] | None = None) -> dict[str, Path]:
    """Run the 5 OOF fine-tunes plus the test fine-tune, skipping any already cached."""
    cfg = cfg or load_config()
    if not torch.cuda.is_available():
        raise RuntimeError("fine-tuning expects a CUDA GPU; install the CUDA build of torch")
    frame = load_split_frame(cfg)
    digest = _fingerprint(cfg, frame)
    paths = {}
    for tag, (train_mask, eval_mask) in finetune_jobs(frame["fold"].to_numpy(), cfg["split"]["n_folds"]).items():
        path = cache_path(tag, digest, cfg)
        if path.exists():
            print(f"[{tag}] cached: {path.name}", flush=True)
        else:
            path = run_split(frame, train_mask, eval_mask, tag, digest, cfg)
        paths[tag] = path
    return paths


def load_finetune_features(
    keys: pd.DataFrame, cfg: dict[str, Any] | None = None, split: str = "cv"
) -> tuple[np.ndarray, np.ndarray]:
    """Fine-tuned (embedding, head probabilities) aligned to keys' row order.

    split="cv" reads the five OOF fold files; split="test" reads the test file.
    """
    cfg = cfg or load_config()
    digest = _fingerprint(cfg, load_split_frame(cfg))
    tags = [f"fold{k}" for k in range(cfg["split"]["n_folds"])] if split == "cv" else ["test"]
    parts = []
    for tag in tags:
        path = cache_path(tag, digest, cfg)
        if not path.exists():
            raise FileNotFoundError(f"{path.name} missing: run python -m intent.finetune")
        own_fold = TEST_FOLD if tag == "test" else int(tag.removeprefix("fold"))
        with np.load(path) as z:
            keep = z["fold"] == own_fold  # guard: a file must only contribute its own fold's rows
            part = pd.DataFrame({"source_row": z["source_row"][keep], "record_pos": z["record_pos"][keep]})
            part["row"] = list(np.hstack([z["embedding"][keep], z["probs"][keep]]))
        parts.append(part)
    feats = keys[ROW_KEY].merge(pd.concat(parts), on=ROW_KEY, how="left", validate="one_to_one")
    if feats["row"].isna().any():
        raise ValueError("fine-tuned features do not cover all requested rows")
    stacked = np.stack(feats["row"].to_numpy())
    n_labels = len(cfg["data"]["labels"])
    return stacked[:, :-n_labels], stacked[:, -n_labels:]


if __name__ == "__main__":
    build_finetune_features()
