"""Leakage-critical pieces of stage 1, tested on CPU without loading the encoder."""

import numpy as np
import pandas as pd
import torch
from torch import nn

from intent.data import load_config
from intent.finetune import _restore, _snapshot, finetune_jobs, inner_split, mean_pool, pos_weight
from intent.split import TEST_FOLD

CFG = load_config()


def test_jobs_never_train_on_their_eval_rows_and_never_touch_test() -> None:
    folds = np.array([TEST_FOLD, 0, 1, 2, 3, 4, 0, 1, TEST_FOLD])
    jobs = finetune_jobs(folds, 5)
    for k in range(5):
        train, evaluate = jobs[f"fold{k}"]
        assert not (train & evaluate).any()
        assert (folds[evaluate] == k).all() and evaluate.sum() == (folds == k).sum()
        assert not train[folds == TEST_FOLD].any()
    train, evaluate = jobs["test"]
    assert (folds[evaluate] == TEST_FOLD).all() and not train[folds == TEST_FOLD].any()


def test_inner_split_is_group_aware() -> None:
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame(rng.random((n, 10)) < 0.2, columns=CFG["data"]["labels"]).astype(int)
    frame["group_key"] = [f"g{i // 2}" for i in range(n)]  # every group has two rows
    tr, val = inner_split(frame, CFG["data"]["labels"], CFG)
    assert set(frame["group_key"].iloc[tr]).isdisjoint(frame["group_key"].iloc[val])
    assert 0.05 < len(val) / n < 0.15


def test_pos_weight_is_neg_over_pos_and_capped() -> None:
    y = np.zeros((100, 2))
    y[:50, 0] = 1  # 50/50 -> 1.0
    y[:2, 1] = 1  # 98/2 -> 49, capped
    assert pos_weight(y, cap=10).tolist() == [1.0, 10.0]


def test_mean_pool_ignores_padding() -> None:
    hidden = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert mean_pool(hidden, mask).item() == 2.0


def test_snapshot_restores_trainable_and_skips_frozen() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    model[0].weight.requires_grad_(False)
    snap = _snapshot(model)
    assert "0.weight" not in snap
    before = model[1].weight.detach().clone()
    with torch.no_grad():
        model[1].weight.add_(1.0)
    _restore(model, snap)
    assert torch.equal(model[1].weight, before)
