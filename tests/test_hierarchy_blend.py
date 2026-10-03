"""E8 routing invariants and E9 cross-fitted blend weights, on synthetic data."""

import numpy as np

from intent.data import load_config
from intent.models import cross_fit_blend, hierarchical_oof

CFG = load_config()
LABELS = CFG["data"]["labels"]
NO = LABELS.index("no")


def _synthetic(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.zeros((n, len(LABELS)), dtype=int)
    y[np.arange(n), rng.integers(0, len(LABELS), n)] = 1
    X = y * 2.0 + rng.normal(0, 1, y.shape)  # informative but noisy features
    return X, y, np.arange(n) % 5


def test_blend_puts_all_weight_on_perfect_scorer() -> None:
    _, y, folds = _synthetic()
    perfect = y * 0.9 + 0.05
    noise = np.random.default_rng(1).random(y.shape) * 100  # large scale: only w = 1 separates perfectly
    _, _, weights = cross_fit_blend(y, perfect, noise, folds, LABELS)
    assert set(weights.values()) == {1.0}


def test_hierarchy_never_mixes_no_with_business_and_always_labels() -> None:
    X, y, folds = _synthetic()
    pred, score, info = hierarchical_oof(X, y, folds, LABELS, CFG)
    business = [j for j in range(len(LABELS)) if j != NO]
    routed_no = pred[:, NO] == 1
    assert not pred[routed_no][:, business].any()
    assert (pred.sum(axis=1) >= 1).all()
    assert np.allclose(score[:, NO] + score[:, business].max(axis=1) <= 1.0 + 1e-9, True)
    assert 0 < info["routed_no_share"] < 1
