"""Metrics, experiment logging and evaluation figures.

Responsibilities:
- Compute macro F1, micro F1, Hamming loss, samples F1, per-label precision,
  recall, F1 and PR-AUC.
- Bootstrap 95% CIs for macro F1 and temp F1, resampling groups (normalized text)
  rather than rows, to match how the split treats duplicates.
- Append exactly one row per experiment to reports/results.csv (with seed and
  timestamp) so every number in the report traces back to a logged run.

Why macro F1 as primary: Hamming loss is dominated by negatives (all-zeros already
scores about 0.115), so it hides failure on rare labels.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from intent.data import load_config, resolve


def result_columns(labels: list[str]) -> list[str]:
    """Schema of results.csv; log_result refuses to append under a different header."""
    return [
        "experiment", "eval_set", "note", "macro_f1", "micro_f1", "hamming_loss", "samples_f1",
        *[f"pr_auc_{lab}" for lab in labels],
        "temp_f1", "macro_f1_ci_low", "macro_f1_ci_high", "temp_f1_ci_low", "temp_f1_ci_high",
        "seed", "timestamp",
    ]


def per_label_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, labels: list[str]
) -> pd.DataFrame:
    """Precision, recall, F1, PR-AUC and support for each label.

    PR-AUC (average precision) is threshold-free, so it separates ranking quality
    from decoding quality; a constant score yields AP equal to prevalence.
    """
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    ap = [average_precision_score(y_true[:, j], y_score[:, j]) for j in range(len(labels))]
    return pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "pr_auc": ap, "support": support},
        index=pd.Index(labels, name="label"),
    )


def summary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Aggregate metrics. zero_division=0 so empty predictions are penalized, not skipped."""
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
    }


def _weighted_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)


def _confusion_parts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, ...]:
    """Row x label indicator arrays for tp, fp, fn; a weighted sum over rows gives counts."""
    return tuple(a.astype(float) for a in (y_true & y_pred, ~y_true & y_pred, y_true & ~y_pred))


def _group_weights(groups: np.ndarray, n_resamples: int, seed: int):
    """Yield per-row weights for each group-level bootstrap draw.

    Drawing groups with replacement and counting how often each was drawn turns a
    resample into a weight vector, so metrics become weighted sums with no row copying.
    The same seed yields the same draws, which is what makes paired comparisons paired.
    """
    rng = np.random.default_rng(seed)
    _, code = np.unique(groups, return_inverse=True)
    n_groups = code.max() + 1
    for _ in range(n_resamples):
        yield np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)[code]


def _weighted_metric(
    metric: str, y_true: np.ndarray, y_pred: np.ndarray, focus_idx: int
) -> Callable[[np.ndarray], float]:
    """Precompute per-row pieces of a metric; return a function of the bootstrap weights."""
    tp, fp, fn = _confusion_parts(y_true, y_pred)
    if metric == "macro_f1":
        return lambda w: _weighted_f1(w @ tp, w @ fp, w @ fn).mean()
    if metric == "temp_f1":
        return lambda w: _weighted_f1(w @ tp, w @ fp, w @ fn)[focus_idx]
    if metric == "micro_f1":
        return lambda w: _weighted_f1(*(np.atleast_1d((w @ a).sum()) for a in (tp, fp, fn)))[0]
    if metric == "samples_f1":
        row = _weighted_f1(tp.sum(1), fp.sum(1), fn.sum(1))
        return lambda w: (w @ row) / w.sum()
    if metric == "hamming_loss":
        row = (fp + fn).mean(1)
        return lambda w: (w @ row) / w.sum()
    raise ValueError(f"unknown metric {metric}")


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    metric: str = "macro_f1",
    n: int = 2000,
    focus_idx: int = -1,
    seed: int = 42,
) -> dict[str, float]:
    """Improvement of b over a (b - a; a - b for Hamming loss) with a group-level 95% CI.

    Both systems are scored on each identical resample, so the shared difficulty of
    a draw cancels out. Two overlapping marginal CIs say nothing about a paired
    difference; this does. p_b_not_better is the share of draws with no improvement.
    """
    y_true = y_true.astype(bool)
    score_a = _weighted_metric(metric, y_true, pred_a.astype(bool), focus_idx)
    score_b = _weighted_metric(metric, y_true, pred_b.astype(bool), focus_idx)
    sign = -1.0 if metric == "hamming_loss" else 1.0  # lower is better: improvement is positive
    ones = np.ones(len(y_true))
    deltas = sign * np.array([score_b(w) - score_a(w) for w in _group_weights(groups, n, seed)])
    return {
        "metric": metric,
        "score_a": score_a(ones), "score_b": score_b(ones),
        "delta": sign * (score_b(ones) - score_a(ones)),
        "ci_low": np.percentile(deltas, 2.5), "ci_high": np.percentile(deltas, 97.5),
        "p_b_not_better": float((deltas <= 0).mean()),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    focus_idx: int,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile CIs for macro F1 and the focus label's F1.

    Resamples groups with replacement and turns the draw into per-row weights, so
    each resample is a weighted sum of precomputed tp/fp/fn instead of a fresh
    f1_score call. Group-level: duplicates move together, as in the split.
    """
    tp, fp, fn = _confusion_parts(y_true.astype(bool), y_pred.astype(bool))
    macro, focus = np.empty(n_resamples), np.empty(n_resamples)
    for b, w in enumerate(_group_weights(groups, n_resamples, seed)):
        f1 = _weighted_f1(w @ tp, w @ fp, w @ fn)
        macro[b], focus[b] = f1.mean(), f1[focus_idx]

    lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    return {
        "macro_f1_ci_low": np.percentile(macro, lo), "macro_f1_ci_high": np.percentile(macro, hi),
        "temp_f1_ci_low": np.percentile(focus, lo), "temp_f1_ci_high": np.percentile(focus, hi),
    }


def log_result(row: dict[str, Any], cfg: dict[str, Any] | None = None) -> None:
    """Append one experiment row to results.csv, adding seed and timestamp.

    Fails loudly if the file's header differs from the current schema, because
    silently appending misaligned columns would corrupt every later comparison.
    """
    cfg = cfg or load_config()
    columns = result_columns(cfg["data"]["labels"])
    path = resolve(cfg["paths"]["results_csv"])
    row = {**row, "seed": cfg["seed"], "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if path.exists() and path.stat().st_size > 0:
        with open(path, encoding="utf-8", newline="") as f:
            header = next(csv.reader(f))
        if header != columns:
            raise ValueError(f"{path} header does not match the results schema; migrate it first")
        write_header = False
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = True
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        if write_header:
            writer.writeheader()
        writer.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})


def _log_per_label(name: str, eval_set: str, table: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Long-format per-label metrics, kept out of results.csv to keep that file one row per run."""
    path = resolve(cfg["paths"]["per_label_csv"])
    out = table.reset_index().assign(experiment=name, eval_set=eval_set).round(5)
    out = out[["experiment", "eval_set", *table.reset_index().columns]]
    out.to_csv(path, mode="a", header=not path.exists(), index=False)


def evaluate(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    cfg: dict[str, Any] | None = None,
    eval_set: str = "cv_oof",
    log: bool = True,
    note: str = "",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute every metric for one experiment and optionally log it.

    Returns (summary row, per-label table). eval_set distinguishes pooled
    out-of-fold CV predictions from the single final test evaluation.
    """
    cfg = cfg or load_config()
    labels = cfg["data"]["labels"]
    focus_idx = labels.index(cfg["evaluation"]["focus_label"])
    y_true, y_pred = y_true.astype(bool), y_pred.astype(bool)

    table = per_label_metrics(y_true, y_pred, y_score, labels)
    boot = cfg["evaluation"]["bootstrap"]
    row: dict[str, Any] = {
        "experiment": name,
        "eval_set": eval_set,
        "note": note,  # e.g. "diagnostic": rows that must never drive model selection
        **summary_metrics(y_true, y_pred),
        **{f"pr_auc_{lab}": table.loc[lab, "pr_auc"] for lab in labels},
        "temp_f1": table.loc[labels[focus_idx], "f1"],
        **bootstrap_ci(y_true, y_pred, groups, focus_idx, boot["n_resamples"], boot["ci"], cfg["seed"]),
    }
    if log:
        log_result(row, cfg)
        _log_per_label(name, eval_set, table, cfg)
    return row, table


# Ablation figure palette: emphasis encoding (one accent, the rest neutral gray).
# Validated with the dataviz validator on the light surface: contrast >= 3:1 for both,
# CVD dE 15.9 and normal-vision dE 17.8 between them. The gray fails the chroma floor
# by design: it is meant to read as "not selected", and every bar has a text label.
_ABLATION_STYLE = {
    "surface": "#fcfcfb", "bar": "#898781", "accent": "#2a78d6", "ink": "#0b0b0b",
    "ink2": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7",
}


def load_results(cfg: dict[str, Any] | None = None, eval_set_prefix: str = "cv") -> pd.DataFrame:
    """results.csv rows whose eval_set starts with eval_set_prefix, indexed by experiment.

    Defaults to CV rows so test-set rows (same experiment names) can never leak
    into the CV ablation or selection. If a name was logged twice, the latest row wins.
    """
    cfg = cfg or load_config()
    df = pd.read_csv(resolve(cfg["paths"]["results_csv"]), keep_default_na=False)
    df = df[df["eval_set"].str.startswith(eval_set_prefix)]
    return df.drop_duplicates("experiment", keep="last").set_index("experiment")


def ablation_table(spec: list[tuple[str, str, str]], cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Rows of results.csv in spec order, as (block, experiment, description) plus metrics."""
    results = load_results(cfg)
    missing = [e for _, e, _ in spec if e not in results.index]
    if missing:
        raise KeyError(f"not in results.csv: {missing}")
    rows = []
    for block, exp, desc in spec:
        r = results.loc[exp]
        pr_auc = r[[c for c in results.columns if c.startswith("pr_auc_")]].astype(float)
        rows.append({
            "block": block, "experiment": exp, "description": desc,
            "macro_f1": float(r["macro_f1"]), "ci_low": float(r["macro_f1_ci_low"]),
            "ci_high": float(r["macro_f1_ci_high"]), "micro_f1": float(r["micro_f1"]),
            "hamming_loss": float(r["hamming_loss"]), "samples_f1": float(r["samples_f1"]),
            "temp_f1": float(r["temp_f1"]), "mean_pr_auc": float(pr_auc.mean()),
        })
    return pd.DataFrame(rows)


def ablation_markdown(table: pd.DataFrame, selected: str) -> str:
    """One markdown table per block; the selected configuration is bolded."""
    out = []
    for block, part in table.groupby("block", sort=False):
        out += [f"**{block}**", "",
                "| Experiment | What changes | Macro F1 [95% CI] | Micro F1 | Hamming | Samples F1 | temp F1 | Mean PR-AUC |",
                "|---|---|---|---|---|---|---|---|"]
        for _, r in part.iterrows():
            name = f"**{r.experiment}** (selected)" if r.experiment == selected else r.experiment
            out.append(
                f"| {name} | {r.description} | {r.macro_f1:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] | "
                f"{r.micro_f1:.3f} | {r.hamming_loss:.4f} | {r.samples_f1:.3f} | {r.temp_f1:.3f} | {r.mean_pr_auc:.3f} |"
            )
        out.append("")
    return "\n".join(out)


def plot_ablation(
    table: pd.DataFrame, selected: str, reference: str, title: str, subtitle: str, path: Path
) -> None:
    """Horizontal bars of macro F1 with 95% CIs, grouped by block, one accent for the selection.

    Bars start at zero (truncating would exaggerate 0.03 gaps). Values are labeled
    only for the reference, the selection and non-comparable rows; the markdown
    table carries every number.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = _ABLATION_STYLE
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})

    # y layout: a header slot per block, one slot per experiment, a spacer between blocks
    ys, labels, headers, y = [], [], [], 0.0
    for block, part in table.groupby("block", sort=False):
        headers.append((y, block))
        y += 1
        for exp in part["experiment"]:
            ys.append(y)
            labels.append(exp)
            y += 1
        y += 0.5
    n_slots = y
    row_in = 0.28
    fig, ax = plt.subplots(figsize=(9, 1.3 + n_slots * row_in), dpi=200)
    fig.patch.set_facecolor(st["surface"])
    ax.set_facecolor(st["surface"])

    bar_h = 24 / (row_in * 200)  # cap bar thickness at 24 px in data units (1 slot = row_in inches)
    ref_value = float(table.loc[table["experiment"] == reference, "macro_f1"].iloc[0])
    for yi, (_, r) in zip(ys, table.iterrows()):
        is_sel = r.experiment == selected
        ax.barh(yi, r.macro_f1, height=bar_h, color=st["accent"] if is_sel else st["bar"], zorder=2)
        ax.errorbar(r.macro_f1, yi, xerr=[[r.macro_f1 - r.ci_low], [r.ci_high - r.macro_f1]],
                    fmt="none", ecolor=st["ink2"], elinewidth=1.2, capsize=2.5, zorder=3)
        if is_sel or r.experiment == reference or r.block.startswith("Not comparable"):
            ax.text(r.ci_high + 0.01, yi, f"{r.macro_f1:.3f}", va="center", ha="left",
                    color=st["ink"], fontsize=8.5, fontweight="bold" if is_sel else "normal")

    ax.axvline(ref_value, color=st["axis"], linestyle=(0, (4, 3)), linewidth=1, zorder=1)
    ax.text(ref_value, -0.9, f"B2 reference {ref_value:.3f}", ha="center", va="bottom",
            color=st["ink2"], fontsize=8)
    from matplotlib.transforms import blended_transform_factory

    header_tf = blended_transform_factory(fig.transFigure, ax.transData)  # x: figure edge, y: row
    for hy, block in headers:
        ax.text(0.02, hy, block, ha="left", va="center", fontweight="bold", color=st["ink"],
                fontsize=9, transform=header_tf)

    ax.set_yticks(ys, labels, color=st["ink2"])
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", colors=st["ink2"], length=0)
    ax.set_ylim(n_slots - 0.2, -1.2)
    ax.set_xlim(0, max(0.8, float(table["ci_high"].max()) + 0.08))
    ax.xaxis.grid(True, color=st["grid"], linewidth=0.6, zorder=0)
    ax.set_xlabel("Macro F1 (out-of-fold, 5-fold grouped CV)", color=st["ink2"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(st["axis"])
    # Block headers sit in the tick-label column, so leave room on the left.
    fig.subplots_adjust(left=0.30, right=0.97, top=1 - 0.95 / fig.get_figheight(), bottom=0.5 / fig.get_figheight())
    fig.text(0.02, 1 - 0.25 / fig.get_figheight(), title, ha="left", va="top", fontsize=12,
             fontweight="bold", color=st["ink"])
    fig.text(0.02, 1 - 0.55 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=8.5,
             color=st["ink2"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=st["surface"])
    plt.close(fig)


def bootstrap_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, groups: np.ndarray, focus_idx: int,
    n_resamples: int = 1000, ci: float = 0.95, seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """(point, ci_low, ci_high) for every reported metric, resampling groups.

    Covers both readings of the company's "average precision": macro precision of
    the decoded labels and mean PR-AUC (average_precision_score accepts the
    bootstrap weights as sample_weight). Micro precision is a third reading.
    """
    y_true, y_pred = y_true.astype(bool), y_pred.astype(bool)
    tp, fp, fn = _confusion_parts(y_true, y_pred)

    def prec(t: np.ndarray, f: np.ndarray) -> np.ndarray:
        return np.divide(t, t + f, out=np.zeros_like(t, dtype=float), where=(t + f) > 0)

    def mean_ap(w: np.ndarray) -> float:
        aps = [average_precision_score(y_true[:, j], y_score[:, j], sample_weight=w)
               for j in range(y_true.shape[1]) if (w * y_true[:, j]).sum() > 0]
        return float(np.mean(aps))

    fns = {name: _weighted_metric(name, y_true, y_pred, focus_idx)
           for name in ["macro_f1", "micro_f1", "hamming_loss", "samples_f1", "temp_f1"]}
    fns["macro_precision"] = lambda w: prec(w @ tp, w @ fp).mean()
    fns["macro_recall"] = lambda w: prec(w @ tp, w @ fn).mean()
    fns["micro_precision"] = lambda w: float(prec(np.atleast_1d((w @ tp).sum()), np.atleast_1d((w @ fp).sum()))[0])
    fns["mean_pr_auc"] = mean_ap

    point = {k: float(f(np.ones(len(y_true)))) for k, f in fns.items()}
    draws = {k: [] for k in fns}
    for w in _group_weights(groups, n_resamples, seed):
        for k, f in fns.items():
            draws[k].append(f(w))
    lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    return {k: (point[k], float(np.percentile(draws[k], lo)), float(np.percentile(draws[k], hi))) for k in fns}


def plot_pr_curves(
    y_true: np.ndarray, scores: dict[str, np.ndarray], labels: list[str], title: str, subtitle: str, path: Path,
) -> None:
    """Small multiples: one PR panel per label, the selected model in the accent, the reference in gray.

    The first entry of scores is drawn in the accent. The dotted line is the label's
    prevalence (a random ranker's PR-AUC). AP values are printed in ink in model order.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    st = _ABLATION_STYLE
    colors = [st["accent"], st["bar"]]
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8.5})
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.6), dpi=200, sharex=True, sharey=True)
    fig.patch.set_facecolor(st["surface"])
    names = list(scores)
    for j, (ax, label) in enumerate(zip(axes.flat, labels)):
        ax.set_facecolor(st["surface"])
        prevalence = y_true[:, j].mean()
        ax.axhline(prevalence, color=st["axis"], linestyle=(0, (1, 2)), linewidth=1)
        aps = []
        for name, color in reversed(list(zip(names, colors))):  # accent drawn last, on top
            precision, recall, _ = precision_recall_curve(y_true[:, j], scores[name][:, j])
            ax.plot(recall, precision, color=color, linewidth=2 if color == st["accent"] else 1.6,
                    drawstyle="steps-post", solid_joinstyle="round")
        for name in names:
            aps.append(average_precision_score(y_true[:, j], scores[name][:, j]))
        ax.set_title(f"{label}  (n={int(y_true[:, j].sum())})", loc="left", color=st["ink"], fontsize=9.5,
                     fontweight="bold")
        ax.text(0.03, 0.04, "AP " + " / ".join(f"{a:.2f}" for a in aps), transform=ax.transAxes,
                color=st["ink"], fontsize=8.5,
                bbox={"facecolor": st["surface"], "edgecolor": "none", "pad": 1.5})  # mask the prevalence line
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, color=st["grid"], linewidth=0.6)
        ax.tick_params(colors=st["ink2"], length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(st["axis"])
    for ax in axes[1]:
        ax.set_xlabel("Recall", color=st["ink2"])
    for ax in axes[:, 0]:
        ax.set_ylabel("Precision", color=st["ink2"])
    handles = [Line2D([], [], color=c, linewidth=2) for c in colors[: len(names)]]
    handles.append(Line2D([], [], color=st["axis"], linestyle=(0, (1, 2)), linewidth=1))
    fig.legend(handles, [*names, "prevalence (random ranker)"], loc="upper right", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.99, 0.955), labelcolor=st["ink2"])
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=12, fontweight="bold", color=st["ink"])
    fig.text(0.01, 0.94, subtitle, ha="left", va="top", fontsize=8.5, color=st["ink2"])
    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.09, wspace=0.12, hspace=0.32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=st["surface"])
    plt.close(fig)


# ----------------------------------------------------------------------------- error analysis
# Pure functions over arrays; models.error_analysis_data assembles the inputs.

GENERIC_TOKENS = frozenset({
    "capital", "trabajo", "crecimiento", "crecer", "inversion", "invertir", "negocio", "empresa",
    "expansion", "expandir", "ampliar", "ampliacion", "mas", "mejorar", "mejora", "general", "operacion",
    "proyecto", "desarrollo", "liquidez", "recurso", "apoyo", "dinero", "necesidad", "seguir", "continuar",
    "actividad", "poder", "tener", "hacer",
})
SEASON_PATTERN = (r"temporada|navid|diciembre|fin de ano|buen fin|vacacion|escolar|regreso a clase"
                  r"|dia de|san valentin|fiestas|epoca|alta demanda")
CYCLE_PATTERN = (r"cobr|factur|credito a (?:mis )?clientes|plazo|\b(?:15|30|45|60|90|120) dias|pedido"
                 r"|contrato|orden(?:es)? de compra|licitaci|obra|pago de (?:mis |los )?clientes|liquidez|flujo")
LIST_PATTERN = r",| y | e |asi como|ademas"
# Priority for the primary category, in the order the causes were requested: text that
# carries too little information first, label noise last (the residual explanation when
# the text is clear and single-intent but the model confidently disagrees).
CATEGORY_ORDER = ["very short or vague", "multiple intents", "temp ambiguity",
                  "personal vs business", "likely label error", "other"]


def label_set(row: np.ndarray, labels: list[str]) -> str:
    """'inv + equ' style rendering of a 0/1 row."""
    return " + ".join(lab for lab, v in zip(labels, row) if v) or "(none)"


def row_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-row F1 between label sets (the samples-F1 summand)."""
    inter = (y_true & y_pred).sum(axis=1)
    denom = y_true.sum(axis=1) + y_pred.sum(axis=1)
    return np.divide(2 * inter, denom, out=np.ones(len(y_true)), where=denom > 0)


def find_label_errors(y: np.ndarray, pred_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cleanlab multi-label issues and label quality scores (lower = more likely mislabeled).

    pred_probs must be out-of-fold: cleanlab compares each given label with what a
    model that never saw that row believes. Our probabilities are per-label
    one-vs-rest, which is what multi-label mode expects.
    """
    from cleanlab.filter import find_label_issues
    from cleanlab.multilabel_classification import get_label_quality_scores

    as_lists = [list(np.flatnonzero(r)) for r in y]
    issues = find_label_issues(labels=as_lists, pred_probs=pred_probs, multi_label=True)
    quality = get_label_quality_scores(as_lists, pred_probs)
    return issues, quality


def conflicting_duplicate_mask(group_keys: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rows whose normalized text appears more than once with different label vectors."""
    vec = pd.Series(["".join(map(str, r)) for r in y.astype(int)])
    n_distinct = vec.groupby(pd.Series(group_keys)).transform("nunique").to_numpy()
    return n_distinct > 1


def _norm(text: str) -> str:
    from intent.preprocess import strip_accents

    return strip_accents(text.lower())


def categorize_errors(
    texts: pd.Series, tfidf_texts: pd.Series, y: np.ndarray, pred: np.ndarray, issues: np.ndarray,
    labels: list[str], score: np.ndarray | None = None,
) -> pd.DataFrame:
    """One row per misclassified example (predicted set != given set) with category flags.

    primary is the first matching flag in CATEGORY_ORDER, so counts sum to the number
    of errors; the flag columns allow overlaps. Definitions:
    - very short or vague: <= 5 words, or every content token is generic (GENERIC_TOKENS).
    - multiple intents: (a) the given set has >= 2 labels, OR (b) the predicted set has
      >= 2 labels AND the lowercased, accent-stripped text matches LIST_PATTERN (a comma,
      " y ", " e ", "asi como" or "ademas"). No parsing: (b) is a lexical proxy for "the text
      enumerates things", so it also fires on coordinated nouns within one use
      ("materiales y personal"). A one-item text with extra predicted labels does not count.
    - temp ambiguity / personal vs business: the error involves temp / no.
    - likely label error: cleanlab flags the row.
    """
    y, pred = y.astype(bool), pred.astype(bool)
    wrong = (y != pred).any(axis=1)
    no, temp = labels.index("no"), labels.index("temp")
    tokens = tfidf_texts.str.split()
    lists_items = texts.map(_norm).str.contains(LIST_PATTERN).to_numpy()
    flags = pd.DataFrame({
        "very short or vague": (texts.str.split().str.len() <= 5).to_numpy()
                                | tokens.map(lambda t: set(t) <= GENERIC_TOKENS).to_numpy(),
        "multiple intents": (y.sum(axis=1) >= 2) | ((pred.sum(axis=1) >= 2) & lists_items),
        "temp ambiguity": y[:, temp] != pred[:, temp],
        "personal vs business": y[:, no] != pred[:, no],
        "likely label error": issues,
    })
    flags["other"] = ~flags.any(axis=1)
    out = flags[wrong].copy()
    out["primary"] = out[CATEGORY_ORDER].idxmax(axis=1)  # first True in priority order
    out["text"] = texts[wrong].to_numpy()
    out["given"] = [label_set(r, labels) for r in y[wrong]]
    out["predicted"] = [label_set(r, labels) for r in pred[wrong]]
    norm = out["text"].map(_norm)
    season, cycle = norm.str.contains(SEASON_PATTERN), norm.str.contains(CYCLE_PATTERN)
    out["temp_reading"] = np.select([season & cycle, season, cycle], ["both", "season", "cycle"], "neither")
    if score is not None:
        # how confidently the model disagrees: mean |score - given| over the mismatched labels
        mism = (y != pred)[wrong]
        out["disagreement"] = (np.abs(score[wrong] - y[wrong]) * mism).sum(axis=1) / mism.sum(axis=1)
    return out


def category_examples(errors: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """k clearest examples per primary category: highest disagreement, repeated texts dropped."""
    order = errors.sort_values("disagreement", ascending=False) if "disagreement" in errors else errors
    order = order.drop_duplicates("text")
    parts = [order[order["primary"] == c].head(k) for c in CATEGORY_ORDER]
    return pd.concat(parts)[["primary", "text", "given", "predicted"]]


def top_label_errors(
    texts: pd.Series, y: np.ndarray, pred: np.ndarray, quality: np.ndarray, issues: np.ndarray,
    conflicting: np.ndarray, labels: list[str], k: int = 10,
) -> pd.DataFrame:
    """The k flagged rows with the lowest cleanlab label quality: given vs suggested labels.

    suggested is the selected model's decoded prediction (its thresholds and rules),
    so it is a label set we would actually output, not a raw argmax.
    """
    idx = np.flatnonzero(issues)
    idx = idx[np.argsort(quality[idx])][:k]
    return pd.DataFrame({
        "label_quality": np.round(quality[idx], 3), "text": texts.iloc[idx].to_numpy(),
        "given": [label_set(r, labels) for r in y[idx]], "suggested": [label_set(r, labels) for r in pred[idx]],
        "conflicting_duplicate": conflicting[idx],
    }, index=idx)


def temp_readings(texts: pd.Series, y: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """How temp-labeled answers read: seasonal demand, receivables/payment cycle, both or neither.

    Also counts non-temp answers that mention a season, the other side of the ambiguity.
    """
    norm = texts.map(_norm)
    season, cycle = norm.str.contains(SEASON_PATTERN).to_numpy(), norm.str.contains(CYCLE_PATTERN).to_numpy()
    reading = np.select([season & cycle, season, cycle], ["both", "season", "cycle"], "neither")
    is_temp = y[:, labels.index("temp")].astype(bool)
    table = pd.DataFrame({
        "temp rows": pd.Series(reading[is_temp]).value_counts(),
        "non-temp rows": pd.Series(reading[~is_temp]).value_counts(),
    }).reindex(["season", "cycle", "both", "neither"]).fillna(0).astype(int)
    table["share of temp rows"] = (table["temp rows"] / is_temp.sum()).round(3)
    return table


def confused_pairs(y: np.ndarray, pred: np.ndarray, labels: list[str], top: int = 10) -> pd.DataFrame:
    """Pairs (missed A, predicted B instead) counted within the same row.

    share_of_A = count / support of A: how often A's positives are mistaken for B.
    """
    y, pred = y.astype(bool), pred.astype(bool)
    missed, extra = y & ~pred, pred & ~y
    counts = missed.T.astype(int) @ extra.astype(int)  # [a, b] = rows missing a while adding b
    rows = [{"missed (given)": labels[a], "predicted instead": labels[b], "rows": int(counts[a, b]),
             "share_of_missed_label": round(counts[a, b] / y[:, a].sum(), 3)}
            for a in range(len(labels)) for b in range(len(labels)) if counts[a, b] > 0]
    return pd.DataFrame(rows).sort_values("rows", ascending=False).head(top).reset_index(drop=True)


def compare_systems(
    texts: pd.Series, y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, score_a: np.ndarray,
    score_b: np.ndarray, labels: list[str], k: int = 5,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Rows where system a beats b and vice versa, by per-row F1.

    Examples are the k rows where the winner is exactly right and the gap is largest,
    ties broken by how confidently the loser was wrong (sum of |score - given| on the
    labels it got wrong), so the examples show clear wins rather than coin flips.
    """
    y = y.astype(bool)
    fa, fb = row_f1(y, pred_a.astype(bool)), row_f1(y, pred_b.astype(bool))
    summary = pd.Series({"a_better": int((fa > fb).sum()), "b_better": int((fb > fa).sum()),
                         "same": int((fa == fb).sum())})

    def examples(f_win, f_lose, pred_win, pred_lose, score_lose):
        wrong = pred_lose.astype(bool) != y
        conf = (np.abs(score_lose - y) * wrong).sum(axis=1)
        idx = np.flatnonzero((f_win == 1) & (f_lose < 1))
        idx = idx[np.lexsort((-conf[idx], -(f_win[idx] - f_lose[idx])))][:k]
        return pd.DataFrame({"text": texts.iloc[idx].to_numpy(),
                             "given": [label_set(r, labels) for r in y[idx]],
                             "winner": [label_set(r, labels) for r in pred_win[idx]],
                             "loser": [label_set(r, labels) for r in pred_lose[idx]]}, index=idx)

    return (summary, examples(fa, fb, pred_a, pred_b, score_b), examples(fb, fa, pred_b, pred_a, score_a))


def duplicate_crosscheck(
    issues: np.ndarray, quality: np.ndarray, conflicting: np.ndarray, top: int = 100
) -> pd.DataFrame:
    """Do cleanlab flags concentrate on conflicting duplicates (independent evidence of noise)?"""
    idx = np.flatnonzero(issues)
    idx = idx[np.argsort(quality[idx])][:top]
    return pd.DataFrame({
        "rows": [len(issues), int(conflicting.sum()), len(idx)],
        "flagged by cleanlab": [issues.mean(), issues[conflicting].mean(), 1.0],
        "in a conflicting duplicate group": [conflicting.mean(), 1.0, conflicting[idx].mean()],
    }, index=["all CV rows", "conflicting-duplicate rows", f"top {top} lowest label quality"]).round(3)


def error_category_counts(errors: pd.DataFrame) -> pd.DataFrame:
    """Primary cause next to "any applicable" counts, each with its share of all errors.

    Primary counts sum to the number of errors but depend on CATEGORY_ORDER; the
    "any applicable" counts do not depend on the order but overlap (a row can count
    in several categories), so they sum to more than the number of errors.
    """
    primary = errors["primary"].value_counts().reindex(CATEGORY_ORDER).fillna(0).astype(int)
    anyc = errors[CATEGORY_ORDER].sum().astype(int)
    return pd.DataFrame({
        "primary cause": primary, "primary share": (primary / len(errors)).round(3),
        "any applicable": anyc, "any applicable share": (anyc / len(errors)).round(3),
    })


def exact_match_ratio(y: np.ndarray, pred: np.ndarray) -> float:
    """Subset accuracy: share of rows whose predicted label set equals the given set exactly."""
    return float((y.astype(bool) == pred.astype(bool)).all(axis=1).mean())


def export_manual_audit(
    errors: pd.DataFrame, rows: pd.DataFrame, path: Path, n: int = 50, seed: int = 42
) -> pd.DataFrame:
    """Random sample of n errors for hand labeling, written to path with an empty manual_category.

    Never overwrites hand-filled work: if path exists and any manual_category cell is
    filled, the file is left as is and returned. utf-8-sig so Excel shows accents.
    """
    if path.exists():
        existing = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        if (existing["manual_category"].astype(str).str.strip() != "").any():
            return existing
    sample = errors.sample(n=n, random_state=seed)
    out = pd.DataFrame({
        "source_row": rows.loc[sample.index, "source_row"].to_numpy(),
        "record_pos": rows.loc[sample.index, "record_pos"].to_numpy(),
        "text": sample["text"].to_numpy(),
        "true_labels": sample["given"].to_numpy(),
        "predicted_labels": sample["predicted"].to_numpy(),
        "assigned_category": sample["primary"].to_numpy(),
        "manual_category": "",
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


# ----------------------------------------------------------------------------- manual audit
# The audit instructions defined "other" as "text clear, label correct, model simply wrong";
# the hand labels call that case model_error, so the two names denote the same category.
MANUAL_ALIASES = {"model_error": "other"}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion: stays inside [0, 1] and behaves at k = 0 or
    small n, where the normal approximation does not (50 audited rows, some cells near 0)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _manual(audit: pd.DataFrame) -> pd.Series:
    return audit["manual_category"].astype(str).str.strip().replace(MANUAL_ALIASES)


def audit_agreement(audit: pd.DataFrame) -> dict[str, Any]:
    """Heuristic (assigned_category) vs hand label: confusion matrix, agreement, Cohen's kappa.

    Kappa corrects agreement for chance, which matters here because one category
    (multiple intents) holds most rows: always guessing it would already agree often.
    strict_agreement compares the raw strings, before mapping model_error to other.
    """
    from sklearn.metrics import cohen_kappa_score

    manual, heur = _manual(audit), audit["assigned_category"].astype(str)
    unknown = set(manual) - set(CATEGORY_ORDER)
    if unknown:
        raise ValueError(f"manual_category has values outside the categories: {sorted(unknown)}")
    cats = [c for c in CATEGORY_ORDER if c in set(manual) | set(heur)]
    confusion = pd.crosstab(heur, manual).reindex(index=cats, columns=cats, fill_value=0)
    confusion.index.name, confusion.columns.name = "heuristic", "manual"
    return {
        "n": len(audit),
        "confusion": confusion,
        "agreement": float((manual == heur).mean()),
        "strict_agreement": float((audit["manual_category"].astype(str).str.strip() == heur).mean()),
        "kappa": float(cohen_kappa_score(heur, manual)),
    }


def audit_distribution(audit: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    """Hand-labeled shares with 95% Wilson intervals, next to the heuristic primary shares
    in the same 50 rows and over all errors (the population the sample was drawn from)."""
    manual, n = _manual(audit), len(audit)
    rows = {}
    for cat in CATEGORY_ORDER:
        k = int((manual == cat).sum())
        lo, hi = wilson_interval(k, n)
        rows[cat] = {
            "manual n": k, "manual share": k / n, "wilson low": lo, "wilson high": hi,
            "heuristic share (same 50)": float((audit["assigned_category"] == cat).mean()),
            "heuristic primary share (all errors)": float((errors["primary"] == cat).mean()),
        }
    table = pd.DataFrame(rows).T
    table["manual n"] = table["manual n"].astype(int)
    return table.rename(index={"other": "other (manual: model_error)"}).round(3)
