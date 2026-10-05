"""Spanish figures for the interview deck, one function per figure, written to reports/figures/deck/.

Why a separate module: the report keeps its English figures (reports/figures/*.png) and
the deck needs the template's dark theme, Spanish labels and slide-sized fonts. Sharing
the numbers but not the drawing code keeps the report figures untouched.

Every number is read from reports/ (results.csv, test_results.csv, test_per_label.csv,
test_paired_vs_B2.csv, manual_audit.csv, latency.csv) or recomputed from the cleaned data
with the same functions the report used, and then checked against the value the report
states (_check). No titles inside images: the slide title carries the message.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from intent import plot_style as ps
from intent.data import load_config, resolve
from intent.evaluate import MANUAL_ALIASES, load_results, temp_readings, wilson_interval
from intent.preprocess import clean_for_transformer

# Short Spanish descriptions used on every slide (reports/glossary_es.md). inv is the
# project's documented assumption: the codebook does not define it.
LABEL_ES = {
    "crec": "crecimiento sin uso específico", "cred": "pago de deudas", "equ": "equipo",
    "inic": "iniciar un negocio", "inv": "inventario o mercancía", "mkt": "marketing",
    "no": "no destinado a capital de trabajo", "renta": "renta", "sueldo": "nómina",
    "temp": "ventas de temporada",
}
SELECTED, REFERENCE = "E9b_blend_B2_E4", "B2_tfidf_lr_br"


def _check(value: float, expected: float, tol: float, what: str) -> None:
    """Refuse to draw a number that disagrees with the one written in reports/report.md."""
    if abs(value - expected) > tol:
        raise ValueError(f"{what}: computed {value} but the report states {expected}")


def _reports(cfg: dict[str, Any], name: str) -> Path:
    return resolve(cfg["paths"]["reports_dir"]) / name


def _clean_with_split(cfg: dict[str, Any]) -> pd.DataFrame:
    """Clean rows with their split and fold (data/interim + data/processed)."""
    clean = pd.read_parquet(resolve(cfg["paths"]["interim_dir"]) / cfg["data"]["clean_file"])
    folds = pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / cfg["split"]["folds_file"])
    return clean.merge(folds[["source_row", "record_pos", "split", "fold"]], on=["source_row", "record_pos"])


def _test_metrics(cfg: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(_reports(cfg, "test_results.csv")).set_index(["model", "metric"])


def _bar_value(ax, x: float, y: float, text: str, bold: bool = False, color: str = ps.INK) -> None:
    ax.text(x, y, text, va="center", ha="left", fontsize=ps.TEXT, color=color,
            fontweight="bold" if bold else "normal",
            bbox={"facecolor": ps.SURFACE, "edgecolor": "none", "pad": 1.5})  # mask grid and reference lines


# ----------------------------------------------------------------------------- figure data
# One function per figure's numbers, each checked against the report, so every rendering
# (this template deck and intent.deck_figures_konfio) draws the same verified values.
ABLATION = [  # (experiment, code, description); the full table is appendix A1
    ("B2_tfidf_lr_br__thr0.5_norules", "B2", "TF-IDF + LR, umbral 0.5"),
    (REFERENCE, "B2", "con umbrales y reglas (ref.)"),
    ("B3_tfidf_lr_chain", "B3", "cadena de clasificadores"),
    ("E3_e5", "E3", "e5 congelado + LR"),
    ("E3_mpnet", "E3", "mpnet congelado + LR"),
    ("E4_ft_head", "E4", "e5 con fine-tuning"),
    ("E7_hybrid_lgbm", "E7", "híbrido con LightGBM"),
    (SELECTED, "E9b", "mezcla B2 + E4"),
]
ABLATION_GAPS = {3: 0.35, 5: 0.35, 7: 0.35}  # small breaks between lexical, frozen, fine-tuned, blend
TEST_F1 = [("macro_f1", "Macro F1"), ("micro_f1", "Micro F1"), ("samples_f1", "Samples F1")]
AUDIT = [  # (category in manual_audit.csv, caused by the model?)
    ("multiple intents", False), ("likely label error", False), ("other", True),
    ("personal vs business", False), ("temp ambiguity", False),
]


def prevalence(cfg: dict[str, Any]) -> pd.Series:
    """Share of clean answers carrying each label, ascending."""
    labels = cfg["data"]["labels"]
    share = _clean_with_split(cfg)[labels].astype(int).mean().sort_values()
    _check(round(100 * share["inv"], 1), 34.7, 0, "inv prevalence")
    _check(round(100 * share["temp"], 1), 2.8, 0, "temp prevalence")
    return share


def ablation(cfg: dict[str, Any]) -> pd.DataFrame:
    """CV macro F1 and its 95% CI for the ABLATION rows, in that order."""
    res = load_results(cfg)
    t = res.loc[[e for e, _, _ in ABLATION], ["macro_f1", "macro_f1_ci_low", "macro_f1_ci_high"]].astype(float)
    _check(round(t.loc[SELECTED, "macro_f1"], 3), 0.715, 0, "E9b CV macro F1")
    _check(round(t.loc[REFERENCE, "macro_f1"], 3), 0.671, 0, "B2 CV macro F1")
    return t


def ablation_positions() -> list[float]:
    """y of each ABLATION row, with the block gaps."""
    ys, y = [], 0.0
    for k in range(len(ABLATION)):
        y += ABLATION_GAPS.get(k, 0)
        ys.append(y)
        y += 1
    return ys


def test_scores(cfg: dict[str, Any]) -> pd.DataFrame:
    """Test metrics with CIs, indexed by (model, metric)."""
    m = _test_metrics(cfg)
    _check(round(m.loc[(SELECTED, "macro_f1"), "value"], 3), 0.732, 0, "E9b test macro F1")
    company = float(m.loc[("company_baseline_reported", "hamming_loss"), "value"])
    _check(company, cfg["evaluation"]["baselines"]["company_hamming"], 1e-9, "company Hamming")
    return m


def per_label_f1(cfg: dict[str, Any]) -> tuple[pd.Series, pd.Series, pd.Series, pd.Index]:
    """Test F1 of E9b and B2 and the support per label, plus labels sorted by E9b F1."""
    t = pd.read_csv(_reports(cfg, "test_per_label.csv")).set_index("label")
    e9, b2, n = (t[f"{SELECTED}__f1"], t[f"{REFERENCE}__f1"], t[f"{SELECTED}__support"].astype(int))
    _check(round(e9["temp"], 3), 0.400, 0, "temp test F1")
    return e9, b2, n, e9.sort_values(ascending=False).index


def audit_shares(cfg: dict[str, Any]) -> pd.DataFrame:
    """Hand-audit categories in AUDIT order: count, share and 95% Wilson interval."""
    audit = pd.read_csv(resolve(cfg["paths"]["manual_audit"]), encoding="utf-8-sig", keep_default_na=False)
    manual = audit["manual_category"].astype(str).str.strip().replace(MANUAL_ALIASES)
    n = len(audit)
    rows = []
    for cat, model in AUDIT:
        k = int((manual == cat).sum())
        lo, hi = wilson_interval(k, n)
        rows.append({"category": cat, "model": model, "k": k, "n": n, "share": k / n, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows).set_index("category")
    _check(out.loc["multiple intents", "share"], 0.58, 1e-9, "audit: labeling convention share")
    _check(out.loc["likely label error", "share"], 0.18, 1e-9, "audit: label error share")
    _check(out.loc["other", "share"], 0.12, 1e-9, "audit: model error share")
    return out


# ----------------------------------------------------------------------------- data and labels
def fig_prevalence(cfg: dict[str, Any]) -> Path:
    """Share of answers carrying each label: strong imbalance, temp the rarest."""
    share = prevalence(cfg)

    fig, ax = ps.figure(ps.WIDE)
    y = np.arange(len(share))
    ax.barh(y, 100 * share.to_numpy(), height=0.6, color=ps.ACCENT, zorder=2)  # one series: one color
    for yi, v in zip(y, share):
        _bar_value(ax, 100 * v + 0.6, yi, f"{100 * v:.1f}%")
    ax.set_yticks(y, [f"{lab}: {LABEL_ES[lab]}" for lab in share.index])
    ax.set_xlim(0, 40)
    ax.set_xticks([])  # every bar is labeled; an axis would repeat the numbers
    ax.set_xlabel("Porcentaje de respuestas con la etiqueta (6,691 respuestas limpias)")
    ps.finish_axes(ax, grid=None)
    fig.subplots_adjust(left=0.43, right=0.97, top=0.98, bottom=0.08)
    return ps.save(fig, "etiquetas_prevalencia", cfg)


def fig_split(cfg: dict[str, Any]) -> Path:
    """The fixed test set next to the 5 grouped CV folds, drawn to scale."""
    rows = _clean_with_split(cfg)
    n_cv, n_test = int((rows["split"] == "cv").sum()), int((rows["split"] == "test").sum())
    _check(n_cv, 5703, 0, "CV rows")
    _check(n_test, 988, 0, "test rows")
    fold_sizes = rows.loc[rows["split"] == "cv", "fold"].value_counts().sort_index().to_numpy()

    fig, ax = ps.figure(ps.WIDE)
    left, h, yb = 0, 0.9, 2.55
    for k, size in enumerate(fold_sizes):
        # the surface-colored edge is the 2 px gap between touching segments
        ax.barh(yb, size, left=left, height=h, color=ps.TEAL, edgecolor=ps.SURFACE, linewidth=3)
        ax.text(left + size / 2, yb, f"Fold {k + 1}", ha="center", va="center", fontsize=ps.TICK,
                color=ps.INK, fontweight="bold")
        left += size
    ax.barh(yb, n_test, left=left, height=h, color=ps.ROSE, edgecolor=ps.SURFACE, linewidth=3)
    ax.text(left + n_test / 2, yb, "Test", ha="center", va="center", fontsize=ps.TICK,
            color=ps.INK, fontweight="bold")
    total = left + n_test

    def part(x0: float, x1: float, head: str, body: str, ha: str = "center") -> None:
        """Bold head over a bracket above the bar; what the part is used for, below the bar."""
        y = yb + h / 2 + 0.2
        ax.plot([x0 + 15, x0 + 15, x1 - 15, x1 - 15], [y - 0.12, y, y, y - 0.12], color=ps.INK_2, lw=1.2)
        x = {"center": (x0 + x1) / 2, "right": x1}[ha]  # the narrow test part anchors at the edge
        ax.text(x, y + 0.15, head, ha=ha, va="bottom", fontsize=ps.TEXT, color=ps.INK, fontweight="bold")
        ax.text(x, yb - h / 2 - 0.2, body, ha=ha, va="top", fontsize=ps.SMALL, color=ps.INK_2,
                linespacing=1.3)

    part(0, left, f"Validación cruzada: {n_cv:,} filas",
         "selección de modelo, peso de la mezcla\ny umbrales con predicciones out-of-fold")
    part(left, total, f"Test: {n_test:,} filas (15%)", "fijo; se evaluó\nuna sola vez", ha="right")
    ax.text(0, 0.35,
            "Grupo = texto normalizado (ftfy, minúsculas, espacios): un texto repetido nunca\n"
            "queda a ambos lados. Estratificación iterativa multietiqueta sobre los grupos.",
            ha="left", va="bottom", fontsize=ps.SMALL, color=ps.INK_2, linespacing=1.35)
    ax.set_xlim(-60, total + 60)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return ps.save(fig, "evaluacion_particion", cfg)


# ----------------------------------------------------------------------------- helpers
# The pipeline (slide 5) and deployment (slide 10) diagrams are native PowerPoint shapes,
# built in intent.deck so they stay editable; only chart-like figures are drawn here.
def _canvas(size: tuple[float, float]):
    """Axes whose data units are inches, so box sizes and font sizes share one scale."""
    fig, ax = ps.figure(size)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, size[0])
    ax.set_ylim(0, size[1])
    ax.axis("off")
    return fig, ax


# ----------------------------------------------------------------------------- results
def fig_ablation(cfg: dict[str, Any]) -> Path:
    """CV macro F1 with 95% CIs for the configurations that carry the story."""
    rows = [(exp, f"{code}: {desc}") for exp, code, desc in ABLATION]
    t = ablation(cfg)

    fig, ax = ps.figure(ps.WIDE)
    ys = ablation_positions()
    for yi, (exp, _) in zip(ys, rows):
        r = t.loc[exp]
        sel = exp == SELECTED
        ax.barh(yi, r.macro_f1, height=0.62, color=ps.ACCENT if sel else ps.NEUTRAL, zorder=2)
        ax.errorbar(r.macro_f1, yi, xerr=[[r.macro_f1 - r.macro_f1_ci_low], [r.macro_f1_ci_high - r.macro_f1]],
                    fmt="none", ecolor=ps.INK_2, elinewidth=1.4, capsize=3, zorder=3)
        _bar_value(ax, r.macro_f1_ci_high + 0.012, yi, ps.num(r.macro_f1), bold=sel)
    ref = t.loc[REFERENCE, "macro_f1"]
    ax.axvline(ref, color=ps.INK_2, linewidth=1, zorder=1)
    ax.text(ref, -1.0, f"referencia B2 {ps.num(ref)}", ha="center", va="center", fontsize=ps.SMALL,
            color=ps.INK_2, bbox={"facecolor": ps.SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.set_yticks(ys, [lab for _, lab in rows])
    for tick, (exp, _) in zip(ax.get_yticklabels(), rows):
        if exp == SELECTED:
            tick.set_fontweight("bold")
    ax.set_ylim(ys[-1] + 0.6, -1.35)
    ax.set_xlim(0, 0.84)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Macro F1 en CV agrupada (5 folds, out-of-fold), IC 95%")
    ps.finish_axes(ax, grid="x")
    fig.subplots_adjust(left=0.42, right=0.98, top=0.97, bottom=0.13)
    return ps.save(fig, "ablacion_cv", cfg)


def fig_test_f1(cfg: dict[str, Any]) -> Path:
    """Test macro, micro and samples F1 with 95% CIs: E9b above B2 on all three."""
    m = test_scores(cfg)
    metrics = TEST_F1
    models = [(SELECTED, "E9b (seleccionado)", ps.ACCENT, -0.17), (REFERENCE, "B2 (referencia)", ps.NEUTRAL, 0.17)]

    fig, ax = ps.figure(ps.HALF_SHORT)
    for i, (metric, _) in enumerate(metrics):
        for model, _, color, dy in models:
            v, lo, hi = m.loc[(model, metric), ["value", "ci_low", "ci_high"]]
            ax.plot([lo, hi], [i + dy, i + dy], color=color, linewidth=2.5, solid_capstyle="round", zorder=2)
            ax.plot(v, i + dy, "o", markersize=10, color=color, markeredgecolor=ps.SURFACE,
                    markeredgewidth=2, zorder=3)  # surface ring keeps the dot legible on its CI line
            sel = model == SELECTED
            ax.text(hi + 0.006, i + dy, ps.num(v), va="center", fontsize=ps.TICK,
                    color=ps.INK if sel else ps.INK_2, fontweight="bold" if sel else "normal")
    ax.set_yticks(range(len(metrics)), [lab for _, lab in metrics])
    ax.set_ylim(len(metrics) - 0.45, -0.75)
    ax.set_xlim(0.64, 0.86)
    ax.set_xticks([0.65, 0.70, 0.75, 0.80, 0.85])
    ax.set_xlabel("Test, 988 filas, IC 95% por bootstrap de grupos")
    handles = [ax.plot([], [], "o", color=c, markersize=10)[0] for _, _, c, _ in models]
    ax.legend(handles, [lab for _, lab, _, _ in models], loc="upper left", bbox_to_anchor=(-0.02, 1.04),
              ncol=2, handletextpad=0.3, columnspacing=1.2)
    ps.finish_axes(ax, grid="x")
    fig.subplots_adjust(left=0.24, right=0.97, top=0.9, bottom=0.18)
    return ps.save(fig, "test_f1", cfg)


def fig_test_hamming(cfg: dict[str, Any]) -> Path:
    """Test Hamming loss: E9b below B2 and below the company's reported figure."""
    m = test_scores(cfg)
    bars = [(SELECTED, "E9b (seleccionado)"), (REFERENCE, "B2 (referencia)"),
            ("company_baseline_reported", "Baseline de la empresa\n(reportado, otro test)")]

    fig, ax = ps.figure(ps.HALF_SHORT)
    for i, (model, _) in enumerate(bars):
        v, lo, hi = m.loc[(model, "hamming_loss"), ["value", "ci_low", "ci_high"]]
        sel = model == SELECTED
        ax.barh(i, v, height=0.55, color=ps.ACCENT if sel else ps.NEUTRAL, zorder=2)
        end = v
        if not np.isnan(lo):
            ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=ps.INK_2, elinewidth=1.4,
                        capsize=3, zorder=3)
            end = hi
        _bar_value(ax, end + 0.0015, i, ps.num(v, 4), bold=sel)
    ax.set_yticks(range(len(bars)), [lab for _, lab in bars])
    for tick, (model, _) in zip(ax.get_yticklabels(), bars):
        if model == SELECTED:
            tick.set_fontweight("bold")
    ax.set_ylim(len(bars) - 0.4, -0.6)
    ax.set_xlim(0, 0.085)
    ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.set_xlabel("Hamming loss en test (menor es mejor), IC 95%")
    ps.finish_axes(ax, grid="x")
    fig.subplots_adjust(left=0.42, right=0.97, top=0.97, bottom=0.18)
    return ps.save(fig, "test_hamming", cfg)


def fig_per_label(cfg: dict[str, Any]) -> Path:
    """Per-label test F1, E9b vs B2, sorted: temp is the clear outlier at the bottom."""
    e9, b2, n, order = per_label_f1(cfg)

    fig, ax = ps.figure(ps.HALF)
    for i, lab in enumerate(order):
        ring = ps.CARD if lab == "temp" else ps.SURFACE  # the ring is a gap in whatever is behind
        if lab == "temp":  # the slide's subject: a quiet band behind its row
            ax.axhspan(i - 0.45, i + 0.45, color=ps.CARD, zorder=0)
        ax.plot([b2[lab], e9[lab]], [i, i], color=ps.INK_2, linewidth=1.2, zorder=1)
        ax.plot(b2[lab], i, "o", markersize=9, color=ps.NEUTRAL, markeredgecolor=ring,
                markeredgewidth=2, zorder=2)
        ax.plot(e9[lab], i, "o", markersize=10, color=ps.ACCENT, markeredgecolor=ring,
                markeredgewidth=2, zorder=3)
    i_temp = list(order).index("temp")
    x = max(e9["temp"], b2["temp"]) + 0.03
    ax.text(x, i_temp, ps.num(e9["temp"], 2), va="center", fontsize=ps.TICK, color=ps.INK, fontweight="bold")
    ax.text(x + 0.075, i_temp, f"vs {ps.num(b2['temp'], 2)} B2", va="center", fontsize=ps.TICK, color=ps.INK_2)
    ax.set_yticks(range(len(order)), [f"{lab} (n={n[lab]})" for lab in order])
    ax.set_ylim(len(order) - 0.4, -1.3)
    ax.set_xlim(0.25, 0.95)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax.set_xlabel("F1 por etiqueta en test")
    handles = [ax.plot([], [], "o", color=c, markersize=10)[0] for c in (ps.ACCENT, ps.NEUTRAL)]
    ax.legend(handles, ["E9b", "B2"], loc="upper left", bbox_to_anchor=(-0.02, 1.03), ncol=2,
              handletextpad=0.3, columnspacing=1.2)
    ps.finish_axes(ax, grid="x")
    fig.subplots_adjust(left=0.29, right=0.97, top=0.97, bottom=0.14)
    return ps.save(fig, "test_f1_por_etiqueta", cfg)


def fig_temp_readings(cfg: dict[str, Any]) -> Path:
    """How CV answers tagged temp read: only one in five mentions a season."""
    labels = cfg["data"]["labels"]
    cv = _clean_with_split(cfg)
    cv = cv[cv["split"] == "cv"].reset_index(drop=True)
    texts = cv[cfg["data"]["text_col"]].map(clean_for_transformer)  # the text error analysis used
    tab = temp_readings(texts, cv[labels].to_numpy(), labels)["temp rows"]
    n = int(tab.sum())
    season, cycle, neither = tab["season"] + tab["both"], tab["cycle"] + tab["both"], tab["neither"]
    _check(round(season / n, 2), 0.21, 0, "temp share mentioning a season")
    _check(round(cycle / n, 2), 0.39, 0, "temp share mentioning a payment cycle")
    _check(round(neither / n, 2), 0.42, 0, "temp share mentioning neither")
    bars = [("Menciona una temporada", season, True), ("Menciona un ciclo de cobro", cycle, False),
            ("No menciona ninguno", neither, False)]

    fig, ax = ps.figure(ps.HALF_SHORT)
    for i, (_, k, hi) in enumerate(bars):
        ax.barh(i, k / n, height=0.55, color=ps.ACCENT if hi else ps.NEUTRAL, zorder=2)
        _bar_value(ax, k / n + 0.012, i, f"{ps.pct(k / n)}  ({k})", bold=hi)
    ax.set_yticks(range(len(bars)), [lab for lab, _, _ in bars])
    ax.set_ylim(len(bars) - 0.45, -0.55)
    ax.set_xlim(0, 0.6)
    ax.set_xticks([])
    fig.text(0.03, 0.03, f"{n} respuestas temp en CV; {int(tab['both'])} mencionan ambos",
             ha="left", va="bottom", fontsize=ps.SMALL, color=ps.INK_2)
    ps.finish_axes(ax, grid=None)
    fig.subplots_adjust(left=0.45, right=0.97, top=0.97, bottom=0.14)
    return ps.save(fig, "temp_lecturas", cfg)


def fig_manual_audit(cfg: dict[str, Any]) -> Path:
    """Causes of 50 hand-reviewed errors: most sit in the labels or the codebook, not the model."""
    shares = audit_shares(cfg)
    n = int(shares["n"].iloc[0])
    names = {  # Spanish label per category in manual_audit.csv
        "multiple intents": "Convención de etiquetado\n(varios usos, otro subconjunto)",
        "likely label error": "Error de etiqueta",
        "other": "Error del modelo\n(texto claro, etiqueta correcta)",
        "personal vs business": "Uso personal vs negocio (no)",
        "temp ambiguity": "Ambigüedad de temp",
    }
    cats = [(c, names[c], model) for c, model in AUDIT]
    counts = shares["k"].to_dict()

    fig, ax = ps.figure(ps.WIDE)
    for i, (c, _, model) in enumerate(cats):
        k = counts[c]
        lo, hi = shares.loc[c, "lo"], shares.loc[c, "hi"]
        ax.barh(i, k / n, height=0.6, color=ps.ROSE if model else ps.TEAL, zorder=2)  # slots 1 and 2
        ax.errorbar(k / n, i, xerr=[[k / n - lo], [hi - k / n]], fmt="none", ecolor=ps.INK_2,
                    elinewidth=1.4, capsize=3, zorder=3)
        _bar_value(ax, hi + 0.015, i, f"{ps.pct(k / n)}  ({k})", bold=i == 0)
    ax.set_yticks(range(len(cats)), [lab for _, lab, _ in cats])
    ax.set_ylim(len(cats) - 0.4, -0.75)
    ax.set_xlim(0, 0.95)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8], ["0%", "20%", "40%", "60%", "80%"])
    ax.set_xlabel(f"Porcentaje de {n} errores revisados a mano (CV), IC 95% de Wilson")
    fig.legend([Patch(color=ps.TEAL), Patch(color=ps.ROSE)], ["Origen en etiquetas o codebook", "Error del modelo"],
               loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=2, handlelength=1.0, columnspacing=1.5)
    ps.finish_axes(ax, grid="x")
    fig.subplots_adjust(left=0.43, right=0.97, top=0.9, bottom=0.13)
    return ps.save(fig, "auditoria_manual", cfg)


def fig_traceability(cfg: dict[str, Any]) -> Path:
    """Order of events on 2026-10-02: selection on CV, then the single test run, then the record."""
    test_sha = pd.read_csv(_reports(cfg, "test_results.csv"))["config_sha"].iloc[0]
    cv_e9b = load_results(cfg).loc[SELECTED, "macro_f1"]
    events = [  # times and facts from reports/TRACEABILITY.md; (time, head, body, code line)
        ("15:11 a 15:12", "E9b elegido en CV", f"macro F1 {ps.num(float(cv_e9b), 4)}; la selección\nqueda escrita en la config", None),
        ("15:37", "Única evaluación en test", "E9b y B2; hash de la config:", test_sha),
        ("19:33 a 19:36", "Registro", "config de test reconstruida con el\nmismo hash; primer commit y tag", None),
    ]
    fig, ax = _canvas(ps.HALF)
    x_line, y0, step = 1.55, 3.5, 1.3
    ax.plot([x_line, x_line], [y0 - 2 * step - 0.25, y0 + 0.25], color=ps.GRID, linewidth=3, zorder=1)
    for k, (when, head, body, code) in enumerate(events):
        y = y0 - k * step
        ax.plot(x_line, y, "o", markersize=14, color=ps.ROSE if k == 1 else ps.TEAL,
                markeredgecolor=ps.SURFACE, markeredgewidth=2.5, zorder=2)
        ax.text(x_line - 0.25, y, when, ha="right", va="center", fontsize=ps.TICK, color=ps.INK_2,
                fontweight="bold")
        ax.text(x_line + 0.3, y, head, ha="left", va="center", fontsize=ps.TEXT, color=ps.INK,
                fontweight="bold")
        ax.text(x_line + 0.3, y - 0.24, body, ha="left", va="top", fontsize=ps.SMALL, color=ps.INK_2,
                linespacing=1.25)
        if code:  # monospace: in Gill Sans MT the digit 1 is a bare stroke, ambiguous inside a hash
            ax.text(x_line + 0.3, y - 0.52, code, ha="left", va="top", fontsize=ps.SMALL, color=ps.INK,
                    family=["Consolas", "DejaVu Sans Mono"])
    ax.text(0.05, 4.15, "2 de octubre de 2026, hora local (UTC-5)", ha="left", va="center",
            fontsize=ps.SMALL, color=ps.INK_2)
    return ps.save(fig, "trazabilidad_linea_tiempo", cfg)


FIGURES = [fig_prevalence, fig_split, fig_ablation, fig_test_f1, fig_test_hamming, fig_per_label,
           fig_temp_readings, fig_manual_audit, fig_traceability]


def build_all(cfg: dict[str, Any] | None = None) -> list[Path]:
    """Regenerate every deck figure; the report's English figures are not touched."""
    cfg = cfg or load_config()
    return [f(cfg) for f in FIGURES]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for p in build_all():
        print(p.relative_to(resolve(".")))
