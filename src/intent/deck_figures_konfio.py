"""Six deck figures in the Konfio palette, sized for half a slide, written to reports/figures/deck_konfio/.

Same numbers and checks as intent.deck_figures (its data functions are reused, not repeated);
only the drawing differs:
- Palette roles (plot_style.KONFIO): primary for E9b and highlighted bars, secondary for the
  fine-tuned family in the ablation, neutral for B2 and everything else, warm only for model
  errors and the temp highlight.
- 6.4 x 5.0 in, half of a 16:9 slide: fonts are 14 to 18 pt at the size they are shown.
- Category labels in two lines (bold name, muted description) where the Spanish description
  is long, so the bars keep their width at half-slide size.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory, offset_copy

from intent import deck_figures as df
from intent import plot_style as ps
from intent.data import load_config, resolve

T = ps.KONFIO
SIZE = ps.KONFIO_SIZE
FOLDER = "konfio_figures_dir"


def _blend(color: str, alpha: float) -> str:
    """color at alpha over the surface, as an opaque hex (dot rings must match what is behind)."""
    c, s = np.array(mcolors.to_rgb(color)), np.array(mcolors.to_rgb(T.surface))
    return mcolors.to_hex(alpha * c + (1 - alpha) * s)


def _value(ax, x: float, y: float, text: str, bold: bool = False, color: str | None = None) -> None:
    ax.text(x, y, text, va="center", ha="left", fontsize=T.text, color=color or T.ink,
            fontweight="bold" if bold else "normal",
            bbox={"facecolor": T.surface, "edgecolor": "none", "pad": 1.5})  # mask grid and reference lines


def _labels(fig, ax, ys, labels: list[tuple[str, str | None, bool]]) -> float:
    """Category labels left of the axis: name (bold if flagged) with an optional muted second line.

    Returns the width in inches of the widest label, so the caller can size the left margin.
    """
    base = blended_transform_factory(ax.transAxes, ax.transData)  # x: axes edge, y: the row
    renderer = fig.canvas.get_renderer()
    widest = 0.0
    for y, (name, sub, strong) in zip(ys, labels):
        weight = "bold" if strong else "normal"
        if sub:
            top = ax.text(0, y, name, ha="right", va="bottom", fontsize=T.tick, color=T.ink, fontweight=weight,
                          transform=offset_copy(base, fig=fig, x=-9, y=0, units="points"))
            bottom = ax.text(0, y, sub, ha="right", va="top", fontsize=T.small, color=T.muted,
                             transform=offset_copy(base, fig=fig, x=-9, y=-1, units="points"))
            parts = [top, bottom]
        else:
            parts = [ax.text(0, y, name, ha="right", va="center", fontsize=T.tick, color=T.ink, fontweight=weight,
                             transform=offset_copy(base, fig=fig, x=-9, y=0, units="points"))]
        widest = max([widest] + [p.get_window_extent(renderer).width / fig.dpi for p in parts])
    ax.set_yticks([])
    return widest + 9 / 72


def _figure():
    fig, ax = ps.figure(SIZE, T)
    return fig, ax


def _finish(fig, ax, widest_label: float, top: float = 0.97, bottom: float = 0.13, grid: str | None = "x") -> None:
    ps.finish_axes(ax, grid=grid, theme=T)
    # text measured before layout runs a few % narrower than drawn (hinting at the save dpi)
    fig.subplots_adjust(left=(1.05 * widest_label + 0.15) / SIZE[0], right=0.97, top=top, bottom=bottom)


def _legend(fig, handles, labels: list[str], y: float = 0.995) -> None:
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, y), ncol=len(labels), fontsize=T.small,
               handlelength=1.0, handletextpad=0.4, columnspacing=1.2, labelcolor=T.ink)


def _save(fig, name: str, cfg: dict[str, Any]) -> Path:
    return ps.save(fig, name, cfg, theme=T, folder=FOLDER)


# ----------------------------------------------------------------------------- figures
def fig_prevalence(cfg: dict[str, Any]) -> Path:
    """Share of answers per label; temp, the rarest and the problem label, in the warm accent."""
    share = df.prevalence(cfg)
    fig, ax = _figure()
    y = np.arange(len(share))
    colors = [T.warm if lab == "temp" else T.primary for lab in share.index]
    ax.barh(y, 100 * share.to_numpy(), height=0.62, color=colors, zorder=2)
    for yi, (lab, v) in zip(y, share.items()):
        _value(ax, 100 * v + 0.8, yi, f"{100 * v:.1f}%", bold=lab == "temp")
    widest = _labels(fig, ax, y, [(lab, df.LABEL_ES[lab], lab == "temp") for lab in share.index])
    ax.set_xlim(0, 44)
    ax.set_ylim(-0.6, len(share) - 0.4)
    ax.set_xticks([])  # every bar is labeled
    ax.spines["bottom"].set_visible(False)
    fig.text(0.02, 0.015, "Porcentaje de 6,691 respuestas limpias", ha="left", va="bottom", fontsize=T.small,
             color=T.muted)
    _finish(fig, ax, widest, top=0.99, bottom=0.07, grid=None)
    return _save(fig, "etiquetas_prevalencia", cfg)


def fig_ablation(cfg: dict[str, Any]) -> Path:
    """CV macro F1 with 95% CIs: E9b in primary, the fine-tuned e5 family in secondary."""
    t = df.ablation(cfg)
    ys = df.ablation_positions()
    fine_tuned = {"E4_ft_head", "E7_hybrid_lgbm"}
    names = {df.REFERENCE: "B2 (referencia)"}  # the reference keeps its role in the name
    fig, ax = _figure()
    for yi, (exp, _, _) in zip(ys, df.ABLATION):
        r = t.loc[exp]
        sel = exp == df.SELECTED
        color = T.primary if sel else T.secondary if exp in fine_tuned else T.neutral
        ax.barh(yi, r.macro_f1, height=0.6, color=color, zorder=2)
        ax.errorbar(r.macro_f1, yi, xerr=[[r.macro_f1 - r.macro_f1_ci_low], [r.macro_f1_ci_high - r.macro_f1]],
                    fmt="none", ecolor=T.muted, elinewidth=1.4, capsize=3, zorder=3)
        _value(ax, r.macro_f1_ci_high + 0.015, yi, ps.num(r.macro_f1), bold=sel)
    ref = t.loc[df.REFERENCE, "macro_f1"]
    ax.axvline(ref, color=T.muted, linewidth=1, zorder=1)
    labels = [(names.get(exp, code), desc.replace(" (ref.)", ""), exp == df.SELECTED) for exp, code, desc in df.ABLATION]
    widest = _labels(fig, ax, ys, labels)
    ax.set_ylim(ys[-1] + 0.55, -0.55)
    ax.set_xlim(0, 0.92)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel(f"Macro F1 en CV, IC 95% (línea: B2 {ps.num(ref)})")
    _legend(fig, [Patch(color=c) for c in (T.primary, T.secondary, T.neutral)],
            ["E9b (seleccionado)", "con e5 fine-tuning", "otras"])
    _finish(fig, ax, widest, top=0.92, bottom=0.11)
    return _save(fig, "ablacion_cv", cfg)


def fig_test_f1(cfg: dict[str, Any]) -> Path:
    """Test macro, micro and samples F1 with 95% CIs: E9b (primary) above B2 (neutral) on all three."""
    m = df.test_scores(cfg)
    models = [(df.SELECTED, "E9b (seleccionado)", T.primary, -0.17), (df.REFERENCE, "B2 (referencia)", T.neutral, 0.17)]
    fig, ax = _figure()
    for i, (metric, _) in enumerate(df.TEST_F1):
        for model, _, color, dy in models:
            v, lo, hi = m.loc[(model, metric), ["value", "ci_low", "ci_high"]]
            ax.plot([lo, hi], [i + dy, i + dy], color=color, linewidth=3, solid_capstyle="round", zorder=2)
            ax.plot(v, i + dy, "o", markersize=12, color=color, markeredgecolor=T.surface,
                    markeredgewidth=2, zorder=3)  # surface ring keeps the dot legible on its CI line
            sel = model == df.SELECTED
            ax.text(hi + 0.006, i + dy, ps.num(v), va="center", fontsize=T.text,
                    color=T.ink if sel else T.muted, fontweight="bold" if sel else "normal")
    widest = _labels(fig, ax, range(len(df.TEST_F1)), [(lab, None, False) for _, lab in df.TEST_F1])
    ax.set_ylim(len(df.TEST_F1) - 0.5, -0.6)
    ax.set_xlim(0.64, 0.88)
    ax.set_xticks([0.65, 0.70, 0.75, 0.80, 0.85])
    ax.set_xlabel("Test (988 filas), IC 95% por bootstrap de grupos")
    _legend(fig, [ax.plot([], [], "o", color=c, markersize=12)[0] for _, _, c, _ in models],
            [lab for _, lab, _, _ in models])
    _finish(fig, ax, widest, top=0.9, bottom=0.12)
    return _save(fig, "test_f1", cfg)


def fig_test_hamming(cfg: dict[str, Any]) -> Path:
    """Test Hamming loss: E9b (primary) below B2 and below the company's reported figure."""
    m = df.test_scores(cfg)
    bars = [(df.SELECTED, "E9b", "seleccionado"), (df.REFERENCE, "B2", "referencia"),
            ("company_baseline_reported", "Baseline de la empresa", "reportado, otro test")]
    fig, ax = _figure()
    for i, (model, _, _) in enumerate(bars):
        v, lo, hi = m.loc[(model, "hamming_loss"), ["value", "ci_low", "ci_high"]]
        sel = model == df.SELECTED
        ax.barh(i, v, height=0.55, color=T.primary if sel else T.neutral, zorder=2)
        end = v
        if not np.isnan(lo):
            ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=T.muted, elinewidth=1.4,
                        capsize=3, zorder=3)
            end = hi
        _value(ax, end + 0.002, i, ps.num(v, 4), bold=sel)
    widest = _labels(fig, ax, range(len(bars)), [(name, sub, model == df.SELECTED) for model, name, sub in bars])
    ax.set_ylim(len(bars) - 0.45, -0.55)
    ax.set_xlim(0, 0.095)
    ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08])
    fig.text(0.02, 0.015, "Hamming loss en test (menor es mejor), IC 95%", ha="left", va="bottom",
             fontsize=T.small, color=T.muted)  # centered under the axis it would reach the right edge
    _finish(fig, ax, widest)
    return _save(fig, "test_hamming", cfg)


def fig_per_label(cfg: dict[str, Any]) -> Path:
    """Per-label test F1, E9b vs B2, sorted; temp, the weakest label, highlighted in the warm accent."""
    e9, b2, n, order = df.per_label_f1(cfg)
    band = _blend(T.warm, 0.28)  # opaque, so the dot rings can match it exactly
    fig, ax = _figure()
    for i, lab in enumerate(order):
        temp = lab == "temp"
        ring = band if temp else T.surface  # the ring is a gap in whatever is behind
        if temp:
            ax.axhspan(i - 0.46, i + 0.46, color=band, zorder=0)
        ax.plot([b2[lab], e9[lab]], [i, i], color=T.warm if temp else T.muted, linewidth=2.2 if temp else 1.2,
                zorder=1)
        ax.plot(b2[lab], i, "o", markersize=10, color=T.neutral, markeredgecolor=ring, markeredgewidth=2, zorder=2)
        ax.plot(e9[lab], i, "o", markersize=11, color=T.primary, markeredgecolor=ring, markeredgewidth=2, zorder=3)
    i_temp = list(order).index("temp")
    x = max(e9["temp"], b2["temp"]) + 0.03
    ax.text(x, i_temp, ps.num(e9["temp"], 2), va="center", fontsize=T.text, color=T.ink, fontweight="bold")
    ax.text(x + 0.085, i_temp, f"vs {ps.num(b2['temp'], 2)} B2", va="center", fontsize=T.text, color=T.ink)
    widest = _labels(fig, ax, range(len(order)), [(f"{lab} (n={n[lab]})", None, lab == "temp") for lab in order])
    ax.set_ylim(len(order) - 0.5, -0.6)
    ax.set_xlim(0.25, 0.97)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax.set_xlabel("F1 por etiqueta en test")
    _legend(fig, [ax.plot([], [], "o", color=c, markersize=12)[0] for c in (T.primary, T.neutral)], ["E9b", "B2"])
    _finish(fig, ax, widest, top=0.91, bottom=0.12)
    return _save(fig, "test_f1_por_etiqueta", cfg)


def fig_manual_audit(cfg: dict[str, Any]) -> Path:
    """Causes of 50 hand-reviewed errors; the model's own errors in the warm accent."""
    shares = df.audit_shares(cfg)
    names = {  # (name, description) per category in manual_audit.csv
        "multiple intents": ("Convención de etiquetado", "varios usos, otro subconjunto"),
        "likely label error": ("Error de etiqueta", None),
        "other": ("Error del modelo", "texto claro, etiqueta correcta"),
        "personal vs business": ("Personal vs negocio", "etiqueta no"),
        "temp ambiguity": ("Ambigüedad de temp", None),
    }
    fig, ax = _figure()
    for i, (cat, r) in enumerate(shares.iterrows()):
        ax.barh(i, r.share, height=0.6, color=T.warm if r.model else T.primary, zorder=2)
        ax.errorbar(r.share, i, xerr=[[r.share - r.lo], [r.hi - r.share]], fmt="none", ecolor=T.muted,
                    elinewidth=1.4, capsize=3, zorder=3)
        _value(ax, r.hi + 0.02, i, f"{ps.pct(r.share)}  ({int(r.k)})", bold=i == 0)
    widest = _labels(fig, ax, range(len(shares)), [(*names[c], False) for c in shares.index])
    ax.set_ylim(len(shares) - 0.45, -0.55)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8], ["0%", "20%", "40%", "60%", "80%"])
    fig.text(0.02, 0.015, f"{int(shares['n'].iloc[0])} errores revisados a mano (CV), IC 95% de Wilson",
             ha="left", va="bottom", fontsize=T.small, color=T.muted)  # too long to center under the axis
    _legend(fig, [Patch(color=T.primary), Patch(color=T.warm)], ["Origen en etiquetas o codebook", "Error del modelo"])
    _finish(fig, ax, widest, top=0.9, bottom=0.13)
    return _save(fig, "auditoria_manual", cfg)


FIGURES = [fig_prevalence, fig_ablation, fig_test_f1, fig_test_hamming, fig_per_label, fig_manual_audit]


def build_all(cfg: dict[str, Any] | None = None) -> list[Path]:
    cfg = cfg or load_config()
    return [f(cfg) for f in FIGURES]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for p in build_all():
        print(p.relative_to(resolve(".")))
