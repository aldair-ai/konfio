"""Matplotlib style for the Spanish deck, derived from assets/template.pptx.

Why these choices (reports/template_inspection.md has the raw template facts):
- The template's master paints every slide with dk2 (#1B192E) and white text. Figures
  are drawn on that same navy so they sit on the slide without a visible box. A white
  figure would read as a pasted card, and a transparent PNG would make the light text
  unreadable when the file is opened on its own.
- Colors are the template's theme accents, checked with the dataviz palette validator
  for a dark surface: OKLCH lightness 0.48 to 0.67, chroma >= 0.10, adjacent CVD
  Delta E >= 8 and normal-vision Delta E >= 15. Base teal and green are slightly too light
  for this surface (L 0.695 and 0.711), so they use the theme's own 10% darker step
  (HSL lumMod 0.90); magenta passes as is; rose needs the same 10% darker step to clear
  the chroma floor.
- Fixed categorical order teal, rose, magenta, green: among the orders that pass, it keeps
  the template's teal first and still clears every check (worst adjacent CVD Delta E 11.1,
  normal vision 19.9). Teal and magenta collapse under deuteranopia (Delta E 4.4), so a
  two-series chart always takes slots 1 and 2, never skips to slot 3.
- Emphasis charts (one highlighted item, the rest "not selected") pair teal with NEUTRAL,
  a darker step of the template's lavender. It fails the chroma floor on purpose (it must
  read as gray) and sits at 2.7:1 contrast, so every NEUTRAL mark carries a visible label.
- Figure sizes equal the template's placeholder sizes in inches, so a point in the figure
  is a point on the slide: the font sizes below are what the audience reads.
- Body font is the theme's minor font, Gill Sans MT. The major font, Walbaum Display, is a
  cloud font that is not installed; Bodoni MT is the closest Didone available. Figures carry
  no titles (the slide title is the message), so they only use the body font.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # figures are written to files only; no display backend needed
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

from intent.data import load_config, resolve  # noqa: E402

# ----------------------------------------------------------------------------- theme tokens
SURFACE = "#1B192E"   # dk2: the master's background (bg2) on every layout
INK = "#FFFFFF"       # lt1: the master's text color (tx1); values and key labels
INK_2 = "#A3A3C1"     # accent5 lavender: ticks, axis titles, notes (6.99:1 on SURFACE)
GRID = "#37335B"      # accent6 plum: hairline grid, one step off the surface
CARD = "#37335B"      # diagram box fill: same plum, so boxes read as part of the theme

TEAL = "#109FAC"      # accent2, 10% darker
ROSE = "#AF5C4F"      # accent4, 10% darker
MAGENTA = "#D40AA8"   # accent3, base
GREEN = "#11AB7B"     # accent1, 10% darker
CATEGORICAL = (TEAL, ROSE, MAGENTA, GREEN)  # fixed order; assign in sequence, never skip
ACCENT = TEAL         # the highlighted item in emphasis charts (selected model, key share)
NEUTRAL = "#5C5C85"   # accent5 lavender, darker step: everything that is not the message

FONT_BODY = ["Gill Sans MT", "Segoe UI", "DejaVu Sans"]  # theme minor font, then fallbacks
FONT_HEADING = ["Bodoni MT", "Georgia", "DejaVu Serif"]  # Walbaum Display is not installed

TEXT = 16    # values and direct labels
TICK = 15    # tick labels
SMALL = 13   # axis titles and notes; nothing on a slide goes smaller

# Placeholder sizes in inches (width, height), from reports/template_inspection.md.
FULL = (12.1, 4.6)        # layout 10 "Tabla": content 12.13 x 4.70
WIDE = (8.3, 4.7)         # layout 8 "Contenido + tabla": right area 8.30 x 4.70
HALF = (5.9, 4.3)         # layout 6 "Dos contenidos 1": each column 5.94 x 4.37
HALF_SHORT = (5.9, 3.6)   # same column, leaving room for one caption line below


@dataclass(frozen=True)
class Theme:
    """Everything a figure needs from a palette: surface, text tokens, series roles, font sizes.

    Series roles: primary is the highlighted item (the selected model), secondary a second
    highlight level, neutral everything that is not the message, warm the one alert color.
    """

    surface: str
    ink: str            # values and category labels
    muted: str          # axis titles, notes, descriptions
    grid: str
    spine: str
    xtick: str          # numeric tick labels
    primary: str
    secondary: str
    neutral: str
    warm: str
    text: float         # values and direct labels
    tick: float         # category labels
    axis: float         # numeric tick labels
    small: float        # axis titles and notes


# The template deck (reports/figures/deck/): the values above, unchanged.
TEMPLATE = Theme(surface=SURFACE, ink=INK, muted=INK_2, grid=GRID, spine=GRID, xtick=INK_2, primary=TEAL,
                 secondary=ROSE, neutral=NEUTRAL, warm=ROSE, text=TEXT, tick=TICK, axis=TICK, small=SMALL)

# Konfio palette (reports/figures/deck_konfio/), given as hex values; checked with the dataviz
# validator on its surface: primary vs neutral passes (CVD dE 20.1, normal 22.4) and primary vs
# warm passes every check (22.5 / 26.9). Warm vs neutral collapses under protanopia (dE 5.1), so
# the two never sit side by side; primary vs secondary is just under the normal-vision floor
# (14.9), so secondary only marks separate, labeled bars. muted is the text color at 72% over
# the surface (8.9:1 contrast): the palette names one text color, the hierarchy needs two.
KONFIO = Theme(surface="#140527", ink="#F2EEF8", muted="#B4ADBE", grid="#2E2245", spine="#F2EEF8",
               xtick="#F2EEF8", primary="#8644F9", secondary="#AD82FB", neutral="#8A8499", warm="#D9637B",
               text=18, tick=17, axis=15, small=14)
# Half of a 16:9 slide (6.67 in wide) less margins, with room above for a title. Fonts are
# sized for this size, so they stay 14 to 18 pt on the slide.
KONFIO_SIZE = (6.4, 5.0)


def apply(theme: Theme = TEMPLATE) -> None:
    """Set rcParams for every deck figure."""
    plt.rcParams.update({
        "font.family": FONT_BODY,
        "font.size": theme.text,
        "text.color": theme.ink,
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "axes.edgecolor": theme.grid,
        "axes.labelcolor": theme.muted,
        "axes.labelsize": theme.small,
        "axes.prop_cycle": cycler(color=CATEGORICAL),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,  # Gill Sans MT has no U+2212; a hyphen renders everywhere
        "xtick.color": theme.xtick,
        "ytick.color": theme.muted,
        "xtick.labelcolor": theme.xtick,
        "ytick.labelcolor": theme.ink,
        "xtick.labelsize": theme.axis,
        "ytick.labelsize": theme.tick,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "grid.color": theme.grid,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": theme.tick,
        "legend.labelcolor": theme.ink,
    })


def figure(size: tuple[float, float], theme: Theme = TEMPLATE) -> tuple[plt.Figure, plt.Axes]:
    """A styled figure of exactly the placeholder size it will fill."""
    apply(theme)
    return plt.subplots(figsize=size)


def finish_axes(ax: plt.Axes, grid: str | None = "x", theme: Theme = TEMPLATE) -> None:
    """Recessive chrome: hairline grid on one axis only, no frame except the baseline."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.spine)
    if grid:
        ax.grid(True, axis=grid, zorder=0)
        ax.set_axisbelow(True)


def num(x: float, digits: int = 3) -> str:
    """Decimal point, thousands comma: the Mexican convention, same as the report."""
    return f"{x:,.{digits}f}"


def pct(x: float, digits: int = 0) -> str:
    """Share in [0, 1] as a percentage string."""
    return f"{100 * x:.{digits}f}%"


def save(fig: plt.Figure, name: str, cfg: dict[str, Any] | None = None, theme: Theme = TEMPLATE,
         folder: str = "figures_dir") -> Path:
    """Write <deck.folder>/<name>.png at the deck dpi and close the figure."""
    cfg = cfg or load_config()
    path = resolve(cfg["deck"][folder]) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg["deck"]["dpi"], facecolor=theme.surface)
    plt.close(fig)
    return path
