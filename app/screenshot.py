"""Capture the demo for the deck and the README: the "Riesgo cred" example in the Probar tab.

Needs the demo running (`python app/run_demo.py`). Drives the installed Microsoft Edge
through Playwright (channel="msedge"), so no browser download is needed. Light theme on
purpose: the demo's charts mark thresholds in near-black, which vanishes on a dark theme.

    python app/screenshot.py   # -> reports/figures/demo.png and reports/figures/deck/demo_app.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intent.data import load_config, resolve  # noqa: E402

UI_URL = os.environ.get("INTENT_UI_URL", "http://localhost:8501")
EXAMPLE = "Riesgo cred"  # cred is a risk label: the case the deployment slide is about
PAD = 12                 # px around the captured panel


def capture(full_path: Path, panel_path: Path) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        # tall viewport: Streamlit scrolls inside its own container, so full_page alone misses content
        page = browser.new_page(viewport={"width": 1280, "height": 1750}, device_scale_factor=2,
                                color_scheme="light")
        page.goto(UI_URL)
        page.get_by_role("button", name=EXAMPLE).click()
        page.get_by_text("Requiere revisión humana").first.wait_for(timeout=120_000)
        # The first request pays lazy-init costs; classify again so the latency shown is a warm
        # one, comparable to the p50 in reports/latency.csv.
        page.get_by_role("button", name="Clasificar").click()
        page.wait_for_timeout(3000)  # the rerun, plus Altair, which draws after the text
        full_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(full_path))
        trim_bottom(full_path)
        panel = page.locator('[data-testid="stColumn"]').filter(has_text="E9b (seleccionado)").last
        box = panel.bounding_box()
        # stop at the chart's bottom: the latency line below it is on the slide already, and a
        # shorter crop is drawn larger in the deck's fixed-height slot
        chart = panel.locator(".vega-embed").first.bounding_box()
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(panel_path), clip={"x": box["x"] - PAD, "y": box["y"] - PAD,
                                                    "width": box["width"] + 2 * PAD,
                                                    "height": chart["y"] + chart["height"] - box["y"] + 2 * PAD})
        trim_bottom(panel_path)
        browser.close()


def trim_bottom(path: Path, margin: int = 40) -> None:
    """Drop the empty page background below the content (columns are as tall as the longest one)."""
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, im.getpixel((im.width - 1, im.height - 1)))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:
        im.crop((0, 0, im.width, min(im.height, bbox[3] + margin))).save(path)


if __name__ == "__main__":
    cfg = load_config()
    full = resolve(cfg["paths"]["figures_dir"]) / "demo.png"
    panel = resolve(cfg["deck"]["figures_dir"]) / "demo_app.png"
    capture(full, panel)
    print(full, panel, sep="\n")
