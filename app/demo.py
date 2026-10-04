"""Interview demo: a Streamlit front end for the loan intent API (src/intent/serve.py).

Predictions come only from the API (INTENT_API_URL, default http://localhost:8000), so
the demo shows exactly what the deployed model returns. The Results tab reads aggregate
numbers from reports/, never from code. Works offline: no CDN, no remote calls (the
mermaid diagram is converted to Graphviz, which Streamlit renders locally).

Run with `python app/run_demo.py` (starts the API too) or `make demo`.
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
from pathlib import Path

import altair as alt
import httpx
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so the demo also runs without `pip install -e .`
from intent.data import load_config, resolve  # noqa: E402

API_URL = os.environ.get("INTENT_API_URL", "http://localhost:8000").rstrip("/")
APP_DIR = Path(__file__).resolve().parent
CFG = load_config()
LABELS: list[str] = CFG["data"]["labels"]
MAX_BATCH: int = CFG["serving"]["max_batch"]
DESCRIPTIONS: dict[str, str] = yaml.safe_load((APP_DIR / "labels.yaml").read_text(encoding="utf-8"))
EXAMPLES: list[dict] = yaml.safe_load((APP_DIR / "examples.yaml").read_text(encoding="utf-8"))["examples"]
REPORTS = resolve(CFG["paths"]["reports_dir"])

# Same validated pair as the report figures: accent for predicted labels, neutral gray otherwise.
ACCENT, NEUTRAL, INK = "#2a78d6", "#898781", "#0b0b0b"

st.set_page_config(page_title="Loan intent demo", page_icon=":bank:", layout="wide")


# ----------------------------------------------------------------------------- API
@st.cache_resource
def http() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=120)


def api_health() -> dict | None:
    try:
        r = http().get("/health")
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def predict(text: str) -> tuple[dict, float]:
    start = time.perf_counter()
    r = http().post("/predict", json={"text": text})
    r.raise_for_status()
    return r.json(), (time.perf_counter() - start) * 1000


# ----------------------------------------------------------------------------- charts
def probability_chart(probs: dict[str, float], thresholds: dict[str, float], predicted: list[str]) -> alt.LayerChart:
    """One bar per label (probability) with a tick at its threshold; predicted labels in the accent."""
    df = pd.DataFrame({
        "label": LABELS,
        "name": [f"{lab}: {DESCRIPTIONS[lab]}" for lab in LABELS],
        "probabilidad": [probs[lab] for lab in LABELS],
        "umbral": [thresholds[lab] for lab in LABELS],
        "predicha": [lab in predicted for lab in LABELS],
    })
    y = alt.Y("name:N", sort=None, title=None, axis=alt.Axis(labelLimit=330, labelFontSize=12))
    tooltip = ["label", alt.Tooltip("probabilidad:Q", format=".3f"), alt.Tooltip("umbral:Q", format=".3f"), "predicha"]
    bars = alt.Chart(df).mark_bar(size=14, cornerRadiusEnd=3).encode(
        x=alt.X("probabilidad:Q", scale=alt.Scale(domain=[0, 1]), title="probabilidad (marca negra = umbral)"),
        y=y, color=alt.condition("datum.predicha", alt.value(ACCENT), alt.value(NEUTRAL)), tooltip=tooltip,
    )
    ticks = alt.Chart(df).mark_tick(color=INK, thickness=2, size=22).encode(x="umbral:Q", y=y, tooltip=tooltip)
    return (bars + ticks).properties(height=34 * len(LABELS))


def review_badge(needs_review: bool, reasons: list[str]) -> None:
    if needs_review:
        st.warning("**Requiere revisión humana**: " + " · ".join(reasons))
    else:
        st.success("**Sin revisión**: confianza suficiente y sin etiquetas de riesgo")


def model_column(title: str, result: dict, thresholds: dict[str, float], latency_ms: float, terms: dict | None) -> None:
    st.subheader(title)
    st.markdown("**Etiquetas:** " + ", ".join(f"`{lab}`" for lab in result["labels"]))
    review_badge(result["needs_review"], result["review_reasons"])
    st.altair_chart(probability_chart(result["probabilities"], thresholds, result["labels"]), width="stretch")
    st.caption(f"Latencia del modelo: {latency_ms:.1f} ms por solicitud (CPU)")
    if terms is not None:
        st.markdown("**Términos que más aportan** (coeficiente x valor TF-IDF, sobre el texto lematizado)")
        for lab, items in terms.items():
            shown = ", ".join(f"`{t['term'].strip() if t['kind'] == 'word' else repr(t['term'])}` ({t['contribution']:.2f})"
                              for t in items)
            st.markdown(f"- **{lab}**: {shown or 'ninguno positivo'}")


# ----------------------------------------------------------------------------- tabs
def tab_try() -> None:
    st.markdown("Escribe un motivo de uso del crédito o elige un ejemplo del **conjunto de test** (nunca visto en entrenamiento).")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex["button"], width="stretch", key=f"ex_{ex['id']}"):
            st.session_state["text"] = ex["text"]
            st.session_state["run"] = True
    text = st.text_area("Motivo", key="text", height=90, placeholder="Ej.: Compra de mercancía para surtir mi tienda")
    if st.button("Clasificar", type="primary") or st.session_state.pop("run", False):
        if not text.strip():
            st.info("Escribe un texto primero.")
            return
        body, round_trip = predict(text.strip())
        st.session_state["last"] = (text.strip(), body, round_trip)
    if "last" not in st.session_state:
        return
    shown_text, body, round_trip = st.session_state["last"]
    left, right = st.columns(2, gap="large")
    with left:
        model_column(f"{body.get('model', 'E9b')} (seleccionado)", body, body["thresholds"], body["latency_ms"], None)
    with right:
        b2 = body["b2"]
        model_column("B2 (TF-IDF + regresión logística)", b2, b2["thresholds"], b2["latency_ms"], b2["top_terms"])
    st.caption(f"Ida y vuelta HTTP (ambos modelos y la explicación de B2): {round_trip:.0f} ms")
    example = next((ex for ex in EXAMPLES if ex["text"] == shown_text), None)
    if example:
        st.info(f"**Anotado en test:** {', '.join(example['annotated'])}  \n{example['explanation']}")


def tab_results() -> None:
    st.image(str(REPORTS / "figures" / "ablation.png"), caption="Ablación: macro F1 en CV con intervalos de 95%")
    res = pd.read_csv(REPORTS / "test_results.csv")

    def fmt(row: pd.Series) -> str:
        digits = 4 if row["metric"] == "hamming_loss" else 3
        ci = "" if pd.isna(row["ci_low"]) else f" [{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"
        return f"{row['value']:.{digits}f}{ci}"

    res["shown"] = res.apply(fmt, axis=1)
    table = res.pivot_table(index="metric", columns="model", values="shown", aggfunc="first")
    table = table.reindex(columns=[c for c in ["E9b_blend_B2_E4", "B2_tfidf_lr_br", "company_baseline_reported"]
                                   if c in table.columns]).fillna("")
    n_test = int(res["n_test_rows"].iloc[0])
    st.subheader(f"Test ({n_test} filas, evaluado una sola vez), intervalos de 95% por bootstrap de grupos")
    st.dataframe(table, width="stretch")
    per_label = pd.read_csv(REPORTS / "test_per_label.csv", index_col="label")
    cols = {c: c.split("__", 1)[1] for c in per_label.columns if c.startswith("E9b_blend_B2_E4__")}
    view = per_label[list(cols)].rename(columns=cols)
    view["B2 f1"] = per_label["B2_tfidf_lr_br__f1"]
    st.subheader("Por etiqueta (E9b en test)")
    st.dataframe(view.round(3), width="stretch")
    st.subheader("Resumen")
    st.markdown(report_section("8. Conclusion"))


def report_section(title: str) -> str:
    """Body of one section of reports/report.md (the summary is quoted, not rewritten)."""
    text = resolve(CFG["paths"]["report"]).read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, flags=re.S | re.M)
    return match.group(1).strip() if match else "(sección no encontrada en reports/report.md)"


def tab_batch() -> None:
    st.markdown(f"Sube un CSV con una columna **motivos** (máximo {MAX_BATCH:,} filas). "
                "Se devuelven las etiquetas, needs_review y las probabilidades.")
    file = st.file_uploader("CSV", type=["csv"])
    if file is None:
        return
    raw = file.getvalue()
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    if "motivos" not in df.columns:
        st.error(f"El CSV no tiene una columna 'motivos'. Columnas: {list(df.columns)}")
        return
    if len(df) > MAX_BATCH:
        st.warning(f"El archivo tiene {len(df):,} filas; se procesan las primeras {MAX_BATCH:,}.")
        df = df.head(MAX_BATCH)
    texts = df["motivos"].fillna("").astype(str).str.strip()
    valid = texts != ""
    with st.spinner(f"Clasificando {int(valid.sum())} textos..."):
        r = http().post("/predict_batch", json={"texts": texts[valid].tolist()})
        r.raise_for_status()
    out = df.copy()
    out["etiquetas"], out["needs_review"], out["motivos_revision"] = "", pd.NA, ""
    preds = r.json()["predictions"]
    rows = out.index[valid]
    out.loc[rows, "etiquetas"] = [", ".join(p["labels"]) for p in preds]
    out.loc[rows, "needs_review"] = [p["needs_review"] for p in preds]
    out.loc[rows, "motivos_revision"] = [" | ".join(p["review_reasons"]) for p in preds]
    for lab in LABELS:
        out.loc[rows, f"p_{lab}"] = [p["probabilities"][lab] for p in preds]
    if (~valid).any():
        st.info(f"{int((~valid).sum())} filas con texto vacío se dejaron sin etiqueta.")
    st.success(f"{len(preds)} textos clasificados en {r.json()['latency_ms']:.0f} ms; "
               f"{int(out['needs_review'].fillna(False).sum())} requieren revisión.")
    st.dataframe(out, width="stretch")
    st.download_button("Descargar CSV", out.to_csv(index=False).encode("utf-8-sig"),
                       file_name="predicciones.csv", mime="text/csv")


def mermaid_to_dot(src: str) -> str:
    """Convert the report's mermaid flowchart (nodes, -->, -->|label|) to Graphviz DOT.

    Streamlit renders DOT locally, so the diagram stays a single source (report.md)
    and still works offline, where mermaid would need a CDN.
    """
    node = re.compile(r'(\w+)(?:\["([^"]*)"\])?')
    names, edges = {}, []
    for line in src.strip().splitlines()[1:]:
        segs = re.split(r"\s*-->(?:\|([^|]*)\|)?\s*", line.strip())
        ids = []
        for seg in segs[0::2]:
            m = node.fullmatch(seg.strip())
            if not m:
                continue
            ids.append(m.group(1))
            if m.group(2):
                names[m.group(1)] = m.group(2).replace("<br/>", "\\n").replace('"', "'")
        edges += [(a, b, lab) for a, b, lab in zip(ids, ids[1:], segs[1::2])]
    lines = ['digraph G { rankdir=LR; bgcolor="transparent";',
             'node [shape=box, style="rounded,filled", fillcolor="#f0efec", color="#c3c2b7", fontname="Helvetica", fontsize=11];',
             'edge [color="#52514e", fontname="Helvetica", fontsize=10];']
    lines += [f'{i} [label="{names.get(i, i)}"];' for i in dict.fromkeys(i for e in edges for i in e[:2])]
    lines += [f'{a} -> {b}' + (f' [label="{lab}"]' if lab else "") + ";" for a, b, lab in edges]
    return "\n".join(lines + ["}"])


def tab_how() -> None:
    report = resolve(CFG["paths"]["report"]).read_text(encoding="utf-8")
    diagrams = re.findall(r"```mermaid\n(.*?)```", report, flags=re.S)
    if diagrams:
        st.graphviz_chart(mermaid_to_dot(diagrams[0]), width="stretch")
        st.caption("Arquitectura del modelo seleccionado (diagrama de reports/report.md)")
    sm = CFG["selected_model"]
    w = sm["blend"]["weight_on_finetuned_head"]
    st.markdown(f"""
- **Dos ramas de texto**: el encoder recibe el texto casi crudo (solo corrección de codificación y espacios); B2 recibe lemas de spaCy sin acentos ni stopwords, conservando negaciones como *no* y *sin*.
- **Etapa 1**: `{sm['finetuned_head']['model']}` ajustado (fine-tuned) con una cabeza de {len(LABELS)} salidas sigmoide, una por etiqueta.
- **B2**: TF-IDF de palabras y de n-gramas de caracteres, con una regresión logística balanceada por etiqueta; es también el modelo de respaldo cuando importa la latencia.
- **Combinación y decisión**: {w:.0%} cabeza ajustada + {1 - w:.0%} B2; umbrales por etiqueta ajustados en predicciones out-of-fold de CV; toda respuesta tiene al menos una etiqueta y *no* excluye a las demás.
- **Selección y servicio**: el modelo se eligió con CV agrupada de 5 folds y macro F1; el test se evaluó una sola vez. La API corre en CPU, sin conexión, y manda a revisión humana los casos cercanos al umbral y las etiquetas de riesgo (*no*, *cred*).
""")


# ----------------------------------------------------------------------------- page
st.title("Clasificación del uso del crédito (PyME)")
health = api_health()
if health is None:
    st.error(f"No hay conexión con la API en {API_URL}. Inicia todo con `python app/run_demo.py`.")
    st.stop()
st.caption(f"API: {API_URL} · modelo {health['model']} · CPU · sin conexión a internet")
try_it, results, batch, how = st.tabs(["Try it", "Results", "Batch", "How it works"])
with try_it:
    tab_try()
with results:
    tab_results()
with batch:
    tab_batch()
with how:
    tab_how()
