"""
app.py
======
Capa de presentación (Streamlit) del Perceptrón Monocapa.
Ejecutar con:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pdf_generator import build_pdf
from perceptron import GATES, INPUTS, Perceptron, fmt, step_to_latex

st.set_page_config(page_title="Perceptrón monocapa", page_icon="🧠", layout="wide")

# Valores preestablecidos (los del ejercicio en el pizarrón)
DEF_W1, DEF_W2, DEF_B, DEF_ETA, DEF_EPOCHS = 0.1, 0.3, 0.2, 0.1, 20


# --------------------------------------------------------------------------- #
# Esquema SVG dinámico (REQ 1)
# --------------------------------------------------------------------------- #
def perceptron_svg(w1, w2, b, x1=None, x2=None, z=None, y=None) -> str:
    v = lambda x: "" if x is None else fmt(x)
    out_fill = "none" if y is None else ("#2e9e5b" if y == 1 else "#8a8f98")
    out_txt = "y" if y is None else f"y = {y}"
    z_txt = "" if z is None else f"z = {fmt(z)}"
    return f"""
<svg viewBox="0 0 660 270" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:760px;color:inherit;font-family:sans-serif">
  <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
        orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
  <g stroke="currentColor" stroke-width="1.8" fill="none">
    <circle cx="70" cy="80" r="26"/><circle cx="70" cy="210" r="26"/>
    <line x1="96" y1="84" x2="262" y2="132" marker-end="url(#arr)"/>
    <line x1="96" y1="206" x2="262" y2="158" marker-end="url(#arr)"/>
    <circle cx="300" cy="145" r="38"/>
    <circle cx="300" cy="28" r="18"/>
    <line x1="300" y1="46" x2="300" y2="105" marker-end="url(#arr)"/>
    <line x1="338" y1="145" x2="398" y2="145" marker-end="url(#arr)"/>
    <rect x="400" y="105" width="90" height="80" rx="8"/>
    <polyline points="412,165 445,165 445,125 478,125" stroke-width="2.4"/>
    <line x1="490" y1="145" x2="556" y2="145" marker-end="url(#arr)"/>
    <circle cx="595" cy="145" r="36" fill="{out_fill}" fill-opacity="0.35"/>
  </g>
  <g fill="currentColor" font-size="15" text-anchor="middle">
    <text x="70" y="86" font-size="18">x₁</text><text x="70" y="216" font-size="18">x₂</text>
    <text x="70" y="124">{v(x1)}</text><text x="70" y="254">{v(x2)}</text>
    <text x="300" y="158" font-size="38">Σ</text>
    <text x="300" y="34" font-size="17">b</text>
    <text x="445" y="203" font-size="13">escalón f(z)</text>
    <text x="445" y="96">{z_txt}</text>
    <text x="595" y="150" font-size="15">{out_txt}</text>
  </g>
  <g fill="currentColor" font-size="14">
    <text x="118" y="74">w₁ = {fmt(w1)}</text>
    <text x="118" y="236">w₂ = {fmt(w2)}</text>
    <text x="312" y="84">b = {fmt(b)}</text>
  </g>
</svg>"""


# --------------------------------------------------------------------------- #
# Estado
# --------------------------------------------------------------------------- #
for key, default in (("result", None), ("cfg", None), ("pdf", None)):
    st.session_state.setdefault(key, default)

st.title("🧠 Perceptrón monocapa — simulador paso a paso")

cfg_col, table_col, calc_col = st.columns([1.05, 1.35, 1.9], gap="large")

# =========================================================================== #
# IZQUIERDA: configuración (tabla de verdad + hiperparámetros)  — REQ 2
# =========================================================================== #
with cfg_col:
    st.subheader("⚙️ Configuración")

    gate = st.selectbox("Compuerta predefinida", list(GATES.keys()), index=0)
    st.caption("Edita la columna **t** para definir tu propia tabla de verdad.")
    base = pd.DataFrame({"x1": [a for a, _ in INPUTS],
                         "x2": [b for _, b in INPUTS],
                         "t": GATES[gate]})
    tt = st.data_editor(
        base, key=f"tt_{gate}", hide_index=True, disabled=["x1", "x2"],
        column_config={"t": st.column_config.SelectboxColumn(
            "t (objetivo)", options=[0, 1], required=True)})
    t_list = [int(t) for t in tt["t"].tolist()]
    gate_name = next((k for k, vals in GATES.items() if vals == t_list), "Personalizada")
    st.markdown(f"**Compuerta actual:** {gate_name}")
    if gate_name == "XOR":
        st.warning("XOR no es linealmente separable: el perceptrón no convergerá.")

    st.markdown("**Parámetros iniciales**")
    c1, c2 = st.columns(2)
    w1 = c1.number_input("w₁", value=DEF_W1, step=0.1, format="%.3f")
    w2 = c2.number_input("w₂", value=DEF_W2, step=0.1, format="%.3f")
    b = c1.number_input("Sesgo b", value=DEF_B, step=0.1, format="%.3f")
    eta = c2.number_input("Tasa η", value=DEF_ETA, min_value=0.001, step=0.01, format="%.3f")
    max_epochs = st.number_input("Límite de épocas", min_value=1, max_value=500,
                                 value=DEF_EPOCHS, step=1)

    if st.button("▶ Entrenar", type="primary"):
        samples = [(float(x1), float(x2), t)
                   for (x1, x2), t in zip(INPUTS, t_list)]
        st.session_state.result = Perceptron(w1, w2, b, eta).train(samples, int(max_epochs))
        st.session_state.cfg = dict(gate=gate_name, samples=samples,
                                    max_epochs=int(max_epochs))
        st.session_state.pdf = None
        st.session_state.epoch_sel = 1

result = st.session_state.result
cfg = st.session_state.cfg

# =========================================================================== #
# CENTRO: tabla resumen por épocas — REQ 4  (y exportación PDF — REQ 5)
# =========================================================================== #
with table_col:
    st.subheader(f"📋 Entrenamiento compuerta {cfg['gate']}" if cfg else "📋 Entrenamiento")
    if result is None:
        st.info("Configura los parámetros y pulsa **Entrenar**.")
    else:
        df = pd.DataFrame([{"Época": e.epoch, "W1": fmt(e.w1), "W2": fmt(e.w2),
                            "b": fmt(e.b), "Errores": e.errors}
                           for e in result.history])
        st.dataframe(df, hide_index=True)
        st.caption(f"Valores iniciales: w₁ = {fmt(result.w1_init)}, "
                   f"w₂ = {fmt(result.w2_init)}, b = {fmt(result.b_init)}, "
                   f"η = {fmt(result.eta)}")
        if result.converged:
            st.success(f"Convergió en la época {result.history[-1].epoch} (0 errores).")
        else:
            st.warning(f"No convergió en {len(result.history)} épocas.")

        st.markdown("**Informe**")
        if st.button("📄 Preparar informe PDF"):
            with st.spinner("Generando PDF…"):
                st.session_state.pdf = build_pdf(cfg["gate"], cfg["samples"],
                                                 cfg["max_epochs"], result)
        if st.session_state.pdf:
            st.download_button("⬇ Descargar informe PDF", data=st.session_state.pdf,
                               file_name="informe_perceptron.pdf",
                               mime="application/pdf")

# =========================================================================== #
# DERECHA: cálculos de la época seleccionada (ocultable) — REQ 3
# =========================================================================== #
with calc_col:
    st.subheader("🧮 Cálculos por época")
    show = st.toggle("Mostrar cálculos", value=True)
    if result is None:
        st.caption("Aquí aparecerá el procedimiento de la época seleccionada.")
    elif show:
        sel = st.selectbox("Época a detallar", [e.epoch for e in result.history],
                           key="epoch_sel")
        ep = result.history[sel - 1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("w₁", fmt(ep.w1)); m2.metric("w₂", fmt(ep.w2))
        m3.metric("b", fmt(ep.b)); m4.metric("Errores", ep.errors)
        for i, s in enumerate(ep.steps, 1):
            with st.container(border=True):
                st.markdown(f"**Muestra {i}:** (x₁, x₂) = ({fmt(s.x1)}, {fmt(s.x2)}), "
                            f"t = {s.t}")
                body = r" \\ ".join(step_to_latex(s))
                st.latex(r"\begin{array}{l}" + body + r"\end{array}")
    else:
        st.caption("Sección oculta.")

# =========================================================================== #
# ABAJO: esquema dinámico (REQ 1) e inferencia (REQ 6)
# =========================================================================== #
st.divider()
st.header("🔬 Esquema del perceptrón y prueba")
sch_col, inf_col = st.columns([2, 1], gap="large")

fw1, fw2, fb = (result.w1, result.w2, result.b) if result else (w1, w2, b)
x1 = x2 = z = y = None

with inf_col:
    st.subheader("Módulo de inferencia")
    if result is None:
        st.info("Entrena el perceptrón para habilitar la prueba.")
    else:
        i1, i2 = st.columns(2)
        x1 = i1.number_input("x₁", value=0.0, step=1.0)
        x2 = i2.number_input("x₂", value=0.0, step=1.0)
        z, y = Perceptron(fw1, fw2, fb, result.eta).predict(x1, x2)
        st.latex(rf"z = ({fmt(x1)} \cdot {fmt(fw1)}) + ({fmt(x2)} \cdot {fmt(fw2)}) "
                 rf"+ {fmt(fb)} = {fmt(z)}")
        st.metric("Salida predicha y", y)
        expected = {(a, c): t for a, c, t in cfg["samples"]}.get((x1, x2))
        if expected is not None:
            st.caption("✅ Coincide con la tabla de verdad." if expected == y
                       else f"⚠️ La tabla de verdad esperaba t = {expected}.")

with sch_col:
    st.caption("Pesos finales tras el entrenamiento." if result
               else "Pesos iniciales configurados.")
    st.html(perceptron_svg(fw1, fw2, fb, x1, x2, z, y))
