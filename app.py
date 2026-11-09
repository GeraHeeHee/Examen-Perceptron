"""
app.py
======
Capa de presentación (Streamlit) del Perceptrón Monocapa.
Ejecutar con:  streamlit run app.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from pdf_generator import build_pdf
from plots import boundary_figure, perceptron_figure
from perceptron import GATES, INPUTS, Perceptron, fmt, step_to_latex

st.set_page_config(page_title="Perceptrón monocapa", page_icon="🧠", layout="wide")

# Valores preestablecidos (los del ejercicio en el pizarrón)
DEF_W1, DEF_W2, DEF_B, DEF_ETA, DEF_EPOCHS = 0.1, 0.3, 0.2, 0.1, 20


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
        st.session_state.graph_epoch = len(st.session_state.result.history)

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
# ABAJO: esquema del perceptrón (REQ 1) + frontera de decisión + inferencia (REQ 6)
# =========================================================================== #
st.divider()
st.header("🔬 Esquema del perceptrón y prueba")

if result is None:
    st.caption("Vista previa con los parámetros iniciales. Entrena para ver el resultado.")
    gsel, gw1, gw2, gb = 0, w1, w2, b
    samples_now = [(float(a), float(c), t) for (a, c), t in zip(INPUTS, t_list)]
    path = [(w1, w2, b)]
else:
    samples_now = cfg["samples"]
    gsel = st.select_slider("Época a graficar (los pesos salen del entrenamiento)",
                            options=list(range(len(result.history) + 1)),
                            key="graph_epoch",
                            format_func=lambda e: "Inicial" if e == 0 else f"Época {e}")
    path = [(result.w1_init, result.w2_init, result.b_init)] + \
           [(e.w1, e.w2, e.b) for e in result.history[:gsel]]
    gw1, gw2, gb = path[-1]

graph_title = "Pesos iniciales" if gsel == 0 else f"Pesos al terminar la época {gsel}"
x1 = x2 = z = y = None
left, right = st.columns([1.7, 1], gap="large")

with right:
    st.subheader("Módulo de inferencia")
    if result is None:
        st.info("Entrena el perceptrón para habilitar la prueba.")
    else:
        i1, i2 = st.columns(2)
        x1 = i1.number_input("x₁", value=0.0, step=1.0)
        x2 = i2.number_input("x₂", value=0.0, step=1.0)
        z, y = Perceptron(gw1, gw2, gb, result.eta).predict(x1, x2)
        st.latex(rf"z = ({fmt(x1)} \cdot {fmt(gw1)}) + ({fmt(x2)} \cdot {fmt(gw2)}) "
                 rf"+ {fmt(gb)} = {fmt(z)}")
        st.metric("Salida predicha y", y)
        expected = {(a, c): t for a, c, t in samples_now}.get((x1, x2))
        if expected is not None:
            st.caption("✅ Coincide con la tabla de verdad." if expected == y
                       else f"⚠️ La tabla de verdad esperaba t = {expected}.")

with left:
    fig = perceptron_figure(gw1, gw2, gb, x1, x2, z, y, graph_title)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Frontera de decisión** " + r"$w_1x_1 + w_2x_2 + b = 0$")
    st.caption("Línea continua: época mostrada. Líneas punteadas: épocas anteriores. "
               "Zona verde: y = 1.")
    fig2 = boundary_figure(samples_now, path, (x1, x2) if x1 is not None else None,
                           graph_title)
    st.pyplot(fig2)
    plt.close(fig2)