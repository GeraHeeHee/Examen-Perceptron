"""
pdf_generator.py
================
Generador dinámico del informe PDF (tabla de épocas + procedimiento paso a paso).
Las ecuaciones se renderizan con matplotlib.mathtext y se incrustan como imágenes.
"""
from __future__ import annotations

import io
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from perceptron import Sample, TrainingResult, fmt, step_to_latex

DPI = 200
MAX_W = 17 * cm


def _latex_image(lines: Sequence[str], fontsize: int = 11) -> Image:
    n = len(lines)
    fig = plt.figure(figsize=(9, 0.34 * n + 0.1))
    for i, ln in enumerate(lines):
        fig.text(0.0, 1 - (i + 0.5) / n, f"${ln}$", fontsize=fontsize,
                 va="center", ha="left")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    buf.seek(0)
    px_w, px_h = ImageReader(buf).getSize()
    buf.seek(0)
    w, h = px_w * 72 / DPI, px_h * 72 / DPI
    if w > MAX_W:
        h *= MAX_W / w
        w = MAX_W
    return Image(buf, width=w, height=h, hAlign="LEFT")


def _table(data, col_widths=None, header_color="#1f3a5f"):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def build_pdf(gate_name: str, samples: Sequence[Sample], max_epochs: int,
              result: TrainingResult) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Informe Perceptron Monocapa")
    ss = getSampleStyleSheet()
    h1, h2, body = ss["Title"], ss["Heading2"], ss["BodyText"]
    h3 = ParagraphStyle("h3", parent=ss["Heading3"], textColor=colors.HexColor("#1f3a5f"))

    story = [Paragraph("Informe del Perceptr&oacute;n Monocapa", h1),
             Paragraph(f"Compuerta l&oacute;gica: <b>{gate_name}</b>", body),
             Spacer(1, 8)]

    # 1. Hiperparámetros
    story.append(Paragraph("1. Par&aacute;metros iniciales", h2))
    story.append(_table(
        [["w1", "w2", "b", "eta", "Epocas max."],
         [fmt(result.w1_init), fmt(result.w2_init), fmt(result.b_init),
          fmt(result.eta), str(max_epochs)]]))
    story.append(Spacer(1, 8))

    # 2. Tabla de verdad
    story.append(Paragraph("2. Tabla de verdad utilizada", h2))
    story.append(_table([["x1", "x2", "t"]] + [[fmt(a), fmt(b), str(t)] for a, b, t in samples],
                        col_widths=[3 * cm] * 3))
    story.append(Spacer(1, 8))

    # 3. Resumen por épocas
    story.append(Paragraph("3. Tabla resumen por &eacute;pocas", h2))
    rows = [["Epoca", "W1", "W2", "b", "Errores"]]
    rows += [[str(e.epoch), fmt(e.w1), fmt(e.w2), fmt(e.b), str(e.errors)]
             for e in result.history]
    story.append(_table(rows))
    story.append(Spacer(1, 6))
    if result.converged:
        story.append(Paragraph(
            f"El perceptr&oacute;n <b>convergi&oacute;</b> en la &eacute;poca {result.history[-1].epoch} "
            f"(0 errores). Pesos finales: w<sub>1</sub> = {fmt(result.w1)}, "
            f"w<sub>2</sub> = {fmt(result.w2)}, b = {fmt(result.b)}.", body))
    else:
        story.append(Paragraph(
            f"El perceptr&oacute;n <b>no convergi&oacute;</b> en {len(result.history)} &eacute;pocas. "
            f"Pesos finales: w<sub>1</sub> = {fmt(result.w1)}, "
            f"w<sub>2</sub> = {fmt(result.w2)}, b = {fmt(result.b)}.", body))

    # 4. Procedimiento paso a paso
    story.append(Paragraph("4. Procedimiento paso a paso", h2))
    for ep in result.history:
        story.append(Paragraph(
            f"&Eacute;poca {ep.epoch} &mdash; errores: {ep.errors}", h3))
        for i, s in enumerate(ep.steps, 1):
            block = [
                Paragraph(f"<b>Muestra {i}:</b> (x<sub>1</sub>, x<sub>2</sub>) = "
                          f"({fmt(s.x1)}, {fmt(s.x2)}), t = {s.t}", body),
                _latex_image(step_to_latex(s)),
                Spacer(1, 6),
            ]
            story.append(KeepTogether(block))
        story.append(Paragraph(
            f"<i>Al terminar la &eacute;poca {ep.epoch}:</i> w<sub>1</sub> = {fmt(ep.w1)}, "
            f"w<sub>2</sub> = {fmt(ep.w2)}, b = {fmt(ep.b)}, errores = {ep.errors}.", body))
        story.append(Spacer(1, 10))

    doc.build(story)
    return buf.getvalue()
