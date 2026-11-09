"""
plots.py
========
Gráficos (matplotlib) de la capa de presentación:
  * perceptron_figure : esquema de la red con los valores calculados.
  * boundary_figure   : frontera de decisión w1*x1 + w2*x2 + b = 0 y su evolución.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from perceptron import Sample, fmt

INK = "#1f2937"
GREEN = "#2e9e5b"
GREY = "#8a8f98"
BLUE = "#2563eb"


def _arrow(ax, start, end, shrink_a=0, shrink_b=0):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6,
                                shrinkA=shrink_a, shrinkB=shrink_b))


def _label(ax, xy, text, **kw):
    ax.text(*xy, text, ha="center", va="center", fontsize=11, color=INK,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"), **kw)


def perceptron_figure(w1: float, w2: float, b: float,
                      x1: Optional[float] = None, x2: Optional[float] = None,
                      z: Optional[float] = None, y: Optional[int] = None,
                      title: str = ""):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0.3, 6.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Entradas
    for cy, name, val in ((4.4, "x_1", x1), (1.8, "x_2", x2)):
        ax.add_patch(Circle((1, cy), 0.6, fc="white", ec=INK, lw=1.8))
        ax.text(1, cy, f"${name}$", ha="center", va="center", fontsize=15, color=INK)
        if val is not None:
            ax.text(1, cy - 0.95, f"= {fmt(val)}", ha="center", va="center",
                    fontsize=11, color=BLUE)

    # Suma ponderada
    ax.add_patch(Circle((6, 3.1), 0.8, fc="white", ec=INK, lw=1.8))
    ax.text(6, 3.1, r"$\Sigma$", ha="center", va="center", fontsize=24, color=INK)

    # Pesos
    _arrow(ax, (1, 4.4), (6, 3.1), shrink_a=32, shrink_b=40)
    _arrow(ax, (1, 1.8), (6, 3.1), shrink_a=32, shrink_b=40)
    _label(ax, (3.5, 3.95), f"$w_1 = {fmt(w1)}$")
    _label(ax, (3.5, 2.25), f"$w_2 = {fmt(w2)}$")

    # Sesgo
    ax.add_patch(Circle((6, 5.5), 0.45, fc="white", ec=INK, lw=1.8))
    ax.text(6, 5.5, "$b$", ha="center", va="center", fontsize=14, color=INK)
    _arrow(ax, (6, 5.5), (6, 3.1), shrink_a=24, shrink_b=40)
    ax.text(6.25, 4.5, f"$b = {fmt(b)}$", ha="left", va="center", fontsize=11, color=INK)

    # Escalón
    _arrow(ax, (6, 3.1), (8.2, 3.1), shrink_a=40)
    ax.add_patch(Rectangle((8.2, 2.2), 1.8, 1.8, fc="white", ec=INK, lw=1.8))
    ax.plot([8.45, 9.1, 9.1, 9.75], [2.55, 2.55, 3.65, 3.65], color=INK, lw=2.4)
    ax.text(9.1, 1.85, "escalón $f(z)$", ha="center", va="center", fontsize=10, color=INK)
    if z is not None:
        ax.text(9.1, 4.35, f"$z = {fmt(z)}$", ha="center", va="center",
                fontsize=11, color=BLUE)

    # Salida
    _arrow(ax, (10, 3.1), (11.0, 3.1))
    fc = "white" if y is None else (GREEN if y == 1 else GREY)
    ax.add_patch(Circle((11.5, 3.1), 0.6, fc=fc, ec=INK, lw=1.8,
                        alpha=1 if y is None else 0.45))
    ax.text(11.5, 3.1, "$y$" if y is None else f"$y={y}$", ha="center", va="center",
            fontsize=13, color=INK)

    if title:
        ax.text(6, 6.1, title, ha="center", va="center", fontsize=12,
                fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def boundary_figure(samples: Sequence[Sample],
                    path: Sequence[Tuple[float, float, float]],
                    test_point: Optional[Tuple[float, float]] = None,
                    title: str = ""):
    """`path` = pesos (w1, w2, b) desde el inicial hasta la época mostrada;
    el último elemento es la frontera actual, los anteriores se dibujan tenues."""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    g = np.linspace(-0.5, 1.5, 150)
    X, Y = np.meshgrid(g, g)

    def zgrid(w):
        return w[0] * X + w[1] * Y + w[2]

    for w in path[:-1]:
        if abs(w[0]) + abs(w[1]) > 0:
            ax.contour(X, Y, zgrid(w), levels=[0], colors=[GREY], linewidths=0.9,
                       linestyles="dashed", alpha=0.7)
    cur = path[-1]
    Z = zgrid(cur)
    if Z.max() >= 0 > Z.min():
        ax.contourf(X, Y, Z, levels=[0, 1e9], colors=[GREEN], alpha=0.14)
        ax.contour(X, Y, Z, levels=[0], colors=[INK], linewidths=2.2)
    elif Z.min() >= 0:
        ax.set_facecolor("#eaf6ef")  # todo el plano da y = 1

    for x1, x2, t in samples:
        ax.scatter([x1], [x2], s=230, c=GREEN if t == 1 else "white",
                   edgecolors=INK, linewidths=1.8, zorder=3)
        ax.text(x1, x2 + 0.13, f"t={t}", ha="center", fontsize=9, color=INK)
    if test_point is not None:
        ax.scatter([test_point[0]], [test_point[1]], marker="*", s=260, c="#dc2626",
                   zorder=4, label="punto de prueba")
        ax.legend(loc="upper left", fontsize=8)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig
