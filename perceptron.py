"""
perceptron.py
=============
Lógica pura del Perceptrón Monocapa (SLP) y almacenamiento del historial.
No depende de Streamlit ni de ninguna librería de documentos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Datos de compuertas
# --------------------------------------------------------------------------- #
INPUTS: List[Tuple[int, int]] = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Salidas esperadas (t) para las entradas INPUTS, en ese orden
GATES = {
    "AND": [0, 0, 0, 1],
    "OR": [0, 1, 1, 1],
    "NAND": [1, 1, 1, 0],
    "NOR": [1, 0, 0, 0],
    "XOR": [0, 1, 1, 0],  # no separable linealmente: el perceptrón no converge
}

Sample = Tuple[float, float, int]  # (x1, x2, t)


# --------------------------------------------------------------------------- #
# Utilidades de formato numérico (evitan ruido de punto flotante: 0.30000000004)
# --------------------------------------------------------------------------- #
def clean(v: float) -> float:
    v = round(float(v), 10)
    return 0.0 if v == 0 else v


def fmt(v: float) -> str:
    """Número sin ceros sobrantes: 0.100000 -> '0.1', -0.0 -> '0'."""
    s = f"{clean(v):.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def paren(v: float) -> str:
    """Igual que fmt, pero los negativos van entre paréntesis: (-0.1)."""
    s = fmt(v)
    return f"({s})" if s.startswith("-") else s


# --------------------------------------------------------------------------- #
# Estructuras del historial
# --------------------------------------------------------------------------- #
@dataclass
class StepRecord:
    """Una muestra evaluada dentro de una época."""
    epoch: int
    x1: float
    x2: float
    t: int
    eta: float
    w1_prev: float
    w2_prev: float
    b_prev: float
    z: float
    y: int
    error: int
    w1_new: float
    w2_new: float
    b_new: float


@dataclass
class EpochRecord:
    epoch: int
    steps: List[StepRecord]
    w1: float  # valores al terminar la época
    w2: float
    b: float
    errors: int


@dataclass
class TrainingResult:
    history: List[EpochRecord] = field(default_factory=list)
    converged: bool = False
    w1_init: float = 0.0
    w2_init: float = 0.0
    b_init: float = 0.0
    eta: float = 0.0

    @property
    def w1(self) -> float:
        return self.history[-1].w1 if self.history else self.w1_init

    @property
    def w2(self) -> float:
        return self.history[-1].w2 if self.history else self.w2_init

    @property
    def b(self) -> float:
        return self.history[-1].b if self.history else self.b_init


# --------------------------------------------------------------------------- #
# Perceptrón
# --------------------------------------------------------------------------- #
def step_function(z: float) -> int:
    """Escalón: 1 si z >= 0, 0 en otro caso."""
    return 1 if z >= 0 else 0


class Perceptron:
    def __init__(self, w1: float, w2: float, b: float, eta: float):
        self.w1, self.w2, self.b, self.eta = w1, w2, b, eta

    def weighted_sum(self, x1: float, x2: float) -> float:
        return clean(x1 * self.w1 + x2 * self.w2 + self.b)

    def predict(self, x1: float, x2: float) -> Tuple[float, int]:
        z = self.weighted_sum(x1, x2)
        return z, step_function(z)

    def train(self, samples: Sequence[Sample], max_epochs: int) -> TrainingResult:
        result = TrainingResult(w1_init=self.w1, w2_init=self.w2,
                                b_init=self.b, eta=self.eta)
        for epoch in range(1, max_epochs + 1):
            steps: List[StepRecord] = []
            errors = 0
            for x1, x2, t in samples:
                w1p, w2p, bp = self.w1, self.w2, self.b
                z = self.weighted_sum(x1, x2)
                y = step_function(z)
                e = t - y
                # Regla delta: w_nuevo = w_previo + eta * e * x ;  b_nuevo = b + eta * e
                self.w1 = clean(w1p + self.eta * e * x1)
                self.w2 = clean(w2p + self.eta * e * x2)
                self.b = clean(bp + self.eta * e)
                if e != 0:
                    errors += 1
                steps.append(StepRecord(epoch, x1, x2, t, self.eta, w1p, w2p, bp,
                                        z, y, e, self.w1, self.w2, self.b))
            result.history.append(EpochRecord(epoch, steps, self.w1, self.w2,
                                              self.b, errors))
            if errors == 0:
                result.converged = True
                break
        return result


# --------------------------------------------------------------------------- #
# Representación LaTeX de un paso (usada por la web y por el PDF)
# Subconjunto compatible con KaTeX (Streamlit) y matplotlib.mathtext (PDF).
# --------------------------------------------------------------------------- #
def step_to_latex(s: StepRecord) -> List[str]:
    cond = r"(z \geq 0)" if s.z >= 0 else r"(z < 0)"
    return [
        rf"z = (x_1 \cdot w_1) + (x_2 \cdot w_2) + b = "
        rf"({fmt(s.x1)} \cdot {paren(s.w1_prev)}) + ({fmt(s.x2)} \cdot {paren(s.w2_prev)}) "
        rf"+ {paren(s.b_prev)} = {fmt(s.z)}",
        rf"y = f(z) = f({fmt(s.z)}) = {s.y} \quad {cond}",
        rf"e = t - y = {s.t} - {s.y} = {fmt(s.error)}",
        rf"w_1^{{(n)}} = w_1^{{(prev)}} + \eta \cdot e \cdot x_1 = "
        rf"{fmt(s.w1_prev)} + {fmt(s.eta)} \cdot {paren(s.error)} \cdot {fmt(s.x1)} = {fmt(s.w1_new)}",
        rf"w_2^{{(n)}} = w_2^{{(prev)}} + \eta \cdot e \cdot x_2 = "
        rf"{fmt(s.w2_prev)} + {fmt(s.eta)} \cdot {paren(s.error)} \cdot {fmt(s.x2)} = {fmt(s.w2_new)}",
        rf"b^{{(n)}} = b^{{(prev)}} + \eta \cdot e = "
        rf"{fmt(s.b_prev)} + {fmt(s.eta)} \cdot {paren(s.error)} = {fmt(s.b_new)}",
    ]
