# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Smooth analytic benchmark surfaces with known optima.

Each surface exposes a sklearn-like ``predict(X)`` (higher = better, since the
QPSO *maximizes* fitness) so it can be passed to the optimizers in place of the
ML surrogate. Purpose: prove the QPSO actually searches on a smooth landscape,
isolating "optimizer works" from "Random Forest is piecewise-constant".
"""
from __future__ import annotations

import numpy as np


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(1, -1) if X.ndim == 1 else X


class Paraboloid:
    """Concave bowl: ``f(x) = -sum((x - c)^2)``.

    A single global maximum of ``0`` at the center ``c``. The easiest possible
    smooth surface — if the optimizer cannot climb this, it is broken.
    """

    name = "paraboloid"

    def __init__(self, dim: int = 4, low: float = 0.0, high: float = 10.0):
        self.dim = int(dim)
        self.bounds_min = np.full(self.dim, float(low))
        self.bounds_max = np.full(self.dim, float(high))
        self.optimum_x = (self.bounds_min + self.bounds_max) / 2.0
        self.optimum_f = 0.0

    def predict(self, X) -> np.ndarray:
        X = _as_2d(X)
        return -np.sum((X - self.optimum_x) ** 2, axis=1)


class StaircaseSurface:
    """Piecewise-constant paraboloid: quantizes ``x`` to a grid before scoring.

    Emulates a Random-Forest-style staircase response — flat plateaus with zero
    gradient almost everywhere — while keeping the *same* global optimum as
    :class:`Paraboloid`. Continuous metaheuristics struggle here because moving
    within a plateau does not change fitness, so selection pressure vanishes.
    """

    name = "staircase"

    def __init__(self, dim: int = 4, low: float = 0.0, high: float = 10.0, step: float = 2.5):
        self.dim = int(dim)
        self.step = float(step)
        self.bounds_min = np.full(self.dim, float(low))
        self.bounds_max = np.full(self.dim, float(high))
        self.optimum_x = (self.bounds_min + self.bounds_max) / 2.0
        self.optimum_f = float(self.predict(self.optimum_x.reshape(1, -1))[0])

    def predict(self, X) -> np.ndarray:
        X = _as_2d(X)
        Xq = np.round(X / self.step) * self.step
        return -np.sum((Xq - self.optimum_x) ** 2, axis=1)
