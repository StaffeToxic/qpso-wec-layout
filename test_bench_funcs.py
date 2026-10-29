# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Tests for bench_funcs: smooth analytic surfaces with known optima.

These surfaces expose a sklearn-like ``predict(X)`` so they can be fed to the
QPSO optimizers in place of the ML surrogate. They let us prove the optimizer
*searches* on a smooth landscape (independent of the staircase Random Forest).

Run directly (no pytest needed):  python test_bench_funcs.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bf = load_module(ROOT / "bench_funcs.py", "bench_funcs")


def test_paraboloid_predict_is_maximal_at_optimum():
    surf = bf.Paraboloid(dim=4)
    # predict returns shape (n,)
    X = np.random.default_rng(0).uniform(surf.bounds_min, surf.bounds_max, size=(20, 4))
    f = surf.predict(X)
    assert f.shape == (20,)
    # the declared optimum must score the declared optimum value...
    f_opt = surf.predict(surf.optimum_x.reshape(1, -1))[0]
    assert np.isclose(f_opt, surf.optimum_f, atol=1e-9)
    # ...and no sampled point may beat the optimum (it is a true maximum)
    assert np.all(f <= surf.optimum_f + 1e-9)


def test_staircase_is_flat_within_a_cell_unlike_smooth():
    """The crux of finding #3 at the surface level: a Random-Forest-like
    staircase has ZERO gradient inside a cell, so a small move changes nothing,
    whereas the smooth paraboloid always responds. This is *why* gradient-free
    continuous search needs a smooth surrogate."""
    smooth = bf.Paraboloid(dim=4)
    stair = bf.StaircaseSurface(dim=4, step=2.5)
    x0 = np.full(4, 1.6)            # interior of a cell, off the grid and off-center
    x1 = x0 + 0.2                   # same cell (|0.2| < step/2), different point
    # staircase: identical score (flat plateau, no gradient)
    assert np.isclose(stair.predict(x0.reshape(1, -1))[0], stair.predict(x1.reshape(1, -1))[0])
    # smooth: score genuinely changes (gradient exists)
    assert not np.isclose(smooth.predict(x0.reshape(1, -1))[0], smooth.predict(x1.reshape(1, -1))[0])


if __name__ == "__main__":
    tests = [
        test_paraboloid_predict_is_maximal_at_optimum,
        test_staircase_is_flat_within_a_cell_unlike_smooth,
    ]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}")
