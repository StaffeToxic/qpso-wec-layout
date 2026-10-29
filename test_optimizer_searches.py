# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Proof test: the QPSO optimizer *searches* on a smooth surface.

Root cause of finding #3 (the pure QPSO returned the seed layout to the cent)
is NOT a broken optimizer. Here we feed the *existing* PureContinuousQPSO_WEC a
smooth analytic paraboloid, seeded FAR from the optimum, and show it climbs ~all
the way to the known maximum. So the optimizer core is sound; the bottleneck was
the piecewise-constant Random Forest. (Why RF specifically blocks improvement is
shown on real data in Fase 1: RF's max over the box ~= the best training layout.)

Run directly (no pytest needed):  python test_optimizer_searches.py
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
pure = load_module(ROOT / "qpso_layout_puro_standalone.py", "qpso_layout_puro_standalone")


def _run_gap(surf, seed=7):
    """Run the pure QPSO seeded FAR from the optimum; return fraction of the
    optimality gap it closed (1.0 = reached the optimum, 0.0 = no progress)."""
    rng = np.random.default_rng(42)
    layout_ref = np.full(surf.dim, 1.5)
    X = np.clip(rng.normal(layout_ref, 0.5, size=(60, surf.dim)), surf.bounds_min, surf.bounds_max)
    y = surf.predict(X)
    opt = pure.PureContinuousQPSO_WEC(
        num_wecs=surf.dim // 2, num_particles=40, max_iter=120, modelo_rf=surf,
        layout_ref=layout_ref, bounds_min=surf.bounds_min, bounds_max=surf.bounds_max,
        dataset_X=X, dataset_y=y, random_state=seed,
    )
    _, best_fit, _ = opt.optimize(verbose=False)
    init = surf.predict(layout_ref.reshape(1, -1))[0]
    return (best_fit - init) / (surf.optimum_f - init)


def test_qpso_climbs_on_smooth_paraboloid():
    gap = _run_gap(bf.Paraboloid(dim=4))
    assert gap >= 0.90, f"optimizer barely moved on a smooth surface: gap_closed={gap:.3f}"


if __name__ == "__main__":
    tests = [test_qpso_climbs_on_smooth_paraboloid]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}")
