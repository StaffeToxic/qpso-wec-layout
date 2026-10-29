# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Tests for qpso_discreto.py — the refactored DISCRETE QPSO (buoy selection).

Unlike the continuous case, the SHAP fitness is additive: P(b) = phi0 + sum b_i*c_i,
so turning a buoy off just subtracts its contribution (interpolation, not
extrapolation). The optimizer should therefore learn to keep high-contribution
buoys and drop low/negative ones, subject to a minimum-count constraint.

Same conventions/fixes as qpso_continuo.py: in-place elitism, single eval,
non-stale operators. Run:  python test_qpso_discreto.py
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


qd = load_module(ROOT / "qpso_discreto.py", "qpso_discreto")


def make_shap_fitness(contribs, base, min_boias, wp=0.7, we=0.3):
    contribs = np.asarray(contribs, float)

    def fitness(bits):
        bits = np.atleast_2d(np.asarray(bits, float))
        n = bits.sum(axis=1)
        power = base + bits @ contribs
        eff = power / np.maximum(n, 1.0)
        f = wp * power + we * eff
        return np.where(n < min_boias, -1e12, f)

    return fitness


def test_discrete_qpso_keeps_high_contribution_buoys():
    N, min_boias = 49, 20
    rng = np.random.default_rng(0)
    contribs = rng.normal(0.0, 1000.0, N)          # mix of + and - contributions
    base = 4_000_000.0
    fitness = make_shap_fitness(contribs, base, min_boias)

    opt = qd.DiscreteQPSO(fitness, num_wecs=N, num_particles=30, max_iter=80,
                          min_boias=min_boias, random_state=1)
    gbest, gfit, history = opt.optimize(verbose=False)

    assert gbest.shape == (N,)
    assert set(np.unique(gbest)).issubset({0, 1})
    assert int(gbest.sum()) >= min_boias                       # constraint respected
    assert gfit >= fitness(np.ones(N))[0]                      # beats keeping all buoys
    # kept buoys must have higher mean contribution than dropped ones
    assert contribs[gbest == 1].mean() > contribs[gbest == 0].mean()


def test_gbest_monotonic_and_min_boias_respected():
    N, min_boias = 49, 20
    contribs = np.random.default_rng(2).normal(0.0, 1000.0, N)
    fitness = make_shap_fitness(contribs, 4_000_000.0, min_boias)
    opt = qd.DiscreteQPSO(fitness, num_wecs=N, num_particles=30, max_iter=60,
                          min_boias=min_boias, random_state=2)
    gbest, _, history = opt.optimize(verbose=False)
    assert np.all(np.diff(history) >= -1e-6), "gbest decreased — elitism broken"
    assert int(gbest.sum()) >= min_boias


def test_pbest_matches_its_fitness():
    """#5a invariant: in-place elitism must keep pbest[i] consistent with pbest_fit[i]."""
    N, min_boias = 49, 20
    contribs = np.random.default_rng(3).normal(0.0, 1000.0, N)
    fitness = make_shap_fitness(contribs, 4_000_000.0, min_boias)
    opt = qd.DiscreteQPSO(fitness, num_wecs=N, num_particles=30, max_iter=60,
                          min_boias=min_boias, random_state=3)
    opt.optimize(verbose=False)
    assert np.allclose(fitness(opt.pbest), opt.pbest_fit, rtol=1e-9, atol=1e-3)


def test_efficiency_weight_reduces_buoy_count():
    """With all-positive contributions and an efficiency-heavy objective, dropping
    buoys raises P/n, so the optimizer should shed buoys (toward the minimum)."""
    N, min_boias = 49, 20
    contribs = np.random.default_rng(4).uniform(100.0, 1000.0, N)
    fitness = make_shap_fitness(contribs, 4_000_000.0, min_boias, wp=0.0, we=1.0)
    opt = qd.DiscreteQPSO(fitness, num_wecs=N, num_particles=30, max_iter=80,
                          min_boias=min_boias, random_state=4)
    gbest, _, _ = opt.optimize(verbose=False)
    assert min_boias <= int(gbest.sum()) < N, f"expected buoy shedding, got {int(gbest.sum())}/{N}"


if __name__ == "__main__":
    tests = [
        test_discrete_qpso_keeps_high_contribution_buoys,
        test_gbest_monotonic_and_min_boias_respected,
        test_pbest_matches_its_fitness,
        test_efficiency_weight_reduces_buoy_count,
    ]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}")
