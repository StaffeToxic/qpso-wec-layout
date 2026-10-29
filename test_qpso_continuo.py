# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Tests for qpso_continuo.py — the refactored continuous QPSO core.

Fixes carried by this module (vs. the legacy qpso_layout_standalone.py):
  #5a elitism is in-place (preserves pbest<->particle identity, gbest monotone)
  #5b reference layout seeding is explicit and consistent
  #4  soft minimum inter-buoy distance repair (d_min)
  LCB trust-region objective (mean - kappa*std) when the surrogate is uncertain

All tests exercise the public `optimize()` only, on smooth surfaces / toy
surrogates with known answers. Run:  python test_qpso_continuo.py
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
qc = load_module(ROOT / "qpso_continuo.py", "qpso_continuo")


def _seed_far(surf, rng, n=60):
    ref = np.full(surf.dim, 1.5)
    X = np.clip(rng.normal(ref, 0.5, size=(n, surf.dim)), surf.bounds_min, surf.bounds_max)
    return X, ref


def test_continuous_qpso_climbs_on_smooth_paraboloid():
    surf = bf.Paraboloid(dim=4)
    rng = np.random.default_rng(42)
    seeds, ref = _seed_far(surf, rng)
    opt = qc.ContinuousQPSO(
        surrogate=surf, bounds_min=surf.bounds_min, bounds_max=surf.bounds_max,
        num_particles=40, max_iter=120, seed_layouts=seeds, layout_ref=ref,
        random_state=7,
    )
    gbest, gbest_fit, history = opt.optimize(verbose=False)
    init = surf.predict(ref.reshape(1, -1))[0]
    gap = (gbest_fit - init) / (surf.optimum_f - init)
    assert gbest.shape == (4,)
    assert len(history) == 120
    assert gap >= 0.90, f"refactored core barely moved: gap_closed={gap:.3f}"


def test_gbest_is_monotonic_nondecreasing():
    """#5a consequence: with in-place elitism the global best can never get
    worse from one iteration to the next."""
    surf = bf.Paraboloid(dim=4)
    rng = np.random.default_rng(0)
    seeds, ref = _seed_far(surf, rng)
    opt = qc.ContinuousQPSO(
        surrogate=surf, bounds_min=surf.bounds_min, bounds_max=surf.bounds_max,
        num_particles=30, max_iter=80, seed_layouts=seeds, layout_ref=ref, random_state=3,
    )
    _, _, history = opt.optimize(verbose=False)
    assert np.all(np.diff(history) >= -1e-9), "gbest decreased — elitism/bests broken"


def test_reference_layout_is_preserved_when_seeded():
    """#5b: seeding the reference must never yield something worse than it
    (the legacy non-pure core discarded the reference entirely)."""
    surf = bf.Paraboloid(dim=4)
    ref = surf.optimum_x.copy()                      # the reference IS the optimum
    rng = np.random.default_rng(1)
    far = np.clip(rng.normal(np.full(4, 1.0), 0.3, size=(40, 4)), surf.bounds_min, surf.bounds_max)
    opt = qc.ContinuousQPSO(
        surrogate=surf, bounds_min=surf.bounds_min, bounds_max=surf.bounds_max,
        num_particles=40, max_iter=2, seed_layouts=far, layout_ref=ref,
        seed_with_ref=True, random_state=5,
    )
    _, gbest_fit, _ = opt.optimize(verbose=False)
    assert gbest_fit >= surf.predict(ref.reshape(1, -1))[0] - 1e-6


def _min_pairwise(layout, n_wecs):
    pts = layout.reshape(n_wecs, 2)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d.min()


def test_repair_method_enforces_distance():
    """#4 (unit): the projection repair separates a tight cluster to >= d_min."""
    dim, n_wecs, d_min = 12, 6, 20.0
    opt = qc.ContinuousQPSO(
        surrogate=bf.Paraboloid(dim=dim, low=0.0, high=100.0),
        bounds_min=np.zeros(dim), bounds_max=np.full(dim, 100.0),
        d_min=d_min, num_particles=4, max_iter=1, random_state=0,
    )
    cluster = np.full(dim, 50.0) + np.random.default_rng(0).normal(0, 2.0, dim)
    repaired = opt._repair_min_dist(cluster.copy())
    assert _min_pairwise(repaired, n_wecs) >= d_min - 1e-6


def test_d_min_repair_separates_buoys():
    """#4 (end-to-end): the optimized layout respects d_min. The pull is
    centered in the box so clustering happens in free space (the real WEC
    surrogate likewise rewards spread, not corner piles)."""
    dim, n_wecs, d_min = 12, 6, 20.0
    surf = bf.Paraboloid(dim=dim, low=0.0, high=100.0)   # optimum at box center
    opt = qc.ContinuousQPSO(
        surrogate=surf, bounds_min=np.zeros(dim), bounds_max=np.full(dim, 100.0),
        num_particles=20, max_iter=25, d_min=d_min, n_elite=2, random_state=0,
    )
    gbest, _, _ = opt.optimize(verbose=False)
    assert _min_pairwise(gbest, n_wecs) >= d_min - 1.0


def test_pbest_stays_consistent_with_its_fitness():
    """#5a invariant: the legacy core copied elites into slots 0..k, so
    pbest[i] no longer matched pbest_fit[i]. With in-place elitism each stored
    personal best must still score its recorded fitness."""
    surf = bf.Paraboloid(dim=4)
    rng = np.random.default_rng(4)
    seeds, ref = _seed_far(surf, rng)
    opt = qc.ContinuousQPSO(
        surrogate=surf, bounds_min=surf.bounds_min, bounds_max=surf.bounds_max,
        num_particles=30, max_iter=60, n_elite=3, seed_layouts=seeds, layout_ref=ref,
        random_state=9,
    )
    opt.optimize(verbose=False)
    assert np.allclose(surf.predict(opt.pbest), opt.pbest_fit, atol=1e-6)


class _TwoRegionSurrogate:
    """Peak A is higher but uncertain; peak B is lower but certain.
    argmax(mean) -> A;  argmax(mean - 2*std) -> B."""
    A = np.array([2.0, 2.0])
    B = np.array([8.0, 8.0])

    def _g(self, X, c, s=4.0):
        X = np.atleast_2d(np.asarray(X, float))
        return np.exp(-((X - c) ** 2).sum(-1) / s)

    def predict(self, X):
        return 100.0 * self._g(X, self.A) + 90.0 * self._g(X, self.B)

    def predict_with_std(self, X):
        return self.predict(X), 50.0 * self._g(X, self.A)


def test_kappa_without_std_support_warns():
    """Footgun guard: kappa>0 with a surrogate lacking predict_with_std (e.g.
    ConsensusSurrogate) silently degrades to greedy search — the user must be
    warned instead of silently losing the trust-region."""
    import warnings as _w

    surf = bf.Paraboloid(dim=4)            # has predict() only, no predict_with_std
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        qc.ContinuousQPSO(
            surrogate=surf, bounds_min=surf.bounds_min, bounds_max=surf.bounds_max,
            num_particles=10, max_iter=5, kappa=2.0, random_state=0,
        )
    assert any("predict_with_std" in str(w.message) for w in caught), (
        "kappa>0 sem predict_with_std deveria emitir warning"
    )
    # and the supported path must stay silent
    with _w.catch_warnings(record=True) as caught2:
        _w.simplefilter("always")
        qc.ContinuousQPSO(
            surrogate=surf, bounds_min=surf.bounds_min, bounds_max=surf.bounds_max,
            num_particles=10, max_iter=5, kappa=0.0, random_state=0,
        )
    assert not any("predict_with_std" in str(w.message) for w in caught2)


def test_lcb_steers_away_from_high_uncertainty():
    surr = _TwoRegionSurrogate()
    bmin, bmax = np.zeros(2), np.full(2, 10.0)
    common = dict(surrogate=surr, bounds_min=bmin, bounds_max=bmax,
                  num_particles=40, max_iter=80, n_elite=3, random_state=2)
    greedy, _, _ = qc.ContinuousQPSO(kappa=0.0, **common).optimize(verbose=False)
    cautious, _, _ = qc.ContinuousQPSO(kappa=2.0, **common).optimize(verbose=False)
    # greedy chases the tall uncertain peak A; cautious prefers the safe peak B
    assert np.linalg.norm(greedy - surr.A) < np.linalg.norm(greedy - surr.B)
    assert np.linalg.norm(cautious - surr.B) < np.linalg.norm(cautious - surr.A)


if __name__ == "__main__":
    tests = [
        test_continuous_qpso_climbs_on_smooth_paraboloid,
        test_gbest_is_monotonic_nondecreasing,
        test_reference_layout_is_preserved_when_seeded,
        test_repair_method_enforces_distance,
        test_d_min_repair_separates_buoys,
        test_pbest_stays_consistent_with_its_fitness,
        test_kappa_without_std_support_warns,
        test_lcb_steers_away_from_high_uncertainty,
    ]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}")
