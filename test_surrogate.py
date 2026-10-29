# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Tests for surrogate.py — the smooth ML surrogate that replaces the
piecewise-constant Random Forest for continuous layout optimization.

Key properties under test (public behavior only):
  1. fits a smooth function accurately (held-out R^2 high);
  2. exposes calibrated uncertainty (ensemble disagreement grows out-of-dist);
  3. lcb(X, kappa) = mean - kappa*std  (the trust-region objective).

Run directly (no pytest needed):  python test_surrogate.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sg = load_module(ROOT / "surrogate.py", "surrogate")


def _smooth_data(n, dim, rng, low=0.0, high=10.0):
    X = rng.uniform(low, high, size=(n, dim))
    c = np.full(dim, (low + high) / 2.0)
    y = -np.sum((X - c) ** 2, axis=1) + 5.0 * np.sin(X[:, 0])  # smooth, nonlinear
    return X, y


def test_ensemble_fits_smooth_function():
    rng = np.random.default_rng(0)
    Xtr, ytr = _smooth_data(500, 6, rng)
    Xte, yte = _smooth_data(150, 6, rng)
    model = sg.MLPEnsembleSurrogate(n_members=4, random_state=0)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    assert pred.shape == (150,)
    r2 = r2_score(yte, pred)
    assert r2 >= 0.90, f"smooth surrogate underfit: R2={r2:.3f}"


def test_uncertainty_grows_out_of_distribution():
    """Ensemble members agree where they were trained and diverge where they
    were not — this is the OOD signal that keeps the optimizer in the trustworthy
    region (via LCB). Train on a narrow band, then probe inside vs far outside."""
    rng = np.random.default_rng(1)
    dim = 6
    Xtr = rng.uniform(4.5, 5.5, size=(600, dim))          # narrow training band
    c = np.full(dim, 5.0)
    ytr = -np.sum((Xtr - c) ** 2, axis=1)
    model = sg.MLPEnsembleSurrogate(n_members=5, random_state=0).fit(Xtr, ytr)

    X_in = rng.uniform(4.5, 5.5, size=(100, dim))         # in-distribution
    X_out = rng.uniform(0.0, 1.0, size=(100, dim))        # far outside the band
    _, std_in = model.predict_with_std(X_in)
    _, std_out = model.predict_with_std(X_out)
    assert std_out.mean() > std_in.mean(), (
        f"uncertainty not calibrated: std_in={std_in.mean():.3f} std_out={std_out.mean():.3f}"
    )


def test_lcb_is_mean_minus_kappa_std():
    rng = np.random.default_rng(2)
    Xtr, ytr = _smooth_data(400, 5, rng)
    model = sg.MLPEnsembleSurrogate(n_members=4, random_state=0).fit(Xtr, ytr)
    Xq = _smooth_data(50, 5, rng)[0]
    mean, std = model.predict_with_std(Xq)
    lcb = model.lcb(Xq, kappa=2.0)
    assert np.allclose(lcb, mean - 2.0 * std)
    assert np.all(lcb <= mean + 1e-12)   # penalizes uncertainty, never rewards it


def test_consensus_is_pessimistic_elementwise_min():
    """A layout is credible only if EVERY surrogate agrees it is good. The
    consensus predict is the elementwise min across models — optimizing it yields
    gains all models endorse (credible by construction), capping the single-model
    extrapolation 'mirage'."""
    class _Const:
        def __init__(self, v):
            self.v = float(v)

        def predict(self, X):
            return np.full(np.atleast_2d(X).shape[0], self.v)

    cons = sg.ConsensusSurrogate([_Const(10.0), _Const(3.0), _Const(7.0)])
    out = cons.predict(np.zeros((5, 4)))
    assert out.shape == (5,)
    assert np.allclose(out, 3.0)   # only as high as the most pessimistic model


if __name__ == "__main__":
    tests = [
        test_ensemble_fits_smooth_function,
        test_uncertainty_grows_out_of_distribution,
        test_lcb_is_mean_minus_kappa_std,
        test_consensus_is_pessimistic_elementwise_min,
    ]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}")
