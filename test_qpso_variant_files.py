# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PURE_FILE = ROOT / "qpso_layout_puro_standalone.py"
COMPARE_FILE = ROOT / "comparar_pso_artigo_qpso.py"


class DummyPowerModel:
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        target = np.array([5.0, 5.0, 10.0, 10.0])
        return 1000.0 - np.sum((X - target) ** 2, axis=1)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pure_qpso_file_exists_and_has_no_hybrid_operators():
    assert PURE_FILE.exists(), f"Missing {PURE_FILE.name}"
    text = PURE_FILE.read_text(encoding="utf-8").lower()
    forbidden = [
        "levy_mu",
        "n_levy",
        "levy_step",
        "elite_slots",
        "top_k",
        "crossover_rate",
        "child = np.where",
    ]
    for token in forbidden:
        assert token not in text, f"{PURE_FILE.name} still contains hybrid operator token: {token}"


def test_pure_qpso_smoke_run_keeps_history_aligned():
    module = load_module(PURE_FILE, "qpso_layout_puro_standalone")
    cls = module.PureContinuousQPSO_WEC
    dataset_X = np.array(
        [
            [1.0, 1.0, 8.0, 8.0],
            [4.0, 4.0, 9.0, 9.0],
            [5.0, 5.0, 10.0, 10.0],
            [6.0, 6.0, 11.0, 11.0],
        ]
    )
    dataset_y = DummyPowerModel().predict(dataset_X)
    optimizer = cls(
        num_wecs=2,
        num_particles=4,
        max_iter=3,
        modelo_rf=DummyPowerModel(),
        layout_ref=dataset_X[0],
        bounds_min=np.zeros(4),
        bounds_max=np.full(4, 12.0),
        dataset_X=dataset_X,
        dataset_y=dataset_y,
        random_state=7,
    )
    best_layout, best_fit, history = optimizer.optimize(verbose=False)
    assert best_layout.shape == (4,)
    assert len(history) == 3
    assert np.isfinite(best_fit)
    assert history[-1] == best_fit


def test_comparison_file_exposes_article_bpso_and_runner():
    assert COMPARE_FILE.exists(), f"Missing {COMPARE_FILE.name}"
    module = load_module(COMPARE_FILE, "comparar_pso_artigo_qpso")
    assert hasattr(module, "ArticleBPSOGrid_WEC")
    assert hasattr(module, "run_comparison")


def test_article_bpso_smoke_run_uses_exact_wec_count():
    module = load_module(COMPARE_FILE, "comparar_pso_artigo_qpso")
    cls = module.ArticleBPSOGrid_WEC
    layout_ref = np.array([5.0, 5.0, 10.0, 10.0])
    optimizer = cls(
        num_wecs=2,
        num_particles=4,
        max_iter=2,
        modelo_rf=DummyPowerModel(),
        layout_ref=layout_ref,
        bounds_min=np.zeros(4),
        bounds_max=np.full(4, 12.0),
        grid_step=5.0,
        random_state=11,
    )
    best_bits, best_layout, best_fit, history = optimizer.optimize(verbose=False)
    assert int(np.sum(best_bits)) == 2
    assert best_layout.shape == (4,)
    assert len(history) == 2
    assert np.isfinite(best_fit)
    assert history[-1] == best_fit


if __name__ == "__main__":
    tests = [
        test_pure_qpso_file_exists_and_has_no_hybrid_operators,
        test_pure_qpso_smoke_run_keeps_history_aligned,
        test_comparison_file_exposes_article_bpso_and_runner,
        test_article_bpso_smoke_run_uses_exact_wec_count,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
