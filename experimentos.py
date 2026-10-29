# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Fase 3 — honest experiments for the continuous QPSO.

Central question raised by Fase 2: the +6.25% gain came with 417 m mean
displacement (likely OOD). Here we test whether a *trustworthy* gain survives.

Design (Sydney 49, surrogate trained once, optimizer re-seeded N times):
  - Baseline ablation: optimize the **RF** surrogate (expected ~0 gain — finding
    #3: RF's max over the box is the best training layout, so it returns the seed).
  - kappa sweep on the **MLP** surrogate: kappa in {0, 0.5, 1, 2, 5}.
  For every run we record, under BOTH surrogates:
    * mlp_gain%  : gain vs best layout, scored by MLP (the optimizer's own view)
    * rf_gain%   : SAME layout scored by RF (independent cross-check)
    * desloc     : mean buoy displacement (m)
    * nn_norm    : normalized distance to the NEAREST training layout (OOD meter)
    * min_dist   : feasibility (>= 50 m)

Reading: a credible gain needs rf_gain% > 0 too, at moderate displacement / small
nn_norm. If only mlp_gain% is positive, the "gain" is surrogate extrapolation.

Usage:  python experimentos.py [Sydney|Perth] [n_seeds]
Outputs: experimentos_<site>.json + experimentos_<site>.png + printed table.
"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from qpso_continuo import ContinuousQPSO
from qpso_layout_standalone import carregar_e_treinar_modelo
from surrogate import MLPEnsembleSurrogate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_WECS = 49
D_MIN = 50.0
KAPPAS = [0.0, 0.5, 1.0, 2.0, 5.0]


def _min_dist(layout):
    pts = layout.reshape(NUM_WECS, 2)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def _mean_disp(layout, ref):
    p, r = layout.reshape(NUM_WECS, 2), ref.reshape(NUM_WECS, 2)
    return float(np.mean(np.sqrt(((p - r) ** 2).sum(1))))


def _nn_norm(layout, X, span):
    """Normalized distance to the nearest training layout (0 = on a data point)."""
    d = np.sqrt((((X - layout) / span) ** 2).mean(axis=1))
    return float(d.min())


def _run_once(surrogate, rf, bmin, bmax, band, best_layout, span, X, kappa, seed):
    opt = ContinuousQPSO(
        surrogate=surrogate, bounds_min=bmin, bounds_max=bmax,
        num_particles=40, max_iter=120, kappa=kappa, d_min=D_MIN, n_elite=3,
        use_levy=True, seed_layouts=band, layout_ref=best_layout, seed_with_ref=True,
        random_state=seed,
    )
    gbest, _, _ = opt.optimize(verbose=False)
    mlp_base = float(_predict(surrogate, best_layout))
    return {
        "mlp_gain": 100 * (float(_predict(surrogate, gbest)) - mlp_base) / mlp_base,
        "rf_gain": 100 * (float(rf.predict(gbest.reshape(1, -1))[0]) - float(rf.predict(best_layout.reshape(1, -1))[0]))
        / float(rf.predict(best_layout.reshape(1, -1))[0]),
        "desloc": _mean_disp(gbest, best_layout),
        "nn_norm": _nn_norm(gbest, X, span),
        "min_dist": _min_dist(gbest),
    }


def _predict(model, layout):
    return model.predict(layout.reshape(1, -1))[0]


def _agg(runs):
    keys = runs[0].keys()
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in keys}


def run(site: str = "Sydney", n_seeds: int = 5):
    db = os.path.join(SCRIPT_DIR, f"WEC_{site}_49.db")
    if not os.path.exists(db):
        print(f"[!] dataset ausente: {db}")
        return
    rf, best_layout, _, rf_r2, _, _, bmin, bmax, X, y = carregar_e_treinar_modelo(db, num_wecs=NUM_WECS, n_estimators=150)
    span = np.maximum(bmax - bmin, 1e-9)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[*] Treinando MLP ensemble ({site}) ...")
    t0 = time.time()
    mlp = MLPEnsembleSurrogate(n_members=5, hidden_layer_sizes=(128, 64), max_iter=400,
                               early_stopping=True, random_state=42).fit(Xtr, ytr)
    print(f"    treino MLP: {time.time() - t0:.1f}s | RF R2={rf_r2:.4f}")

    order = np.argsort(y)
    band = X[order[int(0.5 * len(order)):int(0.9 * len(order))]]

    configs = [("RF", rf, 0.0)] + [("MLP", mlp, k) for k in KAPPAS]
    results = []
    for label, surr, kappa in configs:
        runs = [_run_once(surr, rf, bmin, bmax, band, best_layout, span, X, kappa, s) for s in range(n_seeds)]
        a = _agg(runs)
        a.update({"surrogate": label, "kappa": kappa})
        results.append(a)
        print(f"  {label:>3} k={kappa:<3} | mlp_gain={a['mlp_gain'][0]:+6.2f}% "
              f"rf_gain={a['rf_gain'][0]:+6.2f}% | desloc={a['desloc'][0]:6.1f}m "
              f"nn_norm={a['nn_norm'][0]:.3f} min_dist={a['min_dist'][0]:.1f}m")

    out_json = os.path.join(SCRIPT_DIR, f"experimentos_{site.lower()}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # plot: MLP gain vs RF gain vs displacement, across kappa
    mlp_rows = [r for r in results if r["surrogate"] == "MLP"]
    ks = [r["kappa"] for r in mlp_rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Fase 3 — ganho honesto vs trust-region (κ) — {site} 49", fontweight="bold")
    ax[0].errorbar(ks, [r["mlp_gain"][0] for r in mlp_rows], yerr=[r["mlp_gain"][1] for r in mlp_rows],
                   marker="o", label="ganho sob MLP (otimizador)", color="#1f77b4")
    ax[0].errorbar(ks, [r["rf_gain"][0] for r in mlp_rows], yerr=[r["rf_gain"][1] for r in mlp_rows],
                   marker="s", label="ganho sob RF (cross-check)", color="#d62728")
    ax[0].axhline(0, color="k", lw=1, alpha=0.5)
    ax[0].set_xlabel("κ (peso do LCB)"); ax[0].set_ylabel("ganho vs melhor layout (%)")
    ax[0].set_title("Ganho real só se MLP e RF concordarem"); ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)
    ax[1].errorbar(ks, [r["desloc"][0] for r in mlp_rows], yerr=[r["desloc"][1] for r in mlp_rows],
                   marker="o", color="#2ca02c")
    ax[1].set_xlabel("κ (peso do LCB)"); ax[1].set_ylabel("deslocamento médio (m)")
    ax[1].set_title("κ controla o quanto a busca se afasta dos dados?"); ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(SCRIPT_DIR, f"experimentos_{site.lower()}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[!] salvos: {out_json} + {out_png}")
    return results


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "Sydney"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run(site, n_seeds)
