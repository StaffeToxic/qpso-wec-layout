# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Fase 2 end-to-end: run the refactored ContinuousQPSO on REAL WEC data using
the smooth MLP surrogate, with the 50 m soft distance constraint and the LCB
trust-region. Single-seed run (multi-seed ablations come in Fase 3).

Reports, all under the MLP surrogate:
  - predicted power of the optimized layout vs the best dataset layout;
  - feasibility (minimum inter-buoy distance >= 50 m);
  - mean buoy displacement.

Usage:  python qpso_continuo_run.py [Sydney|Perth] [kappa]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

from qpso_continuo import ContinuousQPSO
from qpso_layout_standalone import carregar_e_treinar_modelo
from surrogate import MLPEnsembleSurrogate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
D_MIN = 50.0
NUM_WECS = 49


def _min_dist(layout):
    pts = layout.reshape(NUM_WECS, 2)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def run(site: str = "Sydney", kappa: float = 0.5):
    db = os.path.join(SCRIPT_DIR, f"WEC_{site}_49.db")
    if not os.path.exists(db):
        print(f"[!] dataset ausente: {db}")
        return

    _, best_layout, _, _, _, _, bmin, bmax, X, y = carregar_e_treinar_modelo(db, num_wecs=NUM_WECS, n_estimators=150)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"[*] Treinando MLP ensemble ({site}) ...")
    t0 = time.time()
    surrogate = MLPEnsembleSurrogate(n_members=5, hidden_layer_sizes=(128, 64), max_iter=400,
                                     early_stopping=True, random_state=42).fit(Xtr, ytr)
    print(f"    treino MLP: {time.time() - t0:.1f}s")

    base_power = float(surrogate.predict(best_layout.reshape(1, -1))[0])

    # sementes: faixa percentil 50-90 do dataset (layouts variados, bons mas nao o melhor)
    order = np.argsort(y)
    band = X[order[int(0.5 * len(order)):int(0.9 * len(order))]]

    print(f"[*] Otimizando (ContinuousQPSO, d_min={D_MIN:.0f} m, kappa={kappa}) ...")
    t0 = time.time()
    opt = ContinuousQPSO(
        surrogate=surrogate, bounds_min=bmin, bounds_max=bmax,
        num_particles=50, max_iter=150, kappa=kappa, d_min=D_MIN, n_elite=3,
        use_levy=True, seed_layouts=band, layout_ref=best_layout, seed_with_ref=True,
        random_state=42,
    )
    gbest, gbest_acq, history = opt.optimize(verbose=True)
    elapsed = time.time() - t0

    opt_power = opt.best_power_pred
    gain = opt_power - base_power
    desloc = float(np.mean([
        np.hypot(gbest[i * 2] - best_layout[i * 2], gbest[i * 2 + 1] - best_layout[i * 2 + 1])
        for i in range(NUM_WECS)
    ]))

    print("\n==== RESULTADO (sob surrogate MLP) ====")
    print(f"  potencia MLP do melhor layout : {base_power:,.0f} W")
    print(f"  potencia MLP do layout otimo  : {opt_power:,.0f} W")
    print(f"  ganho                         : {gain:+,.0f} W ({100 * gain / base_power:+.2f}%)")
    print(f"  acquisition (LCB) final       : {gbest_acq:,.0f}")
    print(f"  viabilidade min dist          : {_min_dist(gbest):.1f} m (alvo >= {D_MIN:.0f})")
    print(f"  deslocamento medio            : {desloc:.1f} m | tempo opt: {elapsed:.1f}s")
    return {"site": site, "base": base_power, "opt": opt_power, "gain": gain, "min_dist": _min_dist(gbest)}


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "Sydney"
    kappa = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    run(site, kappa)
