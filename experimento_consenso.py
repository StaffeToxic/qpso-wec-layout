# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Consensus optimization: move buoys to improve a layout with a CREDIBLE gain
by construction, by optimizing min(MLP, RF) instead of MLP alone.

Fase 3 / experimento_mover showed: optimizing the MLP alone yields gains an
independent RF rejects above the data frontier (mirage). Here the objective is
the pessimistic consensus min(MLP, RF), so the optimizer only climbs where BOTH
models agree. Expected: credible gains from sub-optimal starts (p10/p50), and the
mirage automatically capped near the best layout (p100 -> ~0).

Usage:  python experimento_consenso.py [Sydney|Perth] [n_seeds]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

from qpso_continuo import ContinuousQPSO
from qpso_layout_standalone import carregar_e_treinar_modelo
from surrogate import ConsensusSurrogate, MLPEnsembleSurrogate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_WECS = 49
D_MIN = 50.0
PERCENTIS = [10, 50, 90, 100]


def _gain(model, layout, base):
    return 100 * (float(model.predict(layout.reshape(1, -1))[0]) - base) / base


def run(site: str = "Sydney", n_seeds: int = 3):
    db = os.path.join(SCRIPT_DIR, f"WEC_{site}_49.db")
    if not os.path.exists(db):
        print(f"[!] dataset ausente: {db}")
        return
    rf, _, _, rf_r2, _, _, bmin, bmax, X, y = carregar_e_treinar_modelo(db, num_wecs=NUM_WECS, n_estimators=150)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[*] Treinando MLP ensemble ({site}) ...")
    t0 = time.time()
    mlp = MLPEnsembleSurrogate(n_members=5, hidden_layer_sizes=(128, 64), max_iter=400,
                               early_stopping=True, random_state=42).fit(Xtr, ytr)
    consensus = ConsensusSurrogate([mlp, rf])
    print(f"    treino MLP: {time.time() - t0:.1f}s | RF R2={rf_r2:.4f}\n")

    order = np.argsort(y)
    print("  Objetivo otimizado = min(MLP, RF). Ganhos reportados vs layout de partida:")
    print(f"  {'partida':>8} | {'ganho consenso':>14} {'ganho MLP':>10} {'ganho RF':>10}")
    for p in PERCENTIS:
        idx = order[min(int(p / 100 * (len(order) - 1)), len(order) - 1)]
        start = X[idx].astype(float)
        base_mlp = float(mlp.predict(start.reshape(1, -1))[0])
        base_rf = float(rf.predict(start.reshape(1, -1))[0])
        base_cons = min(base_mlp, base_rf)
        gc, gm, gr = [], [], []
        for s in range(n_seeds):
            opt = ContinuousQPSO(
                surrogate=consensus, bounds_min=bmin, bounds_max=bmax,
                num_particles=40, max_iter=120, kappa=0.0, d_min=D_MIN, n_elite=3,
                use_levy=True, seed_layouts=start.reshape(1, -1), layout_ref=start,
                seed_with_ref=True, random_state=s,
            )
            gbest, _, _ = opt.optimize(verbose=False)
            gc.append(_gain(consensus, gbest, base_cons))
            gm.append(_gain(mlp, gbest, base_mlp))
            gr.append(_gain(rf, gbest, base_rf))
        print(f"  p{p:>3}{'':>3} | {np.mean(gc):>+13.2f}% {np.mean(gm):>+9.2f}% {np.mean(gr):>+9.2f}%")

    print("\nLeitura: o ganho de CONSENSO e o que os dois modelos garantem simultaneamente.")
    print("Esperado: positivo e credivel em p10/p50; ~0 em p100 (miragem cortada por construcao).")


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "Sydney"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    run(site, n_seeds)
