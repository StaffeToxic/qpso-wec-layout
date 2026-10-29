# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Mover boias (contínuo) para MELHORAR um layout dado — teste honesto.

A Fase 3 mostrou que *superar o melhor layout do dataset* movendo boias é inviável
(o máximo do RF é o próprio melhor layout). Aqui testamos o caso que importa de
fato: pegar um layout de qualidade X (percentil de potência) e movê-lo. A pergunta:
o ganho é confirmado pelos DOIS surrogates (RF e MLP)? Se sim, é credível.

Para cada percentil de partida, otimiza-se via MLP (com d_min e LCB) e mede-se o
ganho vs o layout de partida sob MLP e sob RF (cross-check), além de quão dentro
da distribuição o resultado ficou (nn_norm) e o deslocamento médio.

Usage:  python experimento_mover.py [Sydney|Perth] [n_seeds]
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
NUM_WECS = 49
D_MIN = 50.0
PERCENTIS = [10, 50, 90, 100]


def _nn_norm(layout, X, span):
    return float(np.sqrt((((X - layout) / span) ** 2).mean(axis=1)).min())


def _mean_disp(layout, ref):
    p, r = layout.reshape(NUM_WECS, 2), ref.reshape(NUM_WECS, 2)
    return float(np.mean(np.sqrt(((p - r) ** 2).sum(1))))


def run(site: str = "Sydney", n_seeds: int = 3):
    db = os.path.join(SCRIPT_DIR, f"WEC_{site}_49.db")
    if not os.path.exists(db):
        print(f"[!] dataset ausente: {db}")
        return
    rf, _, _, rf_r2, _, _, bmin, bmax, X, y = carregar_e_treinar_modelo(db, num_wecs=NUM_WECS, n_estimators=150)
    span = np.maximum(bmax - bmin, 1e-9)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[*] Treinando MLP ensemble ({site}) ...")
    t0 = time.time()
    mlp = MLPEnsembleSurrogate(n_members=5, hidden_layer_sizes=(128, 64), max_iter=400,
                               early_stopping=True, random_state=42).fit(Xtr, ytr)
    print(f"    treino MLP: {time.time() - t0:.1f}s | RF R2={rf_r2:.4f}\n")

    order = np.argsort(y)
    print(f"  {'partida':>8} {'pot.real':>12} | {'ganho MLP':>10} {'ganho RF':>10} | "
          f"{'desloc':>8} {'nn_norm':>8}")
    for p in PERCENTIS:
        idx = order[min(int(p / 100 * (len(order) - 1)), len(order) - 1)]
        start = X[idx].astype(float)
        base_mlp = float(mlp.predict(start.reshape(1, -1))[0])
        base_rf = float(rf.predict(start.reshape(1, -1))[0])
        mlp_g, rf_g, disp, nn = [], [], [], []
        for s in range(n_seeds):
            opt = ContinuousQPSO(
                surrogate=mlp, bounds_min=bmin, bounds_max=bmax,
                num_particles=40, max_iter=120, kappa=1.0, d_min=D_MIN, n_elite=3,
                use_levy=True, seed_layouts=start.reshape(1, -1), layout_ref=start,
                seed_with_ref=True, random_state=s,
            )
            gbest, _, _ = opt.optimize(verbose=False)
            mlp_g.append(100 * (float(mlp.predict(gbest.reshape(1, -1))[0]) - base_mlp) / base_mlp)
            rf_g.append(100 * (float(rf.predict(gbest.reshape(1, -1))[0]) - base_rf) / base_rf)
            disp.append(_mean_disp(gbest, start))
            nn.append(_nn_norm(gbest, X, span))
        print(f"  p{p:>3}{'':>3} {y[idx]:>12,.0f} | {np.mean(mlp_g):>+9.2f}% {np.mean(rf_g):>+9.2f}% | "
              f"{np.mean(disp):>7.0f}m {np.mean(nn):>8.3f}")

    print("\nLeitura: ganho credível = MLP e RF AMBOS positivos. Espera-se ganho real ao")
    print("partir de layouts ruins (p10/p50) e teto ~0 (ou RF<0) ao partir do melhor (p100).")


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "Sydney"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    run(site, n_seeds)
