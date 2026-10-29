# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Fase 1 acceptance: validate the smooth surrogate on the REAL WEC datasets.

Checks the success criteria for the surrogate swap:
  (A) MLP held-out R^2 >= Random Forest R^2 (same train/test split);
  (B) the MLP response is SMOOTH where RF is a staircase (interpolation plot);
  (C) teaser for #3: how much predicted gain is locally reachable around the
      best layout under each surrogate (RF ~flat leaf vs MLP smooth slope).

Usage:  python surrogate_validation.py [Sydney|Perth]
Outputs: surrogate_validation_<site>.png  +  printed metrics.
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from qpso_layout_standalone import carregar_e_treinar_modelo
from surrogate import MLPEnsembleSurrogate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(site: str = "Sydney"):
    db = os.path.join(SCRIPT_DIR, f"WEC_{site}_49.db")
    if not os.path.exists(db):
        print(f"[!] dataset ausente: {db}")
        return

    # RF + data (carregar treina o RF e devolve X, y completos + R2/MAE do RF)
    rf, best_layout, best_power, rf_r2, rf_mae, _, bmin, bmax, X, y = carregar_e_treinar_modelo(
        db, num_wecs=49, n_estimators=150
    )

    # mesmo split (random_state=42) para comparacao justa MLP vs RF
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n[*] Treinando MLP ensemble ({site}) ...")
    t0 = time.time()
    mlp = MLPEnsembleSurrogate(
        n_members=5, hidden_layer_sizes=(128, 64), max_iter=400,
        early_stopping=True, random_state=42,
    ).fit(Xtr, ytr)
    print(f"    treino MLP: {time.time() - t0:.1f}s")

    mlp_pred = mlp.predict(Xte)
    mlp_r2 = r2_score(yte, mlp_pred)
    mlp_mae = mean_absolute_error(yte, mlp_pred)

    print("\n==== (A) R2 held-out ====")
    print(f"  RF : R2={rf_r2:.4f} | MAE={rf_mae:,.0f} W")
    print(f"  MLP: R2={mlp_r2:.4f} | MAE={mlp_mae:,.0f} W")
    print(f"  criterio (MLP >= RF): {'OK' if mlp_r2 >= rf_r2 else 'FALHOU'}")

    # (C) ganho local alcancavel ao redor do melhor layout
    rng = np.random.default_rng(0)
    span = np.maximum(bmax - bmin, 1e-9)
    perturb = best_layout + rng.normal(0, 0.03, size=(2000, len(best_layout))) * span
    perturb = np.clip(perturb, bmin, bmax)
    rf_base = float(rf.predict(best_layout.reshape(1, -1))[0])
    mlp_base = float(mlp.predict(best_layout.reshape(1, -1))[0])
    rf_gain = float(rf.predict(perturb).max() - rf_base)
    mlp_gain = float(mlp.predict(perturb).max() - mlp_base)
    print("\n==== (C) ganho local reachable (amostragem ao redor do melhor) ====")
    print(f"  RF : max gain ~ {rf_gain:+,.0f} W")
    print(f"  MLP: max gain ~ {mlp_gain:+,.0f} W   (smooth => ha direcao de melhora)")

    # (B) smoothness: interpolar entre dois layouts de teste
    a, b = Xte[0], Xte[1]
    ts = np.linspace(0, 1, 300)
    path = a[None, :] + ts[:, None] * (b - a)[None, :]
    rf_curve = rf.predict(path)
    mlp_curve, mlp_std = mlp.predict_with_std(path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Surrogate suave vs Random Forest — {site} 49", fontweight="bold")
    ax = axes[0]
    ax.plot(ts, rf_curve, color="#d62728", lw=2, label="Random Forest (degraus)")
    ax.plot(ts, mlp_curve, color="#1f77b4", lw=2, label="MLP ensemble (suave)")
    ax.fill_between(ts, mlp_curve - mlp_std, mlp_curve + mlp_std, color="#1f77b4", alpha=0.15,
                    label="±1σ (incerteza)")
    ax.set_title("(B) Resposta ao interpolar entre 2 layouts")
    ax.set_xlabel("t (interpolação a→b)"); ax.set_ylabel("Potência predita (W)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(yte, mlp_pred, s=6, alpha=0.3, color="#1f77b4", label=f"MLP R²={mlp_r2:.3f}")
    lim = [min(yte.min(), mlp_pred.min()), max(yte.max(), mlp_pred.max())]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
    ax.set_title("(A) Predito vs real (held-out)")
    ax.set_xlabel("Potência real (W)"); ax.set_ylabel("Potência predita (W)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, f"surrogate_validation_{site.lower()}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[!] figura salva: {out}")
    return {"site": site, "rf_r2": rf_r2, "mlp_r2": mlp_r2, "rf_gain": rf_gain, "mlp_gain": mlp_gain}


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "Sydney"
    run(site)
