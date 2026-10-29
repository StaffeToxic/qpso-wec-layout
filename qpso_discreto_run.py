# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Discrete QPSO on REAL data via SHAP-additive fitness (buoy selection).

Pipeline: RF surrogate -> SHAP contributions per buoy (TreeExplainer) -> additive
fitness F(b) = wp*(phi0 + sum b_i c_i) + we*(P/n) -> DiscreteQPSO. Because the
objective is additive, removing a buoy subtracts its SHAP contribution exactly
(no OOD extrapolation), so a gain here is well-posed — the opposite of the
continuous case.

Reports the SHAP sanity check (phi0 + sum c_i ~= RF(ref)), which buoys are dropped,
and the power/efficiency change. Usage: python qpso_discreto_run.py [Sydney|Perth]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from qpso_discreto import DiscreteQPSO
from qpso_layout_standalone import carregar_e_treinar_modelo
from qpso_ml_standalone import calcular_contribuicoes_shap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_WECS = 49


def run(site: str = "Sydney", min_boias: int = 20, wp: float = 0.7, we: float = 0.3):
    db = os.path.join(SCRIPT_DIR, f"WEC_{site}_49.db")
    if not os.path.exists(db):
        print(f"[!] dataset ausente: {db}")
        return
    rf, best_layout, _, r2, _, _, _, _, _, _ = carregar_e_treinar_modelo(db, num_wecs=NUM_WECS, n_estimators=150)
    contribs, base = calcular_contribuicoes_shap(rf, best_layout, NUM_WECS, verbose=False)
    contribs = np.asarray(contribs, dtype=float)

    rf_ref = float(rf.predict(best_layout.reshape(1, -1))[0])
    shap_err = abs(base + contribs.sum() - rf_ref)
    print(f"\n[*] {site}: RF R2={r2:.4f} | RF(ref)={rf_ref:,.0f} W")
    print(f"    SHAP sanity: |phi0 + sum c_i - RF(ref)| = {shap_err:,.0f} W "
          f"({100 * shap_err / rf_ref:.3f}%)")

    def fitness(bits):
        bits = np.atleast_2d(np.asarray(bits, float))
        n = bits.sum(axis=1)
        power = base + bits @ contribs
        eff = power / np.maximum(n, 1.0)
        return np.where(n < min_boias, -1e12, wp * power + we * eff)

    print(f"[*] DiscreteQPSO (min_boias={min_boias}, wp={wp}, we={we}) ...")
    t0 = time.time()
    opt = DiscreteQPSO(fitness, num_wecs=NUM_WECS, num_particles=30, max_iter=120,
                       min_boias=min_boias, random_state=42)
    gbest, gfit, _ = opt.optimize(verbose=True)
    elapsed = time.time() - t0

    n_sel = int(gbest.sum())
    power_all = base + contribs.sum()
    power_sel = base + contribs[gbest == 1].sum()
    eff_all, eff_sel = power_all / NUM_WECS, power_sel / n_sel
    removed = np.where(gbest == 0)[0]

    print("\n==== RESULTADO (sob decomposicao SHAP) ====")
    print(f"  boias ativas: {n_sel}/{NUM_WECS}  (removidas: {len(removed)})")
    print(f"  potencia  : todas={power_all:,.0f} W  ->  selecao={power_sel:,.0f} W "
          f"({100 * (power_sel - power_all) / power_all:+.2f}%)")
    print(f"  eficiencia: todas={eff_all:,.0f} W/boia -> selecao={eff_sel:,.0f} W/boia "
          f"({100 * (eff_sel - eff_all) / eff_all:+.2f}%)")
    if len(removed):
        contrib_removed = contribs[removed].sum()
        print(f"  contribuicao SHAP total removida: {contrib_removed:+,.0f} W "
              f"(remover boias de contribuicao baixa/negativa)")
        neg = removed[contribs[removed] < 0]
        print(f"  dentre as removidas, {len(neg)} tinham contribuicao NEGATIVA")
    print(f"  tempo: {elapsed:.1f}s")
    return {"site": site, "n_sel": n_sel, "power_all": power_all, "power_sel": power_sel,
            "eff_gain_pct": 100 * (eff_sel - eff_all) / eff_all, "shap_err_pct": 100 * shap_err / rf_ref}


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "Sydney"
    run(site)
