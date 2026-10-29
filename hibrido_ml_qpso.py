# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""
ML + QPSO Hibrido (somente fluxo dedicado)
===============================================================================
Script dedicado para rodar apenas:
1) Treino do ML Surrogate (RandomForest)
2) Calculo de contribuicoes SHAP
3) Otimizacao com QPSO Hibrido Enhanced
===============================================================================
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt

from shared_utils import carregar_e_treinar_modelo, calcular_contribuicoes_shap
from algoritmos_hibridos_enhanced import HybridQPSO_WEC


def avaliar_layout_binario(bits, modelo, layout_base, num_wecs):
    """Avalia potencia final no modelo RF usando zero nas boias removidas."""
    coords = layout_base.copy()
    for i in range(num_wecs):
        if bits[i] == 0:
            coords[i * 2] = 0
            coords[i * 2 + 1] = 0

    potencia = float(modelo.predict(coords.reshape(1, -1))[0])
    n_ativas = int(np.sum(bits))
    eficiencia = potencia / n_ativas if n_ativas > 0 else 0.0
    return potencia, eficiencia


def main():
    # Configuracao principal
    db_file = "WEC_Sydney_49.db"
    num_wecs = 49
    num_iter = 80
    num_particles = 30
    min_boias = 20
    n_estimators = 150

    print("\n" + "=" * 68)
    print(" ML SURROGATE + QPSO HIBRIDO ENHANCED (APENAS ESTE FLUXO)")
    print("=" * 68)
    print(f"Dataset: {db_file}")
    print(
        f"Config QPSO: iter={num_iter}, particles={num_particles}, min_boias={min_boias}"
    )

    # 1) Treinar ML Surrogate
    modelo, layout_base, best_power, r2, mae, _ = carregar_e_treinar_modelo(
        db_file, num_wecs, n_estimators=n_estimators, verbose=True
    )
    print(f"[ML] R2={r2:.4f} | MAE={mae:,.0f} W | Best raw={best_power:,.2f} W")

    # 2) SHAP por boia
    shap_contribs, shap_base = calcular_contribuicoes_shap(
        modelo, layout_base, num_wecs, verbose=False
    )
    print("[SHAP] Contribuicoes calculadas com sucesso.")

    # 3) Rodar QPSO Enhanced
    qpso = HybridQPSO_WEC(
        num_wecs=num_wecs,
        num_particles=num_particles,
        max_iter=num_iter,
        shap_contribs=shap_contribs,
        shap_base=shap_base,
        min_boias=min_boias,
        levy_mu=0.06,
        crossover_rate=0.35,
    )

    t0 = time.time()
    gbest, fit, hist = qpso.optimize()
    elapsed = time.time() - t0

    potencia, eficiencia = avaliar_layout_binario(gbest, modelo, layout_base, num_wecs)
    n_ativas = int(np.sum(gbest))
    economia_pct = ((num_wecs - n_ativas) / num_wecs) * 100.0
    razao_pct = (potencia / best_power) * 100.0 if best_power > 0 else 0.0

    print("\n--- RESULTADO FINAL (QPSO + ML) ---")
    print(f"Fitness (SHAP):   {fit:,.2f}")
    print(f"Potencia RF:      {potencia:,.2f} W")
    print(f"Boias ativas:     {n_ativas}/{num_wecs}")
    print(f"Retencao potencia:{razao_pct:.2f}%")
    print(f"Economia boias:   {economia_pct:.2f}%")
    print(f"Eficiencia:       {eficiencia:,.2f} W/boia")
    print(f"Tempo total:      {elapsed:.2f}s")

    # Salvar curva de convergencia
    out_plot = "qpso_ml_hibrido_convergencia.png"
    plt.figure(figsize=(9, 5))
    plt.plot(hist, color="#d62728", linewidth=2.2, label=f"QPSO-Enhanced ({n_ativas} boias)")
    plt.title("Convergencia - QPSO Hibrido + ML Surrogate", fontsize=12, fontweight="bold")
    plt.xlabel("Iteracoes")
    plt.ylabel("Fitness (SHAP)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=250)
    plt.close()
    print(f"[OK] Grafico salvo: {out_plot}")

    # Salvar resultado estruturado
    out_json = "resultado_qpso_ml_hibrido.json"
    payload = {
        "dataset": db_file,
        "num_wecs": num_wecs,
        "num_iter": num_iter,
        "num_particles": num_particles,
        "min_boias": min_boias,
        "ml": {
            "n_estimators": n_estimators,
            "r2": float(r2),
            "mae": float(mae),
            "best_power": float(best_power),
        },
        "result": {
            "fitness": float(fit),
            "potencia": float(potencia),
            "eficiencia": float(eficiencia),
            "boias_ativas": n_ativas,
            "economia_pct": float(economia_pct),
            "razao_pct": float(razao_pct),
            "tempo_s": float(elapsed),
            "history": [float(x) for x in hist],
            "gbest": [int(x) for x in gbest],
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Resultado salvo: {out_json}")


if __name__ == "__main__":
    main()

