# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""
=============================================================================
EXECUTOR DE ALGORITMOS HIBRIDOS
Executa todos os algoritmos em todos os datasets e salva resultados em JSON.
Os arquivos de imagem e relatorio sao gerados POR ESTE SCRIPT
apenas com dados da execucao real.
=============================================================================
"""
import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---- Imports dos algoritmos ----
from shared_utils import avaliar_fitness, carregar_e_treinar_modelo, calcular_contribuicoes_shap
from algoritmos_hibridos_wec import HybridSimQAOA_WEC, HybridQEA_WEC, HybridQPSO_WEC
from algoritmos_hibridos_enhanced import (
    HybridSimQAOA_WEC as HybridSimQAOA_WEC_Enhanced,
    HybridQEA_WEC as HybridQEA_WEC_Enhanced,
    HybridQPSO_WEC as HybridQPSO_WEC_Enhanced,
)

# =============================================================================
# CONFIG
# =============================================================================
DB_PATHS     = ["WEC_Sydney_49.db", "WEC_Perth_49.db"]
NUM_WECS     = 49
NUM_ITER     = 60
NUM_POP      = 25
MIN_BOIAS    = 20
JSON_OUT     = "resultados_hibridos.json"
IMG_DIR      = "resultado_imgs"
REPORT_OUT   = "relatorio_resultados.md"

ALGO_COLORS = {
    "QAOA-Hibrido":   "#1f77b4", "QEA-Hibrido":    "#2ca02c",
    "QPSO-Hibrido":   "#d62728", "QAOA-Enhanced":  "#9467bd",
    "QEA-Enhanced":   "#8c564b", "QPSO-Enhanced":  "#e377c2",
}
ALGO_STYLES = {
    "QAOA-Hibrido":   "-", "QEA-Hibrido":    "--",
    "QPSO-Hibrido":   "-.", "QAOA-Enhanced":  ":",
    "QEA-Enhanced":   "-", "QPSO-Enhanced":  "--",
}

def _algos_factory(modelo, layout_base, shap_contribs, shap_base):
    return {
        "QAOA-Hibrido":   HybridSimQAOA_WEC(NUM_WECS, NUM_ITER, NUM_POP, modelo, layout_base, MIN_BOIAS),
        "QEA-Hibrido":    HybridQEA_WEC(NUM_WECS, NUM_POP, NUM_ITER, modelo, layout_base, MIN_BOIAS),
        "QPSO-Hibrido":   HybridQPSO_WEC(NUM_WECS, NUM_POP, NUM_ITER, modelo, layout_base, MIN_BOIAS),
        "QAOA-Enhanced":  HybridSimQAOA_WEC_Enhanced(NUM_WECS, NUM_ITER, NUM_POP, shap_contribs, shap_base, MIN_BOIAS,
                                                   elite_frac=0.1, n_local_search=10),
        "QEA-Enhanced":   HybridQEA_WEC_Enhanced(NUM_WECS, NUM_POP, NUM_ITER, shap_contribs, shap_base, MIN_BOIAS,
                                                rotation_intensity="medium", quantum_perturbation=0.02),
        "QPSO-Enhanced":  HybridQPSO_WEC_Enhanced(NUM_WECS, NUM_POP, NUM_ITER, shap_contribs, shap_base, MIN_BOIAS,
                                                 levy_mu=0.06, crossover_rate=0.35),
    }

def _avaliar_final(gbest, modelo, layout_base):
    coords = layout_base.copy()
    for i in range(NUM_WECS):
        if gbest[i] == 0:
            coords[i*2] = 0; coords[i*2+1] = 0
    potencia = float(modelo.predict(coords.reshape(1, -1))[0])
    n = int(np.sum(gbest))
    eficiência = potencia / n if n > 0 else 0
    return potencia, eficiência

def run_and_save():
    """Executa tudo, salva JSON, gera imagens e relatorio."""
    all_data = {}

    for db_path in DB_PATHS:
        db_name = os.path.basename(db_path)
        if not os.path.exists(db_path):
            print(f"  [SKIP] {db_path} nao encontrado.")
            continue

        print(f"\n{'='*60}\n  {db_name}")
        print(f"{'='*60}")

        modelo, layout_base, raw_power, r2, mae, _ = carregar_e_treinar_modelo(
            db_path, NUM_WECS, n_estimators=150
        )
        shap_contribs, shap_base = calcular_contribuicoes_shap(modelo, layout_base, NUM_WECS, verbose=False)
        eff_orig = raw_power / NUM_WECS

        db_res = {
            "n_wecs": NUM_WECS,
            "n_iter": NUM_ITER,
            "n_pop": NUM_POP,
            "min_boias": MIN_BOIAS,
            "r2": r2,
            "mae": mae,
            "raw_power": raw_power,
            "eff_original": eff_orig,
            "algos": {}
        }

        for nome, opt in _algos_factory(modelo, layout_base, shap_contribs, shap_base).items():
            print(f"  [{nome:>18s}] ...", end="", flush=True)
            t0 = time.time()
            gbest, fit, hist = opt.optimize()
            elapsed = time.time() - t0
            potencia, eficiencia = _avaliar_final(gbest, modelo, layout_base)
            n_ativas = int(np.sum(gbest))
            print(f" fit={fit:,.0f} | {n_ativas} boias | {elapsed:.0f}s")

            db_res["algos"][nome] = {
                "fitness": fit,
                "potencia": potencia,
                "eficiencia": eficiencia,
                "boias_ativas": n_ativas,
                "economia_pct": round(((NUM_WECS - n_ativas) / NUM_WECS) * 100, 2),
                "razao_pct": round((potencia / raw_power) * 100, 2) if raw_power > 0 else 0,
                "ganho_ef_pct": round(((eficiencia / eff_orig) - 1) * 100, 2) if eff_orig > 0 else 0,
                "tempo_s": round(elapsed, 2),
                "history": [float(x) for x in hist],
                "gbest": [int(x) for x in gbest],
                "gbest_count_1": n_ativas,
                "gbest_count_0": NUM_WECS - n_ativas,
            }

        all_data[db_name] = db_res

    # Salvar JSON -- unica fonte de verdade
    with open(JSON_OUT, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nDados reais salvos em {JSON_OUT}")
    return all_data


# =============================================================================
# IMAGE + REPORT (SO COM JSON)
# =============================================================================

def _expand_hist(hist, target):
    """Interpola history para numero fixo de iteracoes."""
    if len(hist) >= target:
        return hist[:target]
    x_old = np.linspace(0, target - 1, len(hist))
    return list(np.interp(np.arange(target), x_old, hist))

def generate_images(data):
    os.makedirs(IMG_DIR, exist_ok=True)
    labels = list(ALGO_COLORS.keys())

    for dn, dd in data.items():
        ds = dn.replace(".", "_").lower()
        algos = dd["algos"]

        # 1. Convergencia
        fig, ax = plt.subplots(figsize=(11, 6))
        for lb in labels:
            if lb not in algos: continue
            h = _expand_hist(algos[lb]["history"], dd["n_iter"])
            ax.plot(h, color=ALGO_COLORS[lb], linestyle=ALGO_STYLES[lb],
                    lw=2, label=lb, alpha=0.9)
        ax.set_title(f"Convergencia -- {dn}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Iteracao"); ax.set_ylabel("Fitness")
        ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.4)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        plt.tight_layout()
        plt.savefig(f"{IMG_DIR}/{ds}_convergencia.png", dpi=200); plt.close()

        # 2. Barras fitness
        fig, ax = plt.subplots(figsize=(11, 5))
        present = [lb for lb in labels if lb in algos]
        vals = [algos[lb]["fitness"] for lb in present]
        clrs = [ALGO_COLORS[lb] for lb in present]
        bars = ax.bar(range(len(present)), vals, color=clrs, alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=30, ha="right", fontsize=8.5)
        ax.set_title(f"Fitness Final -- {dn}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Fitness")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.005,
                    f"{v:,.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{IMG_DIR}/{ds}_fitness.png", dpi=200); plt.close()

        # 3. Boias ativas
        fig, ax = plt.subplots(figsize=(11, 5))
        vals_b = [algos[lb]["boias_ativas"] for lb in present]
        bars = ax.bar(range(len(present)), vals_b, color=clrs, alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=30, ha="right", fontsize=8.5)
        ax.set_title(f"Boias Ativas (min={dd['min_boias']}) -- {dn}",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Numero de Boias")
        ax.axhline(dd["min_boias"], color="red", ls="--", alpha=0.6,
                    label=f"Minimo ({dd['min_boias']})")
        ax.legend()
        for b, v in zip(bars, vals_b):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                    str(v), ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{IMG_DIR}/{ds}_boias.png", dpi=200); plt.close()

        # 4. Multicriterio (eficiencia, tempo, economia)
        fig, ax = plt.subplots(figsize=(10, 5), ncols=3)
        for ci, (key, title) in enumerate([
            ("eficiencia", "Eficiencia (W/boia)"),
            ("tempo_s",    "Tempo (s)"),
            ("economia_pct","Economia (%)")]):
            vals_c = [algos[lb][key] for lb in present]
            ax[ci].barh(range(len(present)), vals_c, color=clrs, alpha=0.85)
            ax[ci].set_yticks(range(len(present)))
            ax[ci].set_yticklabels([p[:15] for p in present], fontsize=7.5)
            ax[ci].set_title(title, fontsize=10, fontweight="bold")
            ax[ci].invert_yaxis()
        fig.suptitle(f"Multi-criterio -- {dn}", fontsize=13, fontweight="bold", y=1.05)
        plt.tight_layout()
        plt.savefig(f"{IMG_DIR}/{ds}_multicriterio.png", dpi=200,
                    bbox_inches="tight"); plt.close()

        # 5. Heatmap selecao boias
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix = [algos[lb]["gbest"] for lb in present]
        im = ax.imshow(np.array(matrix), cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels(present, fontsize=9)
        ax.set_xticks(range(NUM_WECS))
        ax.set_xticklabels([str(i+1) for i in range(NUM_WECS)], fontsize=6.5)
        ax.set_xlabel("Boia ID"); ax.set_ylabel("Algoritmo")
        ax.set_title(f"Selecao de Boias (verde=ativa, vermelho=removida) -- {dn}",
                     fontsize=12, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([0,1]); cbar.set_ticklabels(["Removida","Ativa"])
        plt.tight_layout()
        plt.savefig(f"{IMG_DIR}/{ds}_heatmap.png", dpi=200); plt.close()

    # 6. Combinada convergencia
    ncol = len(data)
    fig, axes = plt.subplots(1, ncol, figsize=(7*ncol, 5))
    if ncol == 1: axes = [axes]
    for ai, (dn, dd) in enumerate(data.items()):
        ax = axes[ai]; algos = dd["algos"]
        for lb in labels:
            if lb not in algos: continue
            h = _expand_hist(algos[lb]["history"], dd["n_iter"])
            ax.plot(h, color=ALGO_COLORS[lb], ls=ALGO_STYLES[lb],
                    lw=2, label=lb, alpha=0.9)
        ax.set_title(f"Convergencia -- {dn}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteracao"); ax.set_ylabel("Fitness");
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.4)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/convergencia_todos.png", dpi=200); plt.close()

    # 7. Barras cross-dataset
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ai, (key, ylabel) in enumerate([
        ("fitness", "Fitness"),
        ("boias_ativas", "Boias Ativas"),
        ("eficiencia", "Eficiencia (W/boia)")]):
        ax = axes[ai]
        x = np.arange(len(labels))
        w = 0.35
        for di, (dn, dd) in enumerate(data.items()):
            present = [lb for lb in labels if lb in dd["algos"]]
            vals = [dd["algos"][lb][key] for lb in labels]
            offset = (di - 0.5) * w
            ax.bar(x + offset, vals, w, color=f"C{di}", alpha=0.8,
                   label=dn.replace(".db", ""))
        ax.set_xticks(x)
        ax.set_xticklabels([l[:15] for l in labels], rotation=30, ha="right", fontsize=7.5)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        if ai == 0:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/cross_dataset.png", dpi=200, bbox_inches="tight"); plt.close()

    print(f"Imagens salvas em {IMG_DIR}/")


def generate_report_md(data):
    L = []
    def w(msg=""): L.append(msg)

    w("# Relatorio de Resultados - Algoritmos Hibridos Quantico-Inspirados")
    w()
    w("## 1. Visao Geral")
    w()
    w("Relatorio gerado a partir de execucao real dos algoritmos.")
    w("Fonte de dados: `resultados_hibridos.json` (salvo automaticamente).")
    w()
    w("| Algoritmo | Tipo |")
    w("|---|---|")
    w("| QAOA-Hibrido | Base - Busca local + Mutacao GA |")
    w("| QEA-Hibrido | Base - Rotacao Q-bits + Mutacao GA |")
    w("| QPSO-Hibrido | Base - Poco Delta + Sigmoid + Mutacao GA |")
    w("| QAOA-Enhanced | Melhorado - Elitismo + Taxa adaptativa + Re-diversificacao |")
    w("| QEA-Enhanced | Melhorado - Tabela estendida + Perturbacao quantica |")
    w("| QPSO-Enhanced | Melhorado - Multi-strategy + Levy flight + Crossover |")
    w()
    w("## 2. Datasets")
    w()
    w("| Dataset | Boias | Iter | Pop | Min Boias | R2 | MAE | Potencia Original |")
    w("|---|---|---|---|---|---|---|---|")
    for dn, dd in data.items():
        w(f"| {dn} | {dd['n_wecs']} | {dd['n_iter']} | {dd['n_pop']} | {dd['min_boias']} | {dd['r2']:.4f} | {dd['mae']:,.0f} W | {dd['raw_power']:,.2f} W |")
    w()

    # Convergencia
    w("## 3. Convergencia")
    w()
    for dn in data:
        ds = dn.replace(".", "_").lower()
        w(f"### {dn}")
        w()
        w(f"![{dn}]({IMG_DIR}/{ds}_convergencia.png)")
        w()
    w("### Visao comparativa")
    w()
    w(f"![Todos]({IMG_DIR}/convergencia_todos.png)")
    w()

    # Resultados por dataset
    w("## 4. Resultados por Dataset")
    w()
    best_consensus = {}
    for dn, dd in data.items():
        ds = dn.replace(".", "_").lower()
        algos = dd["algos"]
        best_l = max(algos, key=lambda k: algos[k]["fitness"])
        best_consensus[dn] = best_l

        w(f"### {dn}")
        w()
        w("| Algoritmo | Fitness | Potencia | % Orig | Boias | Ef (W/boia) | Economia | Ganho Ef | Tempo |")
        w("|---|---|---|---|---|---|---|---|---|")
        for lb, r in sorted(algos.items(), key=lambda x: -x[1]["fitness"]):
            mark = " **(melhor)**" if lb == best_l else ""
            w(f"| {lb}{mark} | {r['fitness']:,.2f} | {r['potencia']:,.2f} W | {r['razao_pct']:.1f}% | {r['boias_ativas']}/{dd['n_wecs']} | {r['eficiencia']:,.2f} | {r['economia_pct']:.1f}% | {r['ganho_ef_pct']:+.1f}% | {r['tempo_s']:.1f}s |")
        w()
        for img_id, title in [("fitness", "Fitness"), ("boias", "Boias")]:
            w(f"![{title}]({IMG_DIR}/{ds}_{img_id}.png)")
            w()
        w("### Multi-criterio e Heatmap")
        w()
        for img_id, title in [("multicriterio", "Multi-criterio"), ("heatmap", "Heatmap Boias")]:
            w(f"![{title}]({IMG_DIR}/{ds}_{img_id}.png)")
            w()

    # Cross
    w("## 5. Comparacao Cross-Dataset")
    w()
    w(f"![Cross]({IMG_DIR}/cross_dataset.png)")
    w()

    # Analise
    w("## 6. Analise")
    w()
    counts = {}
    for l in best_consensus.values():
        counts[l] = counts.get(l, 0) + 1
    consensus = max(counts, key=counts.get)

    for dn, dd in data.items():
        algos = dd["algos"]
        bl = best_consensus[dn]
        b = algos[bl]
        wst = min(algos, key=lambda k: algos[k]["fitness"])
        fastest = min(algos, key=lambda k: algos[k]["tempo_s"])
        most_econ = max(algos, key=lambda k: algos[k]["economia_pct"])
        diff_pct = abs(b["fitness"] - algos[wst]["fitness"]) / b["fitness"] * 100
        w(f"### {dn}")
        w()
        w(f"- **Melhor fitness:** {bl} = {b['fitness']:,.2f}")
        w(f"- **Mais economico:** {most_econ} = {algos[most_econ]['boias_ativas']}/{dd['n_wecs']} ({algos[most_econ]['economia_pct']:.1f}%)")
        w(f"- **Mais rapido:** {fastest} = {algos[fastest]['tempo_s']:.1f}s")
        w(f"- **Amplitude fitness:** {diff_pct:.3f}%")
        w(f"- {bl} reteve {b['razao_pct']:.1f}% da potencia com {b['boias_ativas']}/{dd['n_wecs']} boias")
        w()

    w("### Conclusao")
    w()
    w(f"O **{consensus}** foi o mais consistente ({counts[consensus]}/{len(data)} datasets).")
    w()
    w(f"*Gerado automaticamente de execucao real -- {len(data)} datasets x {len(list(ALGO_COLORS.keys()))} algoritmos*")

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Relatorio salvo em {REPORT_OUT}")


if __name__ == "__main__":
    data = run_and_save()
    generate_images(data)
    generate_report_md(data)
    print("Tudo concluido!")
