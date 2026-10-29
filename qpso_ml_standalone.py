# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sqlite3
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# =============================================================================
# UTILITÁRIOS E ML SURROGATE
# =============================================================================

def sigmoid(x):
    """Mapeia valor real para probabilidade [0, 1] com clipping numérico."""
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))

def _make_coord_cols(num_wecs):
    cols = []
    for i in range(1, num_wecs + 1):
        cols.append(f'X{i}')
        cols.append(f'Y{i}')
    return cols



def carregar_e_treinar_modelo(db_path, num_wecs=49, n_estimators=100, n_jobs=-1, verbose=True):
    """
    Carrega dados do SQLite, treina Random Forest e retorna modelo + metadados.
    """
    if num_wecs <= 0:
        raise ValueError("`num_wecs` deve ser > 0.")
    if n_estimators <= 0:
        raise ValueError("`n_estimators` deve ser > 0.")

    if verbose:
        print(f"[*] Treinando ML Surrogate ({num_wecs} boias, {n_estimators} estimadores)...")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Banco de dados '{db_path}' não encontrado. O dataset real é obrigatório.")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM wec_data", conn)
    conn.close()

    coord_cols = _make_coord_cols(num_wecs)

    missing = [c for c in coord_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing[:5]}...")

    X = df[coord_cols].values
    y = df['Total_Power'].values

    best_idx = np.argmax(y)
    best_layout = X[best_idx]
    best_power = y[best_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=42)
    try:
        modelo.fit(X_train, y_train)
    except PermissionError:
        if n_jobs == 1:
            raise
        if verbose:
            print("    [!] Falha de permissao com n_jobs != 1; aplicando fallback para n_jobs=1.")
        modelo = RandomForestRegressor(n_estimators=n_estimators, n_jobs=1, random_state=42)
        modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    if verbose:
        print(f"    R²={r2:.4f} | MAE={mae:,.0f} W | Best original={best_power:,.2f} W")

    return modelo, best_layout, best_power, r2, mae, coord_cols

def avaliar_fitness(bits, shap_contribs, shap_base, num_wecs, min_boias):
    """
    Avalia a qualidade de uma seleção de boias via decomposição SHAP.

    Potência estimada = base_value + Σ shap_contrib_i (boias ativas)
    Fitness = 0.7 * potencia + 0.3 * eficiencia_por_boia

    Usa SHAP values do melhor layout para decompor a predição do modelo
    em contribuições aditivas por boia, evitando predições Out-of-Distribution.
    """
    if np.sum(bits) < min_boias:
        return -1e6

    potencia = shap_base + np.sum(shap_contribs[bits == 1])
    n_ativas = int(np.sum(bits))
    eficiencia = potencia / n_ativas if n_ativas > 0 else 0

    return 0.7 * potencia + 0.3 * eficiencia

def calcular_contribuicoes_shap(modelo, layout_ref, num_wecs, verbose=True):
    """
    Calcula a contribuição marginal de cada boia usando decomposição SHAP.

    Decomposição aditiva: f(x) = base_value + Σ contrib_i
    Remoção: f(x sem boia_i) ≈ base_value + Σ_{j≠i} contrib_j
    """
    import shap

    if verbose:
        print(f"[*] Calculando SHAP contributions ({num_wecs} WECs)...")

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(layout_ref.reshape(1, -1))
    ev = explainer.expected_value
    ev_arr = np.asarray(ev, dtype=float).reshape(-1)
    if ev_arr.size == 0:
        raise ValueError("expected_value do SHAP vazio.")
    base_value = float(ev_arr[0])

    # Compatibilidade com multiplas formas de saida do SHAP
    # (lista de arrays, array 2D ou formatos com dimensoes extras).
    shap_arr = shap_values[0] if isinstance(shap_values, list) else shap_values
    shap_arr = np.asarray(shap_arr, dtype=float)
    while shap_arr.ndim > 1:
        shap_arr = shap_arr[0]
    if shap_arr.size < num_wecs * 2:
        raise ValueError(
            f"Saida SHAP invalida: esperado >= {num_wecs * 2} features, recebido {shap_arr.size}."
        )

    wec_contribs = np.zeros(num_wecs)
    for i in range(num_wecs):
        wec_contribs[i] = shap_arr[i * 2] + shap_arr[i * 2 + 1]

    total_shap = np.sum(wec_contribs)
    rf_pred = modelo.predict(layout_ref.reshape(1, -1))[0]
    erro_shap = abs(base_value + total_shap - rf_pred)

    if verbose:
        print(f"    SHAP base (E[f(X)]): {base_value:,.0f} W")
        print(f"    Sum SHAP contribs:   {total_shap:,.0f} W")
        print(f"    Predição RF:         {rf_pred:,.0f} W")
        print(f"    Erro decomposição:   {erro_shap:,.0f} W")

        sorted_idx = np.argsort(wec_contribs)
        print(f"    Top-5 boias (maior contribuição):")
        for idx in sorted_idx[-5:][::-1]:
            print(f"      WEC{idx+1}: {wec_contribs[idx]:+,.0f} W")
        print(f"    Bottom-3 boias (menor contribuição):")
        for idx in sorted_idx[:3]:
            print(f"      WEC{idx+1}: {wec_contribs[idx]:+,.0f} W")

    return wec_contribs, base_value

# =============================================================================
# UTILITÁRIOS QPSO
# =============================================================================

def aplicar_mutacao(populacao, fits, num_wecs, min_boias, p_mut=0.25, frozen_indices=None):
    """Operador Genético de Mutação com taxa adaptativa."""
    frozen = set() if frozen_indices is None else set(np.asarray(frozen_indices).tolist())
    valid_fits = fits[fits > -1e5]
    if len(valid_fits) == 0:
        return populacao

    mediana = np.median(valid_fits)
    for i in range(len(populacao)):
        if i in frozen:
            continue
        if fits[i] < mediana:
            if np.random.rand() < p_mut:
                # Multi-bit flip: proporcional à dimensão para melhor escape
                n_flips = max(1, int(p_mut * num_wecs * 0.3))
                for _ in range(n_flips):
                    idx = np.random.randint(0, num_wecs)
                    populacao[i, idx] = 1 - populacao[i, idx]

    for i in range(len(populacao)):
        if i in frozen:
            continue
        n_ativas = int(np.sum(populacao[i]))
        if n_ativas >= min_boias:
            continue
        deficit = int(min_boias - n_ativas)
        zeros = np.where(populacao[i] == 0)[0]
        if len(zeros) > 0:
            chosen = np.random.choice(zeros, min(deficit, len(zeros)), replace=False)
            populacao[i, chosen] = 1

    return populacao

def _garante_min_boias(bits, num_wecs, min_boias):
    """Garante que o vetor binário tenha no mínimo `min_boias` bits ativos."""
    n_ativas = int(np.sum(bits))
    if n_ativas >= min_boias:
        return bits
    deficit = int(min_boias - n_ativas)
    zeros = np.where(bits == 0)[0]
    if len(zeros) >= deficit:
        chosen = np.random.choice(zeros, deficit, replace=False)
    else:
        chosen = zeros
    bits[chosen] = 1
    return bits

def _repair_smart(bits, num_wecs, min_boias, freq_bits):
    """
    Reparação inteligente: ao invés de ativar boias aleatórias,
    prioriza as que têm maior frequência no pbest.
    """
    n_ativas = np.sum(bits)
    if n_ativas >= min_boias:
        return bits
    deficit = int(min_boias - n_ativas)
    zeros = np.where(bits == 0)[0]
    if len(zeros) > 0:
        scores = freq_bits[zeros]
        chosen = zeros[np.argsort(scores)[-deficit:]]
        bits[chosen] = 1
    return bits

def _taxa_mutacao_adaptativa(iter_atual, max_iter, base=0.4, final=0.05):
    """Decaimento linear da taxa de mutação."""
    ratio = iter_atual / max_iter
    return base + (final - base) * ratio

# =============================================================================
# QPSO HÍBRIDO ENHANCED
# =============================================================================

class HybridQPSO_WEC:

    def __init__(self, num_wecs, num_particles, max_iter, shap_contribs, shap_base, min_boias,
                 levy_mu=0.05, crossover_rate=0.3):
        if num_wecs <= 0:
            raise ValueError("`num_wecs` deve ser > 0.")
        if num_particles <= 0:
            raise ValueError("`num_particles` deve ser > 0.")
        if max_iter <= 0:
            raise ValueError("`max_iter` deve ser > 0.")
        if min_boias <= 0 or min_boias > num_wecs:
            raise ValueError("`min_boias` deve estar no intervalo [1, num_wecs].")

        self.num_wecs = num_wecs
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.shap_contribs = shap_contribs
        self.shap_base = shap_base
        self.min_boias = min_boias
        self.levy_mu = levy_mu
        self.crossover_rate = crossover_rate

        self.particles = np.random.randint(0, 2, (num_particles, num_wecs))
        for i in range(num_particles):
            self.particles[i] = _garante_min_boias(
                self.particles[i], self.num_wecs, min_boias
            )
        self.particles[0] = np.ones(num_wecs, dtype=int)

        self.pbest = np.copy(self.particles)
        self.pbest_fit = np.full(num_particles, -np.inf)
        self.gbest = np.ones(num_wecs, dtype=int)
        self.gbest_fit = -np.inf
        self.history = []

    def optimize(self):
        history = []
        n_active_history = []
        for t in range(self.max_iter):
            fits = np.zeros(self.num_particles)
            for i in range(self.num_particles):
                fit = avaliar_fitness(
                    self.particles[i], self.shap_contribs, self.shap_base,
                    self.num_wecs, self.min_boias,
                )
                fits[i] = fit
                if fit > self.pbest_fit[i]:
                    self.pbest_fit[i] = fit
                    self.pbest[i] = np.copy(self.particles[i])
                if fit > self.gbest_fit:
                    self.gbest_fit = fit
                    self.gbest = np.copy(self.particles[i])

            freq = np.mean(self.pbest, axis=0)
            # Usar freq contínuo em vez de binarizar: evita L=0 em
            # dimensões convergidas (mata exploração prematuramente)
            mbest = freq.astype(float)
            alpha = 1.0 - 0.5 * (t / self.max_iter)

            top_k = min(3, self.num_particles)
            top_indices = np.argsort(fits)[-top_k:]
            elites = np.copy(self.particles[top_indices])
            elite_slots = np.arange(top_k)

            for i in range(self.num_particles):
                if np.random.rand() < 0.6:
                    phi = np.random.uniform(0, 1, self.num_wecs)
                    p_real = phi * self.pbest[i].astype(float) + (1 - phi) * self.gbest.astype(float)
                    u = np.random.uniform(1e-10, 1, self.num_wecs)
                    L = alpha * np.abs(mbest - self.particles[i].astype(float))
                    direction = np.where(np.random.rand(self.num_wecs) > 0.5, 1.0, -1.0)
                    x_real = p_real + direction * L * np.log(1.0 / u)
                else:
                    phi = np.random.uniform(0.5, 1, self.num_wecs)
                    p_real = phi * self.pbest[i].astype(float) + (1 - phi) * self.particles[i].astype(float)
                    u = np.random.uniform(1e-10, 1, self.num_wecs)
                    L = alpha * 0.5 * np.abs(self.pbest[i].astype(float) - self.particles[i].astype(float))
                    direction = np.where(np.random.rand(self.num_wecs) > 0.5, 1.0, -1.0)
                    x_real = p_real + direction * L * np.log(1.0 / u)

                x_real = np.where(np.isfinite(x_real), x_real, p_real)
                probs = np.clip(sigmoid(x_real), 0.01, 0.99)
                self.particles[i] = np.where(np.random.rand(self.num_wecs) < probs, 1, 0)

            # Restaurar elites antes dos operadores subsequentes.
            self.particles[:len(elites)] = elites

            p_cx = self.crossover_rate * (1 - t / self.max_iter + 0.1)
            if np.random.rand() < p_cx:
                cx_parent = np.random.randint(0, self.num_particles)
                mask = np.random.rand(self.num_wecs) < 0.5
                child = self.particles[cx_parent].copy()
                child[mask] = self.gbest[mask]
                non_elites = np.setdiff1d(np.arange(self.num_particles), elite_slots, assume_unique=True)
                if len(non_elites) == 0:
                    worst_idx = np.argmin(fits)
                else:
                    worst_idx = non_elites[np.argmin(fits[non_elites])]
                child_fit = avaliar_fitness(
                    child, self.shap_contribs, self.shap_base,
                    self.num_wecs, self.min_boias,
                )
                if child_fit > fits[worst_idx]:
                    self.particles[worst_idx] = child

            levy_rate = self.levy_mu * (1 - 0.5 * (t / self.max_iter))
            sorted_fits = np.argsort(fits)
            n_levy = max(1, self.num_particles // 2)
            levy_candidates = [i for i in sorted_fits if i not in elite_slots][:n_levy]
            for i in levy_candidates:
                if np.random.rand() < levy_rate:
                    levy_beta = 1.5  # índice de estabilidade ∈ (0, 2)
                    sigma_u = (
                        math.gamma(1 + levy_beta) * math.sin(math.pi * levy_beta / 2)
                        / (math.gamma((1 + levy_beta) / 2) * levy_beta * 2 ** ((levy_beta - 1) / 2))
                    ) ** (1 / levy_beta)
                    u_l = np.random.normal(0, sigma_u)
                    v_l = np.random.normal(0, 1)
                    levy_step = abs(u_l / (abs(v_l) ** (1 / levy_beta)))
                    n_flips = min(max(1, int(levy_step)), self.num_wecs - self.min_boias)
                    if n_flips >= 1:
                        flip_indices = np.random.choice(self.num_wecs, n_flips, replace=False)
                        self.particles[i, flip_indices] = 1 - self.particles[i, flip_indices]

            p_mut = _taxa_mutacao_adaptativa(t, self.max_iter, base=0.35, final=0.05)
            self.particles = aplicar_mutacao(
                self.particles, fits, self.num_wecs, self.min_boias, p_mut=p_mut, frozen_indices=elite_slots
            )

            for i in range(self.num_particles):
                if i in elite_slots:
                    continue
                self.particles[i] = _repair_smart(
                    self.particles[i], self.num_wecs, self.min_boias, freq
                )

            # Blindagem final de elitismo: mesmo após operadores, elites são preservados.
            self.particles[:len(elites)] = elites

            # Re-avaliar e atualizar pbest/gbest após mutação/repair
            for i in range(self.num_particles):
                fit = avaliar_fitness(
                    self.particles[i], self.shap_contribs, self.shap_base,
                    self.num_wecs, self.min_boias,
                )
                if fit > self.pbest_fit[i]:
                    self.pbest_fit[i] = fit
                    self.pbest[i] = np.copy(self.particles[i])
                if fit > self.gbest_fit:
                    self.gbest_fit = fit
                    self.gbest = np.copy(self.particles[i])

            # Registrar histórico no estado final da iteração.
            history.append(self.gbest_fit)
            n_active_history.append(int(np.sum(self.gbest)))

        self.history = history
        self.n_active_history = n_active_history
        return self.gbest, self.gbest_fit, history, n_active_history

# =============================================================================
# ROTINA PRINCIPAL DE TESTES
# =============================================================================

if __name__ == "__main__":
    NUM_WECS = 49
    DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WEC_Sydney_49.db")
    NUM_ITER = 80
    NUM_POPULATION = 30
    MIN_BOIAS = 20

    print("\n" + "=" * 60)
    print(" QPSO HÍBRIDO ENHANCED (49 WECs) COM ML SURROGATE")
    print("=" * 60 + "\n")

    modelo, layout_base, best_power, r2, mae, _ = carregar_e_treinar_modelo(
        DB_FILE, NUM_WECS, n_estimators=150
    )
    print(f"    R²={r2:.4f} | MAE={mae:,.0f} W | Best raw={best_power:,.2f} W\n")

    # Calcular contribuições SHAP por boia
    shap_contribs, shap_base = calcular_contribuicoes_shap(modelo, layout_base, NUM_WECS)
    print()

    print("[1] Executando QPSO Enhanced...")
    t0 = time.time()
    qpso = HybridQPSO_WEC(
        NUM_WECS, NUM_POPULATION, NUM_ITER, shap_contribs, shap_base, MIN_BOIAS,
        levy_mu=0.06, crossover_rate=0.35,
    )
    gbest_qpso, fit_qpso, hist_qpso, hist_nactive = qpso.optimize()
    n_ativas_final = int(np.sum(gbest_qpso))
    n_removidas = NUM_WECS - n_ativas_final
    print(f"    Fitness: {fit_qpso:,.2f} | Boias Ativas: {n_ativas_final}/{NUM_WECS} | Tempo: {time.time()-t0:.1f}s\n")

    # =========================================================================
    # RELATÓRIO DE BOIAS DESATIVADAS
    # =========================================================================
    desativadas = np.where(gbest_qpso == 0)[0]
    if len(desativadas) > 0:
        print(f"[2] Boias Desativadas ({len(desativadas)}):")
        # Ordenar por contribuição SHAP (da menor para a maior)
        desativadas_sorted = sorted(desativadas, key=lambda i: shap_contribs[i])
        perda_total = 0.0
        for idx in desativadas_sorted:
            contrib = shap_contribs[idx]
            perda_total += contrib
            sinal = "+" if contrib >= 0 else ""
            print(f"      WEC{idx+1:>2d}: {sinal}{contrib:,.0f} W")
        print(f"    ────────────────────────────────")
        print(f"    Perda total estimada (SHAP): {perda_total:,.0f} W")
        print(f"    Potência base (E[f(X)]):     {shap_base:,.0f} W")
        print(f"    Potência estimada ativas:    {shap_base + sum(shap_contribs[gbest_qpso == 1]):,.0f} W\n")
    else:
        print("[2] Nenhuma boia foi desativada.\n")

    # =========================================================================
    # GRÁFICO 3 PAINÉIS (dados exclusivamente do dataset + otimização)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Painel 1: Convergência ---
    ax1 = axes[0]
    ax1.plot(hist_qpso, color='#9b59b6', linewidth=2.5)
    ax1.set_title('Convergência do QPSO Híbrido Discreto', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Iterações')
    ax1.set_ylabel('Fitness')
    ax1.grid(True, alpha=0.3)

    # --- Painel 2: Mapa de Seleção de Boias ---
    ax2 = axes[1]
    coords_x = layout_base[0::2]  # X1, X2, ..., X49
    coords_y = layout_base[1::2]  # Y1, Y2, ..., Y49
    ativas = gbest_qpso == 1
    removidas = gbest_qpso == 0

    ax2.scatter(coords_x[ativas], coords_y[ativas], c='green', s=60,
               edgecolors='darkgreen', linewidths=0.8, label='Ativa', zorder=3)
    ax2.scatter(coords_x[removidas], coords_y[removidas], c='red', s=50,
               edgecolors='darkred', linewidths=0.8, label='Removida', zorder=3)
    ax2.set_title(f'Seleção de Boias ({n_ativas_final} ativas / {n_removidas} removidas)',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('X (metros)')
    ax2.set_ylabel('Y (metros)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Painel 3: Boias Ativas por Iteração ---
    ax3 = axes[2]
    ax3.plot(hist_nactive, color='#00bcd4', linewidth=2.5)
    ax3.axhline(y=MIN_BOIAS, color='red', linestyle='--', linewidth=1.5,
               label=f'Mínimo ({MIN_BOIAS})', alpha=0.7)
    ax3.set_title('Boias Ativas por Iteração', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Iterações')
    ax3.set_ylabel('Quantidade de Boias Ativas')
    ax3.legend(loc='right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qpso_enhanced_convergence.png')
    plt.savefig(out_path, dpi=300)
    print(f"[!] Gráfico salvo como '{out_path}'")
    print("\n[+] Script concluído com sucesso!")
