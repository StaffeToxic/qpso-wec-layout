# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""
=============================================================================
ALGORITMOS HÍBRIDOS MELHORADOS (QUÂNTICO + GENÉTICO) COM ML SURROGATE
=============================================================================
Versão aprimorada com:
  - Operador de Mutação Adaptativa (taxa decrescente + multi-strategy)
  - Elitismo (melhores soluções preservadas)
  - Diversidade de População monitorada
  - QAOA com schedule otimizado e busca local intensificada
  - QEA com tabela de rotação estendida
  - QPSO com multi-estratégia e reparação inteligente

Usa funções do shared_utils.py para consistência.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from shared_utils import (
    sigmoid,
    avaliar_fitness,
    aplicar_mutacao,
    carregar_e_treinar_modelo,
    calcular_contribuicoes_shap,
)

# =============================================================================
# UTILITÁRIOS INTERNOS
# =============================================================================

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


def _diversidade(pop):
    """Fração de bits não-uniformes: mede quão diversa é a população."""
    freqs = np.mean(pop, axis=0)
    return np.mean(np.minimum(freqs, 1 - freqs))


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
    """Decaimento linear da taxa de mutação: alta no início, baixa no fim."""
    ratio = iter_atual / max_iter
    return base + (final - base) * ratio


# =============================================================================
# 1. QAOA HÍBRIDO ENHANCED
# =============================================================================

class HybridSimQAOA_WEC:
    """
    Simulação Clássica do QAOA com melhorias:
    - Elitismo: top-K indíviduos preservados a cada camada
    - Mutação adaptativa: taxa decrescente ao longo das camadas
    - Busca local mais intensa na fase de problema
    - Diversidade monitorada com re-randomização se estagnar
    """

    def __init__(
        self, num_wecs, p_camadas, num_samples, shap_contribs, shap_base, min_boias,
        elite_frac=0.1, n_local_search=8, diversity_threshold=0.02,
    ):
        self.num_wecs = num_wecs
        self.p_camadas = p_camadas
        self.num_samples = num_samples
        self.shap_contribs = shap_contribs
        self.shap_base = shap_base
        self.min_boias = min_boias

        self.gbest = None
        self.gbest_fit = -np.inf

        # Schedule: rampas suaves gamma (explora->explora) e beta (explora->intensifica)
        self.gammas = np.linspace(0.9, 0.3, p_camadas)
        self.betas  = np.linspace(0.1, 0.5, p_camadas)

        # Hiperparâmetros melhorados
        self.elite_frac = elite_frac
        self.n_local_search = n_local_search
        self.diversity_threshold = diversity_threshold
        self.stagnation_counter = 0
        self.last_gbest_fit = -np.inf

    def optimize(self):
        history = []
        pop = np.random.randint(0, 2, (self.num_samples, self.num_wecs))
        for i in range(self.num_samples):
            pop[i] = _garante_min_boias(pop[i], self.num_wecs, self.min_boias)

        fits = np.full(self.num_samples, -np.inf)

        for t in range(self.p_camadas):
            gamma = self.gammas[t]
            beta = self.betas[t]

            # 1. Fase de Problema: busca local gulosa com intensidade gamma
            for i in range(self.num_samples):
                fit_atual = avaliar_fitness(
                    pop[i], self.shap_contribs, self.shap_base,
                    self.num_wecs, self.min_boias,
                )
                for _ in range(self.n_local_search):
                    if np.random.rand() < gamma:
                        idx = np.random.randint(0, self.num_wecs)
                        pop[i, idx] = 1 - pop[i, idx]
                        novo_fit = avaliar_fitness(
                            pop[i], self.shap_contribs, self.shap_base,
                            self.num_wecs, self.min_boias,
                        )
                        if novo_fit > fit_atual:
                            fit_atual = novo_fit
                        else:
                            pop[i, idx] = 1 - pop[i, idx]

                fits[i] = fit_atual
                if fit_atual > self.gbest_fit:
                    self.gbest_fit = fit_atual
                    self.gbest = np.copy(pop[i])

            # 2. Fase de Mixing: perturbação global
            for i in range(self.num_samples):
                flips = np.random.rand(self.num_wecs) < (beta * 0.15)
                pop[i] = np.where(flips, 1 - pop[i], pop[i])
                fits[i] = avaliar_fitness(
                    pop[i], self.shap_contribs, self.shap_base,
                    self.num_wecs, self.min_boias,
                )

            # 3. Elitismo: preservar top-K
            n_elite = max(1, int(self.elite_frac * self.num_samples))
            elite_indices = np.argsort(fits)[-n_elite:]
            elite = np.copy(pop[elite_indices])

            # 4. Mutação adaptativa
            p_mut = _taxa_mutacao_adaptativa(t, self.p_camadas)
            pop = aplicar_mutacao(pop, fits, self.num_wecs, self.min_boias, p_mut=p_mut)

            # 5. Re-inserir elite
            pop[:n_elite] = elite

            # 6. Re-diversificar se estagnar (diversidade muito baixa)
            div = _diversidade(pop)
            if div < self.diversity_threshold:
                if self.gbest_fit == self.last_gbest_fit:
                    self.stagnation_counter += 1
                    if self.stagnation_counter >= 3:
                        n_replace = self.num_samples // 3
                        pop[-n_replace:] = np.random.randint(
                            0, 2, (n_replace, self.num_wecs)
                        )
                        for i in range(self.num_samples - n_replace, self.num_samples):
                            pop[i] = _garante_min_boias(
                                pop[i], self.num_wecs, self.min_boias
                            )
                        self.stagnation_counter = 0
                else:
                    self.stagnation_counter = 0
            self.last_gbest_fit = self.gbest_fit

            history.append(self.gbest_fit)

        return self.gbest, self.gbest_fit, history


# =============================================================================
# 2. QEA HÍBRIDO ENHANCED (Quantum Evolutionary Algorithm)
# =============================================================================

class HybridQEA_WEC:
    """
    QEA melhorado com:
    - Tabela de rotação com 3 intensidades (fraca, média, forte)
    - Elitismo via injeção periódica do gbest na população quântica
    - Mutação adaptativa no espaço clássico
    - Perturbação quântica: chance de descolamento para evitar convergência precoce
    """

    def __init__(
        self, num_wecs, num_indiv, max_iter, shap_contribs, shap_base, min_boias,
        rotation_intensity="medium", quantum_perturbation=0.01,
    ):
        self.num_wecs = num_wecs
        self.num_indiv = num_indiv
        self.max_iter = max_iter
        self.shap_contribs = shap_contribs
        self.shap_base = shap_base
        self.min_boias = min_boias

        # População quântica: [alpha, beta] normalizados => |alpha|^2 + |beta|^2 = 1
        self.q_pop = np.full((num_indiv, num_wecs, 2), 1.0 / np.sqrt(2))
        self.gbest = None
        self.gbest_fit = -np.inf

        # Rotação: 5 níveis (Han & Kim, 2002)
        intensities = {
            "weak":   0.01 * np.pi,
            "medium": 0.05 * np.pi,
            "strong": 0.10 * np.pi,
        }
        self.theta_base = intensities.get(rotation_intensity, 0.05 * np.pi)
        self.quantum_perturbation = quantum_perturbation

    def optimize(self):
        history = []
        fits = np.zeros(self.num_indiv)

        for t in range(self.max_iter):
            # 1. Colapso: Q-bit -> Clássico
            pop_classica = np.zeros((self.num_indiv, self.num_wecs), dtype=int)
            for i in range(self.num_indiv):
                for j in range(self.num_wecs):
                    prob_1 = self.q_pop[i, j, 1] ** 2
                    pop_classica[i, j] = 1 if np.random.rand() < prob_1 else 0
                pop_classica[i] = _garante_min_boias(
                    pop_classica[i], self.num_wecs, self.min_boias
                )
                fits[i] = avaliar_fitness(
                    pop_classica[i], self.shap_contribs, self.shap_base,
                    self.num_wecs, self.min_boias,
                )
                if fits[i] > self.gbest_fit:
                    self.gbest_fit = fits[i]
                    self.gbest = np.copy(pop_classica[i])

            history.append(self.gbest_fit)

            # 2. Mutação Genética adaptativa
            p_mut = _taxa_mutacao_adaptativa(t, self.max_iter, base=0.35, final=0.05)
            pop_classica = aplicar_mutacao(
                pop_classica, fits, self.num_wecs, self.min_boias, p_mut=p_mut
            )
            # Re-avaliar após mutação
            for i in range(self.num_indiv):
                fits[i] = avaliar_fitness(
                    pop_classica[i], self.shap_contribs, self.shap_base,
                    self.num_wecs, self.min_boias,
                )
                if fits[i] > self.gbest_fit:
                    self.gbest_fit = fits[i]
                    self.gbest = np.copy(pop_classica[i])

            # 3. Rotação Quântica (tabela estendida)
            for i in range(self.num_indiv):
                for j in range(self.num_wecs):
                    x_i = pop_classica[i, j]
                    b_i = self.gbest[j]
                    alpha = self.q_pop[i, j, 0]
                    beta  = self.q_pop[i, j, 1]

                    theta = 0.0
                    if x_i == 0 and b_i == 1:
                        theta = self.theta_base
                    elif x_i == 1 and b_i == 0:
                        theta = -self.theta_base

                    # Perturbação quântica: ruído pequeno para evitar convergência precoce
                    if np.random.rand() < self.quantum_perturbation:
                        theta += np.random.uniform(-0.02, 0.02) * np.pi

                    if theta != 0:
                        n_alpha = alpha * math.cos(theta) - beta * math.sin(theta)
                        n_beta  = alpha * math.sin(theta) + beta * math.cos(theta)
                        norm = math.sqrt(n_alpha**2 + n_beta**2)
                        if norm > 1e-12:
                            self.q_pop[i, j, 0] = n_alpha / norm
                            self.q_pop[i, j, 1] = n_beta / norm

            # 4. Injeção periódica de elite: reset do gbest a cada 20 iterações
            if (t + 1) % 20 == 0 and self.gbest is not None:
                best_ind = np.random.randint(0, self.num_indiv)
                epsilon = 0.01  # Ângulo pequeno para quase-determinismo
                for j in range(self.num_wecs):
                    if self.gbest[j] == 1:
                        # Força alta probabilidade de colapsar para 1
                        # Usando sin/cos garante |α|²+|β|²=1 (regra de Born)
                        self.q_pop[best_ind, j, 0] = math.sin(epsilon)
                        self.q_pop[best_ind, j, 1] = math.cos(epsilon)
                    else:
                        self.q_pop[best_ind, j, 0] = math.cos(epsilon)
                        self.q_pop[best_ind, j, 1] = math.sin(epsilon)

        return self.gbest, self.gbest_fit, history


# =============================================================================
# 3. QPSO HÍBRIDO ENHANCED (Quantum Particle Swarm Optimization)
# =============================================================================

class HybridQPSO_WEC:
    """
    QPSO melhorado com:
    - Multi-estratégia: 70% QPSO global, 30% busca local em pbest
    - Lévy flight mutation: saltos com cauda longa para exploração agressiva
    - Crossover uniforme com gbest para transferência de blocos promissores
    - Elitismo: top-3 partículas preservadas
    - Reparação inteligente: quando min_boias falha, liga boias com maior frequência em pbest
    """

    def __init__(self, num_wecs, num_particles, max_iter, shap_contribs, shap_base, min_boias,
                 levy_mu=0.05, crossover_rate=0.3):
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
        # Partícula 0: referência (todas ativas)
        self.particles[0] = np.ones(num_wecs, dtype=int)

        self.pbest = np.copy(self.particles)
        self.pbest_fit = np.full(num_particles, -np.inf)
        self.gbest = np.ones(num_wecs, dtype=int)
        self.gbest_fit = -np.inf
        self.history = []

    def optimize(self):
        history = []

        for t in range(self.max_iter):
            fits = np.zeros(self.num_particles)

            # Avaliar e atualizar pbest/gbest
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

            # Centro da população (mbest contínuo preserva informação)
            freq = np.mean(self.pbest, axis=0)
            # Usar freq contínuo em vez de binarizar: evita L=0 em
            # dimensões convergidas (mata exploração prematuramente)
            mbest = freq.astype(float)
            alpha = 1.0 - 0.5 * (t / self.max_iter)

            # Salvar top-3 partículas (elitismo)
            top_k = min(3, self.num_particles)
            top_indices = np.argsort(fits)[-top_k:]
            elites = np.copy(self.particles[top_indices])

            # Atualização multi-estratégia
            for i in range(self.num_particles):
                if np.random.rand() < 0.6:
                    # Strategy A: QPSO global (poço delta padrão)
                    phi = np.random.uniform(0, 1, self.num_wecs)
                    p_real = phi * self.pbest[i].astype(float) + (1 - phi) * self.gbest.astype(float)
                    u = np.random.uniform(1e-10, 1, self.num_wecs)
                    L = alpha * np.abs(mbest - self.particles[i].astype(float))
                    direction = np.where(np.random.rand(self.num_wecs) > 0.5, 1.0, -1.0)
                    x_real = p_real + direction * L * np.log(1.0 / u)
                else:
                    # Strategy B: Busca local em torno de pbest (exploração fina)
                    phi = np.random.uniform(0.5, 1, self.num_wecs)
                    p_real = phi * self.pbest[i].astype(float) + (1 - phi) * self.particles[i].astype(float)
                    u = np.random.uniform(1e-10, 1, self.num_wecs)
                    L = alpha * 0.5 * np.abs(self.pbest[i].astype(float) - self.particles[i].astype(float))
                    direction = np.where(np.random.rand(self.num_wecs) > 0.5, 1.0, -1.0)
                    x_real = p_real + direction * L * np.log(1.0 / u)

                # Robustez: fallback para p_real se valores não-numéricos
                x_real = np.where(np.isfinite(x_real), x_real, p_real)

                # Discretização via sigmoid
                probs = np.clip(sigmoid(x_real), 0.01, 0.99)
                self.particles[i] = np.where(np.random.rand(self.num_wecs) < probs, 1, 0)

            # Crossover uniforme com gbest: seleciona bits do gbest em blocos
            p_cx = self.crossover_rate * (1 - t / self.max_iter + 0.1)
            if np.random.rand() < p_cx:
                cx_parent = np.random.randint(0, self.num_particles)
                mask = np.random.rand(self.num_wecs) < 0.5
                child = self.particles[cx_parent].copy()
                child[mask] = self.gbest[mask]
                # Substituir pior partícula pelo filho
                worst_idx = np.argmin(fits)
                child_fit = avaliar_fitness(
                    child, self.shap_contribs, self.shap_base,
                    self.num_wecs, self.min_boias,
                )
                if child_fit > fits[worst_idx]:
                    self.particles[worst_idx] = child

            # Elitismo: preservar top-K
            self.particles[:len(elites)] = elites

            # Operador hibrido: Lévy flight mutation (Mantegna, 1994)
            # Pior metade da população recebe perturbacao com passos de Lévy
            levy_rate = self.levy_mu * (1 - 0.5 * (t / self.max_iter))
            sorted_fits = np.argsort(fits)
            n_levy = max(1, self.num_particles // 2)
            for i in sorted_fits[:n_levy]:
                if np.random.rand() < levy_rate:
                    # Lévy step via algoritmo de Mantegna (distribuição estável)
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

            # Mutação GA adaptativa nas piores
            p_mut = _taxa_mutacao_adaptativa(t, self.max_iter, base=0.35, final=0.05)
            self.particles = aplicar_mutacao(
                self.particles, fits, self.num_wecs, self.min_boias, p_mut=p_mut
            )

            # Reparação inteligente
            for i in range(self.num_particles):
                self.particles[i] = _repair_smart(
                    self.particles[i], self.num_wecs, self.min_boias, freq
                )

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

            # Registrar histórico após todos os operadores e reavaliação
            history.append(self.gbest_fit)

        self.history = history
        return self.gbest, self.gbest_fit, history


# =============================================================================
# ROTINA PRINCIPAL DE TESTES
# =============================================================================

if __name__ == "__main__":
    NUM_WECS = 49
    DB_FILE = "WEC_Sydney_49.db"
    NUM_ITER = 80
    NUM_POPULATION = 30
    MIN_BOIAS = 20

    print("\n" + "=" * 60)
    print(" ALGORITMOS HÍBRIDOS ENHANCED (49 WECs)")
    print(" (QAOA+, QEA+, QPSO+ com Multi-Strategy, Levy, Crossover e Elitismo)")
    print("=" * 60 + "\n")

    modelo, layout_base, best_power, r2, mae, _ = carregar_e_treinar_modelo(
        DB_FILE, NUM_WECS, n_estimators=150
    )
    print(f"    R²={r2:.4f} | MAE={mae:,.0f} W | Best raw={best_power:,.2f} W\n")

    # Calcular contribuições SHAP por boia
    shap_contribs, shap_base = calcular_contribuicoes_shap(modelo, layout_base, NUM_WECS)
    print()

    resultados = {}

    # 1. QAOA Enhanced
    print("[1] QAOA Enhanced (Busca Local + Mixer + Mutação Adaptativa + Elitismo)...")
    t0 = time.time()
    qaoa = HybridSimQAOA_WEC(
        NUM_WECS, NUM_ITER, NUM_POPULATION, shap_contribs, shap_base, MIN_BOIAS,
        elite_frac=0.1, n_local_search=10,
    )
    gbest_qaoa, fit_qaoa, hist_qaoa = qaoa.optimize()
    resultados['QAOA-Enhanced'] = {'fit': fit_qaoa, 'hist': hist_qaoa, 'ativas': int(np.sum(gbest_qaoa))}
    print(f"    Fitness: {fit_qaoa:,.2f} | Boias: {int(np.sum(gbest_qaoa))}/{NUM_WECS} | Tempo: {time.time()-t0:.1f}s\n")

    # 2. QEA Enhanced
    print("[2] QEA Enhanced (Rotação + Mutação Adaptativa + Perturbação Quântica)...")
    t0 = time.time()
    qea = HybridQEA_WEC(
        NUM_WECS, NUM_POPULATION, NUM_ITER, shap_contribs, shap_base, MIN_BOIAS,
        rotation_intensity="medium", quantum_perturbation=0.02,
    )
    gbest_qea, fit_qea, hist_qea = qea.optimize()
    resultados['QEA-Enhanced'] = {'fit': fit_qea, 'hist': hist_qea, 'ativas': int(np.sum(gbest_qea))}
    print(f"    Fitness: {fit_qea:,.2f} | Boias: {int(np.sum(gbest_qea))}/{NUM_WECS} | Tempo: {time.time()-t0:.1f}s\n")

    # 3. QPSO Enhanced
    print("[3] QPSO Enhanced (Multi-Strategy + Levy + Crossover + Elitismo)...")
    t0 = time.time()
    qpso = HybridQPSO_WEC(
        NUM_WECS, NUM_POPULATION, NUM_ITER, shap_contribs, shap_base, MIN_BOIAS,
        levy_mu=0.06, crossover_rate=0.35,
    )
    gbest_qpso, fit_qpso, hist_qpso = qpso.optimize()
    resultados['QPSO-Enhanced'] = {'fit': fit_qpso, 'hist': hist_qpso, 'ativas': int(np.sum(gbest_qpso))}
    print(f"    Fitness: {fit_qpso:,.2f} | Boias: {int(np.sum(gbest_qpso))}/{NUM_WECS} | Tempo: {time.time()-t0:.1f}s\n")

    # Visualização
    print("=" * 60)
    print(" RESUMO DOS RESULTADOS")
    print("=" * 60)

    plt.figure(figsize=(10, 6))
    cores = {'QAOA-Enhanced': '#1f77b4', 'QEA-Enhanced': '#2ca02c', 'QPSO-Enhanced': '#d62728'}
    estilos = {'QAOA-Enhanced': '-', 'QEA-Enhanced': '--', 'QPSO-Enhanced': '-.'}

    for nome, res in resultados.items():
        print(f" [{nome}] Fitness: {res['fit']:,.2f} | Boias: {res['ativas']}/{NUM_WECS}")
        plt.plot(res['hist'], label=f"{nome} (WECs: {res['ativas']})",
                 color=cores[nome], linestyle=estilos[nome], linewidth=2.5)

    plt.title("Convergência Algoritmos Híbridos Enhanced", fontsize=13, fontweight='bold')
    plt.xlabel("Iterações / Camadas")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("comparativo_hibridos_enhanced.png", dpi=300)
    print("\n[!] Gráfico salvo como 'comparativo_hibridos_enhanced.png'")
    print("\n[+] Script concluído com sucesso!")
