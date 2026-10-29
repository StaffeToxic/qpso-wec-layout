# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""
=============================================================================
ALGORITMOS HÍBRIDOS (QUÂNTICO + GENÉTICO) COM ML SURROGATE
=============================================================================
Este script foca na abordagem HÍBRIDA para a Seleção de WECs (49 boias).
Ele implementa versões de inspiração quântica que também incorporam
heurísticas genéticas (Mutação) para escapar de ótimos locais.

Algoritmos Implementados:
1. QAOA Clássico-Simulado Híbrido
2. QEA Híbrido (Quantum Evolutionary Algorithm)
3. QPSO Híbrido (Quantum Particle Swarm Optimization)

Todos rodam exclusivamente sobre layouts de 49 boias via ML Surrogate.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sqlite3
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# =============================================================================
# ETAPA 1: MODELO SURROGATE (MACHINE LEARNING) - EXCLUSIVO 49 BOIAS
# =============================================================================

def carregar_e_treinar_modelo(db_path="WEC_Sydney_49.db", num_wecs=49):
    """Treina o modelo surrogate baseado no banco de dados real de 49 boias."""
    print(f"[*] Treinando ML Surrogate ({num_wecs} boias)...")
    if not os.path.exists(db_path):
        print(f"[AVISO] Banco '{db_path}' não encontrado. Simulando dados dummy.")
        # Simulação para garantir que rode se o DB faltar (fallback local)
        X = np.random.rand(100, num_wecs * 2) * 500
        y = np.sum(X, axis=1) * np.random.uniform(0.8, 1.2, 100)
        best_layout = X[np.argmax(y)]
        best_power = np.max(y)
    else:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM wec_data", conn)
        conn.close()

        coord_cols = []
        for i in range(1, num_wecs + 1):
            coord_cols.append(f'X{i}')
            coord_cols.append(f'Y{i}')

        X = df[coord_cols].values
        y = df['Total_Power'].values
        best_idx = np.argmax(y)
        best_layout = X[best_idx]
        best_power = y[best_idx]

    # RandomForest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42)
    modelo.fit(X_train, y_train)

    return modelo, best_layout, best_power


def avaliar_fitness(bits, modelo, layout_base, num_wecs=49, min_boias=20):
    """Função de custo compartilhada. Recompensa redução de custo (mais eficiência)."""
    if np.sum(bits) < min_boias:
        return -1e6
    coords = layout_base.copy()
    for i in range(num_wecs):
        if bits[i] == 0:
            coords[i * 2] = 0
            coords[i * 2 + 1] = 0
    potencia = modelo.predict(coords.reshape(1, -1))[0]
    n_ativas = np.sum(bits)
    eficiencia = potencia / n_ativas if n_ativas > 0 else 0
    # Otimizar iterativamente potência e eficiência
    return 0.95 * potencia + 0.05 * eficiencia


def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


def aplicar_mutacao(populacao, fits, num_wecs, min_boias, p_mut=0.25):
    """
    Operador Genético de Mutação (A ESÊNCIA DO ALGORITMO HÍBRIDO).
    Aplica inversão de bit nas amostras com fitness abaixo da mediana.
    Isso ajuda os modelos quânticos a evitarem ficar presos em ótimos locais.
    """
    valid_fits = fits[fits > -1e5]
    if len(valid_fits) == 0:
        return populacao
        
    mediana = np.median(valid_fits)
    for i in range(len(populacao)):
        if fits[i] < mediana:
            if np.random.rand() < p_mut:
                idx = np.random.randint(0, num_wecs)
                populacao[i, idx] = 1 - populacao[i, idx]
    
    # Consertar restrições se a mutação quebrou
    for i in range(len(populacao)):
        while np.sum(populacao[i]) < min_boias:
            populacao[i, np.random.randint(0, num_wecs)] = 1
            
    return populacao


# =============================================================================
# 1. QAOA CLÁSSICO-SIMULADO HÍBRIDO
# =============================================================================
class HybridSimQAOA_WEC:
    def __init__(self, num_wecs, p_camadas, num_samples, modelo, layout_base, min_boias):
        self.num_wecs = num_wecs
        self.p_camadas = p_camadas
        self.num_samples = num_samples
        self.modelo = modelo
        self.layout_base = layout_base
        self.min_boias = min_boias
        
        self.gbest = None
        self.gbest_fit = -np.inf
        # Schedules para Busca e Mix
        self.gammas = np.linspace(0.8, 0.2, p_camadas)
        self.betas = np.linspace(0.2, 0.8, p_camadas)

    def optimize(self):
        history = []
        pop = np.random.randint(0, 2, (self.num_samples, self.num_wecs))
        for i in range(self.num_samples):
            while np.sum(pop[i]) < self.min_boias:
                pop[i, np.random.randint(0, self.num_wecs)] = 1
                
        fits = np.full(self.num_samples, -np.inf)
                
        for t in range(self.p_camadas):
            gamma = self.gammas[t]
            beta = self.betas[t]
            
            # 1. QAOA - Fase de Problema (busca local/Hamiltoniano)
            for i in range(self.num_samples):
                fit_atual = avaliar_fitness(pop[i], self.modelo, self.layout_base, self.num_wecs, self.min_boias)
                for _ in range(5):
                    if np.random.rand() < gamma:
                        idx = np.random.randint(0, self.num_wecs)
                        pop[i, idx] = 1 - pop[i, idx]
                        novo_fit = avaliar_fitness(pop[i], self.modelo, self.layout_base, self.num_wecs, self.min_boias)
                        if novo_fit > fit_atual:
                            fit_atual = novo_fit
                        else:
                            pop[i, idx] = 1 - pop[i, idx]
                            
                fits[i] = fit_atual
                if fit_atual > self.gbest_fit:
                    self.gbest_fit = fit_atual
                    self.gbest = np.copy(pop[i])
                    
            # 2. QAOA - Fase de Mixing (perturbador global)
            for i in range(self.num_samples):
                flips = np.random.rand(self.num_wecs) < (beta * 0.1) 
                pop[i] = np.where(flips, 1 - pop[i], pop[i])
                # Avalia de novo após o flip
                fits[i] = avaliar_fitness(pop[i], self.modelo, self.layout_base, self.num_wecs, self.min_boias)

            # 3. ABORDAGEM HÍBRIDA: Mutação Genética
            pop = aplicar_mutacao(pop, fits, self.num_wecs, self.min_boias, p_mut=0.3)
            
            history.append(self.gbest_fit)
        return self.gbest, self.gbest_fit, history


# =============================================================================
# 2. QEA HÍBRIDO (Quantum-Inspired Evolutionary Algorithm + GA)
# =============================================================================
class HybridQEA_WEC:
    def __init__(self, num_wecs, num_indiv, max_iter, modelo, layout_base, min_boias):
        self.num_wecs = num_wecs
        self.num_indiv = num_indiv
        self.max_iter = max_iter
        self.modelo = modelo
        self.layout_base = layout_base
        self.min_boias = min_boias
        
        self.q_pop = np.full((num_indiv, num_wecs, 2), 1.0 / np.sqrt(2))
        self.gbest = None
        self.gbest_fit = -np.inf

    def optimize(self):
        history = []
        for t in range(self.max_iter):
            # 1. QEA - Colapso para estado observável Clássico
            pop_classica = np.zeros((self.num_indiv, self.num_wecs), dtype=int)
            fits = np.zeros(self.num_indiv)
            
            for i in range(self.num_indiv):
                for j in range(self.num_wecs):
                    prob_1 = self.q_pop[i, j, 1] ** 2
                    pop_classica[i, j] = 1 if np.random.rand() < prob_1 else 0
                    
                while np.sum(pop_classica[i]) < self.min_boias:
                    pop_classica[i, np.random.randint(0, self.num_wecs)] = 1
                    
                fits[i] = avaliar_fitness(pop_classica[i], self.modelo, self.layout_base, self.num_wecs, self.min_boias)
                
            # 2. ABORDAGEM HÍBRIDA: Mutação Genética no estado clássico
            pop_classica = aplicar_mutacao(pop_classica, fits, self.num_wecs, self.min_boias, p_mut=0.3)
            
            # Reavaliar após mutação
            for i in range(self.num_indiv):
                fits[i] = avaliar_fitness(pop_classica[i], self.modelo, self.layout_base, self.num_wecs, self.min_boias)
                if fits[i] > self.gbest_fit:
                    self.gbest_fit = fits[i]
                    self.gbest = np.copy(pop_classica[i])
                    
            history.append(self.gbest_fit)
            
            # 3. QEA - Operador de Rotação Quântica (Ajuste dos QuBits)
            for i in range(self.num_indiv):
                for j in range(self.num_wecs):
                    x_i = pop_classica[i, j]
                    b_i = self.gbest[j]
                    alpha = self.q_pop[i, j, 0]
                    beta = self.q_pop[i, j, 1]
                    
                    theta = 0
                    if x_i == 0 and b_i == 1:
                        theta = 0.05 * np.pi
                    elif x_i == 1 and b_i == 0:
                        theta = -0.05 * np.pi
                        
                    if theta != 0:
                        n_alpha = alpha * math.cos(theta) - beta * math.sin(theta)
                        n_beta = alpha * math.sin(theta) + beta * math.cos(theta)
                        norm = math.sqrt(n_alpha**2 + n_beta**2)
                        self.q_pop[i, j, 0] = n_alpha / norm
                        self.q_pop[i, j, 1] = n_beta / norm
                        
        return self.gbest, self.gbest_fit, history


# =============================================================================
# 3. QPSO HÍBRIDO (Quantum-Inspired PSO + Mutação)
# =============================================================================
class HybridQPSO_WEC:
    def __init__(self, num_wecs, num_particles, max_iter, modelo, layout_base, min_boias):
        self.num_wecs = num_wecs
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.modelo = modelo
        self.layout_base = layout_base
        self.min_boias = min_boias
        
        self.particles = np.random.randint(0, 2, (num_particles, num_wecs))
        for i in range(num_particles):
            while np.sum(self.particles[i]) < min_boias:
                self.particles[i, np.random.randint(0, num_wecs)] = 1
                
        self.pbest = np.copy(self.particles)
        self.pbest_fit = np.full(num_particles, -np.inf)
        self.gbest = np.ones(num_wecs, dtype=int)
        self.gbest_fit = -np.inf

    def optimize(self):
        history = []
        for t in range(self.max_iter):
            fits = np.zeros(self.num_particles)
            for i in range(self.num_particles):
                fit = avaliar_fitness(self.particles[i], self.modelo, self.layout_base, self.num_wecs, self.min_boias)
                fits[i] = fit
                if fit > self.pbest_fit[i]:
                    self.pbest_fit[i] = fit
                    self.pbest[i] = np.copy(self.particles[i])
                if fit > self.gbest_fit:
                    self.gbest_fit = fit
                    self.gbest = np.copy(self.particles[i])
                    
            history.append(self.gbest_fit)
            
            freq = np.mean(self.pbest, axis=0)
            mbest = np.where(freq > 0.5, 1, 0).astype(float)
            alpha = 1.0 - 0.5 * (t / self.max_iter)
            
            # 1. QPSO - Poço Delta Schröndiger e Discretização Sigmoid
            for i in range(self.num_particles):
                phi = np.random.uniform(0, 1, self.num_wecs)
                p_real = phi * self.pbest[i].astype(float) + (1 - phi) * self.gbest.astype(float)
                
                u = np.random.uniform(1e-10, 1, self.num_wecs)
                L = alpha * np.abs(mbest - self.particles[i].astype(float))
                direction = np.where(np.random.rand(self.num_wecs) > 0.5, 1.0, -1.0)
                
                x_real = p_real + direction * L * np.log(1.0 / u)
                probs = sigmoid(x_real)
                self.particles[i] = np.where(np.random.rand(self.num_wecs) < probs, 1, 0)
                
            # 2. ABORDAGEM HÍBRIDA: Mutação nas posições atuais das partículas
            self.particles = aplicar_mutacao(self.particles, fits, self.num_wecs, self.min_boias, p_mut=0.3)
            
        return self.gbest, self.gbest_fit, history


# =============================================================================
# ROTINA PRINCIPAL DE TESTES 
# =============================================================================
if __name__ == "__main__":
    NUM_WECS = 49  # FORÇADO PARA FIXAR APENAS AS DE 49 BOIAS COMO SOLICITADO
    DB_FILE = "WEC_Sydney_49.db"
    NUM_ITER = 60
    NUM_POPULATION = 25
    MIN_BOIAS = 20
    
    print("\n" + "="*60)
    print(" INICIANDO TESTE DOS ALGORITMOS HÍBRIDOS (49 WECs)")
    print(" (Combinando QAOA, QEA, QPSO com Operadores Genéticos)")
    print("="*60 + "\n")
    
    modelo, layout_base, _ = carregar_e_treinar_modelo(DB_FILE, NUM_WECS)
    
    resultados = {}
    
    print("\n[1] Rodando QAOA Híbrido (Busca + Mixer + Mutação)...")
    qaoa = HybridSimQAOA_WEC(NUM_WECS, NUM_ITER, NUM_POPULATION, modelo, layout_base, MIN_BOIAS)
    gbest_qaoa, fit_qaoa, hist_qaoa = qaoa.optimize()
    resultados['QAOA-Híbrido'] = {'fit': fit_qaoa, 'hist': hist_qaoa, 'ativas': np.sum(gbest_qaoa)}
    print(f" -> QAOA Concluído. Fitness: {fit_qaoa:,.2f} | Boias: {np.sum(gbest_qaoa)}/49")

    print("\n[2] Rodando QEA Híbrido (Rotação QuBits + Mutação)...")
    qea = HybridQEA_WEC(NUM_WECS, NUM_POPULATION, NUM_ITER, modelo, layout_base, MIN_BOIAS)
    gbest_qea, fit_qea, hist_qea = qea.optimize()
    resultados['QEA-Híbrido'] = {'fit': fit_qea, 'hist': hist_qea, 'ativas': np.sum(gbest_qea)}
    print(f" -> QEA Concluído. Fitness: {fit_qea:,.2f} | Boias: {np.sum(gbest_qea)}/49")

    print("\n[3] Rodando QPSO Híbrido (Schrödinger + Mutação)...")
    qpso = HybridQPSO_WEC(NUM_WECS, NUM_POPULATION, NUM_ITER, modelo, layout_base, MIN_BOIAS)
    gbest_qpso, fit_qpso, hist_qpso = qpso.optimize()
    resultados['QPSO-Híbrido'] = {'fit': fit_qpso, 'hist': hist_qpso, 'ativas': np.sum(gbest_qpso)}
    print(f" -> QPSO Concluído. Fitness: {fit_qpso:,.2f} | Boias: {np.sum(gbest_qpso)}/49")

    print("\n" + "="*60)
    print(" RESUMO DOS RESULTADOS (HÍBRIDOS EM 49 WECs)")
    print("="*60)
    
    plt.figure(figsize=(10, 6))
    cores = {'QAOA-Híbrido': '#1f77b4', 'QEA-Híbrido': '#2ca02c', 'QPSO-Híbrido': '#d62728'}
    estilos = {'QAOA-Híbrido': '-', 'QEA-Híbrido': '--', 'QPSO-Híbrido': '-.'}
    
    for nome, res in resultados.items():
        print(f" [{nome}] -> Melhor Fit: {res['fit']:,.2f} (Utilizando {res['ativas']}/49 WECs)")
        plt.plot(res['hist'], label=f"{nome} (WECs: {int(res['ativas'])})", 
                 color=cores[nome], linestyle=estilos[nome], linewidth=2.5)
                 
    plt.title("Convergência dos Algoritmos Híbridos (Mecânica Quântica + Mutação GA)", fontsize=13, fontweight='bold')
    plt.xlabel("Iterações / Camadas")
    plt.ylabel("Fitness (Potência via Surrogate)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("comparativo_algoritmos_hibridos_49.png", dpi=300)
    print("\n[!] Gráfico de convergência salvo como 'comparativo_algoritmos_hibridos_49.png'")
    
    print("\n[+] Script executado e concluído com sucesso!")
