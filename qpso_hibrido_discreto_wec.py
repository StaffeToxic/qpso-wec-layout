# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""
=============================================================================
QPSO HÍBRIDO DISCRETO COM ML SURROGATE (DATA-DRIVEN)
=============================================================================
Algoritmo de Otimização de Inspiração Quântica Híbrido Multi-Strategy
aplicado à Seleção Inteligente de WECs (Wave Energy Converters).

COMO EXECUTAR:
    pip install numpy pandas scikit-learn matplotlib
    python qpso_hibrido_discreto_wec.py

CONCEITO:
    1. Treina um modelo de Machine Learning (Random Forest) usando TODOS os
       dados reais dos sensores BEM do arquivo CSV (sem inventar nenhuma física).
    2. Usa um QPSO Híbrido Discreto para decidir quais boias manter ligadas
       (bit=1) e quais desligar (bit=0) para maximizar potência/custo.
    3. O operador quântico (Schrödinger) gera probabilidades via sigmoid,
       e o operador genético (Mutação GA) evita ótimos locais.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sqlite3

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# =============================================================================
# ETAPA 1: CARREGAMENTO E TREINAMENTO DO MODELO ML SURROGATE
# =============================================================================

def carregar_e_treinar_modelo(db_path, num_wecs=49):
    """
    Carrega o arquivo real dos sensores (SQLite) e treina um Random Forest Regressor
    para aprender a relação: Coordenadas(X,Y) -> Total_Power.
    
    Não inventa nenhuma fórmula: aprende diretamente dos dados do artigo.
    """
    print("=" * 65)
    print("ETAPA 1: TREINAMENTO DO MODELO DE INTELIGÊNCIA ARTIFICIAL")
    print("=" * 65)
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Banco de dados '{db_path}' não encontrado!")
    
    # Carrega os dados reais do banco
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM wec_data", conn)
    conn.close()
    
    print(f"[+] Banco de dados carregado: {db_path}")
    print(f"    -> {len(df)} layouts de boias disponíveis para treinamento")
    print(f"    -> {len(df.columns)} colunas no dataset")
    
    # Features: Coordenadas X1,Y1,...,X49,Y49 (98 colunas)
    coord_cols = []
    for i in range(1, num_wecs + 1):
        coord_cols.append(f'X{i}')
        coord_cols.append(f'Y{i}')
    
    # Verifica se todas as colunas existem
    missing = [c for c in coord_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando no CSV: {missing[:5]}...")
    
    X = df[coord_cols].values
    y = df['Total_Power'].values
    
    # Também extraindo as potências individuais para análise posterior
    power_cols = [f'Power{i}' for i in range(1, num_wecs + 1)]
    powers_matrix = df[power_cols].values if all(c in df.columns for c in power_cols) else None
    
    # Divisão treino/teste para validação real (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n[*] Treinando Random Forest com {len(X_train)} amostras...")
    print(f"    -> Features: {len(coord_cols)} coordenadas (X,Y de cada boia)")
    print(f"    -> Target: Total_Power (Watts)")
    
    t0 = time.time()
    modelo = RandomForestRegressor(
        n_estimators=200,       # 200 árvores de decisão para precisão
        max_depth=None,         # Sem limite de profundidade (captura interações complexas)
        min_samples_split=5,    # Evita overfitting excessivo
        n_jobs=-1,              # Usa todos os núcleos do CPU
        random_state=42
    )
    modelo.fit(X_train, y_train)
    tempo_treino = time.time() - t0
    
    # Validação com dados NUNCA vistos pelo modelo
    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n[OK] MODELO TREINADO em {tempo_treino:.2f}s")
    print(f"    -> R² Score (Validação): {r2:.4f}  (1.0 = perfeito)")
    print(f"    -> MAE (Erro Médio Absoluto): {mae:.2f} Watts")
    
    if r2 < 0.5:
        print(f"    [!] AVISO: R² baixo ({r2:.2f}). O modelo pode não ser preciso o suficiente.")
    elif r2 > 0.85:
        print(f"    [**] EXCELENTE: O modelo capturou {r2*100:.1f}% da variância dos dados reais!")
    
    # Encontra o melhor layout do CSV para usar como base
    best_idx = np.argmax(y)
    best_layout = X[best_idx]   # Coordenadas do melhor layout
    best_power = y[best_idx]
    best_powers_individual = powers_matrix[best_idx] if powers_matrix is not None else None
    
    print(f"\n[+] Melhor layout original do CSV:")
    print(f"    -> Linha {best_idx + 1}, Total_Power = {best_power:.2f} W")
    if 'qW' in df.columns:
        print(f"    -> qW reportado: {df['qW'].iloc[best_idx]}")
    
    return modelo, best_layout, best_power, best_powers_individual, r2, coord_cols


# =============================================================================
# ETAPA 2: QPSO HÍBRIDO DISCRETO (SELEÇÃO DE BOIAS)
# =============================================================================

def sigmoid(x):
    """Mapeia valor real para probabilidade [0, 1]."""
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


class HybridDiscreteQPSO_WEC:
    """
    QPSO Híbrido Discreto para Seleção de Boias usando ML Surrogate.
    
    Cada partícula é um vetor binário de N bits:
        bit[i] = 1 → Boia i está ATIVA (mantida no layout)
        bit[i] = 0 → Boia i está DESLIGADA (removida do layout)
    
    Híbrido porque combina:
        - Operador Quântico (Poço de Potencial Delta / Schrödinger) para exploração
        - Operador Genético (Mutação GA) para escapar de ótimos locais
    """
    
    def __init__(self, num_wecs, num_particles, max_iter, modelo_ml, 
                 layout_base, min_boias_ativas=20):
        self.num_wecs = num_wecs
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.modelo = modelo_ml
        self.layout_base = layout_base.copy()
        self.min_boias = min_boias_ativas
        
        # Inicializa partículas: cada uma é um vetor binário de 49 bits
        # Começam com ~70% das boias ativas (bias para exploração)
        self.particles = np.zeros((num_particles, num_wecs), dtype=int)
        for i in range(num_particles):
            n_ativas = np.random.randint(self.min_boias, num_wecs + 1)
            indices = np.random.choice(num_wecs, n_ativas, replace=False)
            self.particles[i, indices] = 1
        
        # Partícula 0: todas ativas (referência completa)
        self.particles[0] = np.ones(num_wecs, dtype=int)
        
        # Memória das melhores posições
        self.pbest = np.copy(self.particles)
        self.pbest_fit = np.full(num_particles, -np.inf)
        self.gbest = np.ones(num_wecs, dtype=int)
        self.gbest_fit = -np.inf
    
    def avaliar_fitness(self, bits):
        """
        Avalia a qualidade de uma seleção de boias usando o modelo ML.
        
        Monta o vetor de coordenadas: mantém coords originais para boias ativas,
        zera coords para boias desligadas. O modelo ML prediz a potência total
        dessa configuração baseado nos padrões aprendidos dos dados reais.
        """
        # Garante mínimo de boias ativas
        if np.sum(bits) < self.min_boias:
            return -1e6
        
        # Monta vetor de coordenadas modificado
        coords = self.layout_base.copy()
        for i in range(self.num_wecs):
            if bits[i] == 0:
                coords[i * 2] = 0       # X_i = 0
                coords[i * 2 + 1] = 0   # Y_i = 0
        
        # Prediz potência usando o modelo ML treinado nos dados reais
        potencia_predita = self.modelo.predict(coords.reshape(1, -1))[0]
        
        # Bônus de eficiência: recompensa por usar menos boias
        # (simula redução de custo de instalação/manutenção)
        n_ativas = np.sum(bits)
        eficiencia = potencia_predita / n_ativas if n_ativas > 0 else 0
        
        # Fitness: potência total + bonus por eficiência
        # (peso 0.95 potência, 0.05 eficiência para não dominar)
        fitness = 0.95 * potencia_predita + 0.05 * eficiencia
        
        return fitness
    
    def optimize(self):
        """
        Loop principal: QPSO Híbrido Discreto com Mutação GA.
        """
        print("\n" + "=" * 65)
        print("ETAPA 2: OTIMIZAÇÃO QPSO HÍBRIDO DISCRETO")
        print("=" * 65)
        
        history = []
        boias_ativas_hist = []
        start_time = time.time()
        
        for t in range(self.max_iter):
            # 1. Avaliar fitness de todas as partículas
            for i in range(self.num_particles):
                fit = self.avaliar_fitness(self.particles[i])
                
                if fit > self.pbest_fit[i]:
                    self.pbest_fit[i] = fit
                    self.pbest[i] = np.copy(self.particles[i])
                
                if fit > self.gbest_fit:
                    self.gbest_fit = fit
                    self.gbest = np.copy(self.particles[i])
            
            history.append(self.gbest_fit)
            boias_ativas_hist.append(int(np.sum(self.gbest)))
            
            # 2. Mean Best Position (frequência de cada bit no enxame)
            freq = np.mean(self.pbest, axis=0)
            mbest = np.where(freq > 0.5, 1, 0).astype(float)
            
            # 3. Alpha: Decaimento Contração-Expansão
            alpha = 1.0 - 0.5 * (t / self.max_iter)
            
            # 4. Atualização Quântica Discreta
            for i in range(self.num_particles):
                phi = np.random.uniform(0, 1, self.num_wecs)
                
                # Atrator estocástico no domínio real (entre pbest e gbest)
                p_real = phi * self.pbest[i].astype(float) + (1 - phi) * self.gbest.astype(float)
                
                u = np.random.uniform(1e-10, 1, self.num_wecs)
                L = alpha * np.abs(mbest - self.particles[i].astype(float))
                direction = np.where(np.random.rand(self.num_wecs) > 0.5, 1.0, -1.0)
                
                # Estado real provisório via equação de Schrödinger
                x_real = p_real + direction * L * np.log(1.0 / u)
                
                # Discretização via Sigmoid (Colapso da Função de Onda para binário)
                probs = sigmoid(x_real)
                self.particles[i] = np.where(np.random.rand(self.num_wecs) < probs, 1, 0)
            
            # 5. OPERADOR HÍBRIDO: Mutação Genética (GA)
            # Aplica mutação nas piores partículas para escapar de ótimos locais
            median_fit = np.median(self.pbest_fit[self.pbest_fit > -1e5])
            for i in range(self.num_particles):
                if self.pbest_fit[i] < median_fit:
                    if np.random.rand() < 0.25:    # 25% chance de mutação
                        mut_idx = np.random.randint(0, self.num_wecs)
                        self.particles[i, mut_idx] = 1 - self.particles[i, mut_idx]  # Flip
            
            # Garante mínimo de boias ativas via reparação
            for i in range(self.num_particles):
                while np.sum(self.particles[i]) < self.min_boias:
                    idx = np.random.randint(0, self.num_wecs)
                    self.particles[i, idx] = 1
            
            if (t + 1) % 10 == 0 or t == 0:
                n_ativas = int(np.sum(self.gbest))
                print(f"  Iteração {t+1:03d}/{self.max_iter} | "
                      f"Fitness: {self.gbest_fit:,.2f} | "
                      f"Boias Ativas: {n_ativas}/{self.num_wecs}")
        
        exec_time = time.time() - start_time
        print(f"\n[OK] Otimização concluída em {exec_time:.2f}s")
        
        return self.gbest, self.gbest_fit, history, boias_ativas_hist


# =============================================================================
# ETAPA 3: EXECUÇÃO E VISUALIZAÇÃO
# =============================================================================

if __name__ == "__main__":
    NUM_WECS = 49
    DB_FILE = "WEC_Sydney_49.db"
    NUM_PARTICLES = 30
    MAX_ITER = 80
    MIN_BOIAS = 25  # Mínimo de boias que devem ficar ativas
    
    print("=================================================================")
    print("  QPSO HIBRIDO DISCRETO + ML SURROGATE (DATA-DRIVEN)")
    print("  Otimizacao de Selecao de WECs usando Dados Reais dos CSV")
    print("=================================================================")
    
    # --- ETAPA 1: Treina o Modelo AI ---
    modelo, layout_base, power_original, powers_ind, r2, cols = \
        carregar_e_treinar_modelo(DB_FILE, NUM_WECS)
    
    # --- ETAPA 2: Roda o QPSO Híbrido Discreto ---
    optimizer = HybridDiscreteQPSO_WEC(
        num_wecs=NUM_WECS,
        num_particles=NUM_PARTICLES,
        max_iter=MAX_ITER,
        modelo_ml=modelo,
        layout_base=layout_base,
        min_boias_ativas=MIN_BOIAS
    )
    
    best_bits, best_fit, hist, boias_hist = optimizer.optimize()
    
    # --- ETAPA 3: Resultados ---
    boias_ativas = np.where(best_bits == 1)[0] + 1   # 1-indexed
    boias_desligadas = np.where(best_bits == 0)[0] + 1
    n_ativas = len(boias_ativas)
    
    # Prediz potência final do layout otimizado
    coords_final = layout_base.copy()
    for i in range(NUM_WECS):
        if best_bits[i] == 0:
            coords_final[i * 2] = 0
            coords_final[i * 2 + 1] = 0
    potencia_final = modelo.predict(coords_final.reshape(1, -1))[0]
    
    print("\n" + "=" * 65)
    print("ETAPA 3: RESULTADOS FINAIS")
    print("=" * 65)
    print(f"\n  Modelo ML (R² Score):     {r2:.4f}")
    print(f"  Potência Original (49/49): {power_original:,.2f} W")
    print(f"  Potência Otimizada ({n_ativas}/{NUM_WECS}): {potencia_final:,.2f} W")
    print(f"  Boias Ativas:  {n_ativas} de {NUM_WECS}")
    print(f"  Boias Removidas: {len(boias_desligadas)}")
    
    if n_ativas < NUM_WECS and potencia_final > 0:
        eficiencia_por_boia_original = power_original / NUM_WECS
        eficiencia_por_boia_otimizada = potencia_final / n_ativas
        ganho_eficiencia = (eficiencia_por_boia_otimizada / eficiencia_por_boia_original - 1) * 100
        print(f"\n  Eficiência por boia (original):   {eficiencia_por_boia_original:,.2f} W/boia")
        print(f"  Eficiência por boia (otimizada):  {eficiencia_por_boia_otimizada:,.2f} W/boia")
        print(f"  Ganho de Eficiência:              {ganho_eficiencia:+.2f}%")
    
    print(f"\n  Boias ATIVAS (manter): {list(boias_ativas)}")
    print(f"  Boias REMOVIDAS:       {list(boias_desligadas)}")
    
    # Importância de cada boia (via potência individual, se disponível)
    if powers_ind is not None:
        ranking = np.argsort(powers_ind)[::-1]
        print(f"\n  Top 10 boias mais potentes (layout original):")
        for rank, idx in enumerate(ranking[:10]):
            status = "+ ATIVA" if best_bits[idx] == 1 else "- REMOVIDA"
            print(f"    #{rank+1}: Boia {idx+1} = {powers_ind[idx]:,.2f} W  [{status}]")
    
    # --- VISUALIZAÇÃO ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Gráfico 1: Convergência
    axes[0].plot(hist, color='purple', linewidth=2)
    axes[0].set_title("Convergência do QPSO Híbrido Discreto", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Iterações")
    axes[0].set_ylabel("Fitness")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico 2: Layout com boias ativas/inativas
    wecs_coords = layout_base.reshape((NUM_WECS, 2))
    cores = ['green' if b == 1 else 'red' for b in best_bits]
    tamanhos = [100 if b == 1 else 40 for b in best_bits]
    axes[1].scatter(wecs_coords[:, 0], wecs_coords[:, 1], 
                    c=cores, s=tamanhos, edgecolor='black', alpha=0.85)
    axes[1].set_title(f"Seleção de Boias ({n_ativas} ativas / {len(boias_desligadas)} removidas)", 
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel("X (metros)")
    axes[1].set_ylabel("Y (metros)")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    # Legenda manual
    axes[1].scatter([], [], c='green', s=100, edgecolor='black', label='Ativa')
    axes[1].scatter([], [], c='red', s=40, edgecolor='black', label='Removida')
    axes[1].legend(loc='upper right')
    
    # Gráfico 3: Número de boias ativas ao longo das iterações
    axes[2].plot(boias_hist, color='teal', linewidth=2)
    axes[2].set_title("Boias Ativas por Iteração", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Iterações")
    axes[2].set_ylabel("Quantidade de Boias Ativas")
    axes[2].axhline(y=MIN_BOIAS, color='red', linestyle='--', alpha=0.5, label=f'Mínimo ({MIN_BOIAS})')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_img = "qpso_hibrido_discreto_results.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n[!] Gráficos salvos em '{output_img}'")
    
    # --- RELATÓRIO TXT ---
    report_file = "relatorio_hibrido_discreto.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("RELATÓRIO: QPSO HÍBRIDO DISCRETO + ML SURROGATE\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Arquivo de Dados: {DB_FILE}\n")
        f.write(f"Modelo ML: Random Forest (200 árvores)\n")
        f.write(f"R² Score (Validação): {r2:.4f}\n\n")
        f.write(f"--- CONFIGURAÇÃO DO QPSO ---\n")
        f.write(f"Tipo: Híbrido Discreto (Quântico + Genético)\n")
        f.write(f"Partículas: {NUM_PARTICLES}\n")
        f.write(f"Iterações: {MAX_ITER}\n")
        f.write(f"Mínimo de boias ativas: {MIN_BOIAS}\n\n")
        f.write(f"--- RESULTADOS ---\n")
        f.write(f"Potência Original (todas {NUM_WECS} boias): {power_original:,.2f} W\n")
        f.write(f"Potência Selecionada ({n_ativas} boias):   {potencia_final:,.2f} W\n")
        f.write(f"Boias ATIVAS: {list(boias_ativas)}\n")
        f.write(f"Boias REMOVIDAS: {list(boias_desligadas)}\n\n")
        if n_ativas < NUM_WECS and potencia_final > 0:
            f.write(f"Eficiência por boia (original):  {power_original/NUM_WECS:,.2f} W/boia\n")
            f.write(f"Eficiência por boia (otimizada): {potencia_final/n_ativas:,.2f} W/boia\n")
        f.write(f"\n--- CONCLUSÃO ---\n")
        f.write(f"O algoritmo híbrido discreto selecionou {n_ativas} de {NUM_WECS} boias,\n")
        f.write(f"usando um modelo de IA treinado exclusivamente nos dados reais dos sensores BEM.\n")
        f.write(f"Nenhuma fórmula física foi inventada: toda a predição vem dos dados do CSV.\n")
    
    print(f"[!] Relatório salvo em '{report_file}'")
