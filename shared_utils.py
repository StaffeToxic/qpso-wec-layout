# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""
=============================================================================
UTILITÁRIOS COMPARTILHADOS — QML/WEC
=============================================================================
Funções comuns usadas em múltiplos algoritmos para garantir consistência.
=============================================================================
"""
import numpy as np
import pandas as pd
import os
import sqlite3
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def sigmoid(x):
    """Mapeia valor real para probabilidade [0, 1] com clipping numérico."""
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


def avaliar_fitness(bits, shap_contribs, shap_base, num_wecs, min_boias):
    """
    Avalia a qualidade de uma seleção de boias via decomposição SHAP.

    Potência estimada = base_value + Σ shap_contrib_i (boias ativas)
    Fitness = 0.7 * potencia + 0.3 * eficiencia_por_boia

    Usa SHAP values do melhor layout para decompor a predição do modelo
    em contribuições aditivas por boia, evitando predições Out-of-Distribution
    causadas por zerar coordenadas de boias removidas.
    """
    if np.sum(bits) < min_boias:
        return -1e6

    # Potência via decomposição SHAP aditiva por boia
    potencia = shap_base + np.sum(shap_contribs[bits == 1])
    n_ativas = int(np.sum(bits))
    eficiencia = potencia / n_ativas if n_ativas > 0 else 0

    return 0.7 * potencia + 0.3 * eficiencia


def aplicar_mutacao(populacao, fits, num_wecs, min_boias, p_mut=0.25):
    """
    Operador Genético de Mutação com taxa adaptativa.
    Aplica inversão de bit nas amostras com fitness abaixo da mediana.
    Garante o mínimo de boias ativas após mutação.
    """
    valid_fits = fits[fits > -1e5]
    if len(valid_fits) == 0:
        return populacao

    mediana = np.median(valid_fits)
    for i in range(len(populacao)):
        if fits[i] < mediana:
            if np.random.rand() < p_mut:
                # Multi-bit flip: proporcional à dimensão para melhor escape
                n_flips = max(1, int(p_mut * num_wecs * 0.1))
                for _ in range(n_flips):
                    idx = np.random.randint(0, num_wecs)
                    populacao[i, idx] = 1 - populacao[i, idx]

    # Reparar restrições de mínimo de boias
    for i in range(len(populacao)):
        n_ativas = int(np.sum(populacao[i]))
        if n_ativas >= min_boias:
            continue
        deficit = int(min_boias - n_ativas)
        zeros = np.where(populacao[i] == 0)[0]
        if len(zeros) > 0:
            chosen = np.random.choice(zeros, min(deficit, len(zeros)), replace=False)
            populacao[i, chosen] = 1

    return populacao


def gaussian_mutation(populacao, fits, dim, median_threshold=0.5, p_mut=0.2, sigma=1.0):
    """
    Mutação Gaussiana: adiciona ruído N(0, sigma) a genes de partículas ruins.
    Usada em algoritmos contínuos ou híbridos com espaço real internamente.
    """
    valid_fits = fits[fits > -1e5]
    if len(valid_fits) == 0:
        return populacao

    mediana = np.median(valid_fits)
    for i in range(len(populacao)):
        if fits[i] < mediana and np.random.rand() < p_mut:
            idx = np.random.randint(0, dim)
            populacao[i][idx] += np.random.normal(0, sigma)

    return populacao


# =============================================================================
# ML SURROGATE
# =============================================================================

def carregar_e_treinar_modelo(db_path, num_wecs=49, n_estimators=100, verbose=True):
    """
    Carrega dados do SQLite, treina Random Forest e retorna modelo + metadados.

    Returns:
        modelo, best_layout, best_power, r2, mae, coord_cols
    """
    if verbose:
        print(f"[*] Treinando ML Surrogate ({num_wecs} boias, {n_estimators} estimadores)...")

    if not os.path.exists(db_path):
        if verbose:
            print(f"[AVISO] Banco '{db_path}' não encontrado. Simulando dados.")
        X = np.random.rand(100, num_wecs * 2) * 500
        y = np.sum(X, axis=1) * np.random.uniform(0.8, 1.2, 100)
        best_idx = np.argmax(y)
        best_layout = X[best_idx]
        best_power = np.max(y)
        return modelo_dummy(best_layout, best_power), best_layout, best_power, 0.0, 0.0, _make_coord_cols(num_wecs)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM wec_data", conn)
    conn.close()

    coord_cols = []
    for i in range(1, num_wecs + 1):
        coord_cols.append(f'X{i}')
        coord_cols.append(f'Y{i}')

    missing = [c for c in coord_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing[:5]}...")

    X = df[coord_cols].values
    y = df['Total_Power'].values

    best_idx = np.argmax(y)
    best_layout = X[best_idx]
    best_power = y[best_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    if verbose:
        print(f"    R²={r2:.4f} | MAE={mae:,.0f} W | Best original={best_power:,.2f} W")

    return modelo, best_layout, best_power, r2, mae, coord_cols


def calcular_contribuicoes_shap(modelo, layout_ref, num_wecs, verbose=True):
    """
    Calcula a contribuição marginal de cada boia usando decomposição SHAP.

    Usa TreeExplainer no layout de referência (melhor layout do dataset)
    para decompor a predição em contribuições aditivas por boia:
        f(x) = base_value + Σ contrib_i

    Quando uma boia é removida, sua contribuição é subtraída:
        f(x sem boia_i) ≈ base_value + Σ_{j≠i} contrib_j

    Isso evita predições Out-of-Distribution causadas por zerar coordenadas.

    Args:
        modelo: RandomForest treinado
        layout_ref: array (num_wecs*2,) com coordenadas do melhor layout
        num_wecs: número de WECs
        verbose: se True, imprime diagnósticos

    Returns:
        shap_contribs: array (num_wecs,) com contribuição de cada boia (W)
        shap_base: float, valor base (E[f(X)]) em Watts
    """
    import shap

    if verbose:
        print(f"[*] Calculando SHAP contributions ({num_wecs} WECs)...")

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(layout_ref.reshape(1, -1))  # (1, num_wecs*2)
    ev = explainer.expected_value
    base_value = float(ev.item()) if hasattr(ev, 'item') else float(ev)

    # Agrupar por WEC: contrib_i = SHAP(X_i) + SHAP(Y_i)
    wec_contribs = np.zeros(num_wecs)
    for i in range(num_wecs):
        wec_contribs[i] = shap_values[0, i * 2] + shap_values[0, i * 2 + 1]

    # Sanity check: f(x) = base + Σ SHAP_j
    total_shap = np.sum(wec_contribs)
    rf_pred = modelo.predict(layout_ref.reshape(1, -1))[0]
    erro_shap = abs(base_value + total_shap - rf_pred)

    if verbose:
        print(f"    SHAP base (E[f(X)]): {base_value:,.0f} W")
        print(f"    Sum SHAP contribs:   {total_shap:,.0f} W")
        print(f"    Predição RF:         {rf_pred:,.0f} W")
        print(f"    Erro decomposição:   {erro_shap:,.0f} W")

        # Top contribuidores
        sorted_idx = np.argsort(wec_contribs)
        print(f"    Top-5 boias (maior contribuição):")
        for idx in sorted_idx[-5:][::-1]:
            print(f"      WEC{idx+1}: {wec_contribs[idx]:+,.0f} W")
        print(f"    Bottom-3 boias (menor contribuição):")
        for idx in sorted_idx[:3]:
            print(f"      WEC{idx+1}: {wec_contribs[idx]:+,.0f} W")

    return wec_contribs, base_value


def modelo_dummy(best_layout, best_power):
    """Modelo fallback que retorna valores dummy."""
    class DummyPredictor:
        def predict(self, X):
            return np.ones(X.shape[0]) * best_power
    return DummyPredictor()


def _make_coord_cols(num_wecs):
    cols = []
    for i in range(1, num_wecs + 1):
        cols.append(f'X{i}')
        cols.append(f'Y{i}')
    return cols
