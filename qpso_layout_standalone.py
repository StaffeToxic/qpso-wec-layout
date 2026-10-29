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

def _make_coord_cols(num_wecs):
    cols = []
    for i in range(1, num_wecs + 1):
        cols.append(f'X{i}')
        cols.append(f'Y{i}')
    return cols


def carregar_e_treinar_modelo(db_path, num_wecs=49, n_estimators=100, n_jobs=-1, verbose=True):
    """
    Carrega dados do SQLite, treina Random Forest e retorna modelo + metadados.
    Retorna também os limites (min, max) de cada coordenada para confinamento
    e o dataset (X, y) usado para semear a população inicial.
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

    # Limites reais do dataset para cada coordenada
    bounds_min = X.min(axis=0)
    bounds_max = X.max(axis=0)

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

    return modelo, best_layout, best_power, r2, mae, coord_cols, bounds_min, bounds_max, X, y


# =============================================================================
# QPSO CONTÍNUO PARA OTIMIZAÇÃO DE LAYOUT
# =============================================================================

class ContinuousQPSO_WEC:
    """
    QPSO contínuo para otimizar as posições (X,Y) de todas as boias.
    Todas as 49 boias permanecem ativas — o objetivo é encontrar o layout
    espacial que maximiza a potência total predita pelo modelo RF surrogate.

    Equações de atualização (QPSO padrão contínuo):
        p_i = φ * pbest_i + (1 - φ) * gbest           (atrator local)
        mbest = (1/N) * Σ pbest_i                       (mean best)
        x_i = p_i ± α * |mbest - x_i| * ln(1/u)       (atualização quântica)
        α = 1.0 → 0.5 (decaimento linear)               (coeficiente de contração-expansão)
    """

    def __init__(self, num_wecs, num_particles, max_iter, modelo_rf,
                 layout_ref, bounds_min, bounds_max, dataset_X, dataset_y, levy_mu=0.05):
        if num_wecs <= 0:
            raise ValueError("`num_wecs` deve ser > 0.")
        if num_particles <= 0:
            raise ValueError("`num_particles` deve ser > 0.")
        if max_iter <= 0:
            raise ValueError("`max_iter` deve ser > 0.")
        if len(dataset_y) == 0 or len(dataset_X) == 0:
            raise ValueError("`dataset_X` e `dataset_y` nao podem ser vazios.")

        self.num_wecs = num_wecs
        self.num_dims = num_wecs * 2  # X,Y por boia
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.modelo = modelo_rf
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.levy_mu = levy_mu

        # Selecionar layouts variados do dataset como sementes (percentil 50-90)
        sorted_idx = np.argsort(dataset_y)
        p50 = int(len(sorted_idx) * 0.50)
        p90 = int(len(sorted_idx) * 0.90)
        p90 = max(p90, p50 + 1)
        seed_indices = sorted_idx[p50:p90]
        if len(seed_indices) == 0:
            raise ValueError("Faixa de sementes vazia; verifique o tamanho do dataset.")
        seed_layouts = dataset_X[seed_indices]

        # Inicializar: crossover entre layouts do dataset + perturbacao
        self.particles = np.zeros((num_particles, self.num_dims))
        for i in range(num_particles):
            p1 = seed_layouts[np.random.randint(len(seed_layouts))]
            p2 = seed_layouts[np.random.randint(len(seed_layouts))]
            mask = np.random.rand(self.num_dims) < 0.5
            child = np.where(mask, p1, p2)
            noise = np.random.normal(0, 0.10, self.num_dims) * (bounds_max - bounds_min)
            self.particles[i] = np.clip(child + noise, bounds_min, bounds_max)

        self.pbest = np.copy(self.particles)
        self.pbest_fit = np.full(num_particles, -np.inf)
        self.gbest = layout_ref.copy()
        self.gbest_fit = -np.inf
        self.history = []
        self.layout_ref = layout_ref.copy()

    def _avaliar(self, layout):
        """Avalia a potencia predita pelo RF para um layout continuo."""
        return self.modelo.predict(layout.reshape(1, -1))[0]

    def _avaliar_batch(self, layouts):
        """Avalia batch de layouts de uma vez (muito mais rapido)."""
        return self.modelo.predict(layouts)

    def _confinar(self, x):
        """Garante que todas as coordenadas fiquem dentro dos limites reais."""
        return np.clip(x, self.bounds_min, self.bounds_max)

    def optimize(self):
        history = []

        # Avaliação inicial
        for i in range(self.num_particles):
            fit = self._avaliar(self.particles[i])
            self.pbest_fit[i] = fit
            if fit > self.gbest_fit:
                self.gbest_fit = fit
                self.gbest = np.copy(self.particles[i])

        for t in range(self.max_iter):
            fits = np.zeros(self.num_particles)

            # Avaliar TODAS as particulas em batch (rapido)
            fits = self._avaliar_batch(self.particles)

            for i in range(self.num_particles):
                if fits[i] > self.pbest_fit[i]:
                    self.pbest_fit[i] = fits[i]
                    self.pbest[i] = np.copy(self.particles[i])
                if fits[i] > self.gbest_fit:
                    self.gbest_fit = fits[i]
                    self.gbest = np.copy(self.particles[i])

            # Mean best position
            mbest = np.mean(self.pbest, axis=0)

            # Coeficiente de contração-expansão: decai de 1.0 → 0.5
            alpha = 1.0 - 0.5 * (t / self.max_iter)

            # Elitismo: preservar as top-3 partículas
            top_k = min(3, self.num_particles)
            top_indices = np.argsort(fits)[-top_k:]
            elites = np.copy(self.particles[top_indices])
            elite_slots = np.arange(top_k)

            # Atualização QPSO contínua
            for i in range(self.num_particles):
                phi = np.random.uniform(0, 1, self.num_dims)
                p = phi * self.pbest[i] + (1 - phi) * self.gbest  # atrator local

                u = np.random.uniform(1e-10, 1, self.num_dims)
                L = alpha * np.abs(mbest - self.particles[i])
                direction = np.where(np.random.rand(self.num_dims) > 0.5, 1.0, -1.0)

                new_pos = p + direction * L * np.log(1.0 / u)
                new_pos = np.where(np.isfinite(new_pos), new_pos, p)
                self.particles[i] = self._confinar(new_pos)

            # Restaurar elites
            self.particles[:len(elites)] = elites

            # Voos de Lévy para escape de ótimos locais (metade inferior)
            levy_rate = self.levy_mu * (1 - 0.5 * (t / self.max_iter))
            sorted_fits = np.argsort(fits)
            n_levy = max(1, self.num_particles // 3)
            levy_candidates = [i for i in sorted_fits if i not in elite_slots][:n_levy]
            for i in levy_candidates:
                if np.random.rand() < levy_rate:
                    levy_beta = 1.5
                    sigma_u = (
                        math.gamma(1 + levy_beta) * math.sin(math.pi * levy_beta / 2)
                        / (math.gamma((1 + levy_beta) / 2) * levy_beta * 2 ** ((levy_beta - 1) / 2))
                    ) ** (1 / levy_beta)
                    u_l = np.random.normal(0, sigma_u, self.num_dims)
                    v_l = np.random.normal(0, 1, self.num_dims)
                    levy_step = u_l / (np.abs(v_l) ** (1 / levy_beta))
                    # Escalar passo proporcional ao range do domínio
                    step_scale = 0.05 * (self.bounds_max - self.bounds_min)
                    self.particles[i] = self._confinar(
                        self.particles[i] + levy_step * step_scale
                    )

            # Re-avaliar em batch apos Levy
            fits_post = self._avaliar_batch(self.particles)
            for i in range(self.num_particles):
                if fits_post[i] > self.pbest_fit[i]:
                    self.pbest_fit[i] = fits_post[i]
                    self.pbest[i] = np.copy(self.particles[i])
                if fits_post[i] > self.gbest_fit:
                    self.gbest_fit = fits_post[i]
                    self.gbest = np.copy(self.particles[i])

            # Registrar histórico no estado final da iteração.
            history.append(self.gbest_fit)

            if (t + 1) % 20 == 0 or t == 0:
                print(f"      Iter {t+1:>3d}/{self.max_iter}: Best = {self.gbest_fit:,.2f} W")

        self.history = history
        return self.gbest, self.gbest_fit, history


# =============================================================================
# ROTINA PARA UM DATASET (isolada — sem estado compartilhado)
# =============================================================================

def run_dataset(db_path, dataset_name, num_wecs=49, num_iter=150, num_population=50):
    """
    Executa o pipeline completo para UM dataset:
    treina RF, roda QPSO, gera relatório e salva gráfico.
    Todas as variáveis são locais — sem efeitos colaterais.
    """
    print("\n" + "=" * 60)
    print(f" QPSO CONTINUO ({num_wecs} WECs) - {dataset_name.upper()}")
    print("=" * 60 + "\n")

    # --- Treinar modelo ---
    modelo, layout_base, best_power, r2, mae, _, bounds_min, bounds_max, dataset_X, dataset_y = carregar_e_treinar_modelo(
        db_path, num_wecs, n_estimators=150
    )
    print(f"    R²={r2:.4f} | MAE={mae:,.0f} W | Best original={best_power:,.2f} W\n")

    power_original_rf = modelo.predict(layout_base.reshape(1, -1))[0]
    print(f"[*] Potência RF do layout original: {power_original_rf:,.2f} W\n")

    # --- Executar QPSO ---
    print("[1] Executando QPSO Continuo (otimizacao de posicoes)...")
    t0 = time.time()
    qpso = ContinuousQPSO_WEC(
        num_wecs, num_population, num_iter, modelo,
        layout_base, bounds_min, bounds_max, dataset_X, dataset_y, levy_mu=0.05,
    )
    gbest_layout, fit_otimizado, hist_qpso = qpso.optimize()
    elapsed = time.time() - t0
    print(f"    Potência otimizada: {fit_otimizado:,.2f} W | Tempo: {elapsed:.1f}s")
    ganho = fit_otimizado - power_original_rf
    ganho_pct = (ganho / power_original_rf) * 100 if power_original_rf > 0 else 0
    print(f"    Ganho vs original:  {ganho:+,.2f} W ({ganho_pct:+.2f}%)\n")

    # --- Relatório de deslocamentos ---
    print("[2] Deslocamentos por Boia (Top-10 maiores movimentos):")
    deslocamentos = []
    for i in range(num_wecs):
        x_orig = layout_base[i * 2]
        y_orig = layout_base[i * 2 + 1]
        x_novo = gbest_layout[i * 2]
        y_novo = gbest_layout[i * 2 + 1]
        dist = math.sqrt((x_novo - x_orig) ** 2 + (y_novo - y_orig) ** 2)
        deslocamentos.append((i, x_orig, y_orig, x_novo, y_novo, dist))

    deslocamentos.sort(key=lambda d: d[5], reverse=True)
    for idx, x0, y0, x1, y1, dist in deslocamentos[:10]:
        print(f"      WEC{idx+1:>2d}: ({x0:.0f},{y0:.0f}) -> ({x1:.0f},{y1:.0f})  d={dist:.1f}m")
    media_desloc = np.mean([d[5] for d in deslocamentos])
    print(f"    ------------------------------------")
    print(f"    Deslocamento medio: {media_desloc:.1f} m\n")

    # --- Gráfico 3 painéis ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'QPSO Contínuo — {dataset_name}', fontsize=14, fontweight='bold', y=1.02)

    # Painel 1: Convergência
    ax1 = axes[0]
    ax1.plot(hist_qpso, color='#2196F3', linewidth=2.5)
    ax1.axhline(y=power_original_rf, color='red', linestyle='--', linewidth=1.5,
               label=f'Original ({power_original_rf:,.0f} W)', alpha=0.7)
    ax1.set_title('Convergência', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Iterações')
    ax1.set_ylabel('Potência (W)')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Painel 2: Layout Original vs Otimizado
    ax2 = axes[1]
    orig_x = layout_base[0::2]
    orig_y = layout_base[1::2]
    otim_x = gbest_layout[0::2]
    otim_y = gbest_layout[1::2]

    ax2.scatter(orig_x, orig_y, c='gray', s=50, alpha=0.5, label='Original', zorder=2)
    ax2.scatter(otim_x, otim_y, c='green', s=60, edgecolors='darkgreen',
               linewidths=0.8, label='Otimizado', zorder=3)
    for i in range(num_wecs):
        dx = otim_x[i] - orig_x[i]
        dy = otim_y[i] - orig_y[i]
        if math.sqrt(dx**2 + dy**2) > 5:
            ax2.annotate('', xy=(otim_x[i], otim_y[i]), xytext=(orig_x[i], orig_y[i]),
                        arrowprops=dict(arrowstyle='->', color='#FF5722', lw=1.0, alpha=0.6))

    ax2.set_title(f'Layout: Original → Otimizado ({num_wecs} boias)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('X (metros)')
    ax2.set_ylabel('Y (metros)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Painel 3: Histograma de Deslocamentos
    ax3 = axes[2]
    dists = [d[5] for d in deslocamentos]
    ax3.hist(dists, bins=15, color='#4CAF50', edgecolor='white', alpha=0.8)
    ax3.axvline(x=media_desloc, color='red', linestyle='--', linewidth=1.5,
               label=f'Média ({media_desloc:.1f} m)', alpha=0.7)
    ax3.set_title('Distribuição de Deslocamentos', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Deslocamento (metros)')
    ax3.set_ylabel('Número de Boias')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = dataset_name.lower().replace(' ', '_')
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'qpso_layout_{safe_name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[!] Gráfico salvo como '{out_path}'")
    print(f"\n[+] {dataset_name} concluído com sucesso!")

    return {
        'dataset': dataset_name,
        'r2': r2, 'mae': mae,
        'power_original': power_original_rf,
        'power_otimizado': fit_otimizado,
        'ganho_w': ganho, 'ganho_pct': ganho_pct,
        'media_desloc': media_desloc,
        'tempo': elapsed,
    }


# =============================================================================
# ROTINA PRINCIPAL — EXECUÇÃO SEQUENCIAL DOS DATASETS
# =============================================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    DATASETS = [
        {"db": os.path.join(script_dir, "WEC_Sydney_49.db"), "name": "Sydney 49"},
        {"db": os.path.join(script_dir, "WEC_Perth_49.db"),  "name": "Perth 49"},
    ]

    resultados = []

    for ds in DATASETS:
        if not os.path.exists(ds["db"]):
            print(f"\n[!] Dataset '{ds['db']}' não encontrado. Pulando {ds['name']}.")
            continue
        resultado = run_dataset(ds["db"], ds["name"])
        resultados.append(resultado)

    # =========================================================================
    # RESUMO COMPARATIVO FINAL
    # =========================================================================
    if len(resultados) > 1:
        print("\n\n" + "=" * 60)
        print(" RESUMO COMPARATIVO")
        print("=" * 60)
        print(f"  {'Dataset':<15} {'R²':>6} {'Original (W)':>14} {'Otimizado (W)':>14} {'Ganho':>10} {'Desloc.':>8}")
        print(f"  {'-'*13:<15} {'-'*6:>6} {'-'*14:>14} {'-'*14:>14} {'-'*10:>10} {'-'*8:>8}")
        for r in resultados:
            print(f"  {r['dataset']:<15} {r['r2']:>6.4f} {r['power_original']:>14,.0f} "
                  f"{r['power_otimizado']:>14,.0f} {r['ganho_pct']:>+9.2f}% {r['media_desloc']:>7.1f}m")
        print()

    print("[+] Todos os datasets concluídos!")
