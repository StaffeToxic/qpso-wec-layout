# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

from __future__ import annotations

import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from qpso_layout_standalone import carregar_e_treinar_modelo


class PureContinuousQPSO_WEC:
    """
    QPSO continuo puro para otimizar as coordenadas X/Y das boias.

    Atualizacao usada:
        p_i = phi * pbest_i + (1 - phi) * gbest
        mbest = mean(pbest)
        x_i = p_i +/- alpha * abs(mbest - x_i) * ln(1/u)

    Todas as boias permanecem ativas; o objetivo e maximizar a potencia
    predita pelo RandomForest surrogate.
    """

    def __init__(
        self,
        num_wecs,
        num_particles,
        max_iter,
        modelo_rf,
        layout_ref,
        bounds_min,
        bounds_max,
        dataset_X,
        dataset_y,
        random_state=None,
        alpha_start=1.0,
        alpha_end=0.5,
        noise_scale=0.10,
    ):
        if num_wecs <= 0:
            raise ValueError("`num_wecs` deve ser > 0.")
        if num_particles <= 0:
            raise ValueError("`num_particles` deve ser > 0.")
        if max_iter <= 0:
            raise ValueError("`max_iter` deve ser > 0.")

        self.num_wecs = int(num_wecs)
        self.num_dims = self.num_wecs * 2
        self.num_particles = int(num_particles)
        self.max_iter = int(max_iter)
        self.modelo = modelo_rf
        self.layout_ref = np.asarray(layout_ref, dtype=float).copy()
        self.bounds_min = np.asarray(bounds_min, dtype=float)
        self.bounds_max = np.asarray(bounds_max, dtype=float)
        self.dataset_X = np.asarray(dataset_X, dtype=float)
        self.dataset_y = np.asarray(dataset_y, dtype=float)
        self.alpha_start = float(alpha_start)
        self.alpha_end = float(alpha_end)
        self.noise_scale = float(noise_scale)
        self.rng = np.random.default_rng(random_state)

        if self.layout_ref.shape != (self.num_dims,):
            raise ValueError("`layout_ref` deve ter tamanho num_wecs * 2.")
        if self.bounds_min.shape != (self.num_dims,) or self.bounds_max.shape != (self.num_dims,):
            raise ValueError("`bounds_min` e `bounds_max` devem ter tamanho num_wecs * 2.")
        if self.dataset_X.ndim != 2 or self.dataset_X.shape[1] != self.num_dims:
            raise ValueError("`dataset_X` deve ter shape (n_amostras, num_wecs * 2).")
        if len(self.dataset_X) == 0 or len(self.dataset_y) == 0:
            raise ValueError("`dataset_X` e `dataset_y` nao podem ser vazios.")

        self.particles = self._init_particles()
        self.pbest = self.particles.copy()
        self.pbest_fit = np.full(self.num_particles, -np.inf)
        self.gbest = self.layout_ref.copy()
        self.gbest_fit = -np.inf
        self.history = []

    def _init_particles(self):
        sorted_idx = np.argsort(self.dataset_y)
        p50 = int(len(sorted_idx) * 0.50)
        p90 = max(int(len(sorted_idx) * 0.90), p50 + 1)
        seed_indices = sorted_idx[p50:p90]
        if len(seed_indices) == 0:
            seed_indices = sorted_idx
        seed_layouts = self.dataset_X[seed_indices]

        span = np.maximum(self.bounds_max - self.bounds_min, 1e-9)
        particles = np.zeros((self.num_particles, self.num_dims), dtype=float)
        particles[0] = self._confinar(self.layout_ref)
        for i in range(1, self.num_particles):
            seed = seed_layouts[self.rng.integers(0, len(seed_layouts))]
            noise = self.rng.normal(0.0, self.noise_scale, self.num_dims) * span
            particles[i] = self._confinar(seed + noise)
        return particles

    def _avaliar_batch(self, layouts):
        return np.asarray(self.modelo.predict(layouts), dtype=float)

    def _confinar(self, x):
        return np.clip(x, self.bounds_min, self.bounds_max)

    def _update_bests(self, fits):
        improved = fits > self.pbest_fit
        if np.any(improved):
            self.pbest_fit[improved] = fits[improved]
            self.pbest[improved] = self.particles[improved]

        best_idx = int(np.argmax(fits))
        if fits[best_idx] > self.gbest_fit:
            self.gbest_fit = float(fits[best_idx])
            self.gbest = self.particles[best_idx].copy()

    def optimize(self, verbose=True):
        history = []

        fits = self._avaliar_batch(self.particles)
        self._update_bests(fits)

        for t in range(self.max_iter):
            progress = t / max(1, self.max_iter - 1)
            alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
            mbest = np.mean(self.pbest, axis=0)

            phi = self.rng.random((self.num_particles, self.num_dims))
            attractor = phi * self.pbest + (1.0 - phi) * self.gbest
            u = self.rng.uniform(1e-10, 1.0, (self.num_particles, self.num_dims))
            direction = np.where(
                self.rng.random((self.num_particles, self.num_dims)) > 0.5,
                1.0,
                -1.0,
            )
            step = alpha * np.abs(mbest - self.particles) * np.log(1.0 / u)
            new_positions = attractor + direction * step
            new_positions = np.where(np.isfinite(new_positions), new_positions, attractor)
            self.particles = self._confinar(new_positions)

            fits = self._avaliar_batch(self.particles)
            self._update_bests(fits)
            history.append(self.gbest_fit)

            if verbose and ((t + 1) % 20 == 0 or t == 0):
                print(f"      Iter {t + 1:>3d}/{self.max_iter}: Best = {self.gbest_fit:,.2f} W")

        self.history = history
        return self.gbest, self.gbest_fit, history


def _plot_result(dataset_name, layout_base, gbest_layout, hist_qpso, power_original_rf, deslocamentos):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"QPSO Puro Continuo - {dataset_name}", fontsize=14, fontweight="bold", y=1.02)

    ax1 = axes[0]
    ax1.plot(hist_qpso, color="#1f77b4", linewidth=2.5)
    ax1.axhline(
        y=power_original_rf,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Original ({power_original_rf:,.0f} W)",
        alpha=0.7,
    )
    ax1.set_title("Convergencia", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Iteracoes")
    ax1.set_ylabel("Potencia RF (W)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    orig_x = layout_base[0::2]
    orig_y = layout_base[1::2]
    otim_x = gbest_layout[0::2]
    otim_y = gbest_layout[1::2]
    ax2.scatter(orig_x, orig_y, c="gray", s=50, alpha=0.5, label="Original", zorder=2)
    ax2.scatter(otim_x, otim_y, c="green", s=60, edgecolors="darkgreen", linewidths=0.8, label="QPSO puro", zorder=3)
    for i in range(len(orig_x)):
        dx = otim_x[i] - orig_x[i]
        dy = otim_y[i] - orig_y[i]
        if math.sqrt(dx**2 + dy**2) > 5:
            ax2.annotate(
                "",
                xy=(otim_x[i], otim_y[i]),
                xytext=(orig_x[i], orig_y[i]),
                arrowprops=dict(arrowstyle="->", color="#ff5722", lw=1.0, alpha=0.6),
            )
    ax2.set_title("Layout original -> QPSO puro", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    dists = [d[5] for d in deslocamentos]
    media_desloc = float(np.mean(dists)) if dists else 0.0
    ax3.hist(dists, bins=15, color="#4caf50", edgecolor="white", alpha=0.8)
    ax3.axvline(
        x=media_desloc,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Media ({media_desloc:.1f} m)",
        alpha=0.7,
    )
    ax3.set_title("Deslocamentos", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Deslocamento (m)")
    ax3.set_ylabel("Numero de boias")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_dataset(
    db_path,
    dataset_name,
    num_wecs=49,
    num_iter=150,
    num_population=50,
    random_state=42,
):
    print("\n" + "=" * 60)
    print(f" QPSO PURO CONTINUO ({num_wecs} WECs) - {dataset_name.upper()}")
    print("=" * 60 + "\n")

    modelo, layout_base, best_power, r2, mae, _, bounds_min, bounds_max, dataset_X, dataset_y = carregar_e_treinar_modelo(
        db_path,
        num_wecs,
        n_estimators=150,
    )
    print(f"    R2={r2:.4f} | MAE={mae:,.0f} W | Best dataset={best_power:,.2f} W\n")

    power_original_rf = float(modelo.predict(layout_base.reshape(1, -1))[0])
    print(f"[*] Potencia RF do layout original: {power_original_rf:,.2f} W\n")

    print("[1] Executando QPSO puro continuo...")
    t0 = time.time()
    qpso = PureContinuousQPSO_WEC(
        num_wecs=num_wecs,
        num_particles=num_population,
        max_iter=num_iter,
        modelo_rf=modelo,
        layout_ref=layout_base,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        dataset_X=dataset_X,
        dataset_y=dataset_y,
        random_state=random_state,
    )
    gbest_layout, fit_otimizado, hist_qpso = qpso.optimize()
    elapsed = time.time() - t0

    ganho = fit_otimizado - power_original_rf
    ganho_pct = (ganho / power_original_rf) * 100 if power_original_rf > 0 else 0.0
    print(f"    Potencia otimizada: {fit_otimizado:,.2f} W | Tempo: {elapsed:.1f}s")
    print(f"    Ganho RF vs original: {ganho:+,.2f} W ({ganho_pct:+.2f}%)\n")

    print("[2] Deslocamentos por boia (Top-10 maiores movimentos):")
    deslocamentos = []
    for i in range(num_wecs):
        x_orig = layout_base[i * 2]
        y_orig = layout_base[i * 2 + 1]
        x_new = gbest_layout[i * 2]
        y_new = gbest_layout[i * 2 + 1]
        dist = math.sqrt((x_new - x_orig) ** 2 + (y_new - y_orig) ** 2)
        deslocamentos.append((i, x_orig, y_orig, x_new, y_new, dist))
    deslocamentos.sort(key=lambda d: d[5], reverse=True)

    for idx, x0, y0, x1, y1, dist in deslocamentos[:10]:
        print(f"      WEC{idx + 1:>2d}: ({x0:.0f},{y0:.0f}) -> ({x1:.0f},{y1:.0f})  d={dist:.1f}m")

    media_desloc = float(np.mean([d[5] for d in deslocamentos]))
    print("    ------------------------------------")
    print(f"    Deslocamento medio: {media_desloc:.1f} m\n")

    fig = _plot_result(dataset_name, layout_base, gbest_layout, hist_qpso, power_original_rf, deslocamentos)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    safe_name = dataset_name.lower().replace(" ", "_")
    out_path = os.path.join(script_dir, f"qpso_layout_puro_{safe_name}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[!] Grafico salvo como '{out_path}'")

    return {
        "dataset": dataset_name,
        "r2": float(r2),
        "mae": float(mae),
        "power_original": float(power_original_rf),
        "power_otimizado": float(fit_otimizado),
        "ganho_w": float(ganho),
        "ganho_pct": float(ganho_pct),
        "media_desloc": media_desloc,
        "tempo": float(elapsed),
        "history": [float(v) for v in hist_qpso],
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = [
        {"db": os.path.join(script_dir, "WEC_Sydney_49.db"), "name": "Sydney 49"},
        {"db": os.path.join(script_dir, "WEC_Perth_49.db"), "name": "Perth 49"},
    ]

    resultados = []
    for ds in datasets:
        if not os.path.exists(ds["db"]):
            print(f"\n[!] Dataset '{ds['db']}' nao encontrado. Pulando {ds['name']}.")
            continue
        resultados.append(run_dataset(ds["db"], ds["name"]))

    if len(resultados) > 1:
        print("\n\n" + "=" * 60)
        print(" RESUMO COMPARATIVO - QPSO PURO")
        print("=" * 60)
        print(f"  {'Dataset':<15} {'R2':>6} {'Original (W)':>14} {'QPSO puro (W)':>15} {'Ganho':>10} {'Desloc.':>8}")
        print(f"  {'-' * 13:<15} {'-' * 6:>6} {'-' * 14:>14} {'-' * 15:>15} {'-' * 10:>10} {'-' * 8:>8}")
        for r in resultados:
            print(
                f"  {r['dataset']:<15} {r['r2']:>6.4f} {r['power_original']:>14,.0f} "
                f"{r['power_otimizado']:>15,.0f} {r['ganho_pct']:>+9.2f}% {r['media_desloc']:>7.1f}m"
            )
        out_json = os.path.join(script_dir, "resultado_qpso_layout_puro.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        print(f"\n[!] Resultados salvos em '{out_json}'")

    print("[+] Todos os datasets concluidos!")


if __name__ == "__main__":
    main()
