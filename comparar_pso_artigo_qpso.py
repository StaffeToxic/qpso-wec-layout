# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

from __future__ import annotations

import json
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from qpso_layout_puro_standalone import PureContinuousQPSO_WEC, carregar_e_treinar_modelo


class ArticleBPSOGrid_WEC:
    """
    Nucleo do Enhanced Binary PSO citado no artigo base.

    Adaptacao ao projeto:
    - cada bit representa uma celula candidata de uma grade espacial;
    - exatamente num_wecs bits ficam ativos, formando um layout completo;
    - a transferencia V-shaped do artigo decide se cada bit sera invertido.

    Parametros centrais usados pelo artigo:
        c1 = c2 = 2
        w decai linearmente de 0.9 para 0.4
        velocidade inicial em escala 2
        T(v) = abs((2/pi) * arctan((pi/2) * v))
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
        grid_step=50.0,
        c1=2.0,
        c2=2.0,
        w_max=0.9,
        w_min=0.4,
        initial_velocity=2.0,
        random_state=None,
    ):
        if num_wecs <= 0:
            raise ValueError("`num_wecs` deve ser > 0.")
        if num_particles <= 0:
            raise ValueError("`num_particles` deve ser > 0.")
        if max_iter <= 0:
            raise ValueError("`max_iter` deve ser > 0.")
        if grid_step <= 0:
            raise ValueError("`grid_step` deve ser > 0.")

        self.num_wecs = int(num_wecs)
        self.num_particles = int(num_particles)
        self.max_iter = int(max_iter)
        self.modelo = modelo_rf
        self.layout_ref = np.asarray(layout_ref, dtype=float)
        self.bounds_min = np.asarray(bounds_min, dtype=float)
        self.bounds_max = np.asarray(bounds_max, dtype=float)
        self.grid_step = float(grid_step)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.w_max = float(w_max)
        self.w_min = float(w_min)
        self.initial_velocity = float(initial_velocity)
        self.rng = np.random.default_rng(random_state)

        self.num_dims = self.num_wecs * 2
        if self.layout_ref.shape != (self.num_dims,):
            raise ValueError("`layout_ref` deve ter tamanho num_wecs * 2.")
        if self.bounds_min.shape != (self.num_dims,) or self.bounds_max.shape != (self.num_dims,):
            raise ValueError("`bounds_min` e `bounds_max` devem ter tamanho num_wecs * 2.")

        self.grid_points = self._build_candidate_grid()
        self.num_candidates = len(self.grid_points)
        if self.num_candidates < self.num_wecs:
            raise ValueError("Grade candidata menor que o numero de WECs.")

        self.particles = self._init_particles()
        self.velocities = self.rng.uniform(
            -self.initial_velocity,
            self.initial_velocity,
            size=(self.num_particles, self.num_candidates),
        )
        self.pbest = self.particles.copy()
        self.pbest_fit = np.full(self.num_particles, -np.inf)
        self.gbest = self.particles[0].copy()
        self.gbest_fit = -np.inf
        self.history = []

    def _build_candidate_grid(self):
        x_min = float(np.min(self.bounds_min[0::2]))
        x_max = float(np.max(self.bounds_max[0::2]))
        y_min = float(np.min(self.bounds_min[1::2]))
        y_max = float(np.max(self.bounds_max[1::2]))
        xs = np.arange(x_min, x_max + self.grid_step * 0.5, self.grid_step)
        ys = np.arange(y_min, y_max + self.grid_step * 0.5, self.grid_step)
        grid = np.array([(x, y) for x in xs for y in ys], dtype=float)
        return grid

    def _nearest_grid_bits(self, layout):
        coords = np.asarray(layout, dtype=float).reshape(self.num_wecs, 2)
        distances = np.sum((coords[:, None, :] - self.grid_points[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(distances, axis=1)
        bits = np.zeros(self.num_candidates, dtype=int)
        bits[nearest] = 1
        return self._repair_exact_count(bits)

    def _random_bits(self):
        bits = np.zeros(self.num_candidates, dtype=int)
        chosen = self.rng.choice(self.num_candidates, self.num_wecs, replace=False)
        bits[chosen] = 1
        return bits

    def _repair_exact_count(self, bits):
        bits = np.asarray(bits, dtype=int).copy()
        active = np.flatnonzero(bits)
        if len(active) > self.num_wecs:
            drop = self.rng.choice(active, len(active) - self.num_wecs, replace=False)
            bits[drop] = 0
        elif len(active) < self.num_wecs:
            inactive = np.flatnonzero(bits == 0)
            add = self.rng.choice(inactive, self.num_wecs - len(active), replace=False)
            bits[add] = 1
        return bits

    def _init_particles(self):
        particles = np.zeros((self.num_particles, self.num_candidates), dtype=int)
        particles[0] = self._nearest_grid_bits(self.layout_ref)
        for i in range(1, self.num_particles):
            particles[i] = self._random_bits()
        return particles

    def _bits_to_layout(self, bits):
        fixed = self._repair_exact_count(bits)
        active = np.flatnonzero(fixed)
        points = self.grid_points[active]
        layout = points.reshape(-1)
        layout = np.clip(layout, self.bounds_min, self.bounds_max)
        return layout

    def _avaliar_batch(self, particles):
        layouts = np.array([self._bits_to_layout(bits) for bits in particles], dtype=float)
        return np.asarray(self.modelo.predict(layouts), dtype=float)

    @staticmethod
    def _v_shape_transfer(velocity):
        return np.abs((2.0 / np.pi) * np.arctan((np.pi / 2.0) * velocity))

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
            inertia = self.w_max + (self.w_min - self.w_max) * progress

            r1 = self.rng.random((self.num_particles, self.num_candidates))
            r2 = self.rng.random((self.num_particles, self.num_candidates))
            cognitive = self.c1 * r1 * (self.pbest - self.particles)
            social = self.c2 * r2 * (self.gbest - self.particles)
            self.velocities = inertia * self.velocities + cognitive + social

            transfer = self._v_shape_transfer(self.velocities)
            flip_mask = self.rng.random((self.num_particles, self.num_candidates)) < transfer
            self.particles = np.where(flip_mask, 1 - self.particles, self.particles)
            for i in range(self.num_particles):
                self.particles[i] = self._repair_exact_count(self.particles[i])

            fits = self._avaliar_batch(self.particles)
            self._update_bests(fits)
            history.append(self.gbest_fit)

            if verbose and ((t + 1) % 20 == 0 or t == 0):
                print(f"      Iter {t + 1:>3d}/{self.max_iter}: Best = {self.gbest_fit:,.2f} W")

        self.history = history
        return self.gbest.copy(), self._bits_to_layout(self.gbest), self.gbest_fit, history


def _plot_comparison(dataset_name, original_power, pso_history, qpso_history, pso_layout, qpso_layout, original_layout):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"PSO do artigo vs QPSO puro - {dataset_name}", fontsize=14, fontweight="bold", y=1.02)

    ax1 = axes[0]
    ax1.plot(pso_history, label="PSO artigo (bPSO grade)", color="#ff7f0e", linewidth=2.2)
    ax1.plot(qpso_history, label="QPSO puro continuo", color="#1f77b4", linewidth=2.2)
    ax1.axhline(original_power, color="gray", linestyle="--", linewidth=1.2, label="Original RF")
    ax1.set_title("Convergencia por potencia RF", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Iteracoes")
    ax1.set_ylabel("Potencia RF (W)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2 = axes[1]
    ax2.scatter(original_layout[0::2], original_layout[1::2], c="gray", s=40, alpha=0.45, label="Original")
    ax2.scatter(pso_layout[0::2], pso_layout[1::2], c="#ff7f0e", s=42, alpha=0.8, label="PSO artigo")
    ax2.scatter(qpso_layout[0::2], qpso_layout[1::2], c="#1f77b4", s=42, alpha=0.8, label="QPSO puro")
    ax2.set_title("Layouts finais", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    return fig


def run_comparison(
    db_path,
    dataset_name,
    num_wecs=49,
    num_iter=100,
    num_population=12,
    grid_step=50.0,
    random_state=42,
):
    print("\n" + "=" * 70)
    print(f" COMPARACAO PSO DO ARTIGO vs QPSO PURO - {dataset_name.upper()}")
    print("=" * 70)
    print(" PSO artigo: bPSO em grade discreta com transferencia V-shaped.")
    print(" QPSO puro: coordenadas continuas, todas as boias ativas.\n")

    modelo, layout_base, best_power, r2, mae, _, bounds_min, bounds_max, dataset_X, dataset_y = carregar_e_treinar_modelo(
        db_path,
        num_wecs,
        n_estimators=150,
    )
    original_power = float(modelo.predict(layout_base.reshape(1, -1))[0])
    print(f"    R2={r2:.4f} | MAE={mae:,.0f} W | Best dataset={best_power:,.2f} W")
    print(f"    Potencia RF original: {original_power:,.2f} W\n")

    print("[1] Rodando PSO do artigo (bPSO grade 50m)...")
    t0 = time.time()
    pso = ArticleBPSOGrid_WEC(
        num_wecs=num_wecs,
        num_particles=num_population,
        max_iter=num_iter,
        modelo_rf=modelo,
        layout_ref=layout_base,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        grid_step=grid_step,
        random_state=random_state,
    )
    pso_bits, pso_layout, pso_fit, pso_history = pso.optimize()
    pso_time = time.time() - t0
    print(f"    PSO artigo: {pso_fit:,.2f} W | Tempo: {pso_time:.1f}s | bits ativos: {int(np.sum(pso_bits))}\n")

    print("[2] Rodando QPSO puro continuo...")
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
        random_state=random_state + 1 if random_state is not None else None,
    )
    qpso_layout, qpso_fit, qpso_history = qpso.optimize()
    qpso_time = time.time() - t0
    print(f"    QPSO puro: {qpso_fit:,.2f} W | Tempo: {qpso_time:.1f}s\n")

    diff = qpso_fit - pso_fit
    diff_pct = (diff / pso_fit) * 100 if pso_fit > 0 else 0.0
    print("[3] Resultado direto no RF surrogate:")
    print(f"    QPSO puro - PSO artigo = {diff:+,.2f} W ({diff_pct:+.2f}%)\n")

    fig = _plot_comparison(dataset_name, original_power, pso_history, qpso_history, pso_layout, qpso_layout, layout_base)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    safe_name = dataset_name.lower().replace(" ", "_")
    out_path = os.path.join(script_dir, f"comparacao_pso_artigo_qpso_{safe_name}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[!] Grafico salvo como '{out_path}'")

    return {
        "dataset": dataset_name,
        "r2": float(r2),
        "mae": float(mae),
        "original_power": original_power,
        "article_bpso_power": float(pso_fit),
        "pure_qpso_power": float(qpso_fit),
        "difference_qpso_minus_pso_w": float(diff),
        "difference_qpso_minus_pso_pct": float(diff_pct),
        "article_bpso_time_s": float(pso_time),
        "pure_qpso_time_s": float(qpso_time),
        "article_bpso_history": [float(v) for v in pso_history],
        "pure_qpso_history": [float(v) for v in qpso_history],
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
        resultados.append(run_comparison(ds["db"], ds["name"]))

    if resultados:
        print("\n\n" + "=" * 70)
        print(" RESUMO FINAL")
        print("=" * 70)
        print(f"  {'Dataset':<15} {'PSO artigo':>14} {'QPSO puro':>14} {'Delta':>10} {'Delta %':>9}")
        print(f"  {'-' * 13:<15} {'-' * 14:>14} {'-' * 14:>14} {'-' * 10:>10} {'-' * 9:>9}")
        for r in resultados:
            print(
                f"  {r['dataset']:<15} {r['article_bpso_power']:>14,.0f} {r['pure_qpso_power']:>14,.0f} "
                f"{r['difference_qpso_minus_pso_w']:>+10,.0f} {r['difference_qpso_minus_pso_pct']:>+8.2f}%"
            )

        out_json = os.path.join(script_dir, "comparacao_pso_artigo_qpso.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        print(f"\n[!] Resultados salvos em '{out_json}'")

    print("[+] Comparacao concluida!")


if __name__ == "__main__":
    main()
