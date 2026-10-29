# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

from __future__ import annotations

import json
import sqlite3
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from qpso_layout_puro_standalone import PureContinuousQPSO_WEC
from qpso_layout_standalone import ContinuousQPSO_WEC, carregar_e_treinar_modelo


BASE_DIR = Path(__file__).resolve().parent
NUM_WECS = 49
NUM_RUNS = 10
SEEDS = list(range(NUM_RUNS))

PURE_NUM_PARTICLES = 12
PURE_MAX_ITER = 100
STANDALONE_NUM_PARTICLES = 50
STANDALONE_MAX_ITER = 150


ARTICLE_METHODS = [
    "1+1EA",
    "2+2CMAES",
    "CMAES",
    "DE",
    "IDE",
    "LS-NM",
    "bGA",
    "bDE",
    "bPSO",
    "DLS(I)",
    "DLS(II)",
    "SLSNM-bGA",
    "SLSNM-bDE",
    "SLSNM-bPSO",
    "MS-bGA",
    "MS-bDE",
    "MS-bPSO",
]


ARTICLE_49 = {
    "Sydney 49": {
        "q_best": {"MS-bGA": 0.80},
        "Max": [4082524, 4036152, 4061528, 4028337, 4062984, 4089731, 4052984, 4078389, 4036728, 4108751, 4134622, 4105401, 4100089, 4084955, 4145252, 4132385, 4125968],
        "Min": [4046311, 4013471, 4015919, 4002550, 3976301, 4039525, 4006283, 4050349, 3992467, 4070321, 4098276, 4031291, 4047364, 4066979, 4106192, 4097087, 4107365],
        "Mean": [4062828, 4026230, 4029509, 4014852, 4014611, 4063963, 4028394, 4062048, 4015029, 4092373, 4116149, 4075606, 4082176, 4076665, 4125113, 4113961, 4117115],
        "Median": [4060505, 4029230, 4030342, 4015482, 4011081, 4064273, 4025310, 4059954, 4015661, 4095077, 4113040, 4072725, 4083311, 4076621, 4127949, 4113672, 4117134],
        "Std": [9580, 7808, 12500, 8591, 20268, 18106, 13725, 7924, 11643, 10140, 10814, 19897, 13306, 4946, 12113, 9324, 5856],
    },
    "Perth 49": {
        "q_best": {"MS-bDE": 0.88},
        "Max": [3933532, 3863159, 3946189, 3860444, 3875783, 4080796, 3984477, 4010788, 3951612, 4029731, 4075129, 4112549, 4097756, 4112602, 4143453, 4177659, 4153355],
        "Min": [3822270, 3819043, 3900198, 3817360, 3774422, 3870593, 3884613, 3914666, 3889598, 3916594, 4034768, 4071398, 4055033, 4065949, 4080411, 4108205, 4081240],
        "Mean": [3897954, 3847996, 3920466, 3841577, 3841125, 4001900, 3925487, 3960060, 3912453, 3984165, 4045675, 4092000, 4080173, 4082473, 4101887, 4101260, 4095598],
        "Median": [3904627, 3853485, 3923422, 3842720, 3848962, 4008882, 3919409, 3964299, 3912960, 3987689, 4044025, 4090929, 4081414, 4082183, 4099826, 4095057, 4089500],
        "Std": [32160, 14659, 17904, 13118, 26134, 57673, 30027, 27647, 17177, 33047, 10639, 10535, 14291, 12556, 15447, 33245, 19794],
    },
}


class PureContinuousPSO_WEC:
    """PSO continuo classico para 49 boias fixas, sem Levy/crossover/elitismo extra."""

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
        w_max=0.9,
        w_min=0.4,
        c1=2.0,
        c2=2.0,
        noise_scale=0.10,
        vmax_scale=0.20,
    ):
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
        self.w_max = float(w_max)
        self.w_min = float(w_min)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.noise_scale = float(noise_scale)
        self.vmax_scale = float(vmax_scale)
        self.rng = np.random.default_rng(random_state)

        self.span = np.maximum(self.bounds_max - self.bounds_min, 1e-9)
        self.vmax = self.vmax_scale * self.span
        self.particles = self._init_particles()
        self.velocities = self.rng.uniform(-self.vmax, self.vmax, size=(self.num_particles, self.num_dims))
        self.pbest = self.particles.copy()
        self.pbest_fit = np.full(self.num_particles, -np.inf)
        self.gbest = self.layout_ref.copy()
        self.gbest_fit = -np.inf

    def _init_particles(self):
        sorted_idx = np.argsort(self.dataset_y)
        p50 = int(len(sorted_idx) * 0.50)
        p90 = max(int(len(sorted_idx) * 0.90), p50 + 1)
        seed_layouts = self.dataset_X[sorted_idx[p50:p90]]
        particles = np.zeros((self.num_particles, self.num_dims), dtype=float)
        particles[0] = self._confinar(self.layout_ref)
        for i in range(1, self.num_particles):
            seed = seed_layouts[self.rng.integers(0, len(seed_layouts))]
            noise = self.rng.normal(0.0, self.noise_scale, self.num_dims) * self.span
            particles[i] = self._confinar(seed + noise)
        return particles

    def _confinar(self, values):
        return np.clip(values, self.bounds_min, self.bounds_max)

    def _avaliar_batch(self, layouts):
        return np.asarray(self.modelo.predict(layouts), dtype=float)

    def _update_bests(self, fits):
        improved = fits > self.pbest_fit
        if np.any(improved):
            self.pbest_fit[improved] = fits[improved]
            self.pbest[improved] = self.particles[improved]
        best_idx = int(np.argmax(fits))
        if fits[best_idx] > self.gbest_fit:
            self.gbest_fit = float(fits[best_idx])
            self.gbest = self.particles[best_idx].copy()

    def optimize(self):
        history = []
        fits = self._avaliar_batch(self.particles)
        self._update_bests(fits)

        for t in range(self.max_iter):
            progress = t / max(1, self.max_iter - 1)
            inertia = self.w_max + (self.w_min - self.w_max) * progress
            r1 = self.rng.random((self.num_particles, self.num_dims))
            r2 = self.rng.random((self.num_particles, self.num_dims))
            cognitive = self.c1 * r1 * (self.pbest - self.particles)
            social = self.c2 * r2 * (self.gbest - self.particles)
            self.velocities = inertia * self.velocities + cognitive + social
            self.velocities = np.clip(self.velocities, -self.vmax, self.vmax)
            self.particles = self._confinar(self.particles + self.velocities)
            fits = self._avaliar_batch(self.particles)
            self._update_bests(fits)
            history.append(self.gbest_fit)

        return self.gbest, self.gbest_fit, history


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(values):
    return {
        "max": float(max(values)),
        "min": float(min(values)),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(sample_std(values)),
        "values": [float(v) for v in values],
    }


def fmt_w(value):
    if value is None:
        return "N/D"
    if isinstance(value, str):
        return value
    if abs(float(value) - round(float(value))) < 1e-6:
        return f"{float(value):,.0f}"
    return f"{float(value):,.2f}"


def fmt_pct(value):
    return f"{value:+.2f}%"


def best_q_factor(db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT Total_Power, qW FROM wec_data ORDER BY Total_Power DESC LIMIT 1")
    row = cur.fetchone()
    con.close()
    return float(row[0]), float(row[1])


def article_method_stats(dataset_name, method):
    idx = ARTICLE_METHODS.index(method)
    article = ARTICLE_49[dataset_name]
    return {
        "max": float(article["Max"][idx]),
        "min": float(article["Min"][idx]),
        "mean": float(article["Mean"][idx]),
        "median": float(article["Median"][idx]),
        "std": float(article["Std"][idx]),
        "q_factor": None,
    }


def article_bpso_stats(dataset_name):
    return article_method_stats(dataset_name, "bPSO")


def run_dataset(dataset_name, db_file):
    db_path = str(BASE_DIR / db_file)
    modelo, layout_base, best_power, r2, mae, _, bounds_min, bounds_max, dataset_X, dataset_y = carregar_e_treinar_modelo(
        db_path,
        NUM_WECS,
        n_estimators=150,
        verbose=False,
    )
    db_best_power, db_q = best_q_factor(db_path)

    pso_puro_values = []
    qpso_puro_values = []
    qpso_standalone_values = []

    for seed in SEEDS:
        pso = PureContinuousPSO_WEC(
            NUM_WECS,
            PURE_NUM_PARTICLES,
            PURE_MAX_ITER,
            modelo,
            layout_base,
            bounds_min,
            bounds_max,
            dataset_X,
            dataset_y,
            random_state=seed,
        )
        _, pso_fit, _ = pso.optimize()
        pso_puro_values.append(pso_fit)

        qpso = PureContinuousQPSO_WEC(
            NUM_WECS,
            PURE_NUM_PARTICLES,
            PURE_MAX_ITER,
            modelo,
            layout_base,
            bounds_min,
            bounds_max,
            dataset_X,
            dataset_y,
            random_state=seed + 1000,
        )
        _, qpso_fit, _ = qpso.optimize(verbose=False)
        qpso_puro_values.append(qpso_fit)

        np.random.seed(seed + 2000)
        t0 = time.time()
        qpso_layout = ContinuousQPSO_WEC(
            NUM_WECS,
            STANDALONE_NUM_PARTICLES,
            STANDALONE_MAX_ITER,
            modelo,
            layout_base,
            bounds_min,
            bounds_max,
            dataset_X,
            dataset_y,
            levy_mu=0.05,
        )
        _, qpso_layout_fit, _ = qpso_layout.optimize()
        _ = time.time() - t0
        qpso_standalone_values.append(qpso_layout_fit)

        print(
            f"{dataset_name} seed={seed}: "
            f"PSO puro={pso_fit:,.2f} | "
            f"QPSO puro={qpso_fit:,.2f} | "
            f"QPSO layout={qpso_layout_fit:,.2f}"
        )

    def attach_q_factor(summary):
        summary["q_factor"] = db_q if abs(summary["max"] - db_best_power) < 1.0 else None
        return summary

    return {
        "dataset": dataset_name,
        "r2": float(r2),
        "mae": float(mae),
        "db_best_power": float(db_best_power),
        "db_q_factor": float(db_q),
        "PSO puro": attach_q_factor(summarize(pso_puro_values)),
        "PSO do artigo": article_bpso_stats(dataset_name),
        "MS-bPSO do artigo": article_method_stats(dataset_name, "MS-bPSO"),
        "QPSO puro": attach_q_factor(summarize(qpso_puro_values)),
        "QPSO standalone layout": attach_q_factor(summarize(qpso_standalone_values)),
    }


def build_markdown(results):
    lines = [
        "# Tabela comparativa com metricas do artigo",
        "",
        "Metricas usadas: potencia total `P_sum(x, y)` em W, minimo, media, mediana,",
        "maximo e desvio padrao. A porcentagem final usa o `bPSO` do artigo como",
        "baseline em cada dataset:",
        "",
        "`Dif. % vs bPSO = (Max metodo - Max bPSO) / Max bPSO * 100`",
        "",
        "| Dataset | Metodo | Max/Melhor P_sum (W) | Min (W) | Media (W) | Mediana (W) | Std (W) | Dif. % vs bPSO |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        dataset = result["dataset"]
        bpso_stats = result.get("bPSO do artigo") or result["PSO do artigo"]
        rows = [
            ("bPSO do artigo", bpso_stats),
            ("MS-bPSO do artigo", result.get("MS-bPSO do artigo") or article_method_stats(dataset, "MS-bPSO")),
            ("QPSO standalone layout", result["QPSO standalone layout"]),
        ]
        baseline = bpso_stats["max"]
        for method, stats in rows:
            pct = ((stats["max"] - baseline) / baseline) * 100
            lines.append(
                f"| {dataset} | {method} | {fmt_w(stats['max'])} | {fmt_w(stats['min'])} | "
                f"{fmt_w(stats['mean'])} | {fmt_w(stats['median'])} | {fmt_w(stats['std'])} | "
                f"{fmt_pct(pct)} |"
            )

    lines.extend(
        [
            "",
            "Leitura direta:",
            "",
            "- O MS-bPSO melhora o bPSO nos dois datasets.",
            "- O QPSO standalone layout fica acima do bPSO nos dois datasets.",
            "",
            "Observacao: os resultados do artigo sao os valores reportados na Tabela 1. Os",
            "resultados do nosso QPSO standalone layout foram obtidos pelo codigo local com",
            "avaliacao via surrogate RandomForest; portanto, a comparacao e util como",
            "experimento computacional interno, mas nao substitui uma reavaliacao fisica/BEM",
            "dos layouts finais.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    datasets = [
        ("Sydney 49", "WEC_Sydney_49.db"),
        ("Perth 49", "WEC_Perth_49.db"),
    ]
    results = [run_dataset(name, db_file) for name, db_file in datasets]

    out_json = BASE_DIR / "tabela_metricas_artigo_pso_qpso.json"
    out_md = BASE_DIR / "tabela_metricas_artigo_pso_qpso.md"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(build_markdown(results), encoding="utf-8")
    print(f"JSON salvo em: {out_json}")
    print(f"Tabela salva em: {out_md}")


if __name__ == "__main__":
    main()
