# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Refactored continuous QPSO core for WEC-layout optimization.

Surrogate-agnostic: accepts any object exposing ``predict(X)`` (and optionally
``predict_with_std(X)`` for the LCB trust-region). Designed so Fase 3 ablations
(RF vs MLP, with/without d_min, with/without LCB) are one-flag changes.

Canonical QPSO update (Sun, Feng & Xu, 2004), unchanged in magnitude:

    p_i  = phi * pbest_i + (1 - phi) * gbest                 (stochastic attractor)
    x_i  = p_i  ±  alpha * |mbest - x_i| * ln(1/u)           (quantum step)

Fixes vs. the legacy ContinuousQPSO_WEC:
  #5a  elitism is restored IN PLACE at the elites' own indices (legacy copied
       them into slots 0..k, scrambling the pbest<->particle correspondence);
  #5b  the reference layout is seeded explicitly (legacy discarded it);
  #5c  Lévy selects the worst particles by their CURRENT fitness, not a stale
       pre-move ranking;
  #5d  one full surrogate evaluation per iteration (legacy evaluated twice).

Constraint & risk handling:
  #4   soft minimum inter-buoy distance via projection repair (d_min);
  LCB  acquisition = mean - kappa*std keeps the search inside the trustworthy
       region of the smooth surrogate (avoids OOD over-prediction).
"""
from __future__ import annotations

import math
import warnings

import numpy as np


class ContinuousQPSO:
    def __init__(
        self,
        surrogate,
        bounds_min,
        bounds_max,
        *,
        num_particles: int = 40,
        max_iter: int = 150,
        alpha_start: float = 1.0,
        alpha_end: float = 0.5,
        kappa: float = 0.0,
        d_min: float = 0.0,
        n_elite: int = 3,
        use_levy: bool = False,
        levy_mu: float = 0.05,
        levy_beta: float = 1.5,
        seed_layouts=None,
        layout_ref=None,
        seed_with_ref: bool = True,
        noise_scale: float = 0.10,
        random_state=None,
    ):
        self.bounds_min = np.asarray(bounds_min, dtype=float)
        self.bounds_max = np.asarray(bounds_max, dtype=float)
        self.num_dims = self.bounds_min.size
        if self.bounds_max.size != self.num_dims:
            raise ValueError("`bounds_min` e `bounds_max` devem ter o mesmo tamanho.")
        if self.num_dims % 2 != 0:
            raise ValueError("Layout deve ter dimensao par (pares X,Y por boia).")
        if num_particles <= 0 or max_iter <= 0:
            raise ValueError("`num_particles` e `max_iter` devem ser > 0.")
        if not (0 <= n_elite < num_particles):
            raise ValueError("`n_elite` deve estar em [0, num_particles).")
        if kappa < 0 or d_min < 0:
            raise ValueError("`kappa` e `d_min` nao podem ser negativos.")

        self.surrogate = surrogate
        self.num_wecs = self.num_dims // 2
        self.num_particles = int(num_particles)
        self.max_iter = int(max_iter)
        self.alpha_start = float(alpha_start)
        self.alpha_end = float(alpha_end)
        self.kappa = float(kappa)
        self.d_min = float(d_min)
        self.n_elite = int(n_elite)
        self.use_levy = bool(use_levy)
        self.levy_mu = float(levy_mu)
        self.levy_beta = float(levy_beta)
        self.noise_scale = float(noise_scale)
        self.rng = np.random.default_rng(random_state)
        self._span = np.maximum(self.bounds_max - self.bounds_min, 1e-9)

        if self.kappa > 0 and not hasattr(surrogate, "predict_with_std"):
            warnings.warn(
                "kappa > 0, mas o surrogate nao expoe predict_with_std (ex.: "
                "ConsensusSurrogate); o LCB sera ignorado e a busca fica gulosa "
                "pela media. Use kappa=0 ou um surrogate com incerteza.",
                UserWarning,
                stacklevel=2,
            )

        self.particles = self._init_particles(seed_layouts, layout_ref, seed_with_ref)
        self.particles = self._repair_min_dist(self.particles)
        self.pbest = self.particles.copy()
        self.pbest_fit = np.full(self.num_particles, -np.inf)
        self.gbest = self.particles[0].copy()
        self.gbest_fit = -np.inf
        self.best_power_pred: float | None = None
        self.history: list[float] = []

    # ------------------------------------------------------------------ setup
    def _confine(self, X):
        return np.clip(X, self.bounds_min, self.bounds_max)

    def _init_particles(self, seed_layouts, layout_ref, seed_with_ref):
        P = np.zeros((self.num_particles, self.num_dims), dtype=float)
        if seed_layouts is not None and len(seed_layouts) > 0:
            seeds = np.asarray(seed_layouts, dtype=float)
            for i in range(self.num_particles):
                base = seeds[self.rng.integers(0, len(seeds))]
                P[i] = self._confine(base + self.rng.normal(0.0, self.noise_scale, self.num_dims) * self._span)
        else:
            P = self.rng.uniform(self.bounds_min, self.bounds_max, (self.num_particles, self.num_dims))
        if layout_ref is not None and seed_with_ref:
            P[0] = self._confine(np.asarray(layout_ref, dtype=float))
        return P

    # ------------------------------------------------------------- objectives
    def _acq(self, X):
        """Acquisition the swarm maximizes: mean, or LCB = mean - kappa*std."""
        if self.kappa > 0 and hasattr(self.surrogate, "predict_with_std"):
            mean, std = self.surrogate.predict_with_std(X)
            return np.asarray(mean, float) - self.kappa * np.asarray(std, float)
        return np.asarray(self.surrogate.predict(X), dtype=float)

    def _power(self, X):
        """Raw predicted power (mean) — what we report for a found layout."""
        if hasattr(self.surrogate, "predict_with_std"):
            return np.asarray(self.surrogate.predict_with_std(X)[0], float)
        return np.asarray(self.surrogate.predict(X), dtype=float)

    # ------------------------------------------------------------- constraint
    def _repair_min_dist(self, X, passes: int = 40):
        """Soft projection repair, vectorized over particles: iteratively push
        apart buoy pairs closer than d_min until feasible. Early-breaks when no
        particle violates, so it is cheap on well-separated layouts. Near the
        domain boundary the repair is approximate (soft constraint)."""
        if self.d_min <= 0:
            return X
        single = X.ndim == 1
        X = np.atleast_2d(np.asarray(X, dtype=float)).copy()
        M, N = X.shape[0], self.num_wecs
        lo = self.bounds_min.reshape(N, 2)
        hi = self.bounds_max.reshape(N, 2)
        pts = X.reshape(M, N, 2)
        eye = np.eye(N, dtype=bool)[None]                       # (1,N,N)
        for _ in range(passes):
            diff = pts[:, :, None, :] - pts[:, None, :, :]      # (M,N,N,2): i - j
            dist = np.sqrt((diff ** 2).sum(-1))                 # (M,N,N)
            dist = np.where(eye, np.inf, dist)
            mask = dist < self.d_min
            if not mask.any():
                break
            zero = dist == 0                                    # break exact overlaps
            if zero.any():
                rnd = self.rng.normal(size=diff.shape)
                diff = np.where(zero[..., None], rnd, diff)
                dist = np.where(zero, 1e-6, dist)
            unit = diff / dist[..., None]
            push = (np.clip(self.d_min - dist, 0.0, None) / 2.0) * mask
            pts = np.clip(pts + (unit * push[..., None]).sum(axis=2), lo, hi)
        X = pts.reshape(M, -1)
        return X[0] if single else X

    def _levy_step(self):
        b = self.levy_beta
        sigma_u = (
            math.gamma(1 + b) * math.sin(math.pi * b / 2)
            / (math.gamma((1 + b) / 2) * b * 2 ** ((b - 1) / 2))
        ) ** (1 / b)
        u = self.rng.normal(0, sigma_u, self.num_dims)
        v = self.rng.normal(0, 1, self.num_dims)
        return u / (np.abs(v) ** (1 / b))

    # ------------------------------------------------------------------- loop
    def _update_bests(self, acq):
        improved = acq > self.pbest_fit
        if improved.any():
            self.pbest[improved] = self.particles[improved]
            self.pbest_fit[improved] = acq[improved]
        b = int(np.argmax(acq))
        if acq[b] > self.gbest_fit:
            self.gbest_fit = float(acq[b])
            self.gbest = self.particles[b].copy()

    def optimize(self, verbose: bool = False):
        acq = self._acq(self.particles)
        self._update_bests(acq)
        history: list[float] = []

        for t in range(self.max_iter):
            alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * (t / max(1, self.max_iter - 1))
            mbest = self.pbest.mean(axis=0)

            elite_idx = None
            if self.n_elite > 0:
                elite_idx = np.argsort(acq)[-self.n_elite:]
                elites = self.particles[elite_idx].copy()
                elite_acq = acq[elite_idx].copy()

            phi = self.rng.random((self.num_particles, self.num_dims))
            attractor = phi * self.pbest + (1.0 - phi) * self.gbest
            u = self.rng.uniform(1e-10, 1.0, (self.num_particles, self.num_dims))
            direction = np.where(self.rng.random((self.num_particles, self.num_dims)) > 0.5, 1.0, -1.0)
            step = alpha * np.abs(mbest - self.particles) * np.log(1.0 / u)
            new_pos = attractor + direction * step
            new_pos = np.where(np.isfinite(new_pos), new_pos, attractor)
            self.particles = self._repair_min_dist(self._confine(new_pos))

            acq = self._acq(self.particles)

            # #5c: Lévy on the worst particles by CURRENT acq (not a stale rank)
            if self.use_levy:
                rate = self.levy_mu * (1 - 0.5 * (t / self.max_iter))
                order = np.argsort(acq)
                excluded = set(elite_idx.tolist()) if elite_idx is not None else set()
                cand = [int(i) for i in order if int(i) not in excluded][: max(1, self.num_particles // 3)]
                moved = []
                for i in cand:
                    if self.rng.random() < rate:
                        self.particles[i] = self._confine(self.particles[i] + self._levy_step() * 0.05 * self._span)
                        moved.append(i)
                if moved:
                    mi = np.asarray(moved)
                    self.particles[mi] = self._repair_min_dist(self.particles[mi])
                    acq[mi] = self._acq(self.particles[mi])

            # #5a: restore elites IN PLACE at their own indices
            if elite_idx is not None:
                self.particles[elite_idx] = elites
                acq[elite_idx] = elite_acq

            self._update_bests(acq)
            history.append(self.gbest_fit)
            if verbose and ((t + 1) % 20 == 0 or t == 0):
                print(f"      Iter {t + 1:>3d}/{self.max_iter}: best_acq = {self.gbest_fit:,.2f}")

        self.history = history
        self.best_power_pred = float(self._power(self.gbest.reshape(1, -1))[0])
        return self.gbest, self.gbest_fit, history
