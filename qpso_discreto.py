# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Refactored DISCRETE QPSO for WEC buoy *selection* (which buoys to keep).

Fitness-agnostic: takes a batch fitness ``fitness_fn(bits) -> array``. The intended
use is the SHAP-additive objective ``F(b) = wp*(phi0 + sum b_i c_i) + we*efficiency``,
which is well-posed precisely because it is additive (turning a buoy off subtracts
its SHAP contribution — interpolation, not extrapolation), unlike the continuous
case where the black-box surrogate is queried at OOD layouts.

Quantum collapse (Sun et al., 2004; sigmoid discretization):
    p   = phi*pbest + (1-phi)*gbest                  (continuous attractor in [0,1])
    x   = p ± alpha*|mbest - b|*ln(1/u)              (quantum step, mbest = bit freq)
    b'  = 1 if rand < sigmoid(x) else 0              (Bernoulli collapse)

Carries the same fixes as qpso_continuo.py: in-place elitism (#5a), one full
fitness evaluation per iteration (#5d), operators act on CURRENT fitness (#5c).
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))


class DiscreteQPSO:
    def __init__(
        self,
        fitness_fn,
        num_wecs: int,
        *,
        num_particles: int = 30,
        max_iter: int = 80,
        min_boias: int = 20,
        alpha_start: float = 1.0,
        alpha_end: float = 0.5,
        n_elite: int = 3,
        mutate: bool = True,
        p_mut_base: float = 0.35,
        p_mut_final: float = 0.05,
        init_p_active: float = 0.7,
        random_state=None,
    ):
        if num_wecs <= 0 or num_particles <= 0 or max_iter <= 0:
            raise ValueError("`num_wecs`, `num_particles`, `max_iter` devem ser > 0.")
        if not (0 < min_boias <= num_wecs):
            raise ValueError("`min_boias` deve estar em (0, num_wecs].")
        if not (0 <= n_elite < num_particles):
            raise ValueError("`n_elite` deve estar em [0, num_particles).")

        self.fitness_fn = fitness_fn
        self.num_wecs = int(num_wecs)
        self.num_particles = int(num_particles)
        self.max_iter = int(max_iter)
        self.min_boias = int(min_boias)
        self.alpha_start = float(alpha_start)
        self.alpha_end = float(alpha_end)
        self.n_elite = int(n_elite)
        self.mutate = bool(mutate)
        self.p_mut_base = float(p_mut_base)
        self.p_mut_final = float(p_mut_final)
        self.rng = np.random.default_rng(random_state)

        P = (self.rng.random((self.num_particles, self.num_wecs)) < init_p_active).astype(float)
        P[0] = 1.0  # reference: all buoys active
        self.particles = self._repair_min_boias(P)
        self.pbest = self.particles.copy()
        self.pbest_fit = np.full(self.num_particles, -np.inf)
        self.gbest = self.particles[0].copy()
        self.gbest_fit = -np.inf
        self.history: list[float] = []

    def _repair_min_boias(self, bits, freq=None):
        """Ensure each row has >= min_boias active bits. With `freq` (bit
        frequency among pbest) the smart repair activates the most-agreed buoys."""
        bits = np.atleast_2d(bits).astype(float).copy()
        for r in range(bits.shape[0]):
            n = int(bits[r].sum())
            if n >= self.min_boias:
                continue
            inactive = np.where(bits[r] == 0)[0]
            need = self.min_boias - n
            if freq is not None:
                chosen = inactive[np.argsort(freq[inactive])[::-1][:need]]
            else:
                chosen = self.rng.choice(inactive, size=min(need, len(inactive)), replace=False)
            bits[r, chosen] = 1.0
        return bits

    def _eval(self, bits):
        return np.asarray(self.fitness_fn(bits), dtype=float).ravel()

    def _update_bests(self, fits):
        improved = fits > self.pbest_fit
        if improved.any():
            self.pbest[improved] = self.particles[improved]
            self.pbest_fit[improved] = fits[improved]
        b = int(np.argmax(fits))
        if fits[b] > self.gbest_fit:
            self.gbest_fit = float(fits[b])
            self.gbest = self.particles[b].copy()

    def optimize(self, verbose: bool = False):
        fits = self._eval(self.particles)
        self._update_bests(fits)
        history: list[float] = []

        for t in range(self.max_iter):
            alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * (t / max(1, self.max_iter - 1))
            mbest = self.pbest.mean(axis=0)            # bit frequency in [0,1]

            elite_idx = None
            if self.n_elite > 0:
                elite_idx = np.argsort(fits)[-self.n_elite:]
                elites = self.particles[elite_idx].copy()
                elite_fits = fits[elite_idx].copy()

            phi = self.rng.random((self.num_particles, self.num_wecs))
            p = phi * self.pbest + (1.0 - phi) * self.gbest
            u = self.rng.uniform(1e-10, 1.0, (self.num_particles, self.num_wecs))
            direction = np.where(self.rng.random((self.num_particles, self.num_wecs)) > 0.5, 1.0, -1.0)
            x_real = p + direction * alpha * np.abs(mbest - self.particles) * np.log(1.0 / u)
            x_real = np.where(np.isfinite(x_real), x_real, p)
            prob = np.clip(_sigmoid(x_real), 0.01, 0.99)
            self.particles = (self.rng.random((self.num_particles, self.num_wecs)) < prob).astype(float)
            self.particles = self._repair_min_boias(self.particles, freq=mbest)

            fits = self._eval(self.particles)

            # adaptive mutation on the below-median (non-elite) particles, by CURRENT fits
            if self.mutate:
                excluded = set(elite_idx.tolist()) if elite_idx is not None else set()
                med = np.median(fits)
                p_mut = self.p_mut_base + (self.p_mut_final - self.p_mut_base) * (t / self.max_iter)
                n_flips = max(1, int(p_mut * self.num_wecs * 0.1))
                moved = []
                for i in range(self.num_particles):
                    if i in excluded or fits[i] >= med:
                        continue
                    if self.rng.random() < p_mut:
                        idx = self.rng.choice(self.num_wecs, n_flips, replace=False)
                        self.particles[i, idx] = 1.0 - self.particles[i, idx]
                        moved.append(i)
                if moved:
                    mi = np.asarray(moved)
                    self.particles[mi] = self._repair_min_boias(self.particles[mi], freq=mbest)
                    fits[mi] = self._eval(self.particles[mi])

            if elite_idx is not None:                  # #5a: restore elites in place
                self.particles[elite_idx] = elites
                fits[elite_idx] = elite_fits

            self._update_bests(fits)
            history.append(self.gbest_fit)
            if verbose and ((t + 1) % 20 == 0 or t == 0):
                print(f"      Iter {t + 1:>3d}/{self.max_iter}: best = {self.gbest_fit:,.2f} | "
                      f"ativas = {int(self.gbest.sum())}")

        self.history = history
        return self.gbest.astype(int), self.gbest_fit, history
