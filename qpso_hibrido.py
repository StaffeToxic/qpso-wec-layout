# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

import numpy as np

def fitness(x):
    return np.sum(x**2)

class HybridQPSO:
    """QPSO Híbrido: Inclui Operador de Mutação de Algoritmo Genético (GA)"""
    def __init__(self, dim, num_particles, max_iter):
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.particles = np.random.uniform(-10, 10, (num_particles, dim))
        self.pbest = np.copy(self.particles)
        
    def evaluate(self):
        return np.array([fitness(p) for p in self.particles])

    def optimize(self):
        pbest_fit = self.evaluate()
        gbest = self.particles[np.argmin(pbest_fit)]
        gbest_fit = np.min(pbest_fit)

        for t in range(self.max_iter):
            mbest = np.mean(self.pbest, axis=0)
            alpha = 1.0 - 0.5 * (t / self.max_iter)
            
            for i in range(self.num_particles):
                fit = fitness(self.particles[i])
                if fit < pbest_fit[i]:
                    pbest_fit[i] = fit
                    self.pbest[i] = np.copy(self.particles[i])
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest = np.copy(self.particles[i])

            # 1. Operador Principal: Dinâmica Quântica
            for i in range(self.num_particles):
                phi = np.random.uniform(0, 1, self.dim)
                p = phi * self.pbest[i] + (1 - phi) * gbest
                u = np.random.uniform(0, 1, self.dim)
                L = alpha * np.abs(mbest - self.particles[i])
                direction = np.where(np.random.rand(self.dim) > 0.5, 1, -1)
                self.particles[i] = p + direction * L * np.log(1.0 / u)

            # 2. Operador Híbrido: Mutação GA para escapar de ótimos locais
            # Partículas com baixo desempenho recebem pequenos ruídos para explorarem mais
            median_fit = np.median(pbest_fit)
            for i in range(self.num_particles):
                if pbest_fit[i] > median_fit: # "Piores" partículas (em problema de minimização)
                    if np.random.rand() < 0.2: # 20% de chance de Mutação
                        mut_idx = np.random.randint(0, self.dim)
                        # Adiciona mutação gaussiana ao gene local
                        self.particles[i][mut_idx] += np.random.normal(0, 1.0)
                        
        return gbest, gbest_fit

if __name__ == "__main__":
    opt = HybridQPSO(dim=10, num_particles=30, max_iter=100)
    best_pos, best_fit = opt.optimize()
    print("=== QPSO Híbrido (Quântico + Genético) ===")
    print("Melhor Fitness Alcançado:", best_fit)
