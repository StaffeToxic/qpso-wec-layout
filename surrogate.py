# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""Smooth ML surrogate for continuous WEC-layout optimization.

Why this exists: the Random Forest surrogate is piecewise-constant — its
response is flat inside each leaf and its maximum over the domain is just the
best training layout, so a continuous metaheuristic (QPSO) cannot improve on the
seed. A smooth, differentiable surrogate gives the optimizer real directions of
improvement, and its continuous maximum can land *between* training layouts.

Backend: an ensemble of ``sklearn`` MLPRegressors with ``tanh`` activation
(C-infinity smooth). No torch/gpytorch needed (unavailable on this Python). The
ensemble doubles as an uncertainty estimate: members agree in-distribution and
disagree out-of-distribution, which feeds the LCB trust-region objective
``lcb = mean - kappa * std`` used to keep the search inside the trustworthy region.
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class MLPEnsembleSurrogate:
    """Ensemble of tanh-MLPs with a sklearn-like ``predict`` interface.

    ``predict(X)`` returns the ensemble mean (drop-in for the old ``modelo_rf``).
    ``predict_with_std(X)`` adds the member-disagreement std (epistemic uncertainty).
    ``lcb(X, kappa)`` returns ``mean - kappa*std`` for risk-averse optimization.
    """

    def __init__(
        self,
        n_members: int = 5,
        hidden_layer_sizes=(128, 64),
        activation: str = "tanh",
        max_iter: int = 500,
        random_state: int = 42,
        **mlp_kwargs,
    ):
        if n_members <= 0:
            raise ValueError("`n_members` deve ser > 0.")
        self.n_members = int(n_members)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.mlp_kwargs = mlp_kwargs
        self._members: list[MLPRegressor] = []
        self._x_scaler: StandardScaler | None = None
        self._y_scaler: StandardScaler | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._x_scaler = StandardScaler().fit(X)
        self._y_scaler = StandardScaler().fit(y.reshape(-1, 1))
        Xs = self._x_scaler.transform(X)
        ys = self._y_scaler.transform(y.reshape(-1, 1)).ravel()
        self._members = []
        for i in range(self.n_members):
            member = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                max_iter=self.max_iter,
                random_state=self.random_state + i,
                **self.mlp_kwargs,
            )
            member.fit(Xs, ys)
            self._members.append(member)
        return self

    def _member_preds(self, X) -> np.ndarray:
        if not self._members:
            raise RuntimeError("Surrogate nao treinado: chame fit() antes de predict().")
        Xs = self._x_scaler.transform(np.asarray(X, dtype=float))
        return np.stack(
            [self._y_scaler.inverse_transform(m.predict(Xs).reshape(-1, 1)).ravel() for m in self._members],
            axis=0,
        )  # shape (n_members, n_samples)

    def predict(self, X) -> np.ndarray:
        return self._member_preds(X).mean(axis=0)

    def predict_with_std(self, X) -> tuple[np.ndarray, np.ndarray]:
        preds = self._member_preds(X)
        return preds.mean(axis=0), preds.std(axis=0)

    def lcb(self, X, kappa: float = 1.0) -> np.ndarray:
        mean, std = self.predict_with_std(X)
        return mean - kappa * std


class ConsensusSurrogate:
    """Pessimistic consensus of several independently-trained surrogates.

    ``predict(X)`` returns the **elementwise minimum** across models: a layout
    scores high only if EVERY model agrees it is high. Maximizing this objective
    therefore yields layouts that all models endorse — gains that are credible
    *by construction*, instead of single-model extrapolation ("fantasy peaks").

    Rationale (Fase 3): an MLP alone reports +5–10% from moving buoys, but an
    independent RF disagrees out-of-distribution. Optimizing ``min(MLP, RF)``
    keeps the search where both agree, capping that mirage automatically.
    """

    def __init__(self, models):
        self.models = list(models)
        if not self.models:
            raise ValueError("`models` nao pode ser vazio.")

    def predict(self, X) -> np.ndarray:
        preds = np.stack(
            [np.asarray(m.predict(X), dtype=float).ravel() for m in self.models], axis=0
        )
        return preds.min(axis=0)
