"""Conventional regression and machine-learning baselines.

Every baseline sees the same dimensionless, mechanics-informed features as the
graph network (node rows for displacements, member rows with both end-node
features for member forces) and predicts the same dimensionless targets.
Restrained DOFs are zeroed for all models.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import features as FT


def _make(name, seed):
    if name == 'Linear ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == 'Quadratic ridge':
        return make_pipeline(StandardScaler(), PolynomialFeatures(2), StandardScaler(), Ridge(alpha=10.0))
    if name == 'Random forest':
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_features=0.5,
                                     n_jobs=4, random_state=seed)
    if name == 'Gradient boosting':
        return _PerOutput(lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.08,
                                                                max_leaf_nodes=63, random_state=seed))
    if name == 'MLP':
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 256, 256), learning_rate_init=1e-3,
                                                            max_iter=300, early_stopping=True,
                                                            random_state=seed))
    raise ValueError(name)


class _PerOutput:
    def __init__(self, factory):
        self.factory = factory

    def fit(self, X, Y):
        self.models = [self.factory().fit(X, Y[:, k]) for k in range(Y.shape[1])]
        return self

    def predict(self, X):
        return np.stack([m.predict(X) for m in self.models], 1)


BASELINES = ['Linear ridge', 'Quadratic ridge', 'Random forest', 'Gradient boosting', 'MLP']
MAX_ROWS_POLY = 30000


class TabularSurrogate:
    def __init__(self, name, seed=0):
        self.name, self.seed = name, seed

    def fit(self, graphs):
        Xn, Yn, Mn, Xe, Ye = FT.tabular(graphs)
        self.us = np.sqrt((Yn ** 2 * Mn).sum(0) / Mn.sum(0))
        self.qs = np.sqrt((Ye ** 2).mean(0))
        rng = np.random.default_rng(self.seed)
        if self.name == 'Quadratic ridge':
            kn = rng.permutation(len(Xn))[:MAX_ROWS_POLY]; ke = rng.permutation(len(Xe))[:MAX_ROWS_POLY]
        else:
            kn, ke = slice(None), slice(None)
        self.mn = _make(self.name, self.seed).fit(Xn[kn], (Yn / self.us)[kn])
        self.me = _make(self.name, self.seed).fit(Xe[ke], (Ye / self.qs)[ke])
        return self

    def predict(self, graphs):
        Xn, _, Mn, Xe, _ = FT.tabular(graphs)
        U = self.mn.predict(Xn) * self.us * Mn
        Q = self.me.predict(Xe) * self.qs
        out, on, oe = [], 0, 0
        for g in graphs:
            n, m = len(g['x']), len(g['conn'])
            out.append(dict(u=U[on:on + n], q=Q[oe:oe + m]))
            on += n; oe += m
        return out
