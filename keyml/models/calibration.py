"""Вероятность читерства с доверительным интервалом.

Вместо бинарного «читер / не читер» модель выдаёт
    P(читер | попытка)  и  90%-интервал для этой вероятности.

Три слоя (всё на numpy):

1. БАЙЕСОВСКАЯ ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ (аппроксимация Лапласа).
   Обычная логрегрессия даёт одну точку w*. Но веса оценены по конечной выборке,
   и для «непохожих» на обучающие попыток мы знаем мало. Апостериорное
   распределение весов приближаем гауссианой в точке максимума:
       p(w | D) ≈ N(w*, Σ),   Σ = H^{-1},
       H = Σ_i s_i p_i (1 - p_i) x_i x_iᵀ + λ·n·I     (гессиан −log p(w|D))
   Логит новой попытки z = wᵀx тогда распределён как N(m, v):
       m = w*ᵀx,   v = xᵀ Σ x.
   Интервал для логита: m ± q·√v, для вероятности — σ(m ± q·√v)
   (σ монотонна, поэтому квантили переносятся). Точечная оценка — медиана σ(m),
   она всегда внутри интервала. (Среднее апостериорной вероятности даёт пробит-
   приближение МакКея σ(m / √(1 + πv/8)) — используется в BayesianLogReg.predict_proba.)

2. КАЛИБРОВКА ПЛАТТА.  Модель обучается со взвешиванием классов, поэтому её
   «сырая» вероятность не равна реальной частоте. На out-of-fold предсказаниях
   подбираем p = σ(a·z + b) (логистическая регрессия на один признак, метод Ньютона).
   Применяется к центру и к концам интервала (a > 0 => монотонно).

3. СДВИГ АПРИОРНОЙ ВЕРОЯТНОСТИ (теорема Байеса). Калибровка верна для доли
   читеров в обучающих данных π_data. На сайте доля другая (π_deploy, например 5%).
   Апостериорные шансы = отношение правдоподобий × априорные шансы, значит
       logit p_deploy = logit p_data − logit π_data + logit π_deploy.
"""
from __future__ import annotations

import math

import numpy as np

from .linear import LogisticRegressionNP, _sigmoid

Z90 = 1.6448536269514722  # квантиль N(0,1) уровня 0.95 -> двусторонний 90%


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class BayesianLogReg:
    def __init__(self, l2=1e-2, lr=0.05, epochs=600):
        self.base = LogisticRegressionNP(l2=l2, lr=lr, epochs=epochs, balanced=True)
        self.l2 = l2

    def fit(self, X, y):
        self.base.fit(X, y)
        Xs = self.base.scaler.transform(X)
        y = np.asarray(y, float)
        n = len(y)
        pos = y.mean()
        sw = np.where(y == 1, 0.5 / max(pos, 1e-9), 0.5 / max(1 - pos, 1e-9))
        Xa = np.hstack([Xs, np.ones((n, 1))])           # + столбец для свободного члена
        p = _sigmoid(Xa @ self.w_full)
        H = (Xa * (sw * p * (1 - p))[:, None]).T @ Xa
        reg = np.full(Xa.shape[1], self.l2 * n)
        reg[-1] = 1e-3                                   # на bias почти нет априорного сжатия
        H += np.diag(reg)
        self.cov = np.linalg.inv(H)
        return self

    @property
    def w_full(self):
        return np.r_[self.base.w, self.base.b[0]]

    def logit_dist(self, X):
        """Среднее и дисперсия апостериорного распределения логита."""
        Xs = self.base.scaler.transform(X)
        Xa = np.hstack([Xs, np.ones((len(Xs), 1))])
        m = Xa @ self.w_full
        v = np.einsum("ij,jk,ik->i", Xa, self.cov, Xa)
        return m, np.maximum(v, 0)

    def predict_proba(self, X):
        m, v = self.logit_dist(X)
        return _sigmoid(m / np.sqrt(1 + math.pi * v / 8))


class PlattScaler:
    """p = σ(a·z + b), подбор a, b методом Ньютона на out-of-fold логитах."""

    def fit(self, z, y, iters=50):
        z, y = np.asarray(z, float), np.asarray(y, float)
        # сглаженные метки Платта — защищают от переуверенности
        n1, n0 = y.sum(), len(y) - y.sum()
        t = np.where(y == 1, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))
        a, b = 1.0, 0.0
        Z = np.c_[z, np.ones_like(z)]
        for _ in range(iters):
            p = _sigmoid(a * z + b)
            g = Z.T @ (p - t)
            H = (Z * (p * (1 - p))[:, None]).T @ Z + 1e-9 * np.eye(2)
            step = np.linalg.solve(H, g)
            a, b = a - step[0], b - step[1]
            if np.abs(step).max() < 1e-10:
                break
        self.a, self.b = float(a), float(b)
        return self

    def __call__(self, z):
        return self.a * np.asarray(z, float) + self.b   # калиброванный логит


class CheatProbability:
    """Итоговый ответ: вероятность читерства + доверительный интервал."""

    def __init__(self, model: BayesianLogReg, platt: PlattScaler, pi_data: float,
                 pi_deploy: float | None = None, level_z: float = Z90):
        self.model, self.platt, self.pi_data = model, platt, pi_data
        self.pi_deploy = pi_data if pi_deploy is None else pi_deploy
        self.z = level_z

    def _shift(self):
        return float(_logit(self.pi_deploy) - _logit(self.pi_data))

    def predict(self, X):
        m, v = self.model.logit_dist(X)
        sd = np.sqrt(v)
        shift = self._shift()
        # точечная оценка — медиана апостериорного распределения P (= σ от медианы логита);
        # в отличие от пробит-поправки МакКея она всегда лежит внутри интервала
        mid = _sigmoid(self.platt(m) + shift)
        lo = _sigmoid(self.platt(m - self.z * sd) + shift)
        hi = _sigmoid(self.platt(m + self.z * sd) + shift)
        return mid, lo, hi

    def to_json(self, feature_names):
        d = self.model.base.to_json(feature_names)
        d.update({"type": "bayes_logreg", "cov": self.model.cov.tolist(),
                  "platt_a": self.platt.a, "platt_b": self.platt.b, "pi_data": self.pi_data,
                  "l2": self.model.l2})
        return d

    @staticmethod
    def from_json(d, pi_deploy=None):
        model = BayesianLogReg(l2=d.get("l2", 1e-2))
        model.base = LogisticRegressionNP.from_json(d)
        model.cov = np.array(d["cov"])
        pl = PlattScaler()
        pl.a, pl.b = d["platt_a"], d["platt_b"]
        return CheatProbability(model, pl, d["pi_data"], pi_deploy)


def bootstrap_intervals(X_tr, y_tr, groups_tr, X_te, B=30, level=0.90, seed=0, l2=1e-2):
    """Контрольный (непараметрический) способ: B логрегрессий на бутстреп-выборках
    ГРУПП (пользователей/ботов), интервал = перцентили логита на тесте."""
    rng = np.random.default_rng(seed)
    groups_tr = np.asarray(groups_tr)
    uniq = np.unique(groups_tr)
    idx_by = {g: np.where(groups_tr == g)[0] for g in uniq}
    Z = []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        if len(np.unique(y_tr[idx])) < 2:
            continue
        m = LogisticRegressionNP(l2=l2, epochs=300).fit(X_tr[idx], y_tr[idx])
        Z.append(_logit(m.predict_proba(X_te)))
    Z = np.array(Z)
    a = (1 - level) / 2
    return np.quantile(Z, a, axis=0), np.quantile(Z, 1 - a, axis=0)
