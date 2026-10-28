"""Логистическая регрессия и многослойный перцептрон на numpy (ручной backprop, Adam)."""
from __future__ import annotations

import json

import numpy as np

from ..metrics import StandardScaler


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class LogisticRegressionNP:
    """p(y=1|x) = sigmoid(w·x + b);  loss = взвешенная BCE + l2/2 * ||w||^2.

    Градиент: dL/dw = X^T (p - y) / n + l2 * w. Классы взвешиваются обратно
    пропорционально частоте (ботов может быть меньше/больше, чем людей)."""

    def __init__(self, l2=1e-2, lr=0.05, epochs=600, balanced=True):
        self.l2, self.lr, self.epochs, self.balanced = l2, lr, epochs, balanced

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        y = np.asarray(y, dtype=float)
        n, d = Xs.shape
        sw = np.ones(n)
        if self.balanced:
            pos = y.mean()
            sw = np.where(y == 1, 0.5 / max(pos, 1e-9), 0.5 / max(1 - pos, 1e-9))
        self.w, self.b = np.zeros(d), np.zeros(1)
        opt = Adam([self.w, self.b], lr=self.lr)
        self.loss_history = []
        for _ in range(self.epochs):
            p = _sigmoid(Xs @ self.w + self.b[0])
            r = sw * (p - y) / n
            gw = Xs.T @ r + self.l2 * self.w
            gb = np.array([r.sum()])
            opt.step([gw, gb])
            loss = -(sw * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).mean()
            self.loss_history.append(loss + 0.5 * self.l2 * (self.w ** 2).sum())
        return self

    def predict_proba(self, X):
        return _sigmoid(self.scaler.transform(X) @ self.w + self.b[0])

    def to_json(self, feature_names):
        return {"type": "logreg", "features": list(feature_names), "w": self.w.tolist(),
                "b": float(self.b[0]), "mean": self.scaler.mean_.tolist(),
                "std": self.scaler.std_.tolist()}

    @staticmethod
    def from_json(d):
        m = LogisticRegressionNP()
        m.w, m.b = np.array(d["w"]), np.array([d["b"]])
        m.scaler = StandardScaler()
        m.scaler.mean_, m.scaler.std_ = np.array(d["mean"]), np.array(d["std"])
        return m


class MLPNP:
    """Перцептрон: d -> h1 -> h2 -> 1, ReLU, dropout, мини-батчи, ручной backprop."""

    def __init__(self, hidden=(32, 16), lr=3e-3, epochs=150, batch=64, l2=1e-4,
                 dropout=0.1, seed=0):
        self.hidden, self.lr, self.epochs, self.batch = hidden, lr, epochs, batch
        self.l2, self.dropout, self.seed = l2, dropout, seed

    def _init(self, d):
        rng = np.random.default_rng(self.seed)
        sizes = [d, *self.hidden, 1]
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.B = [np.zeros(b) for b in sizes[1:]]
        self.rng = rng

    def _forward(self, X, train=False):
        acts, masks = [X], []
        h = X
        for i, (W, B) in enumerate(zip(self.W, self.B)):
            z = h @ W + B
            if i < len(self.W) - 1:
                h = np.maximum(z, 0)
                if train and self.dropout > 0:
                    m = (self.rng.random(h.shape) > self.dropout) / (1 - self.dropout)
                    h = h * m
                    masks.append(m)
                else:
                    masks.append(None)
            else:
                h = _sigmoid(z[:, 0])
            acts.append(h)
        return acts, masks

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        y = np.asarray(y, dtype=float)
        self._init(Xs.shape[1])
        pos = y.mean()
        sw_all = np.where(y == 1, 0.5 / max(pos, 1e-9), 0.5 / max(1 - pos, 1e-9))
        opt = Adam(self.W + self.B, lr=self.lr)
        self.loss_history = []
        n = len(y)
        for ep in range(self.epochs):
            perm = self.rng.permutation(n)
            tot = 0.0
            for s in range(0, n, self.batch):
                idx = perm[s:s + self.batch]
                xb, yb, sw = Xs[idx], y[idx], sw_all[idx]
                acts, masks = self._forward(xb, train=True)
                p = acts[-1]
                tot += -(sw * (yb * np.log(p + 1e-12) + (1 - yb) * np.log(1 - p + 1e-12))).sum()
                delta = (sw * (p - yb) / len(idx))[:, None]      # dL/dz последнего слоя
                gW, gB = [None] * len(self.W), [None] * len(self.W)
                for i in range(len(self.W) - 1, -1, -1):
                    gW[i] = acts[i].T @ delta + self.l2 * self.W[i]
                    gB[i] = delta.sum(0)
                    if i > 0:
                        delta = delta @ self.W[i].T
                        if masks[i - 1] is not None:
                            delta = delta * masks[i - 1]
                        delta = delta * (acts[i] > 0)
                opt.step(gW + gB)
            self.loss_history.append(tot / n)
        return self

    def predict_proba(self, X):
        return self._forward(self.scaler.transform(X))[0][-1]


def save_model(model, feature_names, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model.to_json(feature_names), f)
