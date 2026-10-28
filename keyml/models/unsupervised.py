"""Без учителя: k-means++, GMM (EM), PCA, детекторы аномалий для биометрии. numpy."""
from __future__ import annotations

import numpy as np


# ======================================================================= k-means
class KMeansNP:
    def __init__(self, k, n_init=10, max_iter=300, tol=1e-6, seed=0):
        self.k, self.n_init, self.max_iter, self.tol, self.seed = k, n_init, max_iter, tol, seed

    def _kpp(self, X, rng):
        """k-means++: следующий центр выбирается с вероятностью ~ D(x)^2."""
        C = [X[rng.integers(len(X))]]
        for _ in range(1, self.k):
            d2 = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
            C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        return np.array(C)

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            C = self._kpp(X, rng)
            for _ in range(self.max_iter):
                D = ((X[:, None] - C[None]) ** 2).sum(-1)
                lab = D.argmin(1)
                Cn = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(self.k)])
                shift = ((Cn - C) ** 2).sum()
                C = Cn
                if shift < self.tol:
                    break
            inertia = ((X - C[lab]) ** 2).sum()
            if best is None or inertia < best[0]:
                best = (inertia, C, lab)
        self.inertia_, self.centers_, self.labels_ = best
        return self

    def predict(self, X):
        return ((X[:, None] - self.centers_[None]) ** 2).sum(-1).argmin(1)


# ======================================================================= GMM
class GMMNP:
    """Смесь гауссиан с полными ковариациями, EM-алгоритм.
    E-шаг: r_ik = pi_k N(x_i|mu_k,S_k) / sum_j pi_j N(x_i|mu_j,S_j)  (в лог-пространстве)
    M-шаг: N_k = sum_i r_ik, mu_k = sum r_ik x_i / N_k, S_k = sum r_ik (x-mu)(x-mu)^T / N_k + reg*I
    """

    def __init__(self, k, max_iter=300, tol=1e-5, reg=1e-3, n_init=5, seed=0, cov_type="full"):
        self.k, self.max_iter, self.tol, self.reg, self.n_init, self.seed = k, max_iter, tol, reg, n_init, seed
        self.cov_type = cov_type   # "full" или "diag" (меньше параметров — для малых выборок)

    def _log_gauss(self, X, mu, S):
        d = X.shape[1]
        L = np.linalg.cholesky(S)
        z = np.linalg.solve(L, (X - mu).T)
        return -0.5 * (z ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)

    def _estep(self, X):
        lp = np.stack([np.log(self.weights_[j] + 1e-300) + self._log_gauss(X, self.means_[j], self.covs_[j])
                       for j in range(self.k)], 1)
        m = lp.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(lp - m).sum(1))
        return np.exp(lp - lse[:, None]), lse.sum()

    def fit(self, X):
        n, d = X.shape
        best = None
        for init in range(self.n_init):
            km = KMeansNP(self.k, n_init=1, seed=self.seed + init).fit(X)
            self.means_ = km.centers_.copy()
            self.covs_ = np.array([np.cov(X[km.labels_ == j].T) + self.reg * np.eye(d)
                                   if (km.labels_ == j).sum() > d else np.cov(X.T) + self.reg * np.eye(d)
                                   for j in range(self.k)])
            self.weights_ = np.bincount(km.labels_, minlength=self.k) / n + 1e-6
            prev = -np.inf
            for it in range(self.max_iter):
                R, ll = self._estep(X)
                Nk = R.sum(0) + 1e-10
                self.weights_ = Nk / n
                self.means_ = (R.T @ X) / Nk[:, None]
                for j in range(self.k):
                    D = X - self.means_[j]
                    C = (R[:, j, None] * D).T @ D / Nk[j]
                    if self.cov_type == "diag":
                        C = np.diag(np.diag(C))
                    self.covs_[j] = C + self.reg * np.eye(d)
                if ll - prev < self.tol * abs(ll):
                    break
                prev = ll
            if best is None or ll > best[0]:
                best = (ll, self.means_.copy(), self.covs_.copy(), self.weights_.copy())
        self.loglik_, self.means_, self.covs_, self.weights_ = best
        return self

    def n_params(self, d):
        cov = d if self.cov_type == "diag" else d * (d + 1) / 2
        return self.k * (d + cov) + self.k - 1

    def bic(self, X):
        return -2 * self.loglik_ + self.n_params(X.shape[1]) * np.log(len(X))

    def predict_proba(self, X):
        return self._estep(X)[0]

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# ======================================================================= PCA
class PCANP:
    def __init__(self, n_components=2):
        self.n = n_components

    def fit(self, X):
        self.mean_ = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = Vt[: self.n]
        self.explained_variance_ratio_ = (S ** 2 / (S ** 2).sum())[: self.n]
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T


# ======================================================================= биометрия
class MahalanobisDetector:
    """Шаблон пользователя = N(mu, S). Оценка аномальности — расстояние Махаланобиса
    в лог-пространстве таймингов. Ковариация оценивается со «сжатием» (shrinkage)
    к диагонали, т.к. обучающих повторов (10-20) меньше, чем размерность (~90):
        S = (1 - a) * S_выб + a * diag(S_выб)."""

    def __init__(self, shrink=0.5, log=True):
        self.shrink, self.log = shrink, log

    def _t(self, X):
        return np.log(np.clip(X + 200.0, 1.0, None)) if self.log else X  # +200: ud бывает < 0

    def fit(self, X):
        Z = self._t(X)
        self.mu = Z.mean(0)
        S = np.cov(Z.T) if len(Z) > 1 else np.eye(Z.shape[1])
        S = (1 - self.shrink) * S + self.shrink * np.diag(np.diag(S)) + 1e-6 * np.eye(len(S))
        self.P = np.linalg.inv(S)
        return self

    def score(self, X):
        D = self._t(X) - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", D, self.P, D))


class ScaledManhattanDetector:
    """Лучший простой детектор в работе Killourhy & Maxion (2009):
    score = sum_j |x_j - mu_j| / MAD_j."""

    def fit(self, X):
        self.mu = X.mean(0)
        self.mad = np.abs(X - self.mu).mean(0) + 1e-3
        return self

    def score(self, X):
        return (np.abs(X - self.mu) / self.mad).sum(1)
