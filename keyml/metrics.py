"""Метрики и разбиения — собственная реализация (numpy)."""
from __future__ import annotations

import numpy as np


class StandardScaler:
    def fit(self, X):
        self.mean_ = X.mean(0)
        self.std_ = X.std(0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def group_kfold(groups, n_splits=5, seed=0):
    """Как GroupKFold: все попытки одного пользователя/бота — в одном фолде.
    Группы распределяются жадно по размеру, чтобы фолды были сбалансированы."""
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]
    fold_of, sizes = {}, np.zeros(n_splits)
    for i in order:
        f = int(np.argmin(sizes))
        fold_of[uniq[i]] = f
        sizes[f] += counts[i]
    fold = np.array([fold_of[g] for g in groups])
    for f in range(n_splits):
        yield np.where(fold != f)[0], np.where(fold == f)[0]


def stratified_group_kfold(groups, y, n_splits=5, seed=0):
    """Группы людей и ботов раскладываются по фолдам раздельно — в каждом фолде есть оба класса."""
    groups, y = np.asarray(groups), np.asarray(y)
    fold = np.zeros(len(groups), dtype=int)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        for f, (_, te) in enumerate(group_kfold(groups[idx], n_splits, seed)):
            fold[idx[te]] = f
    for f in range(n_splits):
        yield np.where(fold != f)[0], np.where(fold == f)[0]


def roc_curve(y, score):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y, s = y[order], score[order]
    distinct = np.where(np.diff(s))[0]
    thr_idx = np.r_[distinct, len(y) - 1]
    tps = np.cumsum(y)[thr_idx]
    fps = (1 + thr_idx) - tps
    P, N = max(y.sum(), 1), max(len(y) - y.sum(), 1)
    tpr = np.r_[0, tps / P]
    fpr = np.r_[0, fps / N]
    thr = np.r_[np.inf, s[thr_idx]]
    return fpr, tpr, thr


def roc_auc(y, score):
    """AUC через ранги (статистика Манна-Уитни), с учётом связей."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N))


def eer(y, score):
    """Equal Error Rate: точка, где FAR (пропуск чужого) = FRR (отказ своему).
    score: больше = больше похоже на класс 1 (самозванца)."""
    fpr, tpr, _ = roc_curve(y, score)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    # линейная интерполяция между соседними точками
    if i + 1 < len(fpr) and (fnr[i] - fpr[i]) * (fnr[i + 1] - fpr[i + 1]) < 0:
        a = (fnr[i] - fpr[i]) / ((fnr[i] - fpr[i]) - (fnr[i + 1] - fpr[i + 1]))
        return float(fpr[i] + a * (fpr[i + 1] - fpr[i]))
    return float((fpr[i] + fnr[i]) / 2)


def tpr_at_fpr(y, score, max_fpr=0.01):
    fpr, tpr, _ = roc_curve(y, score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def binary_report(y, prob, thr=0.5):
    y = np.asarray(y)
    pred = (np.asarray(prob) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"accuracy": (tp + tn) / max(len(y), 1), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12), "fpr": fp / max(fp + tn, 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def silhouette(X, labels):
    X, labels = np.asarray(X), np.asarray(labels)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = 0.0 if not same.any() else (b - a) / max(a, b, 1e-12)
    return float(s.mean())


def adjusted_rand_index(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    M = np.zeros((len(ua), len(ub)))
    np.add.at(M, (ia, ib), 1)
    c2 = lambda x: x * (x - 1) / 2
    sum_ij = c2(M).sum()
    sa, sb = c2(M.sum(1)).sum(), c2(M.sum(0)).sum()
    exp = sa * sb / c2(len(a))
    mx = (sa + sb) / 2
    return float((sum_ij - exp) / (mx - exp)) if mx != exp else 1.0


# ---------------------------------------------------------------- калибровка вероятностей
def brier(y, p):
    """Средний квадрат ошибки вероятности (0 — идеал, 0.25 — монетка)."""
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss(y, p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_curve(y, p, n_bins=10):
    """Для каждого бина предсказанной вероятности: средний прогноз, реальная доля, размер."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        if m.any():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def ece(y, p, n_bins=10):
    """Expected Calibration Error: взвешенное |прогноз − реальная частота| по бинам."""
    rows = reliability_curve(y, p, n_bins)
    n = sum(r[2] for r in rows)
    return float(sum(r[2] * abs(r[0] - r[1]) for r in rows) / max(n, 1))
