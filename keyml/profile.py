"""Адаптивное профилирование: эмпирическая байесовская оценка слабых биграмм
и генерация персональных упражнений.

Модель (иерархическая нормальная, на log-латентностях):
    log-латентность биграммы b у пользователя u:  x_{u,b,i} ~ N(theta_{u,b}, sigma_b^2)
    истинная «скорость» пользователя на биграмме:   theta_{u,b} ~ N(mu_b + s_u, tau_b^2)
mu_b — средняя сложность биграммы по популяции, s_u — общий темп пользователя.
Гиперпараметры (mu_b, sigma_b^2, tau_b^2) оцениваются по всем пользователям
методом моментов (empirical Bayes). Апостериорное среднее:
    theta_hat = (n/sigma^2 * xbar + 1/tau^2 * (mu_b + s_u)) / (n/sigma^2 + 1/tau^2)
— при малом числе наблюдений n оценка «стягивается» к популяционной, что
убирает шум и не даёт объявить слабой биграмму, встреченную 1-2 раза.

Модуль написан на чистом Python (без numpy), чтобы его мог использовать сервер.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from .data import Keystroke
from .keyboard import LAYOUTS, key_label

MAX_GAP_MS = 1500.0     # паузы длиннее считаем «задумался», а не латентностью
DEFAULT_SIGMA2 = 0.12   # дисперсия log-латентности внутри пользователя (априорно)
DEFAULT_TAU2 = 0.04     # дисперсия между пользователями (априорно)


def bigram_latencies(ks: list[Keystroke]) -> list[tuple[str, str, float]]:
    """(a, b, down-down латентность) для пар без ошибок и без длинных пауз."""
    out = []
    for i in range(1, len(ks)):
        a, b = ks[i - 1], ks[i]
        if a.code == "Backspace" or b.code == "Backspace":
            continue
        if i >= 2 and ks[i - 2].code == "Backspace":
            continue
        dd = b.t_down - a.t_down
        if 15.0 < dd < MAX_GAP_MS:
            out.append((a.code, b.code, dd))
    return out


def key_errors(ks: list[Keystroke]) -> dict[str, int]:
    """Клавиша, нажатая прямо перед Backspace, считается ошибочной."""
    err: dict[str, int] = defaultdict(int)
    for i in range(1, len(ks)):
        if ks[i].code == "Backspace" and ks[i - 1].code != "Backspace":
            err[ks[i - 1].code] += 1
    return dict(err)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class BigramPrior:
    """Популяционные гиперпараметры по каждой биграмме."""

    def __init__(self):
        self.mu: dict[str, float] = {}
        self.sigma2: dict[str, float] = {}
        self.tau2: dict[str, float] = {}
        self.global_mu = math.log(180.0)

    @staticmethod
    def _key(a, b):
        return a + ">" + b

    def fit(self, user_lat: dict[str, list[tuple[str, str, float]]], min_users: int = 3):
        """user_lat: user_id -> список (a, b, dd)."""
        per_user: dict[str, dict[str, list[float]]] = {}
        all_logs = []
        for u, lats in user_lat.items():
            d = defaultdict(list)
            for a, b, dd in lats:
                lx = math.log(dd)
                d[self._key(a, b)].append(lx)
                all_logs.append(lx)
            per_user[u] = d
        if all_logs:
            self.global_mu = _mean(all_logs)
        # общий темп пользователя, чтобы tau отражал именно разброс «сложности»
        offsets = {u: _mean([x for xs in d.values() for x in xs]) - self.global_mu
                   for u, d in per_user.items()}
        bigrams = set(k for d in per_user.values() for k in d)
        for bg in bigrams:
            means, ns, within = [], [], []
            for u, d in per_user.items():
                xs = d.get(bg)
                if xs:
                    means.append(_mean(xs) - offsets[u])
                    ns.append(len(xs))
                    if len(xs) > 1:
                        within.append(_var(xs))
            if len(means) < min_users:
                continue
            s2 = _mean(within) if within else DEFAULT_SIGMA2
            s2 = max(s2, 0.01)
            # метод моментов: Var(xbar_u) = tau^2 + sigma^2 / n_u
            t2 = _var(means) - s2 * _mean([1.0 / n for n in ns])
            self.mu[bg] = _mean(means)
            self.sigma2[bg] = s2
            self.tau2[bg] = max(t2, 0.005)
        return self

    def get(self, a, b):
        k = self._key(a, b)
        return (self.mu.get(k, self.global_mu), self.sigma2.get(k, DEFAULT_SIGMA2),
                self.tau2.get(k, DEFAULT_TAU2))

    def to_json(self) -> dict:
        return {"mu": self.mu, "sigma2": self.sigma2, "tau2": self.tau2,
                "global_mu": self.global_mu}

    @staticmethod
    def from_json(d: dict) -> "BigramPrior":
        p = BigramPrior()
        p.mu, p.sigma2, p.tau2 = d["mu"], d["sigma2"], d["tau2"]
        p.global_mu = d.get("global_mu", p.global_mu)
        return p


def user_profile(lats: list[tuple[str, str, float]], prior: BigramPrior,
                 errors: dict[str, int] | None = None, min_n: int = 1) -> dict:
    """Апостериорные оценки по биграммам пользователя + «слабость»."""
    d = defaultdict(list)
    for a, b, dd in lats:
        d[(a, b)].append(math.log(dd))
    if not d:
        return {"offset": 0.0, "bigrams": [], "keys": []}
    # общий темп пользователя относительно популяции (взвешенно по числу наблюдений)
    num = sum(sum(xs) - len(xs) * prior.get(a, b)[0] for (a, b), xs in d.items())
    den = sum(len(xs) for xs in d.values())
    offset = num / den
    rows = []
    for (a, b), xs in d.items():
        if len(xs) < min_n:
            continue
        mu, s2, t2 = prior.get(a, b)
        n = len(xs)
        prec = n / s2 + 1.0 / t2
        post = (n / s2 * _mean(xs) + (mu + offset) / t2) / prec
        post_sd = math.sqrt(1.0 / prec)
        weakness = post - (mu + offset)          # >0 — медленнее, чем ожидается
        p_weak = _phi(weakness / post_sd)         # P(theta > mu + offset | данные)
        rows.append({"a": a, "b": b, "n": n, "raw_ms": math.exp(_mean(xs)),
                     "post_ms": math.exp(post), "expected_ms": math.exp(mu + offset),
                     "weakness": weakness, "p_weak": p_weak,
                     "label": f"{key_label(a)} → {key_label(b)}"})
    rows.sort(key=lambda r: -r["weakness"])
    keys = []
    if errors:
        cnt = defaultdict(int)
        for a, b, _ in lats:
            cnt[b] += 1
        for k, e in errors.items():
            # бета-биномиальное сглаживание частоты ошибок, априорно 2%
            rate = (e + 1.0) / (cnt.get(k, 0) + e + 50.0)
            keys.append({"key": k, "label": key_label(k), "errors": e, "rate": rate})
        keys.sort(key=lambda r: -r["rate"])
    return {"offset": offset, "speed_factor": math.exp(offset), "bigrams": rows, "keys": keys}


def word_codes(word: str, lang: str) -> list[str]:
    lay = LAYOUTS[lang]
    return [lay[ch][0] for ch in word if ch in lay]


def adaptive_exercise(profile: dict, lang: str, words: list[str], n_words: int = 14,
                      temperature: float = 0.15, seed: int | None = None) -> str:
    """Выбирает слова, содержащие слабые биграммы/клавиши (softmax-сэмплинг).

    score(word) = сумма положительной «слабости» его биграмм (+ бонус за клавиши с
    частыми ошибками), нормированная на sqrt(длины). Температура даёт
    разнообразие: упражнение не повторяется слово в слово.
    """
    rng = random.Random(seed)
    weak = {(r["a"], r["b"]): max(r["weakness"], 0.0) * r["p_weak"]
            for r in profile.get("bigrams", [])}
    key_bonus = {k["key"]: k["rate"] for k in profile.get("keys", [])}
    scored = []
    for w in set(words):
        codes = word_codes(w, lang)
        if len(codes) < 2:
            continue
        s = sum(weak.get((codes[i - 1], codes[i]), 0.0) for i in range(1, len(codes)))
        s += sum(key_bonus.get(c, 0.0) for c in codes)
        scored.append((s / math.sqrt(len(codes)), w))
    if not scored:
        return " ".join(rng.sample(words, min(n_words, len(words))))
    mx = max(s for s, _ in scored)
    weights = [math.exp((s - mx) / max(temperature, 1e-6)) for s, _ in scored]
    chosen = []
    pool = list(zip(weights, [w for _, w in scored]))
    for _ in range(min(n_words, len(pool))):
        tot = sum(w for w, _ in pool)
        r = rng.random() * tot
        acc = 0.0
        for i, (w, word) in enumerate(pool):
            acc += w
            if acc >= r:
                chosen.append(word)
                pool.pop(i)
                break
    return " ".join(chosen)
