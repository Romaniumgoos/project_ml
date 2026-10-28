"""Извлечение признаков из нажатий.

Обозначения для соседних нажатий i-1, i:
  hold  = up_i - down_i                 — время удержания клавиши
  dd    = down_i - down_{i-1}           — интервал «нажатие-нажатие» (латентность биграммы)
  ud    = down_i - up_{i-1}             — «отпустил-нажал»; < 0 означает перекрытие (rollover)
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .data import Attempt, Keystroke
from .keyboard import LAYOUTS

EPS = 1e-9


def _stats(x: np.ndarray, prefix: str) -> dict[str, float]:
    if len(x) < 3:
        return {f"{prefix}_{k}": 0.0 for k in
                ("mean", "std", "cv", "skew", "kurt", "p10", "p50", "p90", "logstd")}
    m, s = x.mean(), x.std()
    z = (x - m) / (s + EPS)
    lx = np.log(np.clip(x, 1.0, None))
    return {
        f"{prefix}_mean": m, f"{prefix}_std": s, f"{prefix}_cv": s / (abs(m) + EPS),
        f"{prefix}_skew": float((z ** 3).mean()), f"{prefix}_kurt": float((z ** 4).mean() - 3),
        f"{prefix}_p10": float(np.percentile(x, 10)), f"{prefix}_p50": float(np.median(x)),
        f"{prefix}_p90": float(np.percentile(x, 90)), f"{prefix}_logstd": float(lx.std()),
    }


def _entropy(x: np.ndarray, lo=math.log(10), hi=math.log(2500), bins=24) -> float:
    if len(x) < 3:
        return 0.0
    h, _ = np.histogram(np.log(np.clip(x, 1.0, None)), bins=bins, range=(lo, hi))
    p = h / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 2:
        return 0.0
    a, b = x[:-lag] - x.mean(), x[lag:] - x.mean()
    return float((a * b).mean() / (x.var() + EPS))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or a.std() < EPS or b.std() < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _rayleigh(x: np.ndarray, period: float) -> float:
    """Насколько значения «прилипают» к сетке с шагом period (таймер ОС у ботов)."""
    if len(x) < 5:
        return 0.0
    ang = 2 * np.pi * x / period
    return float(np.abs(np.exp(1j * ang).mean()))


def timing_arrays(ks: list[Keystroke]):
    hold = np.array([k.hold for k in ks], dtype=float)
    down = np.array([k.t_down for k in ks], dtype=float)
    up = np.array([k.t_up for k in ks], dtype=float)
    dd = np.diff(down)
    ud = down[1:] - up[:-1]
    return hold, dd, ud


class FeatureExtractor:
    """fit() запоминает популяционные средние по биграммам/клавишам (только по
    обучающим людям внутри фолда — без утечки), transform() строит вектор."""

    def __init__(self):
        self.bigram_mu: dict[tuple[str, str], float] = {}
        self.key_hold_mu: dict[str, float] = {}
        self.names: list[str] = []

    # ------------------------------------------------------------ fit
    def fit(self, attempts: list[Attempt]):
        bg = defaultdict(list)
        kh = defaultdict(list)
        for a in attempts:
            if a.label != 0:
                continue
            ks = a.keystrokes()
            for i, k in enumerate(ks):
                if 5 < k.hold < 600:
                    kh[k.code].append(math.log(k.hold))
                if i:
                    dd = k.t_down - ks[i - 1].t_down
                    if 15 < dd < 1500:
                        bg[(ks[i - 1].code, k.code)].append(math.log(dd))
        self.bigram_mu = {b: float(np.mean(v)) for b, v in bg.items() if len(v) >= 3}
        self.key_hold_mu = {c: float(np.mean(v)) for c, v in kh.items() if len(v) >= 3}
        self.names = list(self._features(attempts[0]).keys()) if attempts else []
        return self

    # ------------------------------------------------------------ features
    def _features(self, a: Attempt) -> dict[str, float]:
        ks = a.keystrokes()
        f: dict[str, float] = {}
        n = len(ks)
        # длину текста и долю покрытия таблицей биграмм в признаки НЕ включаем:
        # они описывают текст, а не поведение (иначе модель учится отличать калибровку от текстов)
        if n < 4:
            ks = ks + [Keystroke("KeyA", "a", 0, 80)] * (4 - n)
        hold, dd, ud = timing_arrays(ks)
        dur = (ks[-1].t_up - ks[0].t_down) / 1000.0
        f["wpm"] = (n / 5.0) / (dur / 60.0 + EPS)
        f["bs_rate"] = sum(k.code == "Backspace" for k in ks) / n if n else 0.0
        f.update(_stats(hold, "hold"))
        f.update(_stats(dd, "dd"))
        f["ud_mean"], f["ud_std"] = float(ud.mean()), float(ud.std())
        f["rollover_frac"] = float((ud < 0).mean())
        f["pause_frac"] = float((dd > 1000).mean())
        f["dd_entropy"] = _entropy(dd)
        f["hold_entropy"] = _entropy(hold, math.log(5), math.log(500))
        ldd = np.log(np.clip(dd, 1, None))
        lh = np.log(np.clip(hold, 1, None))
        f["dd_ac1"], f["dd_ac2"] = _autocorr(ldd, 1), _autocorr(ldd, 2)
        f["hold_ac1"] = _autocorr(lh, 1)
        f["dd_smooth"] = float(np.abs(np.diff(ldd)).mean()) if len(ldd) > 1 else 0.0
        f["hold_dd_corr"] = _corr(lh[:-1], ldd)
        # структура: согласуются ли тайминги с «сложностью» биграмм у людей
        pri, obs = [], []
        for i in range(1, len(ks)):
            mu = self.bigram_mu.get((ks[i - 1].code, ks[i].code))
            if mu is not None and 15 < dd[i - 1] < 1500:
                pri.append(mu)
                obs.append(ldd[i - 1])
        pri, obs = np.array(pri), np.array(obs)
        f["corr_dd_prior"] = _corr(obs, pri)
        if len(pri) > 4:
            resid = obs - pri - (obs - pri).mean()
            f["resid_ratio"] = float(resid.var() / (obs.var() + EPS))
        else:
            f["resid_ratio"] = 1.0
        hp = [(lh[i], self.key_hold_mu[k.code]) for i, k in enumerate(ks)
              if k.code in self.key_hold_mu]
        f["corr_hold_prior"] = _corr(np.array([x for x, _ in hp]), np.array([y for _, y in hp])) if hp else 0.0
        sp = np.array([ks[i].code == "Space" for i in range(1, len(ks))])
        f["space_dd_ratio"] = float(dd[sp].mean() / (dd.mean() + EPS)) if sp.any() else 1.0
        after_sp = np.array([ks[i - 1].code == "Space" for i in range(1, len(ks))])
        f["after_space_ratio"] = float(dd[after_sp].mean() / (dd.mean() + EPS)) if after_sp.any() else 1.0
        both = np.concatenate([dd, hold])
        f["quant_15ms"] = max(_rayleigh(both, 15.625), _rayleigh(both, 16.667))
        f["quant_10ms"] = _rayleigh(both, 10.0)
        f["n_dup_dd"] = float(len(dd) - len(np.unique(np.round(dd, 1)))) / max(len(dd), 1)
        return {k: float(v) if np.isfinite(v) else 0.0 for k, v in f.items()}

    def transform(self, attempts: list[Attempt]) -> np.ndarray:
        rows = []
        for a in attempts:
            f = self._features(a)
            rows.append([f[n] for n in self.names])
        return np.array(rows, dtype=float)

    def fit_transform(self, attempts):
        return self.fit(attempts).transform(attempts)

    # ------------------------------------------------------------ последовательности для GRU
    SEQ_DIM = 7

    def sequence(self, a: Attempt, max_len: int = 160) -> np.ndarray:
        ks = a.keystrokes()[:max_len]
        out = np.zeros((len(ks), self.SEQ_DIM), dtype=np.float32)
        for i, k in enumerate(ks):
            h = math.log(max(k.hold, 1.0))
            out[i, 0] = (h - 4.4) / 0.5
            if i:
                dd = k.t_down - ks[i - 1].t_down
                ud = k.t_down - ks[i - 1].t_up
                ldd = math.log(max(dd, 1.0))
                out[i, 1] = (ldd - 5.2) / 0.6
                out[i, 2] = float(np.clip(ud / 150.0, -3, 3))
                mu = self.bigram_mu.get((ks[i - 1].code, k.code))
                out[i, 3] = 0.0 if mu is None else (ldd - mu) / 0.5   # отклонение от популяции
                out[i, 4] = 0.0 if mu is None else 1.0
            mh = self.key_hold_mu.get(k.code)
            out[i, 5] = 0.0 if mh is None else (h - mh) / 0.4
            out[i, 6] = 1.0 if k.code == "Space" else (-1.0 if k.code == "Backspace" else 0.0)
        return out


# ---------------------------------------------------------------- биометрия
def calib_vector(a: Attempt) -> np.ndarray | None:
    """Вектор фиксированной длины для калибровочной фразы (как в датасете CMU):
    hold каждой клавиши + dd и ud каждой биграммы. None, если были ошибки."""
    lay = LAYOUTS.get(a.lang)
    if lay is None:
        return None
    expected = [lay[ch][0] for ch in a.prompt if ch in lay]
    ks = a.keystrokes()
    if [k.code for k in ks] != expected:
        return None
    hold, dd, ud = timing_arrays(ks)
    return np.concatenate([hold, dd, ud]).astype(float)
