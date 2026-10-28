"""Симулятор «синтетических людей».

ВАЖНО: это НЕ замена реальному датасету. Симулятор нужен, чтобы отладить весь
пайплайн (признаки -> модели -> метрики) до того, как набраны реальные
пользователи, и чтобы иметь контролируемый эксперимент, где известна «истина»
(например, какие биграммы у пользователя на самом деле слабые).

Модель печати человека (на log-шкале):
  log dd_i = log base_u + sens_u * cost(b_{i-1}, b_i) + slow_u[палец] + quirk_u[биграмма]
             + AR(1)-дрейф темпа + шум
  log hold_i = log hold_u + offset_u[клавиша] + 0.25 * дрейф темпа + шум
плюс паузы на пробелах, опечатки с исправлением через Backspace, нажатия Shift,
изменение навыка от сессии к сессии.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

from .data import Attempt
from .keyboard import KEY_FINGER, KEY_POS, LAYOUTS, bigram_cost
from . import corpus


def _stable_normal(seed_str: str, sd: float) -> float:
    h = int(hashlib.md5(seed_str.encode()).hexdigest()[:12], 16)
    r = random.Random(h)
    return r.gauss(0.0, sd)


def _neighbors(code: str) -> list[str]:
    p = KEY_POS.get(code)
    if p is None:
        return [code]
    res = [c for c, q in KEY_POS.items()
           if c.startswith("Key") and c != code and abs(q[0] - p[0]) <= 1.3 and abs(q[1] - p[1]) <= 1.0]
    return res or [code]


@dataclass
class SimUser:
    user_id: str
    seed: int
    touch_typist: bool = True
    base_dd: float = 200.0
    sens: float = 1.0
    finger_slow: list = field(default_factory=list)
    quirk_sd: float = 0.15
    noise_sd: float = 0.25
    hold_base: float = 95.0
    hold_sd: float = 0.15
    ar_rho: float = 0.7
    ar_sd: float = 0.07
    pause_p: float = 0.08
    err_p: float = 0.02
    learn: float = 0.97

    archetype: str = "touch"

    # Три «архетипа» печати — скрытая истина, которую кластеризация должна найти сама.
    ARCHETYPES = {
        # слепая печать: быстро, ровно, много перекрытий клавиш
        "touch": dict(base=165, noise=(0.15, 0.22), pause=(0.01, 0.06), err=(0.005, 0.02),
                      hold=88, sens=1.0),
        # смешанная (смотрит на клавиатуру иногда): средняя скорость, паузы на словах
        "hybrid": dict(base=240, noise=(0.22, 0.30), pause=(0.06, 0.15), err=(0.015, 0.035),
                       hold=105, sens=1.2),
        # двумя-тремя пальцами: медленно, долгие удержания, сильная зависимость от расстояния
        "hunt": dict(base=380, noise=(0.30, 0.42), pause=(0.12, 0.25), err=(0.02, 0.05),
                     hold=125, sens=1.6),
    }

    @staticmethod
    def random(user_id: str, seed: int) -> "SimUser":
        r = random.Random(seed)
        arch = r.choices(["touch", "hybrid", "hunt"], weights=[0.45, 0.35, 0.20])[0]
        A = SimUser.ARCHETYPES[arch]
        return SimUser(
            user_id=user_id, seed=seed, touch_typist=(arch == "touch"), archetype=arch,
            base_dd=math.exp(r.gauss(math.log(A["base"]), 0.15)),
            sens=max(0.3, r.gauss(A["sens"], 0.2)),
            finger_slow=[r.gauss(0, 0.10) for _ in range(9)],
            quirk_sd=r.uniform(0.10, 0.22),
            noise_sd=r.uniform(*A["noise"]),
            hold_base=max(55.0, r.gauss(A["hold"], 12)), hold_sd=r.uniform(0.10, 0.22),
            ar_rho=r.uniform(0.5, 0.85), ar_sd=r.uniform(0.04, 0.10),
            pause_p=r.uniform(*A["pause"]), err_p=r.uniform(*A["err"]),
            learn=r.uniform(0.95, 0.995),
        )

    def quirk(self, a: str, b: str) -> float:
        """Личная (устойчивая) особенность пользователя на биграмме — «отпечаток»."""
        return _stable_normal(f"{self.user_id}|{a}|{b}", self.quirk_sd)

    def key_hold_offset(self, c: str) -> float:
        return _stable_normal(f"{self.user_id}|hold|{c}", 0.12)

    # ------------------------------------------------------------------
    def type_text(self, prompt: str, lang: str, session_no: int, rep: int,
                  rng: random.Random) -> list:
        lay = LAYOUTS[lang]
        # сессионный дрейф и обучение
        sess_shift = _stable_normal(f"{self.user_id}|sess|{session_no}", 0.12)
        base = math.log(self.base_dd) + sess_shift + math.log(self.learn) * (session_no - 1)
        base += math.log(max(0.95, 1.0 - 0.005 * rep))   # разгон на повторах фразы
        hold_base = math.log(self.hold_base) + 0.3 * sess_shift
        events = []
        t = rng.uniform(300, 900)
        prev = None
        drift = 0.0

        def press(code, key, t_down, tempo):
            h = math.exp(hold_base + self.key_hold_offset(code) + 0.25 * tempo
                         + rng.gauss(0, self.hold_sd))
            events.append(["d", code, key, round(t_down, 1), 1])
            events.append(["u", code, key, round(t_down + h, 1), 1])
            return t_down + h

        for ch in prompt:
            if ch not in lay:
                continue
            code, shift = lay[ch]
            drift = self.ar_rho * drift + rng.gauss(0, self.ar_sd)
            # опечатка: соседняя клавиша, (иногда ещё один символ), Backspace
            if prev is not None and code != "Space" and rng.random() < self.err_p:
                wrong = rng.choice(_neighbors(code))
                t += math.exp(base + rng.gauss(0, self.noise_sd))
                press(wrong, "?", t, drift)
                t += math.exp(math.log(300) + rng.gauss(0, 0.3))   # заметил ошибку
                press("Backspace", "Backspace", t, drift)
                t += math.exp(math.log(180) + rng.gauss(0, 0.3))
                prev = "Backspace"
            if prev is None:
                ld = base
            else:
                ld = (base + self.sens * bigram_cost(prev, code)
                      + self.finger_slow[KEY_FINGER.get(code, 4)] + self.quirk(prev, code)
                      + _stable_normal(f"{self.user_id}|{session_no}|{prev}|{code}", 0.12)  # день ото дня
                      + drift + rng.gauss(0, self.noise_sd))
            dd = math.exp(ld)
            if code == "Space" and rng.random() < self.pause_p:
                dd += math.exp(rng.gauss(6.0, 0.5))
            if shift:
                sh = "ShiftRight" if KEY_POS.get(code, (0, 0))[0] < 6 else "ShiftLeft"
                ts = t + dd * 0.6
                events.append(["d", sh, "Shift", round(ts, 1), 1])
                dd += math.exp(math.log(70) + rng.gauss(0, 0.3))
                t += dd
                t_up = press(code, ch, t, drift)
                events.append(["u", sh, "Shift", round(t_up + rng.uniform(10, 60), 1), 1])
            else:
                t += dd
                press(code, ch, t, drift)
            prev = code
        events.sort(key=lambda e: e[3])
        return events


def simulate_users(n_users: int = 40, n_sessions: int = 3, calib_reps: int = 10,
                   texts_per_session: int = 8, seed: int = 0) -> tuple[list[Attempt], list[SimUser]]:
    rng = random.Random(seed)
    users = [SimUser.random(f"sim{u:03d}", seed * 1000 + u) for u in range(n_users)]
    attempts = []
    aid = 0
    for u in users:
        lang = "ru" if rng.random() < 0.8 else "en"
        for s in range(1, n_sessions + 1):
            for rep in range(calib_reps):
                ev = u.type_text(corpus.CALIB[lang], lang, s, rep, rng)
                attempts.append(Attempt(f"s{aid}", u.user_id, s, "calib", lang,
                                        corpus.CALIB[lang], ev, typed=corpus.CALIB[lang],
                                        source="sim", label=0))
                aid += 1
            for txt in rng.sample(corpus.TEXTS[lang], texts_per_session):
                ev = u.type_text(txt, lang, s, 0, rng)
                attempts.append(Attempt(f"s{aid}", u.user_id, s, "text", lang, txt, ev,
                                        typed=txt, source="sim", label=0))
                aid += 1
    return attempts, users
