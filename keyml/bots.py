"""Генератор читеров/ботов — негативный класс для античита.

Типы (от простого к сложному):
  const       — постоянная задержка (простейший автокликер)
  uniform     — равномерный случайный джиттер
  gauss       — нормальный джиттер
  ahk         — AutoHotkey/pyautogui: SetKeyDelay с Random, тайминги привязаны к таймеру ОС (~15.6 мс)
  humanized   — «очеловеченный»: логнормальные интервалы как у людей, паузы, фальшивые опечатки,
                но без связи с раскладкой (интервалы независимы от биграмм)
  replay      — воспроизводит записанные тайминги реального человека на другом тексте
  bigram      — продвинутый: знает популяционную сложность биграмм и генерирует по ней
Каждый «экземпляр бота» (один набор параметров) печатает несколько текстов —
это группа для GroupKFold, как отдельный пользователь.
"""
from __future__ import annotations

import math
import random

from .data import Attempt
from .keyboard import LAYOUTS, KEY_POS
from . import corpus

BOT_TYPES = ["const", "uniform", "gauss", "ahk", "humanized", "replay", "bigram"]


def _codes(prompt, lang):
    lay = LAYOUTS[lang]
    return [(lay[ch][0], ch, lay[ch][1]) for ch in prompt if ch in lay]


def _emit(seq, dds, holds, t0=500.0, quant=None, rng=None):
    events = []
    t = t0
    for (code, ch, shift), dd, h in zip(seq, dds, holds):
        t += dd
        td, tu = t, t + h
        if quant:
            td = round(td / quant) * quant + rng.uniform(-0.3, 0.3)
            tu = max(td + 0.1, round(tu / quant) * quant + rng.uniform(-0.3, 0.3))
        if shift:
            sh = "ShiftLeft"
            events.append(["d", sh, "Shift", round(td - 2.0, 1), 1])
            events.append(["u", sh, "Shift", round(tu + 1.0, 1), 1])
        events.append(["d", code, ch, round(td, 1), 1])
        events.append(["u", code, ch, round(tu, 1), 1])
    events.sort(key=lambda e: e[3])
    return events


class BotInstance:
    def __init__(self, bot_type: str, seed: int, human_pool: list[Attempt] | None = None,
                 bigram_mu: dict | None = None, key_hold_mu: dict | None = None):
        self.type = bot_type
        self.rng = random.Random(seed)
        self.id = f"bot_{bot_type}_{seed}"
        r = self.rng
        self.human_pool = human_pool or []
        self.bigram_mu = bigram_mu or {}
        self.key_hold_mu = key_hold_mu or {}
        self.p = {
            "dd": math.exp(r.uniform(math.log(60), math.log(320))),
            "hold": r.uniform(20, 120),
            "spread": r.uniform(0.15, 0.6),
            "sd_log": r.uniform(0.2, 0.5),
            "pause_p": r.uniform(0.0, 0.15),
            "typo_p": r.uniform(0.0, 0.04),
            "jitter": r.uniform(0.0, 0.05),
        }

    def type_text(self, prompt: str, lang: str) -> list:
        r, p = self.rng, self.p
        seq = _codes(prompt, lang)
        n = len(seq)
        t = self.type
        if t == "const":
            dds = [p["dd"]] * n
            holds = [p["hold"]] * n
            return _emit(seq, dds, holds, rng=r)
        if t == "uniform":
            lo, hi = p["dd"] * (1 - p["spread"]), p["dd"] * (1 + p["spread"])
            dds = [r.uniform(lo, hi) for _ in range(n)]
            holds = [r.uniform(p["hold"] * 0.6, p["hold"] * 1.4) for _ in range(n)]
            return _emit(seq, dds, holds, rng=r)
        if t == "gauss":
            dds = [max(5, r.gauss(p["dd"], p["dd"] * p["spread"] / 2)) for _ in range(n)]
            holds = [max(3, r.gauss(p["hold"], p["hold"] * 0.2)) for _ in range(n)]
            return _emit(seq, dds, holds, rng=r)
        if t == "ahk":
            # SetKeyDelay, Random: задержка и длительность нажатия кратны тику таймера
            dds = [r.uniform(p["dd"] * 0.5, p["dd"] * 1.5) for _ in range(n)]
            holds = [r.uniform(0, 60) for _ in range(n)]
            return _emit(seq, dds, holds, quant=15.625, rng=r)
        if t == "humanized":
            mu = math.log(p["dd"])
            dds, holds = [], []
            for code, ch, sh in seq:
                d = math.exp(r.gauss(mu, p["sd_log"]))
                if code == "Space" and r.random() < p["pause_p"]:
                    d += math.exp(r.gauss(6.0, 0.5))
                dds.append(d)
                holds.append(math.exp(r.gauss(math.log(max(p["hold"], 40)), 0.2)))
            ev = _emit(seq, dds, holds, rng=r)
            return self._fake_typos(ev, p["typo_p"])
        if t == "replay":
            src = r.choice([a for a in self.human_pool if a.prompt != prompt] or self.human_pool)
            ks = src.keystrokes()
            sdd = [ks[i].t_down - ks[i - 1].t_down for i in range(1, len(ks))] or [150.0]
            sh = [k.hold for k in ks] or [90.0]
            off = r.randrange(len(sdd))
            j = p["jitter"]
            dds = [sdd[(off + i) % len(sdd)] * (1 + r.uniform(-j, j)) for i in range(n)]
            holds = [sh[(off + i) % len(sh)] * (1 + r.uniform(-j, j)) for i in range(n)]
            return _emit(seq, dds, holds, rng=r)
        if t == "bigram":
            scale = r.gauss(0, 0.25)
            g = sum(self.bigram_mu.values()) / max(len(self.bigram_mu), 1) if self.bigram_mu else 5.2
            gh = sum(self.key_hold_mu.values()) / max(len(self.key_hold_mu), 1) if self.key_hold_mu else 4.5
            dds, holds, prev = [], [], None
            for code, ch, sh in seq:
                mu = self.bigram_mu.get((prev, code), g) if prev else g
                d = math.exp(mu + scale + r.gauss(0, p["sd_log"] * 0.7))
                if code == "Space" and r.random() < p["pause_p"]:
                    d += math.exp(r.gauss(6.0, 0.5))
                dds.append(d)
                holds.append(math.exp(self.key_hold_mu.get(code, gh) + r.gauss(0, 0.15)))
                prev = code
            return _emit(seq, dds, holds, rng=r)
        raise ValueError(t)

    def _fake_typos(self, events, p):
        """Вставляет «ошибка + Backspace», сдвигая время."""
        if p <= 0:
            return events
        out, shift = [], 0.0
        downs = [e for e in events if e[0] == "d" and e[1] != "ShiftLeft"]
        typo_at = {id(e) for e in downs[1:] if self.rng.random() < p}
        for e in events:
            if id(e) in typo_at:
                t = e[3] + shift
                out.append(["d", "KeyJ", "?", round(t, 1), 1])
                out.append(["u", "KeyJ", "?", round(t + 80, 1), 1])
                out.append(["d", "Backspace", "Backspace", round(t + 300, 1), 1])
                out.append(["u", "Backspace", "Backspace", round(t + 380, 1), 1])
                shift += 480
            out.append([e[0], e[1], e[2], round(e[3] + shift, 1), e[4]])
        out.sort(key=lambda e: e[3])
        return out


def generate_bots(n_instances_per_type: int = 6, attempts_per_instance: int = 12,
                  human_pool: list[Attempt] | None = None, bigram_mu=None, key_hold_mu=None,
                  types=BOT_TYPES, seed: int = 123) -> list[Attempt]:
    rng = random.Random(seed)
    out, aid = [], 0
    for t in types:
        for k in range(n_instances_per_type):
            bot = BotInstance(t, seed * 1000 + k * 7 + BOT_TYPES.index(t) * 101, human_pool, bigram_mu, key_hold_mu)
            lang = "ru" if rng.random() < 0.8 else "en"
            for j in range(attempts_per_instance):
                kind = "calib" if j % 2 == 0 else "text"
                prompt = corpus.CALIB[lang] if kind == "calib" else rng.choice(corpus.TEXTS[lang])
                ev = bot.type_text(prompt, lang)
                out.append(Attempt(f"b{aid}", bot.id, 1, kind, lang, prompt, ev, typed=prompt,
                                   mode="cheat", cheat_type=t, source="bot", label=1, bot_type=t))
                aid += 1
    return out
