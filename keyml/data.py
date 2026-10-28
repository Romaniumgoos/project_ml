"""Формат данных и загрузка.

Одна «попытка» (attempt) — один набранный пользователем текст (калибровочная
фраза или текст из корпуса). Сырые события хранятся как список
[тип, code, key, t_ms, isTrusted], где тип 'd' = keydown, 'u' = keyup.

label: 0 — человек печатал честно, 1 — читер/бот.
source: 'real' — собрано тренажёром, 'sim' — синтетический человек, 'bot' — сгенерированный бот.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from .keyboard import MODIFIERS


@dataclass
class Keystroke:
    code: str
    key: str
    t_down: float
    t_up: float

    @property
    def hold(self) -> float:
        return self.t_up - self.t_down


@dataclass
class Attempt:
    attempt_id: str
    user_id: str
    session_no: int
    kind: str            # calib | text | adaptive
    lang: str
    prompt: str
    events: list         # [[type, code, key, t, trusted], ...]
    typed: str = ""
    mode: str = "normal"  # normal | cheat
    cheat_type: str | None = None
    source: str = "real"
    label: int = 0
    bot_type: str | None = None
    flags: dict = field(default_factory=dict)

    # ----- производные -----
    def keystrokes(self, drop_modifiers: bool = True) -> list[Keystroke]:
        return events_to_keystrokes(self.events, drop_modifiers=drop_modifiers)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Attempt":
        keys = Attempt.__dataclass_fields__.keys()
        return Attempt(**{k: v for k, v in d.items() if k in keys})


def events_to_keystrokes(events: list, drop_modifiers: bool = True) -> list[Keystroke]:
    """Склеивает keydown/keyup одной клавиши в нажатие. Автоповтор игнорируется."""
    open_: dict[str, Keystroke] = {}
    out: list[Keystroke] = []
    for ev in sorted(events, key=lambda e: e[3]):
        typ, code, key, t = ev[0], ev[1], ev[2], float(ev[3])
        if typ == "d":
            if code in open_:      # автоповтор: keydown без keyup
                continue
            ks = Keystroke(code, key, t, float("nan"))
            open_[code] = ks
            out.append(ks)
        elif typ == "u":
            ks = open_.pop(code, None)
            if ks is not None:
                ks.t_up = t
    holds = sorted(k.hold for k in out if k.t_up == k.t_up)
    med = holds[len(holds) // 2] if holds else 90.0
    for ks in out:
        if ks.t_up != ks.t_up:   # NaN — keyup потерян (переключение окна и т.п.)
            ks.t_up = ks.t_down + med
    if drop_modifiers:
        out = [k for k in out if k.code not in MODIFIERS]
    return out


# ---------------------------------------------------------------- загрузка
def load_jsonl(path: str | Path) -> list[Attempt]:
    res = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                res.append(Attempt.from_dict(json.loads(line)))
    return res


def save_jsonl(attempts: Iterable[Attempt], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for a in attempts:
            f.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")


def load_sqlite(db_path: str | Path) -> list[Attempt]:
    """Читает базу сборщика (collector/server.py)."""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM attempts WHERE excluded = 0 ORDER BY created").fetchall()
    res = []
    for r in rows:
        mode = r["mode"]
        res.append(Attempt(
            attempt_id=str(r["id"]), user_id=r["user_id"], session_no=r["session_no"],
            kind=r["kind"], lang=r["lang"], prompt=r["prompt"], typed=r["typed"],
            events=json.loads(r["events"]), mode=mode, cheat_type=r["cheat_type"],
            source="real", label=1 if mode == "cheat" else 0,
            flags=json.loads(r["flags"] or "{}"),
        ))
    con.close()
    return res


def load_any(path: str | Path) -> list[Attempt]:
    p = Path(path)
    if p.suffix in (".db", ".sqlite", ".sqlite3"):
        return load_sqlite(p)
    return load_jsonl(p)
