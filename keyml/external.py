"""Открытые датасеты нажатий клавиш: загрузка и приведение к формату проекта.

1. CMU Keystroke Dynamics Benchmark (Killourhy & Maxion, DSN 2009), CC-BY-ND 2.0
   https://roymaxion.github.io/projects/keystroke-benchmark/benchmark-data-set.html
   51 человек × 8 сессий (в разные дни) × 50 повторов пароля ".tie5Roanl".
   Сырых событий нет — только готовые времена (в секундах) по позициям:
     H.key        — удержание клавиши
     DD.k1.k2     — нажатие -> нажатие следующей
     UD.k1.k2     — отпускание -> нажатие следующей (может быть < 0)
   Это ровно вектор, который наша биометрия строит из калибровочной фразы.

2. Aalto 136M Keystrokes (Dhakal, Feit, Kristensson, Oulasvirta, CHI 2018),
   «free to use for research and non-commercial use with attribution»
   https://userinterfaces.aalto.fi/136Mkeystrokes/
   ~169 тыс. участников онлайн-теста печати, по 15 английских предложений, у каждого
   нажатия — время нажатия и отпускания (мс). Одно предложение одного участника =
   одна попытка (Attempt) в нашем формате, участник = пользователь.
"""
from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import Attempt

# --------------------------------------------------------------------------- CMU
CMU_PASSWORD = ".tie5Roanl"


def load_cmu(path: str | Path):
    """-> (X [n×31] в миллисекундах, subject [n], session [n], rep [n], имена колонок)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    cols = header[3:]
    X = np.array([[float(v) * 1000.0 for v in r[3:]] for r in body])
    subj = np.array([r[0] for r in body])
    sess = np.array([int(r[1]) for r in body])
    rep = np.array([int(r[2]) for r in body])
    return X, subj, sess, rep, cols


def reorder_like_ours(X: np.ndarray, cols: list[str]) -> np.ndarray:
    """CMU хранит H, DD, UD вперемешку по позициям; наш calib_vector — [все H, все DD, все UD]."""
    idx = ([i for i, c in enumerate(cols) if c.startswith("H.")]
           + [i for i, c in enumerate(cols) if c.startswith("DD.")]
           + [i for i, c in enumerate(cols) if c.startswith("UD.")])
    return X[:, idx]


# --------------------------------------------------------------------------- Aalto
# JavaScript keyCode -> физический код (event.code) для английской раскладки
_JS = {32: "Space", 8: "Backspace", 13: "Enter", 16: "ShiftLeft", 17: "ControlLeft", 18: "AltLeft",
       20: "CapsLock", 9: "Tab", 186: "Semicolon", 187: "Equal", 188: "Comma", 189: "Minus",
       190: "Period", 191: "Slash", 192: "Backquote", 219: "BracketLeft", 220: "Backslash",
       221: "BracketRight", 222: "Quote", 59: "Semicolon", 173: "Minus", 61: "Equal"}
_JS.update({c: f"Key{chr(c)}" for c in range(65, 91)})
_JS.update({c: f"Digit{chr(c)}" for c in range(48, 58)})


def _find(cols: list[str], *names: str) -> int:
    up = [c.strip().upper() for c in cols]
    for n in names:
        if n in up:
            return up.index(n)
    raise KeyError(f"нет колонки {names} среди {cols}")


def parse_aalto_participant(text: str, max_sections: int | None = None) -> list[Attempt]:
    """Файл одного участника (TSV) -> список попыток (по одной на предложение)."""
    lines = text.splitlines()
    if not lines:
        return []
    cols = lines[0].split("\t")
    i_pid = _find(cols, "PARTICIPANT_ID")
    i_sec = _find(cols, "TEST_SECTION_ID")
    i_sent = _find(cols, "SENTENCE")
    i_inp = _find(cols, "USER_INPUT")
    i_press = _find(cols, "PRESS_TIME")
    i_rel = _find(cols, "RELEASE_TIME")
    i_letter = _find(cols, "LETTER")
    i_code = _find(cols, "KEYCODE")
    by_sec: dict[str, list] = defaultdict(list)
    meta = {}
    for ln in lines[1:]:
        p = ln.split("\t")
        if len(p) < len(cols):
            continue
        try:
            press, rel, kc = float(p[i_press]), float(p[i_rel]), int(float(p[i_code]))
        except ValueError:
            continue
        code = _JS.get(kc)
        if code is None:
            continue
        by_sec[p[i_sec]].append((press, rel, code, p[i_letter]))
        meta[p[i_sec]] = (p[i_pid], p[i_sent], p[i_inp])
    out = []
    for k, sec in enumerate(sorted(by_sec, key=lambda s: by_sec[s][0][0])):
        if max_sections and k >= max_sections:
            break
        ks = sorted(by_sec[sec])
        t0 = ks[0][0]
        events = []
        for press, rel, code, letter in ks:
            if rel < press:            # битые записи
                continue
            events.append(["d", code, letter[:12], round(press - t0, 1), 1])
            events.append(["u", code, letter[:12], round(rel - t0, 1), 1])
        if len(events) < 20:
            continue
        pid, sent, inp = meta[sec]
        out.append(Attempt(attempt_id=f"aalto-{pid}-{sec}", user_id=f"aalto{pid}", session_no=1,
                           kind="text", lang="en", prompt=sent, typed=inp, events=events,
                           source="aalto", label=0))
    return out


def _aalto_meta(text: str) -> dict[str, dict]:
    lines = text.splitlines()
    cols = lines[0].split("\t")
    out = {}
    for ln in lines[1:]:
        p = ln.split("\t")
        if len(p) == len(cols):
            out[p[0]] = dict(zip(cols, p))
    return out


def load_aalto(zip_or_dir: str | Path, n_participants: int = 2000, seed: int = 0,
               max_sections: int | None = None, layouts=("qwerty",),
               keyboard_types=("laptop", "full")) -> list[Attempt]:
    """Случайные n_participants участников из Keystrokes.zip (или распакованной папки).

    Фильтр по метаданным: только раскладка QWERTY (у AZERTY/QWERTZ JS-keycode не совпадает
    с физической клавишей) и физическая клавиатура (без телефонов и экранных клавиатур).
    Метаданные участника (пальцы, курс печати, тип клавиатуры) кладутся в flags["user_meta"]."""
    rng = np.random.default_rng(seed)
    p = Path(zip_or_dir)
    if p.is_dir():
        files = {f.name.split("_")[0]: f for f in p.rglob("*_keystrokes.txt")}
        meta_f = next(p.rglob("metadata_participants.txt"), None)
        meta = _aalto_meta(meta_f.read_text(encoding="utf-8", errors="replace")) if meta_f else {}
        read = lambda pid: files[pid].read_text(encoding="utf-8", errors="replace")
    else:
        z = zipfile.ZipFile(p)
        names = [n for n in z.namelist() if n.endswith("_keystrokes.txt")]
        files = {Path(n).name.split("_")[0]: n for n in names}
        meta_n = next((n for n in z.namelist() if n.endswith("metadata_participants.txt")), None)
        meta = _aalto_meta(z.read(meta_n).decode("utf-8", errors="replace")) if meta_n else {}
        read = lambda pid: z.read(files[pid]).decode("utf-8", errors="replace")
    ok = sorted(pid for pid in files
                if not meta or (meta.get(pid, {}).get("LAYOUT", "").lower() in layouts
                                and meta.get(pid, {}).get("KEYBOARD_TYPE", "").lower() in keyboard_types))
    pick = rng.choice(len(ok), size=min(n_participants, len(ok)), replace=False)
    out = []
    for i in sorted(pick):
        pid = ok[i]
        try:
            atts = parse_aalto_participant(read(pid), max_sections)
        except (KeyError, IndexError):
            continue
        um = {k: meta.get(pid, {}).get(k) for k in ("FINGERS", "HAS_TAKEN_TYPING_COURSE", "KEYBOARD_TYPE",
                                                    "TIME_SPENT_TYPING", "AVG_WPM_15")}
        for a in atts:
            a.flags = {"user_meta": um}
        out += atts
    return out
