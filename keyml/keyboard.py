"""Геометрия клавиатуры и раскладки (ЙЦУКЕН / QWERTY).

Работаем с физическими кодами клавиш (event.code в браузере): KeyA, Digit1, Space...
Код не зависит от раскладки, поэтому биграммы русского и английского текста
сравнимы между собой и между пользователями.
"""
from __future__ import annotations

ROWS = [
    ["Backquote", "Digit1", "Digit2", "Digit3", "Digit4", "Digit5", "Digit6",
     "Digit7", "Digit8", "Digit9", "Digit0", "Minus", "Equal"],
    ["KeyQ", "KeyW", "KeyE", "KeyR", "KeyT", "KeyY", "KeyU", "KeyI", "KeyO",
     "KeyP", "BracketLeft", "BracketRight"],
    ["KeyA", "KeyS", "KeyD", "KeyF", "KeyG", "KeyH", "KeyJ", "KeyK", "KeyL",
     "Semicolon", "Quote"],
    ["KeyZ", "KeyX", "KeyC", "KeyV", "KeyB", "KeyN", "KeyM", "Comma", "Period",
     "Slash"],
]
ROW_OFFSET = [0.0, 0.5, 0.75, 1.25]

# палец: 0..3 левая рука (мизинец..указательный), 4 большой, 5..8 правая (указ..мизинец)
_FINGER_BY_COL = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 5, 7: 5, 8: 6, 9: 7, 10: 8, 11: 8, 12: 8}
_FINGER_BY_COL_ROW1 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 5, 6: 5, 7: 6, 8: 7, 9: 8, 10: 8, 11: 8}

KEY_POS: dict[str, tuple[float, float]] = {}
KEY_FINGER: dict[str, int] = {}
for r, row in enumerate(ROWS):
    for c, code in enumerate(row):
        KEY_POS[code] = (c + ROW_OFFSET[r], float(r))
        if r == 0:
            KEY_FINGER[code] = _FINGER_BY_COL[c]
        else:
            KEY_FINGER[code] = _FINGER_BY_COL_ROW1.get(c, 8)
KEY_POS["Space"] = (5.5, 4.0)
KEY_FINGER["Space"] = 4
KEY_POS["ShiftLeft"] = (0.0, 3.0)
KEY_FINGER["ShiftLeft"] = 0
KEY_POS["ShiftRight"] = (12.0, 3.0)
KEY_FINGER["ShiftRight"] = 8
KEY_POS["Backspace"] = (13.5, 0.0)
KEY_FINGER["Backspace"] = 8
KEY_POS["Enter"] = (12.5, 2.0)
KEY_FINGER["Enter"] = 8

MODIFIERS = {"ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight", "AltLeft",
             "AltRight", "MetaLeft", "MetaRight", "CapsLock"}


def hand(code: str) -> int:
    """0 — левая, 1 — правая, 2 — большой палец/неизвестно."""
    f = KEY_FINGER.get(code)
    if f is None or f == 4:
        return 2
    return 0 if f < 4 else 1


# ---------------- раскладки: символ -> (код, нужен ли Shift) ----------------
EN: dict[str, tuple[str, bool]] = {}
for ch in "abcdefghijklmnopqrstuvwxyz":
    EN[ch] = ("Key" + ch.upper(), False)
    EN[ch.upper()] = ("Key" + ch.upper(), True)
for i, ch in enumerate("1234567890"):
    EN[ch] = ("Digit" + ch, False)
EN.update({" ": ("Space", False), ",": ("Comma", False), ".": ("Period", False),
           ";": ("Semicolon", False), "'": ("Quote", False), "-": ("Minus", False),
           "?": ("Slash", True), "!": ("Digit1", True), ":": ("Semicolon", True),
           '"': ("Quote", True), "/": ("Slash", False), "(": ("Digit9", True),
           ")": ("Digit0", True)})

_RU_LETTERS = {
    "й": "KeyQ", "ц": "KeyW", "у": "KeyE", "к": "KeyR", "е": "KeyT", "н": "KeyY",
    "г": "KeyU", "ш": "KeyI", "щ": "KeyO", "з": "KeyP", "х": "BracketLeft",
    "ъ": "BracketRight", "ф": "KeyA", "ы": "KeyS", "в": "KeyD", "а": "KeyF",
    "п": "KeyG", "р": "KeyH", "о": "KeyJ", "л": "KeyK", "д": "KeyL",
    "ж": "Semicolon", "э": "Quote", "я": "KeyZ", "ч": "KeyX", "с": "KeyC",
    "м": "KeyV", "и": "KeyB", "т": "KeyN", "ь": "KeyM", "б": "Comma", "ю": "Period",
    "ё": "Backquote",
}
RU: dict[str, tuple[str, bool]] = {}
for ch, code in _RU_LETTERS.items():
    RU[ch] = (code, False)
    RU[ch.upper()] = (code, True)
for ch in "1234567890":
    RU[ch] = ("Digit" + ch, False)
RU.update({" ": ("Space", False), ".": ("Slash", False), ",": ("Slash", True),
           "-": ("Minus", False), "!": ("Digit1", True), "?": ("Digit7", True),
           '"': ("Digit2", True), ";": ("Digit4", True), ":": ("Digit6", True),
           "(": ("Digit9", True), ")": ("Digit0", True)})

LAYOUTS = {"en": EN, "ru": RU}


def detect_lang(text: str) -> str:
    cyr = sum(1 for ch in text.lower() if "а" <= ch <= "я" or ch == "ё")
    lat = sum(1 for ch in text.lower() if "a" <= ch <= "z")
    return "ru" if cyr >= lat else "en"


def char_to_code(ch: str, lang: str) -> tuple[str, bool] | None:
    return LAYOUTS[lang].get(ch)


def key_label(code: str) -> str:
    """Человекочитаемое имя клавиши: KeyF -> 'F/А'."""
    ru = {v: k for k, v in _RU_LETTERS.items()}
    if code.startswith("Key"):
        return f"{code[3:]}/{ru.get(code, '').upper()}"
    if code.startswith("Digit"):
        return code[5:]
    names = {"Space": "␣", "Comma": ",/Б", "Period": "./Ю", "Semicolon": ";/Ж",
             "Quote": "'/Э", "BracketLeft": "[/Х", "BracketRight": "]/Ъ",
             "Slash": "/ (.)", "Backquote": "`/Ё", "Minus": "-", "Backspace": "⌫"}
    return names.get(code, code)


def key_distance(a: str, b: str) -> float:
    pa, pb = KEY_POS.get(a), KEY_POS.get(b)
    if pa is None or pb is None:
        return 3.0
    return ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5


def bigram_cost(a: str, b: str) -> float:
    """Априорная «сложность» перехода a->b (только для симулятора и анализа)."""
    fa, fb = KEY_FINGER.get(a, 4), KEY_FINGER.get(b, 4)
    ha, hb = hand(a), hand(b)
    d = key_distance(a, b)
    cost = 0.0
    if a == b:
        cost += 0.35          # повтор той же клавиши
    elif fa == fb and fa != 4:
        cost += 0.45 + 0.08 * d  # тот же палец — самое медленное
    elif ha == hb and ha != 2:
        cost += 0.12 + 0.04 * d  # та же рука
    else:
        cost -= 0.10          # чередование рук — быстро
    ra = KEY_POS.get(a, (0, 2))[1]
    rb = KEY_POS.get(b, (0, 2))[1]
    cost += 0.06 * abs(ra - rb)
    if fb in (0, 8):
        cost += 0.08          # мизинец
    return cost
