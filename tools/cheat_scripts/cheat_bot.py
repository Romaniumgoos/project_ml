"""Бот-читер для сбора РЕАЛЬНЫХ негативных примеров (все типы из keyml/bots.py).

Нажимает настоящие физические клавиши через ОС (pynput), поэтому браузер видит
обычные события с isTrusted=true и правильным event.code — ровно как у читера.

Установка (Windows):   pip install pynput
Запуск из корня проекта:
    python tools/cheat_scripts/cheat_bot.py humanized
Затем на сайте: откройте https://<сайт>/?cheat=1, выберите тип читерства,
начните сессию. Для каждого задания:
    1) нажмите на сайте «Скопировать текст задания» (или выделите текст и Ctrl+C);
    2) нажмите F8 — бот напечатает текст из буфера обмена;
    Esc — выход.
Раскладка ОС должна совпадать с языком задания (русская для русских текстов).

Режимы (повторяют типы ботов из keyml/bots.py):
    const      постоянная задержка и длительность нажатия
    uniform    равномерный случайный джиттер
    gauss      нормальный джиттер
    timer      задержки кратны тику таймера Windows (15.6 мс), как у AutoHotkey/Sleep
    humanized  логнормальные интервалы, паузы на пробелах, фальшивые опечатки с Backspace
    smart      знает популяционную «сложность» биграмм (из collector/model/anticheat.json)
    record     записать СВОЮ печать (печатаете сами, F9 — сохранить запись)
    replay     воспроизвести записанные тайминги на новом тексте
Параметры: --dd 150 (средний интервал, мс), --hold 80 (удержание, мс), --seed 1.
Меняйте параметры между прогонами — разнообразие важнее количества.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from keyml.keyboard import LAYOUTS, detect_lang  # noqa: E402

try:
    from pynput import keyboard as kb
except ImportError:
    sys.exit("Нужен pynput:  pip install pynput")

# физический код клавиши (event.code) -> виртуальный код Windows
VK = {f"Key{c}": ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
VK.update({f"Digit{d}": ord(d) for d in "0123456789"})
VK.update({"Space": 0x20, "Minus": 0xBD, "Equal": 0xBB, "BracketLeft": 0xDB, "BracketRight": 0xDD,
           "Semicolon": 0xBA, "Quote": 0xDE, "Backquote": 0xC0, "Comma": 0xBC, "Period": 0xBE,
           "Slash": 0xBF, "Backspace": 0x08})
REC_FILE = Path(__file__).with_name("my_typing_record.json")

ctl = kb.Controller()


def precise_sleep(ms: float, quant: float | None = None):
    if quant:                       # как Sleep() в Windows: округление вверх до тика таймера
        ms = math.ceil(max(ms, 0) / quant) * quant
    end = time.perf_counter() + max(ms, 0) / 1000
    while True:
        left = end - time.perf_counter()
        if left <= 0:
            return
        time.sleep(left / 2 if left > 0.002 else 0)


def press(code: str, shift: bool, hold_ms: float, quant=None):
    key = kb.KeyCode.from_vk(VK[code])
    if shift:
        ctl.press(kb.Key.shift)
    ctl.press(key)
    precise_sleep(hold_ms, quant)
    ctl.release(key)
    if shift:
        ctl.release(kb.Key.shift)


class Bot:
    def __init__(self, mode, dd, hold, seed):
        self.mode, self.dd, self.hold = mode, dd, hold
        self.r = random.Random(seed)
        self.bigram_mu = {}
        if mode == "smart":
            d = json.loads((ROOT / "collector/model/anticheat.json").read_text(encoding="utf-8"))
            self.bigram_mu = {tuple(k.split(">")): v for k, v in d["bigram_mu"].items()}
            self.key_hold_mu = d["key_hold_mu"]
        self.replay = None
        if mode == "replay":
            if not REC_FILE.exists():
                sys.exit("Сначала запишите свою печать: режим record")
            self.replay = json.loads(REC_FILE.read_text())

    def timings(self, seq):
        """seq: [(code, shift)] -> [(dd, hold)] для каждого нажатия."""
        r, n, out = self.r, len(seq), []
        if self.mode == "replay":
            dds, holds = self.replay["dd"], self.replay["hold"]
            off = r.randrange(len(dds))
            return [(dds[(off + i) % len(dds)] * r.uniform(0.97, 1.03),
                     holds[(off + i) % len(holds)] * r.uniform(0.97, 1.03)) for i in range(n)]
        g = sum(self.bigram_mu.values()) / len(self.bigram_mu) if self.bigram_mu else 5.2
        prev = None
        for code, _ in seq:
            if self.mode == "const":
                dd, h = self.dd, self.hold
            elif self.mode == "uniform":
                dd, h = r.uniform(0.6, 1.4) * self.dd, r.uniform(0.6, 1.4) * self.hold
            elif self.mode == "gauss":
                dd, h = max(10, r.gauss(self.dd, self.dd * 0.2)), max(5, r.gauss(self.hold, self.hold * 0.2))
            elif self.mode == "timer":
                dd, h = r.uniform(0.5, 1.5) * self.dd, r.uniform(0, 60)
            elif self.mode == "humanized":
                dd = math.exp(r.gauss(math.log(self.dd), 0.35))
                h = math.exp(r.gauss(math.log(self.hold), 0.2))
                if code == "Space" and r.random() < 0.08:
                    dd += r.uniform(300, 900)
            else:  # smart
                mu = self.bigram_mu.get((prev, code), g) if prev else g
                dd = math.exp(mu + math.log(self.dd / 180) + r.gauss(0, 0.2))
                h = math.exp(self.key_hold_mu.get(code, 4.5) + r.gauss(0, 0.15))
                if code == "Space" and r.random() < 0.06:
                    dd += r.uniform(300, 800)
            out.append((dd, h))
            prev = code
        return out

    def type_text(self, text: str):
        lang = detect_lang(text)
        lay = LAYOUTS[lang]
        seq = [lay[ch] for ch in text if ch in lay]
        quant = 15.625 if self.mode == "timer" else None
        for (code, shift), (dd, h) in zip(seq, self.timings(seq)):
            precise_sleep(max(dd - h, 1), quant)
            if self.mode == "humanized" and code != "Space" and self.r.random() < 0.03:
                press("KeyJ", False, h)                       # «опечатка» + исправление
                precise_sleep(self.r.uniform(200, 400))
                press("Backspace", False, 70)
                precise_sleep(self.r.uniform(100, 200))
            press(code, shift, h, quant)


def read_clipboard() -> str:
    import tkinter
    root = tkinter.Tk()
    root.withdraw()
    try:
        return root.clipboard_get().replace(" ", " ").strip()
    except tkinter.TclError:
        return ""
    finally:
        root.destroy()


def record_mode():
    """Пишем интервалы своей обычной печати, F9 — сохранить."""
    downs, holds, open_ = [], [], {}

    def on_press(k):
        if k == kb.Key.f9:
            dd = [b - a for a, b in zip(downs, downs[1:]) if 0 < b - a < 2000]
            REC_FILE.write_text(json.dumps({"dd": dd, "hold": holds}))
            print(f"сохранено {len(dd)} интервалов -> {REC_FILE}")
            return False
        t = time.perf_counter() * 1000
        if k not in open_:
            open_[k] = t
            downs.append(t)

    def on_release(k):
        t0 = open_.pop(k, None)
        if t0 is not None:
            holds.append(time.perf_counter() * 1000 - t0)

    print("Печатайте любой текст (лучше 300+ символов) как обычно. F9 — сохранить запись.")
    with kb.Listener(on_press=on_press, on_release=on_release) as l:
        l.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["const", "uniform", "gauss", "timer", "humanized", "smart", "record", "replay"])
    ap.add_argument("--dd", type=float, default=150.0)
    ap.add_argument("--hold", type=float, default=80.0)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "record":
        return record_mode()
    bot = Bot(a.mode, a.dd, a.hold, a.seed)
    q: queue.Queue = queue.Queue()

    def on_press(k):
        if k == kb.Key.f8:
            q.put("go")
        elif k == kb.Key.esc:
            q.put("exit")
            return False

    kb.Listener(on_press=on_press).start()
    print(f"Режим {a.mode}. Скопируйте текст задания и нажмите F8 (Esc — выход).")
    while True:
        cmd = q.get()
        if cmd == "exit":
            break
        text = read_clipboard()
        if not text:
            print("буфер обмена пуст")
            continue
        time.sleep(0.4)  # отпустить F8
        print(f"печатаю {len(text)} симв.: {text[:40]}…")
        bot.type_text(text)


if __name__ == "__main__":
    main()
