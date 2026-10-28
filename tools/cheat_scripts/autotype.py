"""«Читер» на Python для сбора негативных примеров (pynput, работает на Windows/macOS/Linux).

pip install pynput pyperclip
Использование: откройте тренажёр с ?cheat=1, скопируйте текст задания (Ctrl+C),
запустите скрипт, за 3 секунды кликните в поле задания.
  python autotype.py --mode humanized     # логнормальные задержки «как у человека»
  python autotype.py --mode uniform --dd 120
"""
import argparse
import math
import random
import time

import pyperclip
from pynput.keyboard import Controller

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["const", "uniform", "humanized"], default="humanized")
ap.add_argument("--dd", type=float, default=150.0, help="средняя задержка, мс")
ap.add_argument("--hold", type=float, default=60.0, help="длительность нажатия, мс")
args = ap.parse_args()

kb = Controller()
text = pyperclip.paste().replace(" ", " ")
print(f"Печатаю {len(text)} символов через 3 секунды...")
time.sleep(3)
for ch in text:
    if args.mode == "const":
        dd, hold = args.dd, args.hold
    elif args.mode == "uniform":
        dd, hold = random.uniform(0.5, 1.5) * args.dd, random.uniform(0.5, 1.5) * args.hold
    else:
        dd = math.exp(random.gauss(math.log(args.dd), 0.35))
        hold = math.exp(random.gauss(math.log(args.hold), 0.2))
        if ch == " " and random.random() < 0.08:
            dd += random.uniform(300, 900)
    time.sleep(max(dd - hold, 0) / 1000)
    kb.press(ch)
    time.sleep(hold / 1000)
    kb.release(ch)
