"""Скачать собранные данные с развёрнутого сервера в data/real_attempts.jsonl.

python scripts/export_data.py https://ВАШ-ЛОГИН.pythonanywhere.com СЕКРЕТНЫЙ_ТОКЕН
"""
import sys
import urllib.request
from pathlib import Path

base, token = sys.argv[1].rstrip("/"), sys.argv[2]
out = Path(__file__).resolve().parents[1] / "data" / "real_attempts.jsonl"
out.parent.mkdir(exist_ok=True)
with urllib.request.urlopen(f"{base}/api/admin/export?token={token}") as r:
    out.write_bytes(r.read())
n = sum(1 for _ in open(out, encoding="utf-8"))
print(f"сохранено {n} попыток -> {out}")
