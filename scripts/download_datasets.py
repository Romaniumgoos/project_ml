"""Скачать открытые датасеты в data/external/ (один раз).

    python scripts/download_datasets.py           # CMU (5 МБ) + Aalto (1.5 ГБ)
    python scripts/download_datasets.py --cmu     # только CMU

CMU Keystroke Dynamics Benchmark — Killourhy & Maxion (DSN 2009), CC-BY-ND 2.0.
Aalto 136M Keystrokes — Dhakal et al. (CHI 2018), некоммерческое использование с указанием авторов.
Архив Aalto не распаковывается: загрузчик читает нужные файлы прямо из zip.
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
FILES = {
    "cmu": ("https://roymaxion.github.io/projects/keystroke-benchmark/DSL-StrongPasswordData.csv",
            EXT / "cmu" / "DSL-StrongPasswordData.csv", "470235f96568f28f9ea0da62234ec857"),
    "aalto": ("https://userinterfaces.aalto.fi/136Mkeystrokes/data/Keystrokes.zip",
              EXT / "aalto" / "Keystrokes.zip", None),
}


def download(url, dst: Path, md5=None):
    if dst.exists() and dst.stat().st_size > 0:
        print(f"уже есть: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"скачиваю {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r  {done / 2**20:7.0f} / {total / 2**20:.0f} МБ")
                sys.stdout.flush()
    print()
    if md5 and hashlib.md5(tmp.read_bytes()).hexdigest() != md5:
        tmp.unlink()
        raise SystemExit("контрольная сумма не совпала — попробуйте ещё раз")
    tmp.rename(dst)
    print(f"готово: {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmu", action="store_true", help="только CMU")
    a = ap.parse_args()
    for name in (["cmu"] if a.cmu else ["cmu", "aalto"]):
        download(*FILES[name])
