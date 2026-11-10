"""TrainKey: запуск в один клик (сервер + веб-приложение) и открытие браузера.

    python run.py            # или двойной клик по start.bat (Windows) / start.command (macOS)

Ничего внешнего не нужно: сервер — стандартная библиотека Python, страница — локальные
HTML/JS/CSS без CDN. numpy нужен только для онлайн-античита; если его нет, скрипт один раз
попробует поставить его через pip, а при неудаче запустит сайт без оценки читерства.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

if sys.version_info < (3, 9):
    sys.exit("Нужен Python 3.9 или новее: https://www.python.org/downloads/")

if importlib.util.find_spec("numpy") is None:
    print("numpy не найден — пробую установить (нужен только для античита)...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--user", "numpy"])
    if r.returncode != 0:
        print("Не получилось — сайт запустится без оценки вероятности читерства.")

# локальные значения по умолчанию (переменные окружения имеют приоритет)
os.environ.setdefault("KT_DEBUG_TOKEN", "local")          # http://localhost:8000/?debug=local
os.environ.setdefault("KT_ADMIN_TOKEN", "local-admin")    # выгрузка: /api/admin/export?token=local-admin

from collector.server import serve  # noqa: E402

serve(int(os.environ.get("PORT", "8000")), open_browser="--no-browser" not in sys.argv)
