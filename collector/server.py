"""Сервер сборщика данных для клавиатурного тренажёра.

Только стандартная библиотека Python (+ numpy, если есть, — для онлайн-античита).
WSGI-приложение `application` можно запускать:
  * локально:            python collector/server.py            (http://localhost:8000)
  * на PythonAnywhere:   указать `application` из этого файла в WSGI-конфиге
  * через gunicorn:      gunicorn collector.server:application

Переменные окружения:
  KT_DB             путь к SQLite (по умолчанию collector/data/keystrokes.db)
  KT_ADMIN_TOKEN    токен для выгрузки данных (/api/admin/export?token=...)
  KT_MIN_GAP_HOURS  минимум часов между сессиями одного человека (по умолчанию 8; 0 — без ограничения)
  PORT              порт для локального запуска (по умолчанию 8000)
"""
from __future__ import annotations

import json
import math
import mimetypes
import os
import random
import secrets
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from keyml import corpus  # noqa: E402
from keyml.data import Attempt, events_to_keystrokes  # noqa: E402
from keyml.profile import (BigramPrior, adaptive_exercise, bigram_latencies,  # noqa: E402
                           key_errors, user_profile)

DB_PATH = Path(os.environ.get("KT_DB", HERE / "data" / "keystrokes.db"))
ADMIN_TOKEN = os.environ.get("KT_ADMIN_TOKEN", "change-me")
MIN_GAP_H = float(os.environ.get("KT_MIN_GAP_HOURS", "8"))
STATIC = HERE / "static"
MAX_BODY = 2_000_000
CALIB_REPS = 10
TEXTS_PER_SESSION = 6

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, nickname TEXT, created REAL, meta TEXT, consent INTEGER);
CREATE TABLE IF NOT EXISTS sessions (
  user_id TEXT, session_no INTEGER, started REAL, mode TEXT, cheat_type TEXT,
  PRIMARY KEY (user_id, session_no, mode));
CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, session_no INTEGER, kind TEXT,
  lang TEXT, prompt TEXT, typed TEXT, events TEXT, flags TEXT, mode TEXT, cheat_type TEXT,
  client TEXT, created REAL, wpm REAL, acc REAL, bot_score REAL, excluded INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS idx_att_user ON attempts(user_id);
"""


def db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), timeout=10)
    con.row_factory = sqlite3.Row
    return con


with db() as _c:
    _c.executescript(SCHEMA)


# ---------------------------------------------------------------- онлайн-античит (опционально)
class AntiCheat:
    """Загружает логрегрессию, обученную scripts/run_experiments.py (collector/model/anticheat.json)."""

    def __init__(self):
        self.ok = False
        p = HERE / "model" / "anticheat.json"
        try:
            import numpy as np
            from keyml.features import FeatureExtractor
            from keyml.models.linear import LogisticRegressionNP
            d = json.loads(p.read_text(encoding="utf-8"))
            self.fe = FeatureExtractor()
            self.fe.bigram_mu = {tuple(k.split(">")): v for k, v in d["bigram_mu"].items()}
            self.fe.key_hold_mu = d["key_hold_mu"]
            self.fe.names = d["features"]
            self.model = LogisticRegressionNP.from_json(d)
            self.np = np
            self.ok = True
        except Exception as e:  # модели ещё нет — сервер работает без онлайн-оценки
            print("anticheat model not loaded:", e)

    def score(self, att: Attempt) -> float | None:
        if not self.ok or len(att.keystrokes()) < 10:
            return None
        try:
            X = self.fe.transform([att])
            return float(self.model.predict_proba(X)[0])
        except Exception as e:  # модель от старой версии признаков — переобучите run_experiments.py
            print("anticheat scoring failed:", e)
            return None


ANTICHEAT = AntiCheat()


# ---------------------------------------------------------------- популяционный prior
class PriorCache:
    def __init__(self):
        self.prior, self.t, self.lock = None, 0.0, threading.Lock()

    def get(self) -> BigramPrior:
        with self.lock:
            if self.prior is not None and time.time() - self.t < 600:
                return self.prior
            lat = defaultdict(list)
            with db() as con:
                rows = con.execute("SELECT user_id, events FROM attempts WHERE mode='normal' AND excluded=0 "
                                   "ORDER BY id DESC LIMIT 4000").fetchall()
            for r in rows:
                lat[r["user_id"]] += bigram_latencies(events_to_keystrokes(json.loads(r["events"])))
            prior = BigramPrior()
            if len(lat) >= 5:
                prior.fit(lat)
            else:
                p = HERE / "model" / "bigram_prior.json"
                if p.exists():
                    prior = BigramPrior.from_json(json.loads(p.read_text(encoding="utf-8")))
            self.prior, self.t = prior, time.time()
            return prior


PRIOR = PriorCache()


# ---------------------------------------------------------------- утилиты
def new_user_code(con) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        c = "".join(secrets.choice(alphabet) for _ in range(8))
        c = c[:4] + "-" + c[4:]
        if not con.execute("SELECT 1 FROM users WHERE id=?", (c,)).fetchone():
            return c


def wpm_acc(prompt: str, typed: str, events: list) -> tuple[float, float]:
    downs = [e[3] for e in events if e[0] == "d"]
    if len(downs) < 2:
        return 0.0, 0.0
    minutes = (max(downs) - min(downs)) / 60000.0
    wpm = (len(typed) / 5.0) / max(minutes, 1e-6)
    n = max(len(prompt), 1)
    correct = sum(1 for a, b in zip(prompt, typed) if a == b)
    n_bs = sum(1 for e in events if e[0] == "d" and e[1] == "Backspace")
    acc = max(0.0, (correct - n_bs) / n)
    return round(wpm, 1), round(acc, 3)


def user_row(con, uid):
    return con.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()


def sessions_done(con, uid):
    rows = con.execute("SELECT session_no, started FROM sessions WHERE user_id=? AND mode='normal' "
                       "ORDER BY session_no", (uid,)).fetchall()
    done = []
    for r in rows:
        n = con.execute("SELECT COUNT(*) FROM attempts WHERE user_id=? AND session_no=? AND mode='normal'",
                        (uid, r["session_no"])).fetchone()[0]
        done.append({"session_no": r["session_no"], "started": r["started"], "attempts": n})
    return done


def build_profile(con, uid, lang):
    rows = con.execute("SELECT events FROM attempts WHERE user_id=? AND mode='normal' AND excluded=0",
                       (uid,)).fetchall()
    lats, errs = [], defaultdict(int)
    for r in rows:
        ks = events_to_keystrokes(json.loads(r["events"]))
        lats += bigram_latencies(ks)
        for k, v in key_errors(ks).items():
            errs[k] += v
    prof = user_profile(lats, PRIOR.get(), dict(errs), min_n=2)
    prof["exercise"] = adaptive_exercise(prof, lang, corpus.WORDS[lang]) if lats else ""
    prof["bigrams"] = prof["bigrams"][:10]
    prof["keys"] = prof["keys"][:6]
    return prof


# ---------------------------------------------------------------- обработчики API
def api_register(body, q):
    if not body.get("consent"):
        return 400, {"error": "Нужно согласие на участие"}
    meta = body.get("meta") or {}
    nick = str(body.get("nickname", ""))[:24].strip() or "аноним"
    with db() as con:
        uid = new_user_code(con)
        con.execute("INSERT INTO users VALUES (?,?,?,?,1)",
                    (uid, nick, time.time(), json.dumps(meta, ensure_ascii=False)[:4000]))
    return 200, {"user_id": uid}


def api_me(body, q):
    uid = q.get("uid", [""])[0].upper()
    with db() as con:
        u = user_row(con, uid)
        if not u:
            return 404, {"error": "Код не найден"}
        done = sessions_done(con, uid)
        last = max((s["started"] for s in done), default=0)
        wait = max(0.0, MIN_GAP_H * 3600 - (time.time() - last)) if done else 0.0
        best = con.execute("SELECT MAX(wpm) FROM attempts WHERE user_id=? AND mode='normal' AND kind='text' "
                           "AND acc>=0.9", (uid,)).fetchone()[0]
    return 200, {"user_id": uid, "nickname": u["nickname"], "meta": json.loads(u["meta"] or "{}"),
                 "sessions": done, "wait_seconds": int(wait), "best_wpm": best,
                 "min_gap_hours": MIN_GAP_H}


def api_session_start(body, q):
    uid = str(body.get("uid", "")).upper()
    mode = "cheat" if body.get("mode") == "cheat" else "normal"
    cheat_type = str(body.get("cheat_type") or "")[:40] or None
    with db() as con:
        u = user_row(con, uid)
        if not u:
            return 404, {"error": "Код не найден"}
        meta = json.loads(u["meta"] or "{}")
        lang = meta.get("lang", "ru") if meta.get("lang") in ("ru", "en") else "ru"
        done = sessions_done(con, uid)
        incomplete = mode == "normal" and done and done[-1]["attempts"] < CALIB_REPS + TEXTS_PER_SESSION
        if incomplete:
            # незаконченную сессию продолжаем, а не начинаем новую
            sno = done[-1]["session_no"]
        else:
            if mode == "normal" and done and MIN_GAP_H > 0:
                last = max(s["started"] for s in done)
                if time.time() - last < MIN_GAP_H * 3600:
                    return 429, {"error": "Следующая сессия станет доступна позже — нам нужны данные "
                                          "из разных дней", "wait_seconds": int(MIN_GAP_H * 3600 - (time.time() - last))}
            r = con.execute("SELECT MAX(session_no) FROM sessions WHERE user_id=? AND mode=?",
                            (uid, mode)).fetchone()[0]
            sno = (r or 0) + 1
            con.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?)",
                        (uid, sno, time.time(), mode, cheat_type))
        reps = CALIB_REPS if mode == "normal" else 4
        plan = [{"kind": "calib", "lang": lang, "prompt": corpus.CALIB[lang]} for _ in range(reps)]
        rng = random.Random(f"{uid}-{sno}-{mode}")
        for t in rng.sample(corpus.TEXTS[lang], TEXTS_PER_SESSION):
            plan.append({"kind": "text", "lang": lang, "prompt": t})
        if mode == "normal" and sno > 1:
            prof = build_profile(con, uid, lang)
            if prof.get("exercise"):
                plan.append({"kind": "adaptive", "lang": lang, "prompt": prof["exercise"]})
        # уже набранные в этой сессии попытки пропускаем
        n_done = con.execute("SELECT COUNT(*) FROM attempts WHERE user_id=? AND session_no=? AND mode=?",
                             (uid, sno, mode)).fetchone()[0]
    return 200, {"session_no": sno, "mode": mode, "plan": plan[n_done:], "total": len(plan),
                 "offset": n_done}


def api_attempt(body, q):
    uid = str(body.get("uid", "")).upper()
    events = body.get("events")
    if not isinstance(events, list) or not (2 <= len(events) <= 20000):
        return 400, {"error": "bad events"}
    clean = []
    for e in events:
        if not (isinstance(e, list) and len(e) >= 4 and e[0] in ("d", "u")):
            return 400, {"error": "bad event"}
        clean.append([e[0], str(e[1])[:24], str(e[2])[:12], round(float(e[3]), 1),
                      1 if (len(e) < 5 or e[4]) else 0])
    prompt = str(body.get("prompt", ""))[:1000]
    typed = str(body.get("typed", ""))[:1200]
    kind = body.get("kind") if body.get("kind") in ("calib", "text", "adaptive") else "text"
    lang = body.get("lang") if body.get("lang") in ("ru", "en") else "ru"
    mode = "cheat" if body.get("mode") == "cheat" else "normal"
    flags = body.get("flags") if isinstance(body.get("flags"), dict) else {}
    with db() as con:
        if not user_row(con, uid):
            return 404, {"error": "Код не найден"}
        sno = int(body.get("session_no", 1))
        wpm, acc = wpm_acc(prompt, typed, clean)
        att = Attempt("tmp", uid, sno, kind, lang, prompt, clean, typed=typed, mode=mode)
        score = ANTICHEAT.score(att)
        untrusted = sum(1 for e in clean if not e[4])
        if untrusted:
            flags["untrusted_events"] = untrusted
        con.execute(
            "INSERT INTO attempts (user_id, session_no, kind, lang, prompt, typed, events, flags, mode, "
            "cheat_type, client, created, wpm, acc, bot_score) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (uid, sno, kind, lang, prompt, typed, json.dumps(clean), json.dumps(flags), mode,
             str(body.get("cheat_type") or "")[:40] or None,
             json.dumps(body.get("client") or {})[:2000], time.time(), wpm, acc, score))
    verdict = None
    if score is not None:
        verdict = "похоже на бота" if score > 0.9 else ("подозрительно" if score > 0.6 else "человек")
    if untrusted:
        verdict = "синтетические события (isTrusted=false)"
    return 200, {"wpm": wpm, "acc": acc, "bot_score": score, "verdict": verdict}


def api_profile(body, q):
    uid = q.get("uid", [""])[0].upper()
    with db() as con:
        u = user_row(con, uid)
        if not u:
            return 404, {"error": "Код не найден"}
        lang = json.loads(u["meta"] or "{}").get("lang", "ru")
        prof = build_profile(con, uid, lang if lang in ("ru", "en") else "ru")
        hist = [dict(r) for r in con.execute(
            "SELECT session_no, ROUND(AVG(wpm),1) AS wpm, ROUND(AVG(acc),3) AS acc, COUNT(*) AS n "
            "FROM attempts WHERE user_id=? AND mode='normal' AND kind!='calib' GROUP BY session_no",
            (uid,)).fetchall()]
    return 200, {"profile": prof, "history": hist}


def api_leaderboard(body, q):
    with db() as con:
        rows = con.execute(
            "SELECT u.nickname, ROUND(MAX(a.wpm),1) AS best, ROUND(AVG(a.acc),3) AS acc, "
            "COUNT(DISTINCT a.session_no) AS sessions FROM attempts a JOIN users u ON u.id=a.user_id "
            "WHERE a.mode='normal' AND a.kind='text' AND a.acc>=0.9 AND (a.bot_score IS NULL OR a.bot_score<0.9) "
            "GROUP BY a.user_id ORDER BY best DESC LIMIT 15").fetchall()
        tot = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        att = con.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    return 200, {"top": [dict(r) for r in rows], "users": tot, "attempts": att}


def _check_admin(q):
    return q.get("token", [""])[0] == ADMIN_TOKEN and ADMIN_TOKEN != "change-me"


def api_admin_summary(body, q):
    if not _check_admin(q):
        return 403, {"error": "forbidden"}
    with db() as con:
        users = [dict(r) for r in con.execute(
            "SELECT u.id, u.nickname, u.meta, COUNT(a.id) AS attempts, COUNT(DISTINCT a.session_no) AS sessions, "
            "SUM(a.mode='cheat') AS cheat_attempts FROM users u LEFT JOIN attempts a ON a.user_id=u.id "
            "GROUP BY u.id ORDER BY attempts DESC").fetchall()]
    return 200, {"users": users}


def admin_export(q, start_response):
    if not _check_admin(q):
        start_response("403 Forbidden", [("Content-Type", "text/plain")])
        return [b"forbidden"]
    with db() as con:
        rows = con.execute("SELECT * FROM attempts WHERE excluded=0 ORDER BY id").fetchall()
        metas = {r["id"]: json.loads(r["meta"] or "{}") for r in con.execute("SELECT id, meta FROM users")}
    out = []
    for r in rows:
        a = Attempt(str(r["id"]), r["user_id"], r["session_no"], r["kind"], r["lang"], r["prompt"],
                    json.loads(r["events"]), typed=r["typed"], mode=r["mode"], cheat_type=r["cheat_type"],
                    source="real", label=1 if r["mode"] == "cheat" else 0,
                    flags={**json.loads(r["flags"] or "{}"), "user_meta": metas.get(r["user_id"], {})})
        out.append(json.dumps(a.to_dict(), ensure_ascii=False))
    data = ("\n".join(out) + "\n").encode("utf-8")
    start_response("200 OK", [("Content-Type", "application/x-ndjson; charset=utf-8"),
                              ("Content-Disposition", "attachment; filename=attempts.jsonl"),
                              ("Content-Length", str(len(data)))])
    return [data]


ROUTES = {
    ("POST", "/api/register"): api_register,
    ("GET", "/api/me"): api_me,
    ("POST", "/api/session/start"): api_session_start,
    ("POST", "/api/attempt"): api_attempt,
    ("GET", "/api/profile"): api_profile,
    ("GET", "/api/leaderboard"): api_leaderboard,
    ("GET", "/api/admin/summary"): api_admin_summary,
}


# ---------------------------------------------------------------- WSGI
def application(environ, start_response):
    method = environ["REQUEST_METHOD"]
    path = environ.get("PATH_INFO", "/")
    q = parse_qs(environ.get("QUERY_STRING", ""))
    try:
        if path == "/api/admin/export":
            return admin_export(q, start_response)
        handler = ROUTES.get((method, path))
        if handler:
            body = {}
            if method == "POST":
                n = int(environ.get("CONTENT_LENGTH") or 0)
                if n > MAX_BODY:
                    return _json(start_response, 413, {"error": "too large"})
                raw = environ["wsgi.input"].read(n) if n else b"{}"
                body = json.loads(raw.decode("utf-8") or "{}")
            code, payload = handler(body, q)
            return _json(start_response, code, payload)
        return _static(path, start_response)
    except Exception as e:  # pragma: no cover
        import traceback
        traceback.print_exc()
        return _json(start_response, 500, {"error": str(e)})


STATUS = {200: "200 OK", 400: "400 Bad Request", 403: "403 Forbidden", 404: "404 Not Found",
          413: "413 Payload Too Large", 429: "429 Too Many Requests", 500: "500 Internal Server Error"}


def _json(start_response, code, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    start_response(STATUS[code], [("Content-Type", "application/json; charset=utf-8"),
                                  ("Content-Length", str(len(data))), ("Cache-Control", "no-store")])
    return [data]


def _static(path, start_response):
    if path in ("", "/"):
        path = "/index.html"
    f = (STATIC / path.lstrip("/")).resolve()
    if STATIC not in f.parents or not f.is_file():
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"not found"]
    data = f.read_bytes()
    ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
    if ctype.startswith("text/") or ctype.endswith("javascript"):
        ctype += "; charset=utf-8"
    start_response("200 OK", [("Content-Type", ctype), ("Content-Length", str(len(data)))])
    return [data]


if __name__ == "__main__":
    from socketserver import ThreadingMixIn
    from wsgiref.simple_server import WSGIServer, make_server

    class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    port = int(os.environ.get("PORT", "8000"))
    srv = make_server("0.0.0.0", port, application, server_class=ThreadingWSGIServer)
    print(f"Тренажёр: http://localhost:{port}   (база: {DB_PATH})")
    print(f"Режим читера: http://localhost:{port}/?cheat=1")
    srv.serve_forever()
