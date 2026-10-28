"""Сквозные тесты: сервер (WSGI без сети) + загрузка базы + признаки + модели.
Запуск: python -m pytest -q tests  (или просто python tests/test_pipeline.py)"""
import io
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = tempfile.mkdtemp()
os.environ["KT_DB"] = os.path.join(TMP, "test.db")
os.environ["KT_MIN_GAP_HOURS"] = "8"
os.environ["KT_ADMIN_TOKEN"] = "secret"

import numpy as np  # noqa: E402

from collector import server  # noqa: E402
from keyml.data import load_jsonl, load_sqlite  # noqa: E402
from keyml.simulate import SimUser  # noqa: E402
from keyml.metrics import roc_auc, eer, adjusted_rand_index  # noqa: E402


def call(method, path, body=None, query=""):
    raw = json.dumps(body or {}).encode()
    env = {"REQUEST_METHOD": method, "PATH_INFO": path, "QUERY_STRING": query,
           "CONTENT_LENGTH": str(len(raw)), "wsgi.input": io.BytesIO(raw)}
    status = {}

    def sr(s, h):
        status["s"] = s
    data = b"".join(server.application(env, sr))
    code = int(status["s"].split()[0])
    try:
        return code, json.loads(data)
    except Exception:
        return code, data


def test_server_flow():
    import random
    rng = random.Random(0)
    code, r = call("POST", "/api/register", {"nickname": "t", "consent": False})
    assert code == 400
    uids = []
    for u in range(6):
        code, r = call("POST", "/api/register", {"nickname": f"user{u}", "consent": True,
                                                 "meta": {"lang": "ru", "typing": "touch"}})
        assert code == 200
        uids.append(r["user_id"])
    for i, uid in enumerate(uids):
        sim = SimUser.random(uid, i)
        code, s = call("POST", "/api/session/start", {"uid": uid, "mode": "normal"})
        assert code == 200 and s["session_no"] == 1 and len(s["plan"]) == 16
        for item in s["plan"]:
            ev = sim.type_text(item["prompt"], item["lang"], 1, 0, rng)
            code, res = call("POST", "/api/attempt", {"uid": uid, "session_no": 1, "kind": item["kind"],
                                                      "lang": item["lang"], "prompt": item["prompt"],
                                                      "typed": item["prompt"], "events": ev})
            assert code == 200, res
            assert res["wpm"] > 5
    # вторая сессия сразу — запрещена (нужен другой день)
    code, r = call("POST", "/api/session/start", {"uid": uids[0]})
    assert code == 429
    # режим читера разрешён всегда и нумеруется отдельно
    code, r = call("POST", "/api/session/start", {"uid": uids[0], "mode": "cheat", "cheat_type": "ahk"})
    assert code == 200 and r["mode"] == "cheat"
    code, p = call("GET", "/api/profile", query=f"uid={uids[0]}")
    assert code == 200 and p["profile"]["bigrams"], p
    assert p["profile"]["exercise"]
    code, lb = call("GET", "/api/leaderboard")
    assert code == 200 and lb["users"] == 6
    code, _ = call("GET", "/api/admin/export", query="token=wrong")
    assert code == 403
    code, data = call("GET", "/api/admin/export", query="token=secret")
    assert code == 200
    out = Path(TMP) / "export.jsonl"
    out.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
    atts = load_jsonl(out)
    assert len(atts) == 6 * 16
    assert len(load_sqlite(os.environ["KT_DB"])) == 6 * 16
    assert all(len(a.keystrokes()) > 10 for a in atts)


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert eer(y, [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert abs(adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) - 1.0) < 1e-9
    from sklearn.metrics import roc_auc_score
    r = np.random.default_rng(0)
    yy, ss = r.integers(0, 2, 300), r.random(300)
    assert abs(roc_auc(yy, ss) - roc_auc_score(yy, ss)) < 1e-9


def test_own_models_match_sklearn():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from keyml.models.linear import LogisticRegressionNP, MLPNP
    from keyml.models.unsupervised import KMeansNP
    from sklearn.cluster import KMeans
    X, y = make_classification(600, 10, random_state=0)
    a = roc_auc(y, LogisticRegressionNP(l2=1e-3, epochs=800).fit(X, y).predict_proba(X))
    b = roc_auc(y, LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)[:, 1])
    assert abs(a - b) < 0.01
    assert roc_auc(y, MLPNP(epochs=60).fit(X, y).predict_proba(X)) > 0.9
    km = KMeansNP(4, seed=0).fit(X)
    sk = KMeans(4, n_init=10, random_state=0).fit(X)
    assert km.inertia_ <= sk.inertia_ * 1.02




def test_torch_models():
    """Быстрая проверка нейросетей (пропускается, если PyTorch не установлен)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch not installed — skip")
        return
    from keyml.simulate import simulate_users
    from keyml.bots import generate_bots
    from keyml.features import FeatureExtractor
    from keyml.models.seq_torch import train_gru_classifier, predict_gru, train_siamese, embed
    humans, _ = simulate_users(8, 2, calib_reps=3, texts_per_session=3, seed=1)
    fe = FeatureExtractor().fit(humans)
    bots = generate_bots(2, 4, human_pool=humans, bigram_mu=fe.bigram_mu, key_hold_mu=fe.key_hold_mu,
                         types=["const", "uniform"])
    data = humans + bots
    seqs = [fe.sequence(a, 60) for a in data]
    y = np.array([a.label for a in data])
    m = train_gru_classifier(seqs, y, epochs=3, batch=32)
    p = predict_gru(m, seqs)
    assert p.shape == (len(data),) and np.all((p >= 0) & (p <= 1))
    assert m.history[-1] < m.history[0]
    sm = train_siamese(seqs[:len(humans)], [a.user_id for a in humans], epochs=2, P=4, K=3, steps_per_epoch=5)
    E = embed(sm, seqs[:10])
    assert E.shape == (10, 32) and np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-4)


def test_cheat_probability_interval():
    """Байесовская логрегрессия: интервал содержит точку, расширяется вдали от данных; Платт калибрует."""
    from sklearn.datasets import make_classification
    from keyml.models.calibration import BayesianLogReg, PlattScaler, CheatProbability
    from keyml.metrics import ece
    X, y = make_classification(1500, 8, random_state=1, weights=[0.7])
    bl = BayesianLogReg(l2=1e-2).fit(X[:1000], y[:1000])
    m, v = bl.logit_dist(X[1000:])
    pl = PlattScaler().fit(m, y[1000:])
    cp = CheatProbability(bl, pl, pi_data=float(y.mean()))
    p, lo, hi = cp.predict(X[1000:])
    assert np.all(lo <= p + 1e-9) and np.all(p <= hi + 1e-9)
    _, v_far = bl.logit_dist(X[1000:1010] * 6)   # далеко от обучающих данных — дисперсия логита больше
    assert np.all(v_far > v[:10])
    assert ece(y[1000:], p) < 0.08
    # сдвиг априорной вероятности уменьшает P при меньшей доле читеров
    cp2 = CheatProbability(bl, pl, pi_data=float(y.mean()), pi_deploy=0.05)
    assert np.all(cp2.predict(X[1000:])[0] <= p + 1e-9)


if __name__ == "__main__":
    test_metrics()
    test_own_models_match_sklearn()
    test_cheat_probability_interval()
    test_server_flow()
    test_torch_models()
    print("all tests passed")
