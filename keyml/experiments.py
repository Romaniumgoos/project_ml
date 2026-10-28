"""Все эксперименты проекта. Запуск: python scripts/run_experiments.py --help"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .bots import BOT_TYPES, generate_bots
from .data import Attempt
from .features import FeatureExtractor, calib_vector
from .metrics import (StandardScaler, adjusted_rand_index, binary_report, eer, roc_auc,
                      roc_curve, silhouette, stratified_group_kfold, tpr_at_fpr)
from .models.linear import LogisticRegressionNP, MLPNP
from .models.unsupervised import (GMMNP, KMeansNP, MahalanobisDetector, PCANP,
                                  ScaledManhattanDetector)
from .profile import BigramPrior, bigram_latencies, key_errors, user_profile, adaptive_exercise
from . import corpus

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    return plt


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# =====================================================================================
# 0. Датасет
# =====================================================================================
def dataset_summary(humans: list[Attempt], cheats: list[Attempt], out: Path) -> dict:
    users = sorted({a.user_id for a in humans})
    sess = defaultdict(set)
    for a in humans:
        sess[a.user_id].add(a.session_no)
    ks = sum(len(a.keystrokes()) for a in humans)
    summ = {
        "human_attempts": len(humans), "users": len(users),
        "users_with_3plus_sessions": sum(len(v) >= 3 for v in sess.values()),
        "calib_attempts": sum(a.kind == "calib" for a in humans),
        "text_attempts": sum(a.kind != "calib" for a in humans),
        "keystrokes": ks, "cheat_attempts": len(cheats),
        "cheat_by_type": {t: sum(a.bot_type == t or a.cheat_type == t for a in cheats)
                          for t in sorted({a.bot_type or a.cheat_type or "?" for a in cheats})},
        "sources": sorted({a.source for a in humans + cheats}),
    }
    plt = _plt()
    fe = FeatureExtractor().fit(humans)
    Xh = fe.transform(humans)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    wi = fe.names.index("wpm")
    ax[0].hist(Xh[:, wi], bins=40, color="#3b6ea5")
    ax[0].set_title("Скорость людей, WPM")
    hold = np.concatenate([[k.hold for k in a.keystrokes()] for a in humans[:600]])
    dd = np.concatenate([np.diff([k.t_down for k in a.keystrokes()]) for a in humans[:600]])
    ax[1].hist(hold[(hold > 0) & (hold < 300)], bins=60, color="#3b6ea5")
    ax[1].set_title("Время удержания (hold), мс")
    ax[2].hist(np.log10(dd[(dd > 5) & (dd < 3000)]), bins=60, color="#3b6ea5")
    ax[2].set_title("log10 интервала нажатие-нажатие (dd)")
    fig.tight_layout()
    fig.savefig(out / "fig_dataset.png")
    plt.close(fig)
    return summ


# =====================================================================================
# 1. Античит
# =====================================================================================
def _sk_models():
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler as SkScaler
    return {
        "sklearn LogisticRegression": lambda: make_pipeline(SkScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000)),
        "sklearn RandomForest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=0),
        "sklearn HistGradientBoosting": lambda: HistGradientBoostingClassifier(random_state=0),
    }


def anticheat_experiment(humans, cheats, out: Path, n_folds=5, use_torch=True, gru_epochs=15,
                         seed=0) -> dict:
    data = humans + cheats
    y = np.array([a.label for a in data])
    groups = np.array([a.user_id if a.label == 0 else (a.user_id + "#cheat") for a in data])
    btype = np.array([a.bot_type or a.cheat_type or "human" for a in data])
    own = {"LogReg (numpy, своя)": lambda: LogisticRegressionNP(l2=1e-2),
           "MLP (numpy, своя)": lambda: MLPNP(hidden=(32, 16), epochs=120)}
    sk = _sk_models()
    names = list(own) + (["BiGRU (PyTorch, своя ячейка)"] if (use_torch and HAS_TORCH) else []) + list(sk)
    oof = {n: np.zeros(len(data)) for n in names}
    lr_losses = None
    folds = list(stratified_group_kfold(groups, y, n_folds, seed))
    fold_X = []   # признаки всех попыток с популяционными таблицами, оценёнными на train-людях фолда
    for f, (tr, te) in enumerate(folds):
        log(f"античит: фолд {f + 1}/{n_folds} (train {len(tr)}, test {len(te)})")
        fe = FeatureExtractor().fit([data[i] for i in tr])
        Xf = fe.transform(data)
        fold_X.append(Xf)
        Xtr, Xte = Xf[tr], Xf[te]
        for n, mk in own.items():
            m = mk().fit(Xtr, y[tr])
            oof[n][te] = m.predict_proba(Xte)
            if f == 0 and n.startswith("LogReg"):
                lr_losses = m.loss_history
        for n, mk in sk.items():
            m = mk().fit(Xtr, y[tr])
            oof[n][te] = m.predict_proba(Xte)[:, 1]
        if use_torch and HAS_TORCH:
            from .models.seq_torch import predict_gru, train_gru_classifier
            str_ = [fe.sequence(data[i], 120) for i in tr]
            ste = [fe.sequence(data[i], 120) for i in te]
            m = train_gru_classifier(str_, y[tr], epochs=gru_epochs, seed=seed)
            oof["BiGRU (PyTorch, своя ячейка)"][te] = predict_gru(m, ste)
    # ---- метрики
    res = {}
    for n in names:
        p = oof[n]
        thr1 = np.quantile(p[y == 0], 0.99)          # порог, при котором ~1% людей ложно заблокированы
        per_type = {t: float((p[(btype == t)] > thr1).mean()) for t in sorted(set(btype) - {"human"})}
        rep = binary_report(y, p, 0.5)
        res[n] = {"roc_auc": roc_auc(y, p), "f1@0.5": rep["f1"], "accuracy@0.5": rep["accuracy"],
                  "fpr@0.5": rep["fpr"], "tpr@fpr1%": tpr_at_fpr(y, p, 0.01),
                  "detect_rate_by_type@fpr1%": per_type}
        log(f"  {n}: AUC={res[n]['roc_auc']:.4f} F1={rep['f1']:.3f} TPR@1%FPR={res[n]['tpr@fpr1%']:.3f}")
    # ---- обобщение на неизвестный тип бота (leave-one-bot-type-out)
    loto = defaultdict(dict)
    types = [t for t in sorted(set(btype)) if t != "human"]
    for t in types:
        for n, mk in [("LogReg (numpy, своя)", own["LogReg (numpy, своя)"]),
                      ("sklearn RandomForest", sk["sklearn RandomForest"])]:
            scores = np.zeros(len(data))
            mask_te = np.zeros(len(data), bool)
            for (tr, te), Xf in zip(folds, fold_X):
                # таблицы FeatureExtractor строятся только по людям => исключение ботов их не меняет
                tr2 = tr[btype[tr] != t]
                te2 = te[(btype[te] == "human") | (btype[te] == t)]
                m = mk().fit(Xf[tr2], y[tr2])
                pr = m.predict_proba(Xf[te2])
                scores[te2] = pr if pr.ndim == 1 else pr[:, 1]
                mask_te[te2] = True
            loto[t][n] = roc_auc(y[mask_te], scores[mask_te])
    # ---- финальная модель для сервера + важность признаков
    fe_all = FeatureExtractor().fit(data)
    X_all = fe_all.transform(data)
    final = LogisticRegressionNP(l2=1e-2).fit(X_all, y)
    imp = sorted(zip(fe_all.names, final.w), key=lambda t: -abs(t[1]))[:12]
    # ---- графики
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n in names:
        fpr, tpr, _ = roc_curve(y, oof[n])
        ax[0].plot(fpr, tpr, label=f"{n} (AUC {res[n]['roc_auc']:.3f})", lw=1.3,
                   ls="-" if "своя" in n else "--")
    ax[0].set_xscale("log")
    ax[0].set_xlim(1e-3, 1)
    ax[0].set_xlabel("FPR (люди, принятые за читеров) — лог. шкала")
    ax[0].set_ylabel("TPR (пойманные читеры)")
    ax[0].set_title("ROC античита (out-of-fold)")
    ax[0].legend(fontsize=7)
    if lr_losses:
        ax[1].plot(lr_losses)
        ax[1].set_title("Сходимость своей логрегрессии (фолд 1)")
        ax[1].set_xlabel("итерация Adam")
        ax[1].set_ylabel("loss")
    fig.tight_layout()
    fig.savefig(out / "fig_anticheat_roc.png")
    plt.close(fig)
    # распределения ключевых признаков
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    for ax_, (fname, _) in zip(axs.ravel(), imp[:6]):
        j = fe_all.names.index(fname)
        for t in ["human"] + types:
            v = X_all[btype == t, j]
            ax_.hist(v, bins=30, alpha=0.45 if t == "human" else 0.3, density=True,
                     label=t, histtype="stepfilled" if t == "human" else "step", lw=1.2)
        ax_.set_title(fname)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_anticheat_features.png")
    plt.close(fig)
    return {"models": res, "leave_one_type_out_auc": dict(loto),
            "top_features_logreg": [(n, float(w)) for n, w in imp],
            "final_model": final.to_json(fe_all.names),
            "feature_tables": {"bigram_mu": {f"{a}>{b}": v for (a, b), v in fe_all.bigram_mu.items()},
                               "key_hold_mu": fe_all.key_hold_mu}}


# =====================================================================================
# 2. Биометрическая верификация
# =====================================================================================
def _calib_sets(humans):
    by = defaultdict(lambda: defaultdict(list))
    for a in humans:
        if a.kind != "calib":
            continue
        v = calib_vector(a)
        if v is not None:
            by[(a.user_id, a.lang)][a.session_no].append(v)
    return by


def verification_experiment(humans, out: Path, n_impostor_per_user=5) -> dict:
    by = _calib_sets(humans)
    detectors = {
        "Махаланобис + shrinkage (своя)": lambda: MahalanobisDetector(0.5),
        "Scaled Manhattan (своя)": lambda: ScaledManhattanDetector(),
    }
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM

        class _SkWrap:
            def __init__(self, mk):
                self.mk = mk

            def fit(self, X):
                Z = np.log(np.clip(X + 200, 1, None))
                self.sc = StandardScaler().fit(Z)
                self.m = self.mk().fit(self.sc.transform(Z))
                return self

            def score(self, X):
                return -self.m.score_samples(self.sc.transform(np.log(np.clip(X + 200, 1, None))))

        detectors["sklearn OneClassSVM"] = lambda: _SkWrap(lambda: OneClassSVM(nu=0.1, gamma="scale"))
        detectors["sklearn IsolationForest"] = lambda: _SkWrap(lambda: IsolationForest(n_estimators=200, random_state=0))
    except Exception:
        pass

    def run(enroll_fn, label):
        res = {n: [] for n in detectors}
        for (u, lang), sess in by.items():
            split = enroll_fn(sess)
            if split is None:
                continue
            enroll, genuine = split
            if len(enroll) < 4 or len(genuine) < 2:
                continue
            impostors = []
            for (u2, l2), s2 in by.items():
                if u2 != u and l2 == lang:
                    allv = [v for s in sorted(s2) for v in s2[s]]
                    impostors += allv[:n_impostor_per_user]
            if len(impostors) < 5:
                continue
            Xe, Xg, Xi = np.array(enroll), np.array(genuine), np.array(impostors)
            yt = np.r_[np.zeros(len(Xg)), np.ones(len(Xi))]
            for n, mk in detectors.items():
                d = mk().fit(Xe)
                res[n].append(eer(yt, np.r_[d.score(Xg), d.score(Xi)]))
        return {n: {"mean_eer": float(np.mean(v)) if v else float("nan"),
                    "std_eer": float(np.std(v)) if v else float("nan"), "users": len(v)}
                for n, v in res.items()}

    def cross_session(sess):
        ss = sorted(sess)
        if len(ss) < 2:
            return None
        return [v for s in ss[:-1] for v in sess[s]], sess[ss[-1]]

    # Честное сравнение: шаблон одинаковый (первая половина 1-й сессии),
    # меняется только то, откуда берутся «свои» тестовые попытки.
    def same_session(sess):
        ss = sorted(sess)
        v = sess[ss[0]]
        if len(v) < 6 or len(ss) < 2:
            return None
        h = len(v) // 2
        return v[:h], v[h:]

    def other_session_same_enroll(sess):
        ss = sorted(sess)
        v = sess[ss[0]]
        if len(v) < 6 or len(ss) < 2:
            return None
        return v[: len(v) // 2], sess[ss[-1]]

    cross = run(cross_session, "cross")
    same = run(same_session, "same")
    other = run(other_session_same_enroll, "other")
    # кривая: EER от числа повторов в шаблоне
    curve = {}
    for n_rep in [3, 5, 8, 12, 16, 20]:
        def fn(sess, n_rep=n_rep):
            ss = sorted(sess)
            if len(ss) < 2:
                return None
            e = [v for s in ss[:-1] for v in sess[s]]
            return (e[:n_rep], sess[ss[-1]]) if len(e) >= n_rep else None
        r = run(fn, "curve")
        curve[n_rep] = {k: v["mean_eer"] for k, v in r.items() if k.endswith("(своя)")}
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for n in next(iter(curve.values())):
        ax.plot(list(curve), [curve[k][n] for k in curve], marker="o", label=n)
    ax.set_xlabel("повторов калибровочной фразы в шаблоне")
    ax.set_ylabel("средний EER")
    ax.set_title("Верификация: сколько нужно данных для шаблона")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_verification_curve.png")
    plt.close(fig)
    return {"cross_session": cross, "same_session": same, "other_session_small_enroll": other,
            "eer_vs_enrollment": curve}


def siamese_experiment(humans, out: Path, seed=0, epochs=30) -> dict:
    """Свободный текст: обучаем эмбеддинг на одних пользователях, проверяем на других."""
    texts = [a for a in humans if a.kind != "calib"]
    users = sorted({a.user_id for a in texts})
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n_tr = int(len(users) * 0.6)
    tr_u, te_u = set(users[:n_tr]), set(users[n_tr:])
    tr = [a for a in texts if a.user_id in tr_u]
    te = [a for a in texts if a.user_id in te_u]
    fe = FeatureExtractor().fit(tr)

    def eval_scores(vecs, att, dist):
        eers = []
        by = defaultdict(list)
        for v, a in zip(vecs, att):
            by[a.user_id].append((a.session_no, v))
        for u, items in by.items():
            last = max(s for s, _ in items)
            enroll = [v for s, v in items if s < last]
            gen = [v for s, v in items if s == last]
            imp = [v for u2, it in by.items() if u2 != u for _, v in it[:4]]
            if len(enroll) < 3 or not gen:
                continue
            c = np.mean(enroll, 0)
            sc = np.r_[[dist(c, v, enroll) for v in gen], [dist(c, v, enroll) for v in imp]]
            eers.append(eer(np.r_[np.zeros(len(gen)), np.ones(len(imp))], sc))
        return float(np.mean(eers)), float(np.std(eers)), len(eers)

    # бейзлайн: статистические признаки + scaled manhattan
    Xtr = fe.transform(tr)
    sc = StandardScaler().fit(Xtr)
    Xte = sc.transform(fe.transform(te))

    def manh(c, v, enroll):
        mad = np.abs(np.array(enroll) - c).mean(0) + 0.1
        return float((np.abs(v - c) / mad).sum())
    base = eval_scores(list(Xte), te, manh)
    res = {"features+ScaledManhattan (своя)": {"mean_eer": base[0], "std_eer": base[1], "users": base[2]}}
    if HAS_TORCH:
        from .models.seq_torch import embed, train_siamese
        log(f"сиамская сеть: {len(tr_u)} обучающих пользователей, {len(te_u)} тестовых")
        m = train_siamese([fe.sequence(a, 120) for a in tr], [a.user_id for a in tr],
                          epochs=epochs, seed=seed)
        E = embed(m, [fe.sequence(a, 120) for a in te])
        cos = lambda c, v, e: float(1 - v @ (c / (np.linalg.norm(c) + 1e-9)))
        r = eval_scores(list(E), te, cos)
        res["Siamese BiGRU + triplet (PyTorch, своя)"] = {"mean_eer": r[0], "std_eer": r[1], "users": r[2]}
        plt = _plt()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(m.history)
        ax.set_title("Triplet loss сиамской сети")
        ax.set_xlabel("эпоха")
        fig.tight_layout()
        fig.savefig(out / "fig_siamese_loss.png")
        plt.close(fig)
    return res


# =====================================================================================
# 3. Профилирование (кластеризация стилей печати)
# =====================================================================================
PROFILE_FEATURES = ["wpm", "hold_mean", "dd_logstd", "rollover_frac", "bs_rate", "pause_frac",
                    "space_dd_ratio", "hold_logstd"]


def clustering_experiment(humans, out: Path, sim_users=None, seed=0) -> dict:
    fe = FeatureExtractor().fit(humans)
    texts = [a for a in humans if a.kind != "calib"]
    X = fe.transform(texts)
    idx = [fe.names.index(n) for n in PROFILE_FEATURES]
    by = defaultdict(list)
    for a, row in zip(texts, X[:, idx]):
        by[a.user_id].append(row)
    uids = sorted(by)
    U = np.array([np.mean(by[u], 0) for u in uids])
    Z = StandardScaler().fit_transform(U)
    sil, bic, inertia = {}, {}, {}
    for k in range(2, 6):
        km = KMeansNP(k, seed=seed).fit(Z)
        sil[k] = silhouette(Z, km.labels_)
        inertia[k] = float(km.inertia_)
        try:
            g = GMMNP(k, reg=1e-2, seed=seed, cov_type="diag").fit(Z)
            bic[k] = float(g.bic(Z))
        except np.linalg.LinAlgError:
            bic[k] = float("nan")
    k_best = max(sil, key=sil.get)
    km = KMeansNP(k_best, seed=seed).fit(Z)
    gm = GMMNP(k_best, reg=1e-2, seed=seed, cov_type="diag").fit(Z)
    from sklearn.cluster import KMeans
    skm = KMeans(k_best, n_init=10, random_state=seed).fit(Z)
    res = {"k_selected_by_silhouette": k_best, "silhouette": sil, "gmm_bic": bic, "inertia": inertia,
           "ARI(own kmeans, sklearn kmeans)": adjusted_rand_index(km.labels_, skm.labels_),
           "ARI(own kmeans, own GMM)": adjusted_rand_index(km.labels_, gm.predict(Z)),
           "own_inertia": float(km.inertia_), "sklearn_inertia": float(skm.inertia_)}
    clusters = []
    for j in range(k_best):
        m = km.labels_ == j
        clusters.append({"cluster": j, "size": int(m.sum()),
                         **{n: float(U[m, i].mean()) for i, n in enumerate(PROFILE_FEATURES)}})
    res["clusters"] = clusters
    if sim_users is not None:
        truth = {u.user_id: u.archetype for u in sim_users}
        t = [truth.get(u, "?") for u in uids]
        res["ARI(k-means k=3, истинный архетип симулятора)"] = adjusted_rand_index(
            KMeansNP(3, seed=seed).fit(Z).labels_, t)
        res["ARI(GMM k=3, истинный архетип симулятора)"] = adjusted_rand_index(
            GMMNP(3, reg=1e-2, seed=seed, cov_type="diag").fit(Z).predict(Z), t)
        res["ARI(выбранная модель, истинный архетип)"] = adjusted_rand_index(km.labels_, t)
    pca = PCANP(2).fit(Z)
    P = pca.transform(Z)
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sc = ax[0].scatter(P[:, 0], P[:, 1], c=km.labels_, cmap="tab10", s=30)
    for j in range(k_best):
        c = pca.transform(km.centers_[j:j + 1])[0]
        ax[0].scatter(*c, marker="X", s=150, c="k")
    ax[0].set_title(f"Стили печати: k-means (k={k_best}) в проекции PCA "
                    f"({pca.explained_variance_ratio_.sum():.0%} дисперсии)")
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    ax[1].plot(list(sil), list(sil.values()), marker="o", label="силуэт (k-means)")
    ax2 = ax[1].twinx()
    ax2.plot(list(bic), list(bic.values()), marker="s", c="C1", label="BIC (GMM)")
    ax[1].set_xlabel("k")
    ax[1].set_title("Выбор числа кластеров")
    ax[1].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_clusters.png")
    plt.close(fig)
    return res


# =====================================================================================
# 4. Слабые биграммы: эмпирический Байес vs выборочное среднее
# =====================================================================================
def bayes_bigram_experiment(humans, out: Path, sim_users=None) -> dict:
    by_user = defaultdict(list)
    for a in humans:
        by_user[a.user_id].append(a)
    train_lat, test_lat = {}, {}
    for u, atts in by_user.items():
        last = max(a.session_no for a in atts)
        if last < 2:
            continue
        train_lat[u] = [x for a in atts if a.session_no < last for x in bigram_latencies(a.keystrokes())]
        test_lat[u] = [x for a in atts if a.session_no == last for x in bigram_latencies(a.keystrokes())]
    prior = BigramPrior().fit(train_lat)
    buckets = {"n=1": (1, 1), "n=2-3": (2, 3), "n=4-9": (4, 9), "n≥10": (10, 10 ** 9)}
    err = {m: {b: [] for b in buckets} for m in ("выборочное среднее", "популяция + темп", "эмп. Байес")}
    weak_corr = []
    for u in train_lat:
        prof = user_profile(train_lat[u], prior)
        post = {(r["a"], r["b"]): r for r in prof["bigrams"]}
        tst = defaultdict(list)
        for a, b, dd in test_lat[u]:
            tst[(a, b)].append(math.log(dd))
        for bg, r in post.items():
            if bg not in tst:
                continue
            target = np.mean(tst[bg])
            for bname, (lo, hi) in buckets.items():
                if lo <= r["n"] <= hi:
                    err["выборочное среднее"][bname].append((math.log(r["raw_ms"]) - target) ** 2)
                    err["популяция + темп"][bname].append((math.log(r["expected_ms"]) - target) ** 2)
                    err["эмп. Байес"][bname].append((math.log(r["post_ms"]) - target) ** 2)
        if sim_users is not None:
            su = next((s for s in sim_users if s.user_id == u), None)
            if su is not None:
                rows = [r for r in prof["bigrams"] if r["n"] >= 2]
                if len(rows) > 10:
                    est = np.array([r["weakness"] for r in rows])
                    raw = np.array([math.log(r["raw_ms"]) - math.log(r["expected_ms"]) for r in rows])
                    true = np.array([su.quirk(r["a"], r["b"]) for r in rows])
                    rk = lambda v: np.argsort(np.argsort(v))
                    weak_corr.append((np.corrcoef(rk(est), rk(true))[0, 1], np.corrcoef(rk(raw), rk(true))[0, 1]))
    mse = {m: {b: (float(np.mean(v)) if v else float("nan")) for b, v in d.items()} for m, d in err.items()}
    counts = {b: len(v) for b, v in err["эмп. Байес"].items()}
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    w = 0.27
    for i, m in enumerate(mse):
        ax.bar(np.arange(len(buckets)) + (i - 1) * w, list(mse[m].values()), w, label=m)
    ax.set_xticks(range(len(buckets)), [f"{b}\n({counts[b]} пар)" for b in buckets])
    ax.set_ylabel("MSE log-латентности на след. сессии")
    ax.set_title("Прогноз скорости биграмм: чем меньше наблюдений, тем важнее усадка")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_bayes_bigrams.png")
    plt.close(fig)
    res = {"mse_by_n": mse, "pairs_by_n": counts}
    if weak_corr:
        res["spearman(оценка слабости, истинная особенность) — Байес"] = float(np.mean([c[0] for c in weak_corr]))
        res["spearman(оценка слабости, истинная особенность) — сырое среднее"] = float(np.mean([c[1] for c in weak_corr]))
    # демонстрация адаптивного упражнения
    u0 = next(iter(train_lat))
    lang = next(a.lang for a in by_user[u0])
    errs = {}
    for a in by_user[u0]:
        for k, v in key_errors(a.keystrokes(drop_modifiers=False)).items():
            errs[k] = errs.get(k, 0) + v
    prof = user_profile(train_lat[u0], prior, errs, min_n=2)
    res["demo"] = {"user": u0, "top_weak_bigrams": [(r["label"], round(r["post_ms"]), round(r["expected_ms"]),
                                                     round(r["p_weak"], 2)) for r in prof["bigrams"][:8]],
                   "exercise": adaptive_exercise(prof, lang, corpus.WORDS[lang], seed=1)}
    res["prior"] = prior.to_json()
    return res
