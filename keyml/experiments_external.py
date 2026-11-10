"""Эксперименты на открытых датасетах (CMU, Aalto) — доказательство, что модели реально обучаются.

Запуск: python scripts/run_external.py --help
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .bots import BOT_TYPES, generate_bots
from .data import Attempt
from .experiments import PROFILE_FEATURES, _plt, _sk_models, log
from .external import load_cmu, reorder_like_ours
from .features import FeatureExtractor
from .metrics import (StandardScaler, adjusted_rand_index, brier, ece, eer, roc_auc, roc_curve,
                      silhouette, stratified_group_kfold, tpr_at_fpr)
from .models.calibration import BayesianLogReg, CheatProbability, PlattScaler
from .models.linear import LogisticRegressionNP, MLPNP
from .models.unsupervised import (GMMNP, KMeansNP, MahalanobisDetector, PCANP,
                                  ScaledManhattanDetector)

# Опубликованные результаты (таблица на странице датасета CMU, средний EER и std по 51 человеку)
CMU_PAPER = {
    "Manhattan (scaled)": (0.0962, 0.0694),
    "Nearest Neighbor (Mahalanobis)": (0.0996, 0.0642),
    "SVM (one-class)": (0.1025, 0.0650),
    "Mahalanobis": (0.1101, 0.0645),
    "Manhattan": (0.1529, 0.0925),
    "Euclidean": (0.1706, 0.0952),
}


def boot_ci(values, B=2000, seed=0, level=0.95):
    """Бутстреп-интервал для среднего (ресэмплинг по людям)."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


# =====================================================================================
# 1. CMU: воспроизводим протокол Killourhy & Maxion (2009)
# =====================================================================================
class _Euclid:
    def fit(self, X):
        self.mu = X.mean(0)
        return self

    def score(self, X):
        return np.sqrt(((X - self.mu) ** 2).sum(1))


class _Manhattan(_Euclid):
    def score(self, X):
        return np.abs(X - self.mu).sum(1)


class _NNMahalanobis:
    """Ближайший сосед из обучающих повторов в метрике Махаланобиса."""

    def fit(self, X):
        self.X = X
        self.P = np.linalg.pinv(np.cov(X.T))
        return self

    def score(self, X):
        D = X[:, None, :] - self.X[None, :, :]
        return np.einsum("nmd,de,nme->nm", D, self.P, D).min(1)


def cmu_benchmark(csv_path, out: Path) -> dict:
    X, subj, sess, rep, cols = load_cmu(csv_path)
    X = reorder_like_ours(X, cols)
    subjects = sorted(set(subj))
    detectors = {
        "Manhattan (scaled)": lambda: ScaledManhattanDetector(),
        "Nearest Neighbor (Mahalanobis)": lambda: _NNMahalanobis(),
        "Mahalanobis": lambda: MahalanobisDetector(shrink=0.0, log=False),
        "Manhattan": lambda: _Manhattan(),
        "Euclidean": lambda: _Euclid(),
        "Наш: Махаланобис + log + shrinkage": lambda: MahalanobisDetector(shrink=0.5, log=True),
    }
    try:
        from sklearn.svm import OneClassSVM

        class _OCSVM:
            def fit(self, X_):
                self.sc = StandardScaler().fit(X_)
                self.m = OneClassSVM(nu=0.1, gamma="scale").fit(self.sc.transform(X_))
                return self

            def score(self, X_):
                return -self.m.score_samples(self.sc.transform(X_))
        detectors["SVM (one-class)"] = lambda: _OCSVM()
    except Exception:
        pass
    eers = {n: [] for n in detectors}
    # протокол статьи: шаблон — первые 200 повторов, свои — последние 200,
    # самозванцы — первые 5 повторов каждого из остальных 50 человек
    first5 = {s: X[(subj == s)][:5] for s in subjects}
    for s in subjects:
        Xs = X[subj == s]
        train, genuine = Xs[:200], Xs[200:]
        impostor = np.vstack([first5[o] for o in subjects if o != s])
        y = np.r_[np.zeros(len(genuine)), np.ones(len(impostor))]
        for n, mk in detectors.items():
            d = mk().fit(train)
            eers[n].append(eer(y, np.r_[d.score(genuine), d.score(impostor)]))
    res = {}
    for n, v in eers.items():
        lo, hi = boot_ci(v)
        res[n] = {"mean_eer": float(np.mean(v)), "std": float(np.std(v)), "ci95": [lo, hi],
                  "paper_mean_eer": CMU_PAPER.get(n, (None, None))[0]}
    # кривая: сколько повторов нужно в шаблоне (наш детектор и scaled Manhattan)
    curve = {}
    for n_tr in [5, 10, 20, 50, 100, 200]:
        row = {}
        for n in ["Manhattan (scaled)", "Наш: Махаланобис + log + shrinkage"]:
            v = []
            for s in subjects:
                Xs = X[subj == s]
                impostor = np.vstack([first5[o] for o in subjects if o != s])
                d = detectors[n]().fit(Xs[:n_tr])
                y = np.r_[np.zeros(200), np.ones(len(impostor))]
                v.append(eer(y, np.r_[d.score(Xs[200:]), d.score(impostor)]))
            row[n] = float(np.mean(v))
        curve[n_tr] = row
    res_curve = curve
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    names = [n for n in res if res[n]["paper_mean_eer"] is not None] + \
            [n for n in res if res[n]["paper_mean_eer"] is None]
    xs = np.arange(len(names))
    ours = [res[n]["mean_eer"] for n in names]
    err = np.array([[res[n]["mean_eer"] - res[n]["ci95"][0], res[n]["ci95"][1] - res[n]["mean_eer"]] for n in names]).T
    ax[0].bar(xs - 0.2, ours, 0.4, yerr=err, label="наша реализация (95% ДИ)", capsize=3)
    ax[0].bar(xs + 0.2, [res[n]["paper_mean_eer"] or 0 for n in names], 0.4, label="статья (2009)")
    short = {"Наш: Махаланобис + log + shrinkage": "Наш:\nМахаланобис\n+log+shrink"}
    ax[0].set_xticks(xs, [short.get(n, n.replace(" (", "\n(")) for n in names], fontsize=7, rotation=25, ha="right")
    ax[0].set_ylabel("средний EER (меньше — лучше)")
    ax[0].set_title("CMU benchmark: наша реализация против опубликованной")
    ax[0].legend(fontsize=8)
    for n in next(iter(curve.values())):
        ax[1].plot(list(curve), [curve[k][n] for k in curve], marker="o", label=n)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("повторов фразы в шаблоне")
    ax[1].set_ylabel("средний EER")
    ax[1].set_title("Кривая обучения биометрии (CMU)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_cmu.png")
    plt.close(fig)
    return {"detectors": res, "eer_vs_train_reps": res_curve, "subjects": len(subjects)}


# =====================================================================================
# 2. Aalto: античит на реальных людях
# =====================================================================================
def make_aalto_bots(humans: list[Attempt], fe: FeatureExtractor, share=0.25, seed=7):
    """Боты печатают те же предложения, что и люди (иначе модель отличала бы тексты, а не печать)."""
    prompts = [("en", a.prompt) for a in humans]
    n_needed = int(len(humans) * share / (1 - share))
    per_type = max(2, n_needed // (len(BOT_TYPES) * 12))
    return generate_bots(per_type, 12, human_pool=humans, bigram_mu=fe.bigram_mu,
                         key_hold_mu=fe.key_hold_mu, seed=seed, prompts=prompts)


def _best_single_feature(Xtr, ytr, Xte):
    """Бейзлайн «одно правило-порог»: признак с лучшим AUC на обучении."""
    best, sign, j = -1, 1, 0
    for k in range(Xtr.shape[1]):
        a = roc_auc(ytr, Xtr[:, k])
        a2 = max(a, 1 - a)
        if a2 > best:
            best, sign, j = a2, (1 if a >= 0.5 else -1), k
    return sign * Xte[:, j], j


def aalto_anticheat(humans, out: Path, sim_humans=None, n_folds=5, n_perm=30, seed=0) -> dict:
    fe0 = FeatureExtractor().fit(humans)
    bots = make_aalto_bots(humans, fe0, seed=seed + 7)
    data = humans + bots
    y = np.array([a.label for a in data])
    groups = np.array([a.user_id for a in data])
    btype = np.array([a.bot_type or "human" for a in data])
    folds = list(stratified_group_kfold(groups, y, n_folds, seed))
    log(f"Aalto: {len(humans)} попыток людей ({len(set(groups[y == 0]))} чел.), {len(bots)} ботов")
    fold_X = []
    for f, (tr, te) in enumerate(folds):
        fe = FeatureExtractor().fit([data[i] for i in tr])
        fold_X.append(fe.transform(data))
        log(f"  признаки фолда {f + 1}/{n_folds}")
    names = fe.names
    models = {"LogReg (numpy, своя)": lambda: LogisticRegressionNP(l2=1e-2),
              "MLP (numpy, своя)": lambda: MLPNP(hidden=(32, 16), epochs=60),
              **{k: v for k, v in _sk_models().items() if "Logistic" not in k}}
    oof = {n: np.zeros(len(data)) for n in models}
    oof["Бейзлайн: один признак-порог"] = np.zeros(len(data))
    chosen = []
    oof_m = np.zeros(len(data))
    for (tr, te), Xf in zip(folds, fold_X):
        for n, mk in models.items():
            m = mk().fit(Xf[tr], y[tr])
            p = m.predict_proba(Xf[te])
            oof[n][te] = p if p.ndim == 1 else p[:, 1]
        s, j = _best_single_feature(Xf[tr], y[tr], Xf[te])
        oof["Бейзлайн: один признак-порог"][te] = s
        chosen.append(names[j])
        oof_m[te] = BayesianLogReg(l2=1e-2).fit(Xf[tr], y[tr]).logit_dist(Xf[te])[0]
    # калиброванная вероятность (Платт кросс-фиттингом)
    p_cal = np.zeros(len(data))
    for tr, te in folds:
        pl = PlattScaler().fit(oof_m[tr], y[tr])
        p_cal[te] = 1 / (1 + np.exp(-pl(oof_m[te])))
    oof["Байес. LogReg + Платт (итоговая)"] = p_cal
    pi = y.mean()
    res_models = {}
    for n, p in oof.items():
        is_prob = n != "Бейзлайн: один признак-порог"
        res_models[n] = {"ROC-AUC": roc_auc(y, p), "TPR@FPR1%": tpr_at_fpr(y, p, 0.01),
                         "Brier": brier(y, p) if is_prob else None,
                         "ECE": ece(y, p) if is_prob else None}
    res_models["Бейзлайн: константа (доля читеров)"] = {"ROC-AUC": 0.5, "TPR@FPR1%": 0.01,
                                                        "Brier": float(pi * (1 - pi)), "ECE": 0.0}
    by_type = {t: float(np.median(p_cal[btype == t])) for t in ["human"] + sorted(set(btype) - {"human"})}
    # ---- перестановочный тест: метки перемешиваются ПО ГРУППАМ (человек/бот целиком)
    log(f"  перестановочный тест ({n_perm} перестановок)...")
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    glabel = {g: y[groups == g][0] for g in ug}
    real_auc = roc_auc(y, oof["LogReg (numpy, своя)"])
    perm_aucs = []
    for _ in range(n_perm):
        perm = dict(zip(ug, rng.permutation([glabel[g] for g in ug])))
        yp = np.array([perm[g] for g in groups])
        sc = np.zeros(len(data))
        for (tr, te), Xf in zip(folds, fold_X):
            if len(set(yp[tr])) < 2:
                continue
            sc[te] = LogisticRegressionNP(l2=1e-2, epochs=200).fit(Xf[tr], yp[tr]).predict_proba(Xf[te])
        perm_aucs.append(roc_auc(yp, sc))
    p_value = (1 + sum(a >= real_auc for a in perm_aucs)) / (1 + len(perm_aucs))
    # ---- кривая обучения: сколько ЛЮДЕЙ нужно в обучении
    log("  кривая обучения...")
    tr0, te0 = folds[0]
    Xf = fold_X[0]
    hum_groups = sorted(set(groups[tr0][y[tr0] == 0]))
    bot_groups = sorted(set(groups[tr0][y[tr0] == 1]))
    curve = defaultdict(lambda: defaultdict(list))
    sizes = [s for s in [3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000] if s <= len(hum_groups)]
    if sizes[-1] != len(hum_groups):
        sizes.append(len(hum_groups))
    for rep in range(3):
        r = np.random.default_rng(100 + rep)
        for k in sizes:
            hg = set(r.choice(hum_groups, size=k, replace=False))
            idx = np.array([i for i in tr0 if (y[i] == 1 or groups[i] in hg)])
            for n, mk in [("LogReg (numpy, своя)", models["LogReg (numpy, своя)"]),
                          ("MLP (numpy, своя)", models["MLP (numpy, своя)"])]:
                m = mk().fit(Xf[idx], y[idx])
                p = m.predict_proba(Xf[te0])
                curve[n][k].append((roc_auc(y[te0], p), float(np.mean(p[y[te0] == 0] > 0.5))))
    lc = {n: {int(k): {"AUC": float(np.mean([a for a, _ in v])), "AUC_std": float(np.std([a for a, _ in v])),
                       "доля людей с P>0.5": float(np.mean([b for _, b in v]))} for k, v in d.items()}
          for n, d in curve.items()}
    # ---- перенос: модель, обученная на СИНТЕТИКЕ, проверяется на реальных людях Aalto
    transfer = {}
    if sim_humans:
        fe_s = FeatureExtractor().fit(sim_humans)
        sim_bots = generate_bots(8, 12, human_pool=sim_humans, bigram_mu=fe_s.bigram_mu,
                                 key_hold_mu=fe_s.key_hold_mu, seed=seed + 99, ru_share=0.0)
        sim_data = sim_humans + sim_bots
        ys = np.array([a.label for a in sim_data])
        te_att = [data[i] for i in te0]
        for name, train_att, ytr, fe_t in [
                ("обучена на синтетике", sim_data, ys, fe_s),
                ("обучена на Aalto", [data[i] for i in tr0], y[tr0], FeatureExtractor().fit([data[i] for i in tr0]))]:
            m = LogisticRegressionNP(l2=1e-2).fit(fe_t.transform(train_att), ytr)
            p = m.predict_proba(fe_t.transform(te_att))
            transfer[name] = {"ROC-AUC на Aalto": roc_auc(y[te0], p),
                              "доля реальных людей с P>0.5": float(np.mean(p[y[te0] == 0] > 0.5)),
                              "доля ботов с P>0.5": float(np.mean(p[y[te0] == 1] > 0.5))}
    # ---- графики
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for n in ["LogReg (numpy, своя)", "MLP (numpy, своя)", "sklearn HistGradientBoosting",
              "Бейзлайн: один признак-порог"]:
        if n in oof:
            fpr, tpr, _ = roc_curve(y, oof[n])
            ax[0].plot(fpr, tpr, label=f"{n} ({res_models[n]['ROC-AUC']:.3f})",
                       ls=":" if "Бейзлайн" in n else "-")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.7, label="константа / случайно (0.5)")
    ax[0].set_title("ROC на реальных людях Aalto (out-of-fold)")
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].legend(fontsize=7)
    ax[1].hist(perm_aucs, bins=15, color="gray", label="перемешанные метки")
    ax[1].axvline(real_auc, color="C3", lw=2, label=f"настоящие метки ({real_auc:.3f})")
    ax[1].set_title(f"Перестановочный тест, p = {p_value:.3f}")
    ax[1].set_xlabel("ROC-AUC")
    ax[1].legend(fontsize=8)
    for n, d in lc.items():
        ks = sorted(d)
        ax[2].errorbar(ks, [d[k]["AUC"] for k in ks], yerr=[d[k]["AUC_std"] for k in ks], marker="o",
                       capsize=3, label=n)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("людей в обучении")
    ax[2].set_ylabel("ROC-AUC на отложенных людях")
    ax[2].set_title("Кривая обучения")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_aalto_anticheat.png")
    plt.close(fig)
    # синтетика vs реальность: распределения признаков людей
    if sim_humans:
        feats = ["dd_logstd", "hold_mean", "rollover_frac", "bs_rate", "corr_dd_prior", "quant_15ms"]
        fe_a, fe_s2 = FeatureExtractor().fit(humans), FeatureExtractor().fit(sim_humans)
        Xa = fe_a.transform(humans[:3000])
        Xs = fe_s2.transform(sim_humans)
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        for ax_, f_ in zip(axs.ravel(), feats):
            j = fe_a.names.index(f_)
            lo, hi = np.percentile(np.r_[Xa[:, j], Xs[:, j]], [1, 99])
            ax_.hist(Xa[:, j], bins=40, range=(lo, hi), density=True, alpha=0.5, label="реальные (Aalto)")
            ax_.hist(Xs[:, j], bins=40, range=(lo, hi), density=True, alpha=0.5, label="симулятор")
            ax_.set_title(f_)
        axs[0, 0].legend(fontsize=8)
        fig.suptitle("Люди: симулятор против реальных данных")
        fig.tight_layout()
        fig.savefig(out / "fig_sim_vs_real.png")
        plt.close(fig)
    return {"n_humans": int((y == 0).sum()), "n_people": len(set(groups[y == 0])), "n_bots": int((y == 1).sum()),
            "models": res_models, "median_p_by_type": by_type,
            "baseline_feature_chosen": Counter(chosen).most_common(1)[0][0],
            "permutation": {"real_auc": real_auc, "perm_auc_mean": float(np.mean(perm_aucs)),
                            "perm_auc_max": float(np.max(perm_aucs)), "p_value": float(p_value), "n": n_perm},
            "learning_curve": lc, "transfer": transfer}


# =====================================================================================
# 3. Aalto: стили печати и слабые биграммы на реальных людях
# =====================================================================================
def _fingers_group(v):
    return {"1-2": "1–4 пальца", "3-4": "1–4 пальца", "5-6": "5–8", "7-8": "5–8", "9-10": "9–10"}.get(v)


def aalto_profiles(humans, out: Path, seed=0) -> dict:
    fe = FeatureExtractor().fit(humans)
    X = fe.transform(humans)
    idx = [fe.names.index(n) for n in PROFILE_FEATURES]
    by, meta = defaultdict(list), {}
    for a, row in zip(humans, X[:, idx]):
        by[a.user_id].append(row)
        meta[a.user_id] = (a.flags or {}).get("user_meta", {})
    uids = [u for u in sorted(by) if _fingers_group(meta[u].get("FINGERS"))]
    U = np.array([np.mean(by[u], 0) for u in uids])
    Z = StandardScaler().fit_transform(U)
    truth = [_fingers_group(meta[u]["FINGERS"]) for u in uids]
    sil = {k: silhouette(Z[:800], KMeansNP(k, seed=seed).fit(Z[:800]).labels_) for k in range(2, 6)}
    km = KMeansNP(3, seed=seed).fit(Z)
    gm = GMMNP(3, reg=1e-2, seed=seed, cov_type="diag").fit(Z)
    from sklearn.cluster import KMeans
    sk = KMeans(3, n_init=10, random_state=seed).fit(Z)
    clusters = []
    for j in range(3):
        m = km.labels_ == j
        fing = Counter(t for t, mm in zip(truth, m) if mm)
        clusters.append({"кластер": j, "людей": int(m.sum()),
                         **{n: float(U[m, i].mean()) for i, n in enumerate(PROFILE_FEATURES[:4])},
                         "доля «9–10 пальцев»": fing["9–10"] / max(m.sum(), 1)})
    pca = PCANP(2).fit(Z)
    P = pca.transform(Z)
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"1–4 пальца": "C3", "5–8": "C1", "9–10": "C0"}
    for t, c in colors.items():
        m = np.array(truth) == t
        ax[0].scatter(P[m, 0], P[m, 1], s=6, c=c, alpha=0.5, label=t)
    ax[0].set_title("Реальные люди (Aalto) в PCA: цвет — сколько пальцев (анкета)")
    ax[0].legend(fontsize=8)
    for j in range(3):
        m = km.labels_ == j
        ax[1].scatter(P[m, 0], P[m, 1], s=6, alpha=0.5, label=f"кластер {j}")
    ax[1].set_title("k-means (k=3), без анкеты")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_aalto_clusters.png")
    plt.close(fig)
    return {"people": len(uids), "silhouette": sil,
            "ARI(k-means, пальцы по анкете)": adjusted_rand_index(km.labels_, truth),
            "ARI(GMM, пальцы по анкете)": adjusted_rand_index(gm.predict(Z), truth),
            "ARI(своя k-means, sklearn k-means)": adjusted_rand_index(km.labels_, sk.labels_),
            "инерция своей k-means (меньше — лучше)": float(km.inertia_),
            "инерция sklearn k-means": float(sk.inertia_),
            "clusters": clusters}
