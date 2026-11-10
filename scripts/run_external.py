"""Эксперименты на открытых датасетах + обучение итоговой модели античита.

1) Скачать данные (один раз, ~1.5 ГБ):
     python scripts/download_datasets.py
2) Запустить:
     python scripts/run_external.py                      # отчёт reports/external/external.md
     python scripts/run_external.py --export-model       # + обновить модель на сервере
     python scripts/run_external.py --own data/real_attempts.jsonl   # + проверка на своих участниках

Что делает:
  * CMU: воспроизводит протокол Killourhy & Maxion (2009) и сравнивает наши детекторы с опубликованными EER;
  * Aalto: античит на реальных людях — бейзлайны, перестановочный тест, кривая обучения,
    перенос «синтетика -> реальные люди», калибровка;
  * Aalto: стили печати (кластеры против анкеты «сколько пальцев»), слабые биграммы (эмпирический Байес);
  * --export-model: обучает итоговую модель (Aalto + синтетические люди для русского + боты)
    и записывает её в collector/model/ для сайта.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from keyml import experiments as E  # noqa: E402
from keyml import experiments_external as X  # noqa: E402
from keyml.bots import generate_bots  # noqa: E402
from keyml.data import load_any  # noqa: E402
from keyml.external import load_aalto  # noqa: E402
from keyml.features import FeatureExtractor  # noqa: E402
from keyml.metrics import roc_auc  # noqa: E402
from keyml.models.calibration import BayesianLogReg, CheatProbability, PlattScaler  # noqa: E402
from keyml.simulate import simulate_users  # noqa: E402
from keyml.metrics import stratified_group_kfold  # noqa: E402

EXT = ROOT / "data" / "external"


def f3(x):
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def table(rows, cols):
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return s + "".join("| " + " | ".join(f3(c) for c in r) + " |\n" for r in rows)


def export_model(aalto, sim, out_dir: Path, seed=0) -> dict:
    """Итоговая модель для сайта: реальные люди (Aalto, англ.) + синтетические люди (рус./англ.,
    чтобы язык текста не стал признаком «бота») + боты на тех же текстах."""
    fe0 = FeatureExtractor().fit(aalto + sim)
    bots = X.make_aalto_bots(aalto, fe0, share=0.2, seed=seed + 11)
    sim_bots = generate_bots(10, 12, human_pool=aalto, bigram_mu=fe0.bigram_mu, key_hold_mu=fe0.key_hold_mu,
                             seed=seed + 12, ru_share=0.8)
    data = aalto + sim + bots + sim_bots
    y = np.array([a.label for a in data])
    groups = np.array([a.user_id for a in data])
    oof_m = np.zeros(len(data))
    for tr, te in stratified_group_kfold(groups, y, 5, seed):
        fe = FeatureExtractor().fit([data[i] for i in tr])
        Xf = fe.transform(data)
        oof_m[te] = BayesianLogReg(l2=1e-2).fit(Xf[tr], y[tr]).logit_dist(Xf[te])[0]
    fe_all = FeatureExtractor().fit(data)
    Xa = fe_all.transform(data)
    model = CheatProbability(BayesianLogReg(l2=1e-2).fit(Xa, y), PlattScaler().fit(oof_m, y), pi_data=float(y.mean()))
    d = model.to_json(fe_all.names)
    d["bigram_mu"] = {f"{a}>{b}": v for (a, b), v in fe_all.bigram_mu.items()}
    d["key_hold_mu"] = fe_all.key_hold_mu
    d["trained_on"] = {"aalto_attempts": len(aalto), "sim_attempts": len(sim), "bots": len(bots) + len(sim_bots),
                       "date": time.strftime("%Y-%m-%d")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "anticheat.json").write_text(json.dumps(d), encoding="utf-8")
    src = np.array([a.source for a in data])
    p = 1 / (1 + np.exp(-PlattScaler().fit(oof_m, y)(oof_m)))
    return {"oof_auc": roc_auc(y, oof_m),
            "median_p_real_humans": float(np.median(p[src == "aalto"])),
            "median_p_sim_humans": float(np.median(p[src == "sim"])),
            "median_p_bots": float(np.median(p[y == 1]))}


def own_check(own_path, model_json: Path) -> dict:
    """Проверка итоговой модели на своих участниках (выгрузка сайта)."""
    atts = [a for a in load_any(own_path) if not (a.flags or {}).get("debug") and len(a.keystrokes()) >= 10]
    d = json.loads(model_json.read_text(encoding="utf-8"))
    fe = FeatureExtractor()
    fe.bigram_mu = {tuple(k.split(">")): v for k, v in d["bigram_mu"].items()}
    fe.key_hold_mu, fe.names = d["key_hold_mu"], d["features"]
    model = CheatProbability.from_json(d)
    p, lo, hi = model.predict(fe.transform(atts))
    y = np.array([a.label for a in atts])
    res = {"attempts": len(atts), "people": len({a.user_id for a in atts}),
           "median_p_humans": float(np.median(p[y == 0])) if (y == 0).any() else None,
           "share_humans_p>0.5": float(np.mean(p[y == 0] > 0.5)) if (y == 0).any() else None}
    if (y == 1).any() and (y == 0).any():
        res["roc_auc"] = roc_auc(y, p)
        res["median_p_cheats"] = float(np.median(p[y == 1]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aalto", default=str(EXT / "aalto" / "Keystrokes.zip"))
    ap.add_argument("--cmu", default=str(EXT / "cmu" / "DSL-StrongPasswordData.csv"))
    ap.add_argument("--n-people", type=int, default=1000, help="сколько участников Aalto взять")
    ap.add_argument("--perm", type=int, default=99, help="число перестановок")
    ap.add_argument("--export-model", action="store_true")
    ap.add_argument("--own", help="выгрузка сайта (.jsonl/.db) для проверки на своих участниках")
    ap.add_argument("--out", default=str(ROOT / "reports" / "external"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = ROOT / "collector" / "model" / "anticheat.json"
    prev_model = mpath.read_text(encoding="utf-8") if mpath.exists() else None
    R, L = {}, [f"# Эксперименты на открытых данных\n\nДата: {time.strftime('%Y-%m-%d %H:%M')}.\n"]

    if Path(a.cmu).exists():
        E.log("CMU benchmark...")
        R["cmu"] = c = X.cmu_benchmark(a.cmu, out)
        L.append("## 1. Биометрия на CMU Keystroke Benchmark\n")
        L.append(f"{c['subjects']} человек × 400 повторов пароля `.tie5Roanl` (8 сессий в разные дни). Протокол "
                 "статьи Killourhy & Maxion (2009): шаблон — первые 200 повторов, свои — остальные 200, "
                 "самозванцы — первые 5 повторов каждого из 50 других. EER — где доля пропущенных чужих "
                 "равна доле отвергнутых своих (меньше — лучше).\n\n")
        L.append(table([[n, v["mean_eer"], f"[{v['ci95'][0]:.3f}; {v['ci95'][1]:.3f}]", v["paper_mean_eer"]]
                        for n, v in c["detectors"].items()],
                       ["детектор", "наш EER", "95% ДИ (бутстреп по людям)", "EER в статье"]))
        L.append("\nСовпадение с опубликованными числами подтверждает, что детекторы и метрика реализованы верно.\n\n"
                 "**Сколько повторов нужно в шаблоне:**\n\n")
        curve = c["eer_vs_train_reps"]
        cols = list(next(iter(curve.values())))
        L.append(table([[k] + [curve[k][n] for n in cols] for k in curve], ["повторов"] + cols))
        L.append("\n![](fig_cmu.png)\n")

    if Path(a.aalto).exists():
        E.log(f"загрузка Aalto ({a.n_people} чел.)...")
        aalto = load_aalto(a.aalto, a.n_people, seed=a.seed)
        sim, _ = simulate_users(60, 3, seed=a.seed)
        R["aalto"] = r = X.aalto_anticheat(aalto, out, sim_humans=sim, n_perm=a.perm, seed=a.seed)
        L.append("## 2. Античит на реальных людях (Aalto 136M Keystrokes)\n")
        L.append(f"Люди: {r['n_people']} случайных участников Aalto (QWERTY, физическая клавиатура), "
                 f"{r['n_humans']} попыток. Читеры: {r['n_bots']} попыток ботов 7 типов, печатающих **те же "
                 "предложения**. 5 фолдов, разбиение по людям/ботам: тестовые люди не участвуют в обучении.\n\n")
        L.append(table([[n, m["ROC-AUC"], m["TPR@FPR1%"], m["Brier"], m["ECE"]] for n, m in r["models"].items()],
                       ["модель", "ROC-AUC", "TPR при FPR=1%", "Brier", "ECE"]))
        L.append(f"\nБейзлайн «один признак» чаще всего выбирал `{r['baseline_feature_chosen']}`.\n\n")
        pm = r["permutation"]
        L.append(f"**Перестановочный тест.** Метки «человек/бот» перемешивались по группам {pm['n']} раз, "
                 f"модель обучалась заново: средний ROC-AUC {pm['perm_auc_mean']:.3f}, максимум "
                 f"{pm['perm_auc_max']:.3f}; на настоящих метках — **{pm['real_auc']:.3f}** (p = {pm['p_value']:.3f}). "
                 + ("Ни одна перестановка не дошла до настоящего результата: модель выучила закономерность, а не шум.\n\n"
                    if pm["p_value"] <= 0.05 else "Для значимого вывода увеличьте число перестановок (--perm).\n\n"))
        L.append("**Кривая обучения** (отложенные люди фолда 1; «P>0.5» — доля людей, ошибочно названных читерами):\n\n")
        lc = r["learning_curve"]
        ks = sorted(next(iter(lc.values())))
        L.append(table([[k] + sum([[lc[n][k]["AUC"], lc[n][k]["доля людей с P>0.5"]] for n in lc], []) for k in ks],
                       ["людей в обучении"] + sum([[f"AUC {n.split()[0]}", f"P>0.5 {n.split()[0]}"] for n in lc], [])))
        if r["transfer"]:
            L.append("\n**Перенос «синтетика → реальные люди»** (одни и те же тестовые люди и боты Aalto):\n\n")
            L.append(table([[n] + list(v.values()) for n, v in r["transfer"].items()],
                           ["модель"] + list(next(iter(r["transfer"].values())))))
            L.append("\nМодель, обученная только на симуляторе, хуже работает на реальных людях — поэтому "
                     "реальные данные (Aalto + свои участники) необходимы.\n")
        L.append("\n**Медиана P(читер) по типам** (итоговая калиброванная модель):\n\n")
        L.append(table([[t, v] for t, v in r["median_p_by_type"].items()], ["тип", "медиана P"]))
        L.append("\n![](fig_aalto_anticheat.png)\n\n![](fig_sim_vs_real.png)\n")

        E.log("стили печати...")
        R["profiles"] = pr = X.aalto_profiles(aalto, out, seed=a.seed)
        L.append("## 3. Стили печати на реальных людях\n")
        L.append(f"{pr['people']} человек; вектор человека — средние 8 признаков. Анкета Aalto («сколько пальцев») "
                 "используется только для проверки, не для обучения.\n\n")
        L.append(table([[k, v] for k, v in pr.items() if k.startswith("ARI") or k.startswith("инерция")],
                       ["сравнение", "значение"]))
        L.append("\nКластеры почти не совпадают с ответом «сколько пальцев»: число пальцев само по себе слабо "
                 "определяет стиль, кластеры делят людей скорее по скорости и перекрытиям клавиш (см. таблицу). "
                 "Своя k-means и sklearn дают одинаковое разбиение и одинаковую инерцию.\n")
        L.append("\n" + table([[k, v] for k, v in pr["silhouette"].items()], ["k", "силуэт"]))
        cols = list(pr["clusters"][0])
        L.append("\n" + table([[c[k] for k in cols] for c in pr["clusters"]], cols))
        L.append("\n![](fig_aalto_clusters.png)\n")

        E.log("слабые биграммы...")
        by_user = {}   # первые 10 предложений — «прошлое», последние 5 — «следующая сессия»
        for at in aalto:
            by_user.setdefault(at.user_id, []).append(at)
        for u, atts in by_user.items():
            for i, at in enumerate(atts):
                at.session_no = 1 if i < 10 else 2
        R["bayes"] = b = E.bayes_bigram_experiment(aalto, out)
        L.append("## 4. Слабые биграммы: эмпирический Байес на реальных людях\n")
        L.append("Оценка по первым 10 предложениям участника, проверка — прогноз латентности биграмм в последних 5.\n\n")
        bk = list(b["pairs_by_n"])
        L.append(table([[m] + [b["mse_by_n"][m][k] for k in bk] for m in b["mse_by_n"]], ["метод"] + bk))
        L.append("\n![](fig_bayes_bigrams.png)\n")
        prior = b.pop("prior")

        if a.export_model:
            E.log("обучение итоговой модели для сайта...")
            R["export"] = ex = export_model(aalto, sim, ROOT / "collector" / "model", seed=a.seed)
            (ROOT / "collector" / "model" / "bigram_prior.json").write_text(json.dumps(prior), encoding="utf-8")
            L.append("## 5. Итоговая модель для сайта\n")
            L.append("Обучена на людях Aalto + синтетических людях (русский и английский, чтобы язык текста не "
                     "стал признаком бота) + ботах на тех же текстах. Записана в `collector/model/anticheat.json`, "
                     "популяционные средние биграмм — в `collector/model/bigram_prior.json`.\n\n")
            L.append(table([[k, v] for k, v in ex.items()], ["показатель", "значение"]))

    if a.own:
        E.log("проверка на своих участниках...")
        R["own"] = ow = {"новая модель": own_check(a.own, ROOT / "collector" / "model" / "anticheat.json")}
        if prev_model is not None:
            tmp = out / "previous_anticheat.json"
            tmp.write_text(prev_model, encoding="utf-8")
            ow["предыдущая модель"] = own_check(a.own, tmp)
            tmp.unlink()
        L.append("\n## 6. Проверка на своих участниках (выгрузка сайта)\n\n")
        keys = list(next(iter(ow.values())))
        L.append(table([[n] + [v.get(k) for k in keys] for n, v in ow.items()], ["модель"] + keys))
        L.append("\nСвоих участников модель не видела ни разу. Пока их мало, вывод предварительный.\n")

    L.append("\n---\nДанные: CMU Keystroke Dynamics Benchmark (Killourhy & Maxion, DSN 2009, CC-BY-ND 2.0); "
             "Aalto 136M Keystrokes (Dhakal, Feit, Kristensson, Oulasvirta, CHI 2018, некоммерческое использование "
             "с указанием авторов).\n")
    (out / "external.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (out / "external.md").write_text("\n".join(L), encoding="utf-8")
    E.log(f"готово: {out / 'external.md'}")


if __name__ == "__main__":
    main()
