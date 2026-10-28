"""Полный прогон экспериментов.

Примеры:
  # отладка на синтетических пользователях (пока нет реальных данных)
  python scripts/run_experiments.py --simulate 50
  # реальные данные из базы сборщика (+ сгенерированные боты как негативы)
  python scripts/run_experiments.py --data collector/data/keystrokes.db
  # быстрый прогон без PyTorch
  python scripts/run_experiments.py --simulate 30 --no-torch --fast
Результаты: reports/results.md, reports/results.json, reports/fig_*.png,
модель для сервера: collector/model/anticheat.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from keyml import experiments as E  # noqa: E402
from keyml.bots import generate_bots  # noqa: E402
from keyml.data import load_any, save_jsonl  # noqa: E402
from keyml.features import FeatureExtractor  # noqa: E402
from keyml.simulate import simulate_users  # noqa: E402


def fmt(x, nd=3):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def md_table(rows, cols):
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        s += "| " + " | ".join(fmt(c) for c in r) + " |\n"
    return s


def render_md(R, args) -> str:
    L = []
    src = "СИНТЕТИЧЕСКИЕ пользователи (симулятор) — для отладки пайплайна" if args.simulate else f"реальные данные: {args.data}"
    L.append(f"# Результаты экспериментов\n\nИсточник людей: **{src}**. "
             f"Негативный класс: сгенерированные боты 7 типов"
             f"{' + реальные попытки в режиме читера' if not args.simulate else ''}.\n"
             f"Дата прогона: {time.strftime('%Y-%m-%d %H:%M')}.\n")
    d = R["dataset"]
    L.append("## 0. Датасет\n")
    L.append(md_table([[k, v] for k, v in d.items()], ["показатель", "значение"]))
    L.append("\n![](fig_dataset.png)\n")

    a = R["anticheat"]
    L.append("## 1. Античит: человек или бот\n")
    L.append("Валидация: 5 фолдов, разбиение по группам (пользователь / экземпляр бота) — одна и та же "
             "группа не попадает одновременно в train и test. Метрики out-of-fold.\n")
    rows = [[n, m["roc_auc"], m["f1@0.5"], m["accuracy@0.5"], m["fpr@0.5"], m["tpr@fpr1%"]] for n, m in a["models"].items()]
    L.append(md_table(rows, ["модель", "ROC-AUC", "F1@0.5", "Accuracy@0.5", "FPR@0.5", "TPR при FPR=1%"]))
    types = sorted(next(iter(a["models"].values()))["detect_rate_by_type@fpr1%"])
    L.append("\n**Доля пойманных читеров по типам** (порог: ложно заблокированы ~1% людей):\n\n")
    L.append(md_table([[n] + [m["detect_rate_by_type@fpr1%"][t] for t in types] for n, m in a["models"].items()],
                      ["модель"] + types))
    L.append("\n**Обобщение на неизвестный тип бота** (тип исключён из обучения, ROC-AUC на людях vs этот тип):\n\n")
    lo = a["leave_one_type_out_auc"]
    mn = list(next(iter(lo.values())))
    L.append(md_table([[t] + [lo[t][m] for m in mn] for t in lo], ["исключённый тип"] + mn))
    L.append("\n**Самые весомые признаки логрегрессии** (стандартизованные веса, + = «похоже на бота»):\n\n")
    L.append(md_table(a["top_features_logreg"], ["признак", "вес"]))
    L.append("\n![](fig_anticheat_roc.png)\n\n![](fig_anticheat_features.png)\n")

    v = R["verification"]
    L.append("## 2. Биометрическая верификация (калибровочная фраза)\n")
    L.append("Шаблон строится по прошлым сессиям, проверка — на последней сессии пользователя; "
             "«самозванцы» — первые 5 попыток каждого другого пользователя. Метрика — EER (меньше = лучше).\n\n")
    L.append(md_table([[n, m["mean_eer"], m["std_eer"], m["users"]] for n, m in v["cross_session"].items()],
                      ["детектор", "EER (шаблон: все прошлые сессии)", "std", "польз."]))
    L.append("\n**Влияние «дня»** — шаблон одинаковый (первая половина 1-й сессии), "
             "меняется источник своих тестовых попыток:\n\n")
    L.append(md_table([[n, v["same_session"][n]["mean_eer"], v["other_session_small_enroll"][n]["mean_eer"]]
                       for n in v["same_session"]], ["детектор", "тест в той же сессии", "тест в другой сессии"]))
    own = next(n for n in v["same_session"] if "Махаланобис" in n)
    a_, b_ = v["same_session"][own]["mean_eer"], v["other_session_small_enroll"][own]["mean_eer"]
    if a_ < b_:
        L.append(f"\nПри проверке в той же сессии EER ниже ({a_:.3f} против {b_:.3f}): оценка «внутри дня» "
                 "оптимистична, поэтому протокол сбора требует нескольких сессий в разные дни.\n")
    else:
        L.append(f"\nЗдесь EER в той же сессии не ниже ({a_:.3f} против {b_:.3f}): сильнее влияет разгон "
                 "внутри сессии (повторы 6–10 быстрее повторов 1–5), чем смена дня.\n")
    L.append("\n![](fig_verification_curve.png)\n")
    if "siamese" in R:
        L.append("\n### Свободный текст: сиамская сеть (тестовые пользователи не участвовали в обучении)\n\n")
        L.append(md_table([[n, m["mean_eer"], m["std_eer"], m["users"]] for n, m in R["siamese"].items()],
                          ["метод", "EER", "std", "польз."]))
        L.append("\n![](fig_siamese_loss.png)\n")

    c = R["clustering"]
    L.append("## 3. Профилирование: стили печати (кластеризация)\n")
    L.append(f"Число кластеров по силуэту: **k = {c['k_selected_by_silhouette']}**.\n\n")
    L.append(md_table([[k, c["silhouette"][k], c["gmm_bic"][k], c["inertia"][k]] for k in c["silhouette"]],
                      ["k", "силуэт (k-means)", "BIC (GMM)", "инерция"]))
    L.append("\n" + md_table([[k, v] for k, v in c.items() if k.startswith("ARI") or k in ("own_inertia", "sklearn_inertia")],
                              ["сравнение", "значение"]))
    L.append("\n**Профили кластеров (средние):**\n\n")
    cols = list(c["clusters"][0])
    L.append(md_table([[r[k] for k in cols] for r in c["clusters"]], cols))
    L.append("\n![](fig_clusters.png)\n")

    b = R["bayes"]
    L.append("## 4. Слабые биграммы: эмпирический Байес\n")
    L.append("Оценка «скорости» каждой биграммы по прошлым сессиям, проверка — насколько хорошо она "
             "предсказывает латентность на следующей сессии (MSE log-латентности).\n\n")
    bk = list(b["pairs_by_n"])
    L.append(md_table([[m] + [b["mse_by_n"][m][k] for k in bk] for m in b["mse_by_n"]], ["метод"] + bk))
    for k, val in b.items():
        if k.startswith("spearman"):
            L.append(f"\n- {k}: **{val:.3f}**")
    L.append("\n\n![](fig_bayes_bigrams.png)\n")
    L.append(f"\n**Пример персонального отчёта** ({b['demo']['user']}):\n\n")
    L.append(md_table(b["demo"]["top_weak_bigrams"], ["биграмма", "оценка, мс", "ожидалось, мс", "P(слабая)"]))
    L.append(f"\nСгенерированное упражнение: _{b['demo']['exercise']}_\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", help="файлы .db / .jsonl с реальными данными")
    ap.add_argument("--simulate", type=int, default=0, help="число синтетических пользователей")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--bots-per-type", type=int, default=8)
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--fast", action="store_true", help="меньше эпох, без LOTO для GRU")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sim_users = None
    if args.simulate:
        E.log(f"симуляция {args.simulate} пользователей...")
        humans, sim_users = simulate_users(args.simulate, args.sessions, seed=args.seed)
        real_cheats = []
    elif args.data:
        allA = [a for p in args.data for a in load_any(p)]
        humans = [a for a in allA if a.label == 0]
        real_cheats = [a for a in allA if a.label == 1]
    else:
        ap.error("укажите --data или --simulate")
    humans = [a for a in humans if len(a.keystrokes()) >= 10]
    E.log(f"людей-попыток: {len(humans)}, реальных читерских: {len(real_cheats)}")
    fe = FeatureExtractor().fit(humans)
    bots = generate_bots(args.bots_per_type, 12, human_pool=humans, bigram_mu=fe.bigram_mu,
                         key_hold_mu=fe.key_hold_mu, seed=args.seed + 7)
    cheats = real_cheats + bots
    save_jsonl(bots, ROOT / "data" / "generated_bots.jsonl")
    if args.simulate:
        save_jsonl(humans, ROOT / "data" / "simulated_humans.jsonl")

    R = {}
    R["dataset"] = E.dataset_summary(humans, cheats, out)
    use_torch = not args.no_torch
    R["anticheat"] = E.anticheat_experiment(humans, cheats, out, use_torch=use_torch,
                                            gru_epochs=8 if args.fast else 15, seed=args.seed)
    E.log("верификация...")
    R["verification"] = E.verification_experiment(humans, out)
    if use_torch and E.HAS_TORCH:
        R["siamese"] = E.siamese_experiment(humans, out, seed=args.seed, epochs=15 if args.fast else 30)
    elif not args.no_torch:
        E.log("PyTorch не установлен — нейросетевые эксперименты пропущены")
    E.log("кластеризация...")
    R["clustering"] = E.clustering_experiment(humans, out, sim_users, seed=args.seed)
    E.log("байесовские биграммы...")
    R["bayes"] = E.bayes_bigram_experiment(humans, out, sim_users)

    # модель для сервера
    mdir = ROOT / "collector" / "model"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "anticheat.json").write_text(json.dumps({**R["anticheat"].pop("final_model"),
                                                     **R["anticheat"].pop("feature_tables")}), encoding="utf-8")
    (mdir / "bigram_prior.json").write_text(json.dumps(R["bayes"].pop("prior")), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (out / "results.md").write_text(render_md(R, args), encoding="utf-8")
    E.log(f"готово: {out / 'results.md'}")


if __name__ == "__main__":
    main()
