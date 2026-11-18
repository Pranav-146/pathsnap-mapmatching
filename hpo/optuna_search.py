"""Bayesian hyperparameter optimisation with Optuna (P2).

    python hpo/optuna_search.py                    # HistGradientBoosting, 40 trials
    python hpo/optuna_search.py --model rf --trials 30

* Sampler: TPE (Bayesian), seed 42. A median pruner stops clearly bad trials after 2 folds.
* Objective: mean 5-fold GroupKFold Macro F1 on the 80 TRAIN trips (grouped by trip_id).
  The student_test trips are not used for the search, the search space or the choice of
  the best trial; they are only used afterwards to report the tuned vs untuned comparison.
* Every trial is logged to results/hpo_trials.csv; Optuna plots go to results/hpo_plots/.
* The best configuration is refit on all 80 trips and compared with the untuned default
  (same metrics, same partitions) in results/hpo_tuned_vs_untuned.csv.
* The model with the better CV Macro F1 (train data only) is exported to artifacts/:
  model.pkl, feature_config.json (scaler.pkl only if the model uses a scaler).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.candidates import K_DEFAULT, R_DEFAULT  # noqa: E402
from models.data import ARTIFACTS, FEATURE_COLS, RESULTS, arcs_for_trips, load_split, valid_only  # noqa: E402
from models.metrics import binary_log_loss, evaluate_scores, macro_f1  # noqa: E402

SEED = 42


def suggest(model: str, trial: optuna.Trial) -> dict:
    if model == "hgb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 300),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 16]),
        }
    if model == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 12, 20, 30]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    raise ValueError(model)


def build(model: str, params: dict):
    if model == "hgb":
        return HistGradientBoostingClassifier(random_state=SEED, **params)
    return RandomForestClassifier(n_jobs=-1, random_state=SEED, **params)


DEFAULTS = {"hgb": {"max_iter": 200, "learning_rate": 0.1}, "rf": {"n_estimators": 150, "min_samples_leaf": 5}}


def cv_score(model, params, X, y, groups, folds, trial=None):
    scores = []
    for i, (tr, va) in enumerate(folds):
        m = build(model, params).fit(X.iloc[tr], y.iloc[tr])
        s = m.predict_proba(X.iloc[va])[:, 1]
        scores.append(macro_f1(y.iloc[va], s))
        if trial is not None:
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return scores


def evaluate(model, params, train, train_pts, test, test_pts, test_arcs, folds):
    X, y, g = train[FEATURE_COLS], train["label"], train["trip_id"]
    cv = cv_score(model, params, X, y, g, folds)
    t0 = time.perf_counter()
    m = build(model, params).fit(X, y)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = m.predict_proba(test[FEATURE_COLS])[:, 1]
    infer_s = time.perf_counter() - t0
    res = evaluate_scores(test.assign(score=s), test_pts, test_arcs)
    return m, {"cv_macro_f1_mean": float(np.mean(cv)), "cv_macro_f1_std": float(np.std(cv)),
               "test_top1": res["top1"], "test_macro_f1": res["macro_f1"], "test_log_loss": res["log_loss"],
               "test_route_continuity_pct": res["route_continuity_pct"],
               "train_time_s": train_s, "inference_time_s": infer_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hgb", choices=["hgb", "rf"])
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()

    train_all, train_pts = load_split("train")
    test_all, test_pts = load_split("studtest")
    train, test = valid_only(train_all), valid_only(test_all)
    test_arcs = arcs_for_trips("student_test")
    X, y, groups = train[FEATURE_COLS], train["label"], train["trip_id"]
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
                                study_name=f"pathsnap_{args.model}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = suggest(args.model, trial)
        t0 = time.perf_counter()
        scores = cv_score(args.model, params, X, y, groups, folds, trial)
        trial.set_user_attr("fold_macro_f1", [float(s) for s in scores])
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return float(np.mean(scores))

    def log_trial(study, trial):
        print(f"trial {trial.number:2d} {trial.state.name:8s} value={trial.value if trial.value is not None else float('nan'):.4f} "
              f"best={study.best_value:.4f}", flush=True)

    study.optimize(objective, n_trials=args.trials, callbacks=[log_trial])

    rows = []
    for t in study.trials:
        r = {"trial": t.number, "state": t.state.name, "cv_macro_f1_mean": t.value,
             "seconds": t.user_attrs.get("seconds"), **{f"param_{k}": v for k, v in t.params.items()}}
        for i, v in enumerate(t.user_attrs.get("fold_macro_f1", []), 1):
            r[f"fold{i}_f1"] = v
        rows.append(r)
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "hpo_trials.csv", index=False)

    plots = RESULTS / "hpo_plots"
    plots.mkdir(exist_ok=True)
    from optuna.visualization import matplotlib as ovm
    for name, fn in (("optimization_history", ovm.plot_optimization_history), ("param_importances", ovm.plot_param_importances),
                     ("slice", ovm.plot_slice), ("parallel_coordinate", ovm.plot_parallel_coordinate)):
        try:
            fn(study)
            plt.gcf().set_size_inches(11, 5) if name != "slice" else None
            plt.tight_layout()
            plt.savefig(plots / f"{name}.png", dpi=130)
        except Exception as e:  # a plot that cannot be drawn should not lose the search record
            print(f"plot {name} skipped: {e}")
        plt.close("all")

    best = {**study.best_params}
    print("best params:", best, "value:", round(study.best_value, 4))
    m_def, r_def = evaluate(args.model, DEFAULTS[args.model], train, train_pts, test, test_pts, test_arcs, folds)
    m_tun, r_tun = evaluate(args.model, best, train, train_pts, test, test_pts, test_arcs, folds)
    cmp = pd.DataFrame([{"variant": "untuned (default)", "params": json.dumps(DEFAULTS[args.model]), **r_def},
                        {"variant": "tuned (Optuna TPE)", "params": json.dumps(best, default=str), **r_tun}])
    cmp["n_trials"] = [0, len(study.trials)]
    cmp["search_seconds"] = [0.0, float(sum(t.user_attrs.get("seconds", 0) for t in study.trials))]
    cmp.to_csv(RESULTS / "hpo_tuned_vs_untuned.csv", index=False)
    print(cmp.drop(columns=["params"]).round(4).to_string(index=False))

    # export: choose by CV Macro F1 on train data only (never by student_test)
    use_tuned = r_tun["cv_macro_f1_mean"] > r_def["cv_macro_f1_mean"]
    model, params, label = (m_tun, best, "tuned") if use_tuned else (m_def, DEFAULTS[args.model], "untuned")
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "model.pkl")
    cfg = {
        "model": args.model, "variant": label, "params": params, "random_state": SEED,
        "feature_cols": FEATURE_COLS, "scaler": None,
        "scaler_note": "not used: tree-based model, no feature scaling needed",
        "candidate_radius_m": R_DEFAULT, "candidate_k": K_DEFAULT,
        "emission": "predict_proba(X)[:, 1] normalised across each point's candidates",
        "trained_on": "all 80 train trips, label_valid == 1 rows",
        "n_train_rows": int(len(train)), "sklearn_version": sklearn.__version__,
        "selection_rule": "tuned model exported only if its CV Macro F1 (train trips) beats the untuned default",
    }
    (ARTIFACTS / "feature_config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(f"exported {label} model to artifacts/model.pkl")


if __name__ == "__main__":
    main()
