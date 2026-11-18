"""Train and compare the candidate-scoring models (P2).

    python models/train_all.py                      # all six models
    python models/train_all.py --models logreg rf hgb

Every model is evaluated with
  * 5-fold GroupKFold CV on the 80 train trips (grouped by trip_id, so one trip never
    sits in both train and validation): Macro F1, Top-1, log loss
  * a refit on all train trips, then scored on the 10 student_test trips: Top-1, Macro F1,
    log loss, miss rate, route continuity
  * training time, inference time and the configuration used.
Rows are written to results/model_comparison.csv (re-running a model replaces its row).
Only label-valid points (covered by the .route) are used. random_state=42 everywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.data import ARTIFACTS, FEATURE_COLS, RESULTS, arcs_for_trips, load_split, valid_only  # noqa: E402
from models.metrics import binary_log_loss, evaluate_scores, macro_f1, top1_accuracy  # noqa: E402

SEED = 42
# max_rows: KNN prediction cost grows with training size, so it is fitted on a random
# subsample of candidate rows (stated in the config column of the results).
MODEL_SPECS = {
    "logreg": dict(
        build=lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
        config="StandardScaler + LogisticRegression(max_iter=1000)"),
    "gnb": dict(
        build=lambda: GaussianNB(),
        config="GaussianNB(default var_smoothing)"),
    "knn": dict(
        build=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
        config="StandardScaler + KNN(k=15, weights=distance), fitted on <=150k random rows", max_rows=150_000),
    "rf": dict(
        build=lambda: RandomForestClassifier(n_estimators=150, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        config="RandomForest(n_estimators=150, min_samples_leaf=5)"),
    "hgb": dict(
        build=lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED),
        config="HistGradientBoosting(max_iter=200, learning_rate=0.1)"),
    "mlp": dict(
        build=lambda: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=60, random_state=SEED)),
        config="StandardScaler + MLP(64,32, early_stopping, max_iter=60)"),
}


def _fit(name, X, y):
    spec = MODEL_SPECS[name]
    max_rows = spec.get("max_rows")
    if max_rows and len(X) > max_rows:
        idx = np.random.RandomState(SEED).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    m = spec["build"]()
    m.fit(X, y)
    return m


def run_model(name, train, train_pts, test, test_pts, test_arcs) -> dict:
    X, y, groups = train[FEATURE_COLS], train["label"], train["trip_id"]
    cv_f1, cv_top1, cv_ll, cv_fit = [], [], [], []
    for fold, (tr_i, va_i) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        t0 = time.perf_counter()
        m = _fit(name, X.iloc[tr_i], y.iloc[tr_i])
        cv_fit.append(time.perf_counter() - t0)
        va = train.iloc[va_i].copy()
        va["score"] = m.predict_proba(va[FEATURE_COLS])[:, 1]
        n_pts = int(train_pts[(train_pts.label_valid == 1) & train_pts.trip_id.isin(va.trip_id.unique())].shape[0])
        cv_f1.append(macro_f1(va["label"], va["score"]))
        cv_top1.append(top1_accuracy(va, n_points=n_pts))
        cv_ll.append(binary_log_loss(va["label"], va["score"]))
        print(f"  [{name}] fold {fold + 1}/5  F1={cv_f1[-1]:.4f} top1={cv_top1[-1]:.4f} ({cv_fit[-1]:.1f}s)", flush=True)

    t0 = time.perf_counter()
    final = _fit(name, X, y)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = final.predict_proba(test[FEATURE_COLS])[:, 1]
    infer_s = time.perf_counter() - t0
    res = evaluate_scores(test.assign(score=scores), test_pts, test_arcs)

    ARTIFACTS.joinpath("all_models").mkdir(parents=True, exist_ok=True)
    joblib.dump(final, ARTIFACTS / "all_models" / f"{name}.joblib")
    if name == "mlp":  # only the MLP has per-epoch histories; saved for the loss-curve plot
        est = final[-1]
        RESULTS.joinpath("mlp_history.json").write_text(json.dumps({
            "loss_curve": est.loss_curve_, "validation_score_curve": est.validation_scores_,
            "note": "train loss per epoch of the final refit; validation_scores = accuracy on the internal 10% split"}))
    return {
        "model": name,
        "cv_macro_f1_mean": float(np.mean(cv_f1)), "cv_macro_f1_std": float(np.std(cv_f1)),
        "cv_top1_mean": float(np.mean(cv_top1)), "cv_log_loss_mean": float(np.mean(cv_ll)),
        "test_top1": res["top1"], "test_macro_f1": res["macro_f1"], "test_log_loss": res["log_loss"],
        "test_miss_rate": res["miss_rate"], "test_route_continuity_pct": res["route_continuity_pct"],
        "train_time_s": train_s, "cv_fit_time_mean_s": float(np.mean(cv_fit)),
        "inference_time_s": infer_s, "inference_ms_per_1k_rows": 1000 * infer_s / (len(test) / 1000),
        "n_train_rows": len(train), "n_test_rows": len(test),
        "config": MODEL_SPECS[name]["config"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    args = ap.parse_args()

    train_all, train_pts = load_split("train")
    test_all, test_pts = load_split("studtest")
    train, test = valid_only(train_all), valid_only(test_all)
    test_arcs = arcs_for_trips("student_test")
    print(f"train rows {len(train)} (trips {train.trip_id.nunique()}), test rows {len(test)} (trips {test.trip_id.nunique()})")

    out = RESULTS / "model_comparison.csv"
    for name in args.models:
        print(f"== {name}", flush=True)
        row = run_model(name, train, train_pts, test, test_pts, test_arcs)
        old = pd.read_csv(out) if out.exists() else pd.DataFrame()
        if len(old):
            old = old[old["model"] != name]
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(out, index=False)
        print(f"   test top1={row['test_top1']:.4f}  F1={row['test_macro_f1']:.4f}  logloss={row['test_log_loss']:.4f}  "
              f"cvF1={row['cv_macro_f1_mean']:.4f}  train {row['train_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
