"""Does feature v2 help? Same HistGradientBoosting, v1 features vs v1+v2 features (P1).

5-fold GroupKFold CV on the 80 train trips, then a refit on all train trips scored on the 10
student_test trips. Writes results/feature_v2_ablation.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.build_features import FEATURE_COLS, FEATURE_COLS_V1  # noqa: E402
from models.data import RESULTS, load_split, valid_only  # noqa: E402
from models.metrics import binary_log_loss, evaluate_scores, macro_f1, top1_accuracy  # noqa: E402


def run(name, cols, train, train_pts, test, test_pts):
    X, y, g = train[cols], train["label"], train["trip_id"]
    f1s, t1s, lls = [], [], []
    for tr, va in GroupKFold(n_splits=5).split(X, y, g):
        m = HistGradientBoostingClassifier(max_iter=200, random_state=42).fit(X.iloc[tr], y.iloc[tr])
        v = train.iloc[va].assign(score=m.predict_proba(X.iloc[va])[:, 1])
        n = int(train_pts[(train_pts.label_valid == 1) & train_pts.trip_id.isin(v.trip_id.unique())].shape[0])
        f1s.append(macro_f1(v["label"], v["score"]))
        t1s.append(top1_accuracy(v, n_points=n))
        lls.append(binary_log_loss(v["label"], v["score"]))
    m = HistGradientBoostingClassifier(max_iter=200, random_state=42).fit(X, y)
    res = evaluate_scores(test.assign(score=m.predict_proba(test[cols])[:, 1]), test_pts)
    return {"features": name, "n_features": len(cols), "cv_macro_f1_mean": np.mean(f1s), "cv_macro_f1_std": np.std(f1s),
            "cv_top1_mean": np.mean(t1s), "cv_log_loss_mean": np.mean(lls),
            "test_top1": res["top1"], "test_macro_f1": res["macro_f1"], "test_log_loss": res["log_loss"]}


def main():
    train_all, train_pts = load_split("train")
    test_all, test_pts = load_split("studtest")
    train, test = valid_only(train_all), valid_only(test_all)
    rows = [run("v1 (10 features)", FEATURE_COLS_V1, train, train_pts, test, test_pts),
            run("v2 (v1 + 6 context features)", FEATURE_COLS, train, train_pts, test, test_pts)]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "feature_v2_ablation.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
