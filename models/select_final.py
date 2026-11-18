"""Final-model selection (P2).

Rule, stated before looking at the numbers and using TRAIN-trip evidence only:
  1. primary metric: mean 5-fold GroupKFold Macro F1 (higher is better);
  2. every model within 0.003 of the best CV Macro F1 is a contender;
  3. among contenders prefer the lowest CV log loss, because the HMM uses the scores as
     probabilities (emission), so probability quality matters more than the last 0.001 of F1;
  4. remaining ties: lower test calibration error, then faster inference.
student_test numbers are shown for information (development-stage comparison) but are not used
by the rule. Writes results/model_selection.csv and prints the ranking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.data import RESULTS  # noqa: E402

TOL = 0.003


def select(df: pd.DataFrame) -> pd.DataFrame:
    best = df["cv_macro_f1_mean"].max()
    df = df.copy()
    df["contender"] = df["cv_macro_f1_mean"] >= best - TOL
    order = df.sort_values(["contender", "cv_log_loss_mean", "test_ece", "inference_time_s"],
                           ascending=[False, True, True, True])
    order["rank"] = range(1, len(order) + 1)
    return order


def main():
    df = pd.read_csv(RESULTS / "model_comparison.csv")
    r = select(df)
    cols = ["rank", "model", "contender", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_log_loss_mean", "test_ece",
            "test_top1", "test_macro_f1", "inference_time_s", "train_time_s"]
    r[cols].to_csv(RESULTS / "model_selection.csv", index=False)
    print(r[cols].round(4).to_string(index=False))
    print("\nselected:", r.iloc[0]["model"])


if __name__ == "__main__":
    main()
