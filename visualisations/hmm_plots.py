"""HMM comparison plot (P3), drawn from results/hmm_comparison.csv.

    python visualisations/hmm_plots.py     # -> results/plots/hmm_comparison.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.load import REPO_ROOT  # noqa: E402


def main():
    df = pd.read_csv(REPO_ROOT / "results" / "hmm_comparison.csv")
    x, w = np.arange(len(df)), 0.27
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for k, (col, lab, c) in enumerate((("top1_nearest_arc", "nearest arc (no model, no HMM)", "#b8c2cd"),
                                       ("top1_classifier_only", "classifier only", "#e07b39"),
                                       ("top1_classifier_hmm", "classifier + HMM", "#c0392b"))):
        ax.bar(x + (k - 1) * w, df[col], w, label=lab, color=c)
    ax.set_xticks(x, [f"{t}\n({n:,} pts)" for t, n in zip(df.trip_id, df.n_points)], fontsize=8)
    ax.set_ylim(max(0, df[["top1_nearest_arc", "top1_classifier_only"]].min().min() - 0.1), 1.02)
    ax.set_ylabel("pointwise Top-1"); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("student_test: pointwise Top-1 per trip, nearest arc vs classifier vs classifier + HMM")
    fig.tight_layout()
    out = REPO_ROOT / "results" / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "hmm_comparison.png", dpi=140)
    print("wrote", out / "hmm_comparison.png")


if __name__ == "__main__":
    main()
