"""Model comparison plots (P2). Every plot is drawn from files in results/; nothing is simulated.

    python visualisations/model_plots.py

Outputs in results/plots/:
  cv_macro_f1.png            5-fold GroupKFold CV Macro F1 (mean, std as error bar)
  test_top1_f1.png           student_test Top-1 and Macro F1
  log_loss.png               CV and student_test log loss
  confusion_matrices.png     student_test confusion matrices of every model (diagnostic)
  overfitting_gap.png        Macro F1 on train trips (in-sample) vs CV vs student_test
  efficiency.png             training time and inference time
  mlp_loss_curve.png         MLP only: the only model here with epoch-wise training history
  hpo_tuned_vs_untuned.png   if results/hpo_tuned_vs_untuned.csv exists
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.load import REPO_ROOT  # noqa: E402

RESULTS = REPO_ROOT / "results"
OUT = RESULTS / "plots"
LABELS = {"logreg": "Logistic\nRegression", "gnb": "Gaussian\nNaive Bayes", "knn": "Distance-\nweighted KNN",
          "rf": "Random\nForest", "hgb": "HistGradient-\nBoosting", "mlp": "MLP\n(64,32)"}
C_A, C_B, C_C = "#4c78a8", "#e07b39", "#54a24b"


def _names(df):
    return [LABELS.get(m, m) for m in df["model"]]


def _bar_labels(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")


def cv_f1(df):
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(_names(df), df["cv_macro_f1_mean"], yerr=df["cv_macro_f1_std"], capsize=4, color=C_A)
    _bar_labels(ax, bars)
    ax.set_ylabel("Macro F1"); ax.set_ylim(max(0, df["cv_macro_f1_mean"].min() - 0.08), 1.0)
    ax.set_title("5-fold GroupKFold CV Macro F1, 80 train trips (error bar = std over folds)")
    fig.tight_layout(); fig.savefig(OUT / "cv_macro_f1.png", dpi=140); plt.close(fig)


def test_metrics(df):
    x, w = np.arange(len(df)), 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(x - w / 2, df["test_top1"], w, label="Observation Top-1", color=C_A)
    b2 = ax.bar(x + w / 2, df["test_macro_f1"], w, label="Macro F1 (candidates)", color=C_B)
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_xticks(x, _names(df)); ax.set_ylim(max(0, min(df["test_top1"].min(), df["test_macro_f1"].min()) - 0.08), 1.0)
    ax.legend(loc="lower right"); ax.set_title("student_test (10 trips): Top-1 accuracy and Macro F1")
    fig.tight_layout(); fig.savefig(OUT / "test_top1_f1.png", dpi=140); plt.close(fig)


def log_loss(df):
    x, w = np.arange(len(df)), 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(x - w / 2, df["cv_log_loss_mean"], w, label="CV (train trips)", color=C_A)
    b2 = ax.bar(x + w / 2, df["test_log_loss"], w, label="student_test", color=C_B)
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_xticks(x, _names(df)); ax.set_ylabel("log loss (lower is better)"); ax.legend()
    ax.set_title("Log loss: quality of the scores as probabilities (HMM emission)")
    fig.tight_layout(); fig.savefig(OUT / "log_loss.png", dpi=140); plt.close(fig)


def confusion_matrices(df):
    cms = json.loads((RESULTS / "confusion_matrices.json").read_text())
    models = [m for m in df["model"] if m in cms]
    cols = min(3, len(models)); rows = int(np.ceil(len(models) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, m in zip(axs.ravel(), models):
        cm = np.array(cms[m]["matrix_tn_fp_fn_tp"])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.axis("on"); ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.1%})", ha="center", va="center",
                        color="white" if norm[i, j] > .5 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(LABELS.get(m, m).replace("\n", " "), fontsize=10)
    fig.suptitle("student_test confusion matrices (candidate level, threshold 0.5; % = share of the true class)")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=140); plt.close(fig)


def overfitting(df):
    x, w = np.arange(len(df)), 0.27
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, (col, lab, c) in enumerate((("train_macro_f1_20trips", "train trips (in-sample, 20 trips)", C_C),
                                       ("cv_macro_f1_mean", "5-fold CV (unseen trips)", C_A),
                                       ("test_macro_f1", "student_test (unseen trips)", C_B))):
        _bar_labels(ax, ax.bar(x + (k - 1) * w, df[col], w, label=lab, color=c), "{:.3f}")
    ax.set_xticks(x, _names(df)); ax.set_ylim(max(0, df["cv_macro_f1_mean"].min() - 0.1), 1.02)
    ax.set_ylabel("Macro F1"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    ax.set_title("Over/under-fitting check: a large train-vs-unseen gap means overfitting")
    fig.tight_layout(); fig.savefig(OUT / "overfitting_gap.png", dpi=140); plt.close(fig)


def efficiency(df):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    axs[0].bar(_names(df), df["train_time_s"], color=C_A); axs[0].set_yscale("log")
    axs[0].set_title("Training time on all 80 trips (s, log)")
    axs[1].bar(_names(df), df["inference_time_s"], color=C_B); axs[1].set_yscale("log")
    axs[1].set_title("Inference time on student_test (s, log)")
    for ax in axs:
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(OUT / "efficiency.png", dpi=140); plt.close(fig)


def mlp_curve():
    p = RESULTS / "mlp_history.json"
    if not p.exists():
        return
    h = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(range(1, len(h["loss_curve"]) + 1), h["loss_curve"], color=C_A, label="training loss (log loss)")
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax2 = ax.twinx()
    ax2.plot(range(1, len(h["validation_score_curve"]) + 1), h["validation_score_curve"], color=C_B, label="validation accuracy")
    ax2.set_ylabel("validation accuracy (internal 10% hold-out)")
    ax.set_title("MLP epoch history (the only model here with epoch-wise training)")
    l1, n1 = ax.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, n1 + n2, loc="center right")
    fig.tight_layout(); fig.savefig(OUT / "mlp_loss_curve.png", dpi=140); plt.close(fig)


def tuned_vs_untuned():
    p = RESULTS / "hpo_tuned_vs_untuned.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    metrics = [("cv_macro_f1_mean", "CV Macro F1"), ("test_top1", "Test Top-1"), ("test_macro_f1", "Test Macro F1")]
    x, w = np.arange(len(metrics)), 0.38
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 1.3]})
    for k, (_, r) in enumerate(d.iterrows()):
        bars = axs[0].bar(x + (k - .5) * w, [r[m] for m, _ in metrics], w, label=r["variant"], color=[C_A, C_B][k])
        _bar_labels(axs[0], bars)
    axs[0].set_xticks(x, [n for _, n in metrics]); axs[0].set_ylim(0.8, 0.95); axs[0].legend(loc="lower right")
    axs[0].set_title("Tuned vs untuned (same model, same partitions)")
    bars = axs[1].bar(["untuned", "tuned"], d["test_log_loss"], color=[C_A, C_B]); _bar_labels(axs[1], bars, "{:.4f}")
    axs[1].set_title("student_test log loss")
    fig.tight_layout(); fig.savefig(OUT / "hpo_tuned_vs_untuned.png", dpi=140); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS / "model_comparison.csv")
    order = {m: i for i, m in enumerate(["logreg", "gnb", "knn", "rf", "hgb", "mlp"])}
    df = df.sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)
    for fn in (cv_f1, test_metrics, log_loss, confusion_matrices, overfitting, efficiency):
        fn(df)
    mlp_curve()
    tuned_vs_untuned()
    print("plots written to", OUT)


if __name__ == "__main__":
    main()
