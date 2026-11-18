"""Assemble the technical report PDF (P4).

    python report/build_report.py            # -> report/PathSnap_Report.pdf

Each section module reads its numbers from results/ and artifacts/ (nothing is typed in by hand):
  sec_eda.py (P1)  section 2.2        sec_models.py (P2)  sections 3 and 4
  sec_hmm.py (P3)  sections 5 and 6   sec_api.py (P4)     section 7
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report import sec_api, sec_eda, sec_hmm, sec_models  # noqa: E402
from report.common import ARTIFACTS, PAGE_W, RESULTS, H1, H2, BODY, SMALL, P, bullets, f, pb, read, sp, table  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib import colors  # noqa: E402

OUT = Path(__file__).resolve().parent / "PathSnap_Report.pdf"
REPO_URL = "https://github.com/Pranav-146/pathsnap-mapmatching"


def front_matter(story):
    title = ParagraphStyle("t", parent=H1, fontSize=22, leading=27, spaceAfter=4)
    story += [sp(30), P("PathSnap: Distributed Map-Matching and Route Reconstruction", title),
              P("Technical report, UE24CS352A Machine Learning hackathon", H2), sp(6),
              P(f"Repository: {REPO_URL}<br/>Date: {date.today():%d %B %Y}", BODY), sp(10)]
    story.append(P("Team and contributions", H2))
    story.append(table([["Member", "Part", "Report sections"],
                        ["P1 (Pranav-146)", "Data loading, EDA, labels, candidates, features", "2.2"],
                        ["P2 (srijan-nanda)", "Metrics, model comparison, Bayesian tuning, final model export", "3, 4"],
                        ["P3 (skandanadig)", "Road graph, HMM/Viterbi, pipeline, decoder tuning, Folium maps", "5, 6"],
                        ["P4 (sneha1a408-stud)", "FastAPI backend, UI, repository, report assembly", "7, assembly"]],
                       [PAGE_W * 0.26, PAGE_W * 0.54, PAGE_W * 0.2]))
    story.append(sp(10))

    # headline numbers, all from files
    mc = read("model_comparison.csv")
    cfg = __import__("json").loads((ARTIFACTS / "feature_config.json").read_text())
    tv = read("hpo_tuned_vs_untuned.csv")
    hc = read("hmm_comparison.csv")
    w = hc["n_label_valid"]
    pooled = lambda c: float((hc[c] * w).sum() / w.sum())
    final = tv.iloc[1] if cfg["variant"] == "tuned" else tv.iloc[0]
    story.append(P("Summary of results", H2))
    story += bullets([
        f"<b>Models.</b> {len(mc)} models from different families were compared with 5-fold GroupKFold cross-validation over the 80 train trips and scored on the 10 student_test trips (section 3).",
        f"<b>Final model.</b> {cfg['model'].upper()} ({cfg['variant']} configuration, {len(cfg['feature_cols'])} features), exported to artifacts/model.pkl: CV Macro F1 {f(final.cv_macro_f1_mean)}, "
        f"student_test Top-1 {f(final.test_top1)}, Macro F1 {f(final.test_macro_f1)}, log loss {f(final.test_log_loss)}.",
        f"<b>Bayesian tuning.</b> Optuna TPE, {int(tv.iloc[1].n_trials)} trials; tuned vs untuned CV Macro F1 {f(tv.iloc[1].cv_macro_f1_mean)} vs {f(tv.iloc[0].cv_macro_f1_mean)} (section 4).",
        f"<b>HMM.</b> On the 10 student_test trips, pooled pointwise Top-1 rises from {f(pooled('top1_classifier_only'))} (classifier only) to {f(pooled('top1_classifier_hmm'))} (classifier + HMM/Viterbi) (section 5).",
        "<b>System.</b> FastAPI backend and Streamlit UI on two machines; accuracy is computed on the server and ground truth never leaves it (section 7).",
    ])
    story.append(P("All numbers and figures in this report are read from files in the repository's results/ and artifacts/ folders by report/build_report.py, "
                   "and can be regenerated with the commands in the README. The 10 evaluator trips were never used for training, tuning, feature selection or any threshold.", SMALL))
    story.append(pb())


def back_matter(story):
    story.append(P("8. Reproducibility, limitations and AI-use disclosure", H1))
    story.append(P("Reproducibility", H2))
    story += bullets([
        "<font name='Courier'>random_state=42</font> everywhere; GroupKFold(5) grouped by trip so one trip never sits in both train and validation.",
        "The README lists the commands in order (build_features, train_all, select_final, optuna_search, model_plots, tune_hmm, evaluate_hmm, capture_screenshots, build_report).",
        "Experiment logs are kept in results/: model_comparison.csv, cv_folds.csv, hpo_trials.csv, hmm_tuning_log.csv, hmm_comparison.csv and the raw API console log.",
    ])
    story.append(P("Limitations", H2))
    story += bullets([
        "The reference route is used as ground truth but is not guaranteed to be error-free; for two train trips it covers only part of the track, and those points are excluded rather than guessed.",
        "Labels are derived (nearest arc on the route, forward-only), so U-turns, duplicated two-way geometry and a reference route that is itself approximate can make some labels ambiguous.",
        "The decoder parameters were tuned on 12 train trips and the 10 student_test trips; the student_test numbers for the HMM are therefore slightly optimistic, which is why a hold-out of 12 other train trips is reported next to them (section 5).",
        "The candidate radius (50 m) and k (8) were chosen from the miss-rate table; points whose reference arc is outside the candidate set cannot be recovered by any scorer.",
        "The screenshots were captured with API and UI on one machine; a cross-machine run only changes the API URL.",
    ])
    story.append(P("AI-use disclosure", H2))
    story.append(P("The source code of this project and the first draft of this report's text were produced with the help of an AI coding assistant (Claude Code, Anthropic), working from the team's plan and the course guidelines. "
                   "The AI assistant did not replace the experiments: every model was trained, every number computed, and every figure drawn by the code in this repository from the provided data, "
                   "and the report builder reads each number from a result file. The team reviewed the results, and each member is responsible for explaining the part they own "
                   "(data and features, models and tuning, HMM and maps, API and UI) in the viva. Per-model discussion and analysis text in sections 3 to 5 was drafted from the measured results and should be read, "
                   "checked and adjusted by the owner of each section before submission.", BODY))


def main():
    story: list = []
    front_matter(story)
    sec_eda.build(story)
    sec_models.build(story)
    sec_hmm.build(story)
    sec_api.build(story)
    back_matter(story)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(2 * cm, 1.1 * cm, "PathSnap technical report")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="PathSnap technical report", author="PathSnap team")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
