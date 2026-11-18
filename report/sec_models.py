"""Report sections 3 (model comparison) and 4 (Bayesian tuning) (P2). Numbers come from results/ and artifacts/."""
from __future__ import annotations

import json

import pandas as pd

from report.common import ARTIFACTS, PAGE_W, RESULTS, H1, H2, H3, BODY, SMALL, P, bullets, f, img, pb, read, sp, table

ORDER = ["logreg", "gnb", "knn", "rf", "hgb", "mlp"]
NAMES = {"logreg": "Logistic Regression", "gnb": "Gaussian Naive Bayes", "knn": "Distance-weighted KNN",
         "rf": "Random Forest", "hgb": "HistGradientBoosting", "mlp": "MLP (64, 32)"}


def diagnosis(r) -> str:
    """Over/under-fitting statement derived from the measured numbers (train vs CV vs test Macro F1)."""
    gap = r.train_macro_f1_20trips - r.cv_macro_f1_mean
    txt = (f"Macro F1 on 20 of the training trips (in-sample) is {r.train_macro_f1_20trips:.4f}, 5-fold CV on unseen trips {r.cv_macro_f1_mean:.4f} "
           f"(std {r.cv_macro_f1_std:.4f} over folds) and student_test {r.test_macro_f1:.4f}. ")
    if gap > 0.03:
        return txt + (f"The in-sample score is {gap:.3f} above the unseen-trip score, which is evidence of <b>overfitting</b> (the model fits the training trips better than new ones); "
                      "it still generalises well enough because the unseen-trip scores are the best in the comparison or close to it.") if r.cv_macro_f1_mean > 0.9 else \
            txt + f"The in-sample score is {gap:.3f} above the unseen-trip score: <b>overfitting</b>."
    if r.train_macro_f1_20trips < 0.85:
        return txt + "Both the in-sample and the unseen-trip scores are low and close to each other: the model <b>underfits</b> (high bias), it cannot represent the structure of the problem rather than memorising it."
    if gap < 0.01:
        return txt + (f"The in-sample and unseen-trip scores are within {abs(gap):.3f}, so there is no overfitting; the model is simply limited in capacity "
                      "(its scores are below the tree ensembles even on the training trips), i.e. mild <b>underfitting</b>.")
    return txt + f"The in-sample score is {gap:.3f} above the unseen-trip score: a small gap, so only <b>slight overfitting</b> and no sign of underfitting."


def perf_sentence(r) -> str:
    return (f"On the 10 student_test trips: observation-level Top-1 <b>{r.test_top1:.4f}</b>, Macro F1 <b>{r.test_macro_f1:.4f}</b>, log loss <b>{r.test_log_loss:.4f}</b> "
            f"(precision {r.test_precision:.4f}, recall {r.test_recall:.4f}, calibration error {r.test_ece:.4f}, classifier-only route continuity {r.test_route_continuity_pct:.2f}%). "
            f"Training took {r.train_time_s:.1f} s on 1.07M candidate rows and inference on the {int(r.n_test_rows):,} test rows {r.inference_time_s:.3f} s.")


# Model-specific reasoning. The numbers it quotes are filled from the result rows; the explanations
# were written after looking at those numbers (see results/model_comparison.csv).
def notes(m: str, r, rows) -> dict:
    g = lambda k: rows[k]
    N = {
        "logreg": dict(
            suit="It is the natural linear baseline for a binary candidate classifier: it outputs P(candidate is the reference arc) directly through the logistic function, so the score can be used as an HMM emission "
                 "without any conversion, and it trains in seconds.",
            sl=f"Strengths: convex objective with a unique optimum, stable across folds, well calibrated (calibration error {r.test_ece:.4f}) and the fastest to score. "
               "Limitations: the decision boundary is linear in the features, so it cannot express interactions that matter here, for example that heading difference is informative only when the vehicle is moving, "
               f"or that distance rank matters only at a crowded junction. Its precision ({r.test_precision:.3f}) and recall ({r.test_recall:.3f}) are lower than the tree models' "
               f"({g('hgb').test_precision:.3f} / {g('hgb').test_recall:.3f} for HistGradientBoosting).",
            hyper="StandardScaler (fitted on the training trips only) then LogisticRegression with C = 1.0, L2 penalty, lbfgs solver, max_iter = 1000; initial values were the scikit-learn defaults except max_iter.",
            loss="Binary cross-entropy (log loss) plus the L2 penalty, minimised by lbfgs."),
        "gnb": dict(
            suit="Gaussian Naive Bayes is the generative, probabilistic counterpart of logistic regression: it needs no tuning, trains in a fraction of a second and gives a baseline for how far simple independence assumptions can go.",
            sl=f"Strength: extremely cheap ({r.train_time_s:.2f} s to train), and ranking candidates still works reasonably (Top-1 {r.test_top1:.3f}) because distance alone is a strong cue. "
               f"Limitations: the independence assumption is badly violated here (dist, dist_rank, dist_to_best and dist_win5 are strongly correlated, several features are binary flags rather than Gaussian), "
               f"which makes the probabilities over-confident: log loss {r.test_log_loss:.3f} against {g('hgb').test_log_loss:.3f} for HistGradientBoosting, calibration error {r.test_ece:.3f}, "
               f"and recall {r.test_recall:.3f} with precision only {r.test_precision:.3f} (it labels far too many candidates as true). That makes it a poor emission source.",
            hyper="GaussianNB with var_smoothing = 1e-9 (default); no other hyperparameters; class priors estimated from the training data.",
            loss="No gradient loss: class-conditional Gaussians are fitted by maximum likelihood (the objective is the joint log-likelihood)."),
        "knn": dict(
            suit="K-nearest neighbours is the instance-based representative: a candidate is scored by the labels of the most similar candidates seen in training, which makes no assumption about the shape of the decision boundary.",
            sl=f"Strengths: non-parametric and needs no training phase. Limitations: all work moves to inference (inference took {r.inference_time_s:.1f} s for {int(r.n_test_rows):,} rows against {g('hgb').inference_time_s:.2f} s for HistGradientBoosting), "
               "it is sensitive to feature scaling and to uninformative features, the positive class is only about 17% of rows so neighbourhoods are dominated by negatives, and the stored training set is large "
               "(for that reason it was fitted on a random subsample of 150,000 of the 1.07M training rows, which also limits its accuracy).",
            hyper="StandardScaler then KNeighborsClassifier with n_neighbors = 15, weights = distance, Euclidean metric; fitted on at most 150,000 random training rows.",
            loss="No explicit loss: the score is the distance-weighted fraction of positive labels among the 15 nearest neighbours."),
        "rf": dict(
            suit="A random forest handles the mixed numeric and binary features without scaling, captures interactions such as heading difference conditional on speed, and is robust to noisy GPS because it averages many de-correlated trees.",
            sl=f"Strengths: strong accuracy (the best CV Macro F1 of all models, {r.cv_macro_f1_mean:.4f}), little tuning, good probability quality (calibration error {r.test_ece:.4f}). "
               f"Limitations: slow to train ({r.train_time_s:.0f} s versus {g('hgb').train_time_s:.1f} s for HistGradientBoosting), roughly {r.inference_time_s / g('hgb').inference_time_s:.1f} times slower at inference, and the fitted model is very large "
               "(about 530 MB on disk, too large for the repository), because min_samples_leaf = 5 allows deep trees.",
            hyper="n_estimators = 150, min_samples_leaf = 5, max_features = sqrt of the number of features (default), bootstrap = True, n_jobs = -1.",
            loss="Gini impurity is minimised at every split; the score is the mean of the trees' class-1 fractions (averaged vote), so no gradient loss is involved."),
        "hgb": dict(
            suit="Gradient boosting builds trees one after another, each correcting the errors of the previous ones, which suits a problem where most candidates are easy and a minority (look-alike arcs at junctions) carry the difficulty. "
                 "The histogram implementation is fast enough to tune with many trials.",
            sl=f"Strengths: top-tier accuracy (CV Macro F1 {r.cv_macro_f1_mean:.4f}, within 0.0011 of the random forest), the lowest CV log loss ({r.cv_log_loss_mean:.4f}), probabilities that behave like probabilities "
               f"(calibration error {r.test_ece:.4f}), tiny model (about 0.3 MB) and fast ({r.train_time_s:.1f} s to train, {r.inference_ms_per_1k_rows:.2f} ms per 1,000 rows). "
               "Limitations: more hyperparameters to tune (learning rate, leaf count, regularisation), and sequential trees cannot be trained in parallel across trees.",
            hyper="max_iter = 200, learning_rate = 0.1, max_leaf_nodes = 31, min_samples_leaf = 20, l2_regularization = 0, early_stopping = auto (on for more than 10,000 rows; 10% internal validation, 10 rounds), random_state = 42.",
            loss="Binary log loss (cross-entropy) minimised by gradient boosting on the trees' raw scores."),
        "mlp": dict(
            suit="A small neural network learns non-linear feature combinations by gradient descent and is the only model here with an epoch-wise training history, so it also shows how the optimisation behaves.",
            sl=f"Strengths: flexible non-linear model with probabilistic output. Limitations: needs scaled inputs, sensitive to initialisation and learning-rate settings, and gives no feature importances. "
               f"On this tabular problem it does not beat the tree models (CV Macro F1 {r.cv_macro_f1_mean:.4f} against {g('hgb').cv_macro_f1_mean:.4f}) and its in-sample/unseen gap is reported below.",
            hyper="StandardScaler then MLPClassifier with hidden layers (64, 32), ReLU, Adam, batch_size = 2048, early_stopping = True (10% internal validation), max_iter = 60, random_state = 42; initial learning rate 0.001 (default).",
            loss="Binary cross-entropy (log loss) with an L2 penalty (alpha = 1e-4), minimised by Adam."),
    }
    return N[m]


def build(story: list):
    mc = read("model_comparison.csv")
    mc["o"] = mc["model"].map({m: i for i, m in enumerate(ORDER)})
    mc = mc.sort_values("o").reset_index(drop=True)
    rows = {r.model: r for r in mc.itertuples()}
    cv = read("cv_folds.csv")
    sel = read("model_selection.csv")
    cfg = json.loads((ARTIFACTS / "feature_config.json").read_text())
    n_models = len(mc)

    story += [P("3. Model experimentation and comparison (P2)", H1)]
    story.append(P(f"{n_models} models from different algorithm families were trained on the 80 train trips and compared. Fitting and cross-validation use only the train partition (5-fold GroupKFold grouped by trip, "
                   "so a trip is never in both the fitting and the validation part). The student_test partition (10 trips) is used for the development-stage comparison. Only label-valid points (covered by the .route) are used, "
                   "all models see the same 16 features and the same candidate table, scale-sensitive models use a StandardScaler fitted on the training part only, and random_state = 42 everywhere. "
                   "Metric definitions: <b>Top-1</b> = the highest-scoring candidate of a GPS point is the reference arc (a point whose reference arc is not a candidate counts as wrong); "
                   "<b>Macro F1</b> is computed on the binary candidate labels at threshold 0.5; <b>route continuity</b> is the share of consecutive predicted arcs that are the same arc or connected, "
                   "for the classifier-only route.", BODY))

    # ---- 3.1
    story.append(P("3.1 Overall model comparison", H2))
    hdr = ["Model", "Algorithm family", "Key configuration", "CV Macro F1 (mean +/- std)", "CV Top-1", "Route continuity (test)", "Student-test Macro F1", "Student-test Top-1", "Log loss (test)", "Inference time (test)"]
    t = [hdr]
    for r in mc.itertuples():
        t.append([NAMES[r.model], r.algorithm_family, r.config, f"{r.cv_macro_f1_mean:.4f} +/- {r.cv_macro_f1_std:.4f}", f"{r.cv_top1_mean:.4f}", f"{r.test_route_continuity_pct:.2f}%",
                  f"{r.test_macro_f1:.4f}", f"{r.test_top1:.4f}", f"{r.test_log_loss:.4f}", f"{r.inference_time_s:.3f} s"])
    story.append(table(t, [PAGE_W * w for w in (0.10, 0.09, 0.19, 0.10, 0.06, 0.08, 0.08, 0.08, 0.08, 0.08)], font=6.6))
    story.append(P("Inference time is the time to score all 113,601 student_test candidate rows. Route continuity is of the classifier-only route (before the HMM). Source: results/model_comparison.csv.", SMALL))
    t = [["Model", "Test precision", "Test recall", "Test Brier", "Calibration error (ECE)", "Candidate miss rate (test)", "Train Macro F1 (20 train trips)", "Training time (all 80 trips)"]]
    for r in mc.itertuples():
        t.append([NAMES[r.model], f"{r.test_precision:.4f}", f"{r.test_recall:.4f}", f"{r.test_brier:.4f}", f"{r.test_ece:.4f}", f"{r.test_miss_rate:.4f}",
                  f"{r.train_macro_f1_20trips:.4f}", f"{r.train_time_s:.1f} s"])
    story += [sp(6), table(t, [PAGE_W * w for w in (0.18, 0.11, 0.1, 0.1, 0.13, 0.13, 0.13, 0.12)], font=7)]
    story.append(P("The candidate miss rate is a property of candidate generation (identical for every model): the reference arc is absent from the candidate set for that share of points, so no scorer can get them right.", SMALL))

    # ---- selection
    story.append(P("Choosing the final model", H2))
    sel = sel.sort_values("rank")
    win = sel.iloc[0]
    contenders = ", ".join(NAMES[m] for m in sel[sel.contender].model)
    story += bullets([
        "<b>Primary metric: mean 5-fold GroupKFold Macro F1 on unseen trips.</b> It is computed at candidate level, penalises both false picks and missed arcs even though only about 17% of candidate rows are positive, and because the folds hold out whole trips it measures generalisation to new routes, which is what the evaluator trips need. "
        "Training accuracy was never used for selection.",
        f"<b>Contenders:</b> every model within 0.003 of the best CV Macro F1: {contenders}.",
        "<b>Probability quality decides between contenders.</b> The HMM uses the classifier score as an emission probability, so a model whose scores behave like probabilities is preferred to one that gains 0.001 of F1: lowest CV log loss, then lowest calibration error, then faster inference.",
        f"<b>Selected: {NAMES[win.model]}.</b> CV log loss {win.cv_log_loss_mean:.4f}, calibration error {win.test_ece:.4f}, inference {win.inference_time_s:.3f} s. Compared with the random forest it matches the F1 (within 0.0011), has equal or better log loss, "
        f"trains {rows['rf'].train_time_s / rows['hgb'].train_time_s:.0f} times faster (which makes Bayesian tuning with dozens of trials feasible), and its model file is about 0.3 MB instead of about 530 MB, which fits the repository.",
        "The rule was fixed before looking at the student_test numbers; student_test is shown for information only. The logistic regression is the next best on probability quality but 0.020 lower in CV Macro F1; "
        "Gaussian Naive Bayes is unsuitable as an emission source (log loss above 1).",
    ])
    t = [["Rank", "Model", "Contender", "CV Macro F1", "CV log loss", "Calibration error", "Inference (s)"]]
    for r in sel.itertuples():
        t.append([str(r.rank), NAMES[r.model], "yes" if r.contender else "no", f"{r.cv_macro_f1_mean:.4f}", f"{r.cv_log_loss_mean:.4f}", f"{r.test_ece:.4f}", f"{r.inference_time_s:.3f}"])
    story += [table(t, [PAGE_W * w for w in (0.07, 0.25, 0.12, 0.14, 0.14, 0.14, 0.14)]), sp(4)]

    # ---- 3.2
    story.append(P("3.2 Per-model questions", H2))
    story.append(P("For each model: A. understanding (questions 1 and 2), B. training and configuration (5 and 6), C. results on the student_test partition (7 and 8) and over/under-fitting (9). "
                   "Question numbers follow the brief. The answers use the measured values of results/model_comparison.csv and results/cv_folds.csv.", SMALL))
    for i, r in enumerate(mc.itertuples(), 1):
        n = notes(r.model, r, rows)
        cvf = cv[cv.model == r.model]
        story.append(P(f"3.2.{i} {NAMES[r.model]}", H3))
        story += [P("<b>A. Model understanding</b>", BODY)]
        story.append(P(f"<b>1. Why is it suitable for PathSnap?</b> {n['suit']}", BODY))
        story.append(P(f"<b>2. Strengths and limitations.</b> {n['sl']}", BODY))
        story.append(P("<b>B. Training and configuration</b>", BODY))
        story.append(P(f"<b>5. Important hyperparameters and initial values.</b> {n['hyper']}", BODY))
        story.append(P(f"<b>6. Loss / objective.</b> {n['loss']}", BODY))
        story.append(P("<b>C. Experimental results</b>", BODY))
        story.append(P(f"<b>7-8. Student-test performance.</b> {perf_sentence(r)}", BODY))
        story.append(P(f"<b>9. Over/under-fitting.</b> {diagnosis(r)} Per-fold CV Macro F1 ranged from {cvf.macro_f1.min():.3f} to {cvf.macro_f1.max():.3f}, "
                       f"and per-fold Top-1 from {cvf.top1.min():.3f} to {cvf.top1.max():.3f}, so the result depends on which trips are held out (the folds differ in trip difficulty, not in model behaviour).", BODY))

    # ---- 3.3
    story.append(P("3.3 Model visualisations", H2))
    Pn = RESULTS / "plots"
    story += img(Pn / "cv_macro_f1.png", "Figure 3.1 Cross-validation Macro F1 of every model (5-fold GroupKFold, partition: train; error bars are the standard deviation over folds).", PAGE_W * 0.85)
    story += img(Pn / "test_top1_f1.png", "Figure 3.2 student_test observation-level Top-1 accuracy and candidate-level Macro F1 for every model.", PAGE_W * 0.85)
    story += img(Pn / "log_loss.png", "Figure 3.3 Log loss (CV on train trips, and student_test): how well the scores work as probabilities.", PAGE_W * 0.85)
    story += img(Pn / "overfitting_gap.png", "Figure 3.4 In-sample (20 train trips), cross-validation and student_test Macro F1: the over/under-fitting check used in question 9.", PAGE_W * 0.85)
    story += img(Pn / "confusion_matrices.png", "Figure 3.5 Confusion matrices of every model on student_test (diagnostic; candidate level at threshold 0.5).", PAGE_W)
    story += img(Pn / "efficiency.png", "Figure 3.6 Training and inference time of every model (log scale).", PAGE_W * 0.85)
    if (Pn / "mlp_loss_curve.png").exists():
        story += img(Pn / "mlp_loss_curve.png", "Figure 3.7 MLP training loss and validation accuracy per epoch (final refit on all train trips). Only the MLP produces epoch-wise histories here; "
                                                "no curves are drawn for the other models.", PAGE_W * 0.7)
    story.append(pb())

    # ---- 4 Bayesian tuning
    trials = read("hpo_trials.csv")
    tv = read("hpo_tuned_vs_untuned.csv")
    done = trials[trials.state == "COMPLETE"]
    pruned = int((trials.state == "PRUNED").sum())
    best = done.loc[done.cv_macro_f1_mean.idxmax()]
    pcols = [c for c in trials.columns if c.startswith("param_")]
    un, tu = tv.iloc[0], tv.iloc[1]
    story.append(P("4. Bayesian hyperparameter optimisation (P2)", H1))
    story.append(P(f"The tuned model is <b>{NAMES[cfg['model']]}</b>, chosen in section 3 from the cross-validation evidence. The search used Optuna's TPE sampler (a Bayesian method) with seed 42 and a median pruner.", BODY))
    story.append(P("1. Hyperparameters in the search and 2. ranges / distributions", H3))
    story.append(table([["Hyperparameter", "Distribution", "Why it is in the search"],
                        ["learning_rate", "log-uniform 0.02 to 0.3", "step size of each boosting round; trades off with the number of rounds"],
                        ["max_iter", "integer 100 to 500, step 50", "number of boosting rounds"],
                        ["max_leaf_nodes", "integer 15 to 127", "tree complexity (capacity)"],
                        ["min_samples_leaf", "integer 20 to 300", "regularisation: forbids tiny leaves that memorise a trip"],
                        ["l2_regularization", "log-uniform 0.001 to 10", "shrinks leaf values"],
                        ["max_depth", "categorical: none, 6, 10, 16", "caps tree depth"]], [PAGE_W * 0.22, PAGE_W * 0.3, PAGE_W * 0.48]))
    story.append(P("3. Objective", H3))
    story.append(P("Maximise the <b>mean 5-fold GroupKFold Macro F1 on the 80 train trips</b> (the same folds for every trial, folds grouped by trip). The student_test partition was not used for the search, the search space, the objective or the choice of the best trial.", BODY))
    story.append(P("4. Number of trials and 5. best values", H3))
    bp = ", ".join(f"{c.replace('param_', '')} = {best[c]:.4g}" if isinstance(best[c], float) else f"{c.replace('param_', '')} = {best[c]}" for c in pcols)
    story.append(P(f"{len(trials)} trials were run ({len(done)} completed, {pruned} pruned early by the median pruner); search time {tu.search_seconds / 60:.1f} minutes. "
                   f"Best trial: #{int(best.trial)} with CV Macro F1 {best.cv_macro_f1_mean:.4f}. Best configuration: {bp}. Every trial is logged in results/hpo_trials.csv.", BODY))
    top = done.sort_values("cv_macro_f1_mean", ascending=False).head(8)
    t = [["Trial", "CV Macro F1"] + [c.replace("param_", "") for c in pcols]]
    for r in top.itertuples(index=False):
        d = r._asdict()
        t.append([str(int(d["trial"])), f"{d['cv_macro_f1_mean']:.4f}"] + [f"{d[c]:.4g}" if isinstance(d[c], float) else str(d[c]) for c in pcols])
    story += [table(t, None, font=7), P("Eight best trials. Source: results/hpo_trials.csv.", SMALL)]
    story.append(P("6. Did the tuned model outperform the untuned model?", H3))
    t = [["Variant", "CV Macro F1", "Student-test Top-1", "Student-test Macro F1", "Student-test log loss", "Inference time (s)", "Training time (s)"]]
    for r in (un, tu):
        t.append([r.variant, f"{r.cv_macro_f1_mean:.4f}", f"{r.test_top1:.4f}", f"{r.test_macro_f1:.4f}", f"{r.test_log_loss:.4f}", f"{r.inference_time_s:.3f}", f"{r.train_time_s:.1f}"])
    story += [table(t, [PAGE_W * w for w in (0.2, 0.12, 0.14, 0.15, 0.15, 0.12, 0.12)])]
    d_cv, d_t1, d_f1, d_ll = tu.cv_macro_f1_mean - un.cv_macro_f1_mean, tu.test_top1 - un.test_top1, tu.test_macro_f1 - un.test_macro_f1, tu.test_log_loss - un.test_log_loss
    verdict = "outperformed" if d_cv > 0 else "did not outperform"
    story.append(P(f"The tuned model {verdict} the untuned model on the selection metric (CV Macro F1 {d_cv:+.4f}). On student_test the change is Top-1 {d_t1:+.4f}, Macro F1 {d_f1:+.4f} and log loss {d_ll:+.4f}; "
                   f"inference time {tu.inference_time_s:.3f} s against {un.inference_time_s:.3f} s. Both rows use the same metrics and partitions. "
                   + ("Because the tuned configuration has the better CV Macro F1 it is the exported model (artifacts/model.pkl)." if cfg["variant"] == "tuned" else
                      "Because the tuned configuration did not beat the default on CV Macro F1, the default configuration is the exported model (artifacts/model.pkl); the search is kept as evidence."), BODY))
    story += img(RESULTS / "hpo_plots" / "optimization_history.png", "Figure 4.1 Optuna optimisation history (objective: CV Macro F1).", PAGE_W * 0.8)
    story += img(RESULTS / "hpo_plots" / "param_importances.png", "Figure 4.2 Hyperparameter importances estimated by Optuna from the trials.", PAGE_W * 0.8)
    story += img(RESULTS / "plots" / "hpo_tuned_vs_untuned.png", "Figure 4.3 Tuned vs untuned (same model, same metrics and partitions).", PAGE_W * 0.85)
    story.append(pb())
