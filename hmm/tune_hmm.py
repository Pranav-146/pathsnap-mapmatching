"""Tune the HMM decoder parameters (P3). Uses TRAIN and student_test data only.

    python hmm/tune_hmm.py

Tuning set   : 12 train trips (scored with out-of-fold classifier scores, so the classifier never saw
               the trip it is scoring) + the 10 student_test trips (scored by the final model, which
               was fitted on train only).
Hold-out set : 12 OTHER train trips, also out-of-fold. They are not used for tuning and give a check that
               is not tuned on the data it is measured on.
Objective    : pointwise Top-1 pooled over the tuning set's label-valid points.
Search       : two rounds of coordinate descent over beta, emission_weight, same/adjacent bonus,
               gap_T and straight_mode (road distances are computed once per trip, so each
               evaluation is only the Viterbi recurrence). sigma is used only by the distance-only
               emission variant, so it is tuned separately for that variant.
The evaluator trips are never touched. Writes artifacts/hmm_config.json, results/hmm_tuning_log.csv,
results/hmm_holdout_train_trips.csv.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.load import DATA_ROOT, REPO_ROOT, load_trip  # noqa: E402
from features.candidates import TripGeometry  # noqa: E402
from hmm.graph import RoadGraph  # noqa: E402
from hmm.viterbi import (HMMParams, classifier_only, emission_distance_only, emission_from_scores,  # noqa: E402
                         prepare, viterbi)
from hpo.optuna_search import build  # noqa: E402
from models.data import FEATURE_COLS, RESULTS, load_split, valid_only  # noqa: E402
from models.predict import load_artifacts, predict_scores  # noqa: E402

SEED = 42
N_TUNE_TRAIN, N_HOLDOUT = 12, 12
GRID = {
    "beta": [5.0, 10.0, 20.0, 40.0, 80.0, 160.0],
    "emission_weight": [0.1, 0.25, 0.5, 1.0, 2.0, 3.0],
    "same_bonus": [0.0, 0.5, 1.0, 2.0],
    "adjacent_bonus": [0.0, 0.25, 0.5, 1.0],
    "gap_T": [10.0, 30.0, 60.0, 120.0],
    "straight_mode": ["projected", "gps"],
}
SIGMAS = [3.0, 5.0, 8.0, 12.0, 20.0, 30.0]


def oof_scores(train_all: pd.DataFrame, model: str, params: dict) -> pd.DataFrame:
    """Out-of-fold classifier scores for every candidate row of the train trips."""
    fit = train_all[train_all["label_valid"] == 1]
    trips = np.array(sorted(train_all["trip_id"].unique()))
    folds = GroupKFold(n_splits=5).split(fit[FEATURE_COLS], fit["label"], fit["trip_id"])
    parts = []
    for tr, va in folds:
        held = set(fit.iloc[va]["trip_id"].unique())
        m = build(model, params).fit(fit.iloc[tr][FEATURE_COLS], fit.iloc[tr]["label"])
        rows = train_all[train_all["trip_id"].isin(held)]
        parts.append(rows.assign(score=m.predict_proba(rows[FEATURE_COLS])[:, 1]))
    out = pd.concat(parts, ignore_index=True)
    out[["trip_id", "obs_idx", "arc_id", "score"]].to_parquet(RESULTS / "oof_scores_train.parquet", index=False)
    return out


class TripCase:
    """One trip prepared for fast repeated decoding."""

    def __init__(self, partition: str, trip_dir: Path, scored: pd.DataFrame, labels: pd.DataFrame):
        trip = load_trip(trip_dir)
        geom = TripGeometry(trip["track"], trip["nodes"], trip["arcs"])
        self.partition, self.trip_id = partition, trip["trip_id"]
        self.t = trip["track"]["t"].to_numpy()
        self.n = len(self.t)
        self.rg = RoadGraph(geom)
        self.prep = prepare(scored, self.rg, self.t, geom.track_xy, self.n)
        self.em = emission_from_scores(self.prep.c)
        v = labels[labels["label_valid"] == 1]
        self.obs, self.truth = v["obs_idx"].to_numpy(), v["label_arc"].to_numpy()
        self.clf_arcs = np.full(self.n, -1)
        best = classifier_only(self.prep.c, self.em)
        self.clf_arcs[best["obs_idx"].to_numpy()] = best["arc_id"].to_numpy()

    def correct_hmm(self, params: HMMParams, distance_only_sigma: float | None = None) -> int:
        em = self.em if distance_only_sigma is None else emission_distance_only(self.prep.c, distance_only_sigma)
        arcs = viterbi(None, em, self.rg, self.t, params, prepared=self.prep)["arc_id"].to_numpy()
        return int((arcs[self.obs] == self.truth).sum())

    def correct_clf(self) -> int:
        return int((self.clf_arcs[self.obs] == self.truth).sum())


def pooled(cases, params, sigma=None) -> float:
    ok = sum(c.correct_hmm(params, sigma) for c in cases)
    return ok / sum(len(c.obs) for c in cases)


def main():
    cfg = json.loads((REPO_ROOT / "artifacts" / "feature_config.json").read_text())
    train_all, train_pts = load_split("train")
    test_all, test_pts = load_split("studtest")

    t0 = time.perf_counter()
    oof = oof_scores(train_all, cfg["model"], cfg["params"])
    print(f"out-of-fold train scores ready ({time.perf_counter() - t0:.0f}s)", flush=True)
    load_artifacts()
    test_scored = predict_scores(test_all)

    # eligible train trips: label-valid >= 95% and 300..4000 points
    g = train_pts.groupby("trip_id").agg(n=("obs_idx", "size"), valid=("label_valid", "mean"))
    elig = sorted(g[(g.valid >= 0.95) & g.n.between(300, 4000)].index)
    pick = list(np.random.RandomState(SEED).permutation(elig))
    tune_ids, hold_ids = sorted(pick[:N_TUNE_TRAIN]), sorted(pick[N_TUNE_TRAIN:N_TUNE_TRAIN + N_HOLDOUT])
    print("tuning train trips:", tune_ids, "\nhold-out train trips:", hold_ids, flush=True)

    def case(part, tid, scored_all, pts):
        sc = scored_all[scored_all["trip_id"] == tid]
        return TripCase(part, DATA_ROOT / ("train" if part == "train" else "student_test") / tid, sc, pts[pts["trip_id"] == tid])

    tune_cases = [case("train", t, oof, train_pts) for t in tune_ids]
    tune_cases += [case("student_test", t, test_scored, test_pts) for t in sorted(test_all["trip_id"].unique())]
    hold_cases = [case("train", t, oof, train_pts) for t in hold_ids]
    print(f"prepared {len(tune_cases)} tuning + {len(hold_cases)} hold-out trips ({time.perf_counter() - t0:.0f}s)", flush=True)

    n_tune = sum(len(c.obs) for c in tune_cases)
    clf_top1 = sum(c.correct_clf() for c in tune_cases) / n_tune
    default = HMMParams()
    log = [{"round": 0, "param": "default", "value": "", "tuning_top1": pooled(tune_cases, default)}]
    print(f"classifier-only Top-1 on tuning set: {clf_top1:.4f}; HMM with default params: {log[0]['tuning_top1']:.4f}", flush=True)

    best, best_score = HMMParams(**default.to_dict()), log[0]["tuning_top1"]
    for rnd in (1, 2):
        for key, values in GRID.items():
            for v in values:
                cand = HMMParams(**{**best.to_dict(), key: v})
                s = pooled(tune_cases, cand)
                log.append({"round": rnd, "param": key, "value": v, "tuning_top1": s})
                if s > best_score + 1e-9:
                    best, best_score = cand, s
            print(f"round {rnd} {key:16s} -> {getattr(best, key)}  (Top-1 {best_score:.4f})", flush=True)

    sig = []
    for s_ in SIGMAS:
        sc = pooled(tune_cases, HMMParams(**{**best.to_dict()}), s_)
        log.append({"round": 3, "param": "sigma (distance-only emission)", "value": s_, "tuning_top1": sc})
        sig.append((sc, s_))
    best_sigma = max(sig)[1]
    best.sigma = best_sigma

    pd.DataFrame(log).to_csv(RESULTS / "hmm_tuning_log.csv", index=False)
    hold = []
    for c in hold_cases:
        n = len(c.obs)
        hold.append({"trip_id": c.trip_id, "n_valid_points": n,
                     "top1_classifier_only": c.correct_clf() / n,
                     "top1_hmm_default_params": c.correct_hmm(default) / n,
                     "top1_hmm_tuned_params": c.correct_hmm(best) / n,
                     "top1_hmm_distance_only": c.correct_hmm(best, best_sigma) / n})
    hd = pd.DataFrame(hold)
    hd.to_csv(RESULTS / "hmm_holdout_train_trips.csv", index=False)
    w = hd["n_valid_points"]
    print("hold-out (pooled):", {k: round(float(np.average(hd[k], weights=w)), 4) for k in hd.columns if k.startswith("top1")})

    out = {"params": best.to_dict(), "tuned_on": {"train_trips (out-of-fold scores)": tune_ids,
                                                    "student_test_trips": sorted(test_all["trip_id"].unique().tolist())},
           "held_out_train_trips": hold_ids, "objective": "pooled pointwise Top-1 on label-valid points",
           "tuning_top1_classifier_only": clf_top1, "tuning_top1_hmm_tuned": best_score,
           "evaluator_trips_used": False}
    (REPO_ROOT / "artifacts" / "hmm_config.json").write_text(json.dumps(out, indent=1))
    print("saved artifacts/hmm_config.json:", best.to_dict())


if __name__ == "__main__":
    main()
