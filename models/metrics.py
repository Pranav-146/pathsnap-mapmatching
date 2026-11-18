"""Evaluation metrics shared by every model (P2).

Definitions
-----------
Top-1 (observation level)
    For each (trip_id, obs_idx) the highest-scoring candidate is the prediction; it is
    correct if that candidate is the reference arc (label == 1). If `n_points` is
    given, it is the denominator, so points whose reference arc was never a candidate
    (a candidate miss) count as wrong. Otherwise only points with candidates count.
Macro F1
    Computed on the BINARY candidate labels (score >= threshold -> 1), averaged over
    the two classes (0 and 1).
Log loss
    Binary cross-entropy of the candidate score vs the candidate label.
Candidate miss rate
    Share of label-valid points whose reference arc is not in the candidate set.
Route continuity
    Percentage of consecutive predicted arcs that are the same arc or connected
    (to_node of one == from_node of the next).
Training and inference time
    Measured by the caller (see `timed`).
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

KEYS = ["trip_id", "obs_idx"]


def predicted_arcs(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Highest-scoring candidate per (trip_id, obs_idx), sorted by trip then observation."""
    idx = df.groupby(KEYS, sort=True)[score_col].idxmax()
    return df.loc[idx.to_numpy()].sort_values(KEYS)


def top1_accuracy(df: pd.DataFrame, score_col: str = "score", n_points: int | None = None) -> float:
    best = predicted_arcs(df, score_col)
    correct = int((best["label"] == 1).sum())
    denom = n_points if n_points is not None else len(best)
    return correct / denom if denom else float("nan")


def macro_f1(y_true, score, threshold: float = 0.5) -> float:
    return float(f1_score(y_true, (np.asarray(score) >= threshold).astype(int), average="macro", zero_division=0))


def binary_log_loss(y_true, score, eps: float = 1e-15) -> float:
    return float(log_loss(y_true, np.clip(np.asarray(score, dtype=float), eps, 1 - eps), labels=[0, 1]))


def candidate_miss_rate(points: pd.DataFrame, cands: pd.DataFrame) -> float:
    """Share of label-valid points with no positive candidate row."""
    valid = points[points["label_valid"] == 1][KEYS]
    hit = cands.loc[cands["label"] == 1, KEYS].drop_duplicates()
    merged = valid.merge(hit.assign(_hit=1), on=KEYS, how="left")
    return float(merged["_hit"].isna().mean())


def route_continuity(arc_ids, from_node, to_node) -> float:
    """% of consecutive predicted arcs that are identical or connected end-to-start."""
    a = np.asarray(arc_ids)
    if len(a) < 2:
        return float("nan")
    prev, nxt = a[:-1], a[1:]
    ok = (prev == nxt) | (np.asarray(to_node)[prev] == np.asarray(from_node)[nxt])
    return float(100.0 * ok.mean())


def route_continuity_by_trip(pred: pd.DataFrame, arcs_by_trip: dict) -> float:
    """Continuity pooled over trips. pred: one row per point (trip_id, obs_idx, arc_id)."""
    ok = tot = 0
    for tid, g in pred.sort_values(KEYS).groupby("trip_id"):
        arcs = arcs_by_trip[tid]
        a = g["arc_id"].to_numpy()
        if len(a) < 2:
            continue
        fn, tn = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
        ok += int(((a[:-1] == a[1:]) | (tn[a[:-1]] == fn[a[1:]])).sum())
        tot += len(a) - 1
    return float(100.0 * ok / tot) if tot else float("nan")


def evaluate_scores(cands: pd.DataFrame, points: pd.DataFrame | None = None,
                    arcs_by_trip: dict | None = None, score_col: str = "score") -> dict:
    """Top-1, macro F1, log loss, miss rate and continuity for a scored candidate table.

    Only label-valid points are used (points the .route does not cover have no reference arc).
    """
    c = cands[cands["label_valid"] == 1]
    n_points = int((points["label_valid"] == 1).sum()) if points is not None else None
    out = {
        "top1": top1_accuracy(c, score_col, n_points),
        "macro_f1": macro_f1(c["label"], c[score_col]),
        "log_loss": binary_log_loss(c["label"], c[score_col]),
    }
    if points is not None:
        out["miss_rate"] = candidate_miss_rate(points, c)
    if arcs_by_trip is not None:
        out["route_continuity_pct"] = route_continuity_by_trip(predicted_arcs(c, score_col), arcs_by_trip)
    return out


@contextmanager
def timed(store: dict, key: str):
    """with timed(d, 'train_s'): ...  -> d['train_s'] = seconds."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = time.perf_counter() - t0


if __name__ == "__main__":  # tiny self-check on dummy data
    d = pd.DataFrame({"trip_id": "a", "obs_idx": [0, 0, 1, 1, 2, 2], "arc_id": [5, 6, 6, 7, 7, 9],
                      "label": [1, 0, 1, 0, 0, 0], "label_valid": 1,
                      "score": [.9, .1, .8, .3, .6, .2]})
    pts = pd.DataFrame({"trip_id": "a", "obs_idx": [0, 1, 2], "label_valid": 1})
    print(evaluate_scores(d, pts, {"a": pd.DataFrame({"from_node": [0] * 10, "to_node": [0] * 10})}))
