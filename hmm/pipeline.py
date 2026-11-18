"""End-to-end inference pipeline (P3), called by the API (P4).

    run_trip(trip_dir) = load -> candidates + features -> classifier score -> Viterbi

Returns the classifier-only prediction AND the HMM prediction for every GPS point. Ground-truth
arcs (.route) are used only inside `compute_metrics`, on the server, to report accuracy; they never
enter candidate generation, the classifier, the decoder or the returned payload.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.labels import LABEL_MAX_M, derive_labels  # noqa: E402
from data_processing.load import REPO_ROOT, load_trip  # noqa: E402
from features.build_features import candidates_for_trip  # noqa: E402
from hmm.graph import RoadGraph  # noqa: E402
from hmm.viterbi import (HMMParams, classifier_only, emission_from_scores, prepare, viterbi)  # noqa: E402
from models.metrics import route_continuity  # noqa: E402
from models.predict import load_artifacts, predict_scores  # noqa: E402

HMM_CONFIG = REPO_ROOT / "artifacts" / "hmm_config.json"


def load_hmm_params(path: Path = HMM_CONFIG) -> HMMParams:
    """Decoder parameters tuned on train + student_test data (artifacts/hmm_config.json)."""
    if path.exists():
        return HMMParams(**json.loads(path.read_text())["params"])
    return HMMParams()


def per_point_arcs(df: pd.DataFrame, n_points: int) -> np.ndarray:
    arr = np.full(n_points, -1, dtype=np.int64)
    arr[df["obs_idx"].to_numpy()] = df["arc_id"].to_numpy()
    return arr


def connect_route(rg: RoadGraph, point_arcs: np.ndarray, nodes: pd.DataFrame) -> list[list[list[float]]]:
    """Turn a per-point arc sequence into connected polylines [[lat, lon], ...].

    Consecutive distinct arcs that do not touch are joined by the shortest road path; if no path
    exists the polyline is split there instead of drawing a false straight jump.
    """
    g = rg.geom
    lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    a = point_arcs[point_arcs >= 0]
    if len(a) == 0:
        return []
    a = a[np.concatenate([[True], a[1:] != a[:-1]])]
    segments, cur = [], []

    def xy(n):
        return [float(lat[n]), float(lon[n])]

    for k, arc in enumerate(a):
        u, v = int(g.from_node[arc]), int(g.to_node[arc])
        if not cur:
            cur = [xy(u)]
        else:
            last = int(g.to_node[a[k - 1]])
            if last != u:
                path = rg.path_nodes(last, u)
                if path is None:
                    segments.append(cur)
                    cur = [xy(u)]
                else:
                    cur.extend(xy(n) for n in path[1:])
        cur.append(xy(v))
    segments.append(cur)
    return segments


def compute_metrics(trip: dict, geom, clf_arcs: np.ndarray, hmm_arcs: np.ndarray) -> dict:
    """Server-side accuracy against the reference route. Returns plain numbers only."""
    route = trip["route"]
    if len(route) == 0:
        return {}
    label_arc, _, label_dist = derive_labels(geom.track_xy, route, geom.arc_a, geom.arc_b)
    valid = label_dist <= LABEL_MAX_M
    ref_set = set(int(x) for x in np.unique(route))
    out = {"n_points": int(len(label_arc)), "n_label_valid": int(valid.sum())}
    for name, arcs in (("classifier", clf_arcs), ("hmm", hmm_arcs)):
        ok = arcs[valid] == label_arc[valid]
        seq = arcs[arcs >= 0]
        pred_set = set(int(x) for x in np.unique(seq))
        inter = len(pred_set & ref_set)
        prec = inter / len(pred_set) if pred_set else 0.0
        rec = inter / len(ref_set) if ref_set else 0.0
        out[f"top1_{name}"] = float(ok.mean()) if valid.any() else None
        out[f"continuity_{name}_pct"] = route_continuity(seq, geom.from_node, geom.to_node)
        out[f"arc_precision_{name}"] = prec
        out[f"arc_recall_{name}"] = rec
        out[f"arc_f1_{name}"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return out


def run_trip(trip_dir, params: HMMParams | None = None, with_metrics: bool = True,
             keep_internal: bool = False) -> dict:
    """Run the whole pipeline on one trip folder.

    Returned keys: trip_id, n_points, gps (lat/lon list), clf_arcs, hmm_arcs (per point, index = obs_idx),
    route_segments (connected HMM route as polylines), clf_route_segments, metrics, timing, trip.
    `trip` (the loaded dict) is for make_map in-process and must be dropped before sending anything
    to a client. `keep_internal=True` adds `internals` (scored candidates, road graph, prepared
    decoder inputs, geometry) for evaluation scripts.
    """
    params = params or load_hmm_params()
    trip = load_trip(trip_dir)
    n = len(trip["track"])
    t0 = time.perf_counter()
    cands, geom = candidates_for_trip(trip["track"], trip["nodes"], trip["arcs"],
                                      trip_id=trip["trip_id"], return_geometry=True)
    t1 = time.perf_counter()
    scored = predict_scores(cands)
    t2 = time.perf_counter()
    emission = emission_from_scores(scored)
    clf_arcs = per_point_arcs(classifier_only(scored, emission), n)
    rg = RoadGraph(geom)
    prep = prepare(scored, rg, trip["track"]["t"].to_numpy(), geom.track_xy, n)
    # `prepare` sorts rows by (obs_idx, arc_id); emission must follow the same order
    em_sorted = emission_from_scores(prep.c)
    hmm_df = viterbi(None, em_sorted, rg, trip["track"]["t"].to_numpy(), params, prepared=prep)
    hmm_arcs = hmm_df["arc_id"].to_numpy()
    t3 = time.perf_counter()
    routes = connect_route(rg, hmm_arcs, trip["nodes"])
    clf_routes = connect_route(rg, clf_arcs, trip["nodes"])
    t4 = time.perf_counter()

    res = {
        "trip_id": trip["trip_id"], "n_points": n,
        "gps": trip["track"][["lat", "lon"]].to_numpy().tolist(),
        "clf_arcs": clf_arcs, "hmm_arcs": hmm_arcs,
        "route_segments": routes, "clf_route_segments": clf_routes,
        "n_hmm_segments": int(hmm_df["segment"].nunique()),
        "timing": {"candidates_s": t1 - t0, "scoring_s": t2 - t1, "hmm_s": t3 - t2, "route_join_s": t4 - t3,
                   "inference_total_s": t4 - t0},
        "trip": trip,
    }
    res["metrics"] = compute_metrics(trip, geom, clf_arcs, hmm_arcs) if with_metrics else {}
    if keep_internal:
        res["internals"] = {"scored": scored, "rg": rg, "prep": prep, "geom": geom}
    return res


def warm_up():
    """Load the model once (the API calls this at startup)."""
    load_artifacts()
    return load_hmm_params()
