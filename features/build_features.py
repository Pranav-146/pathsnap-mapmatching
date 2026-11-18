"""Candidate feature table (P1).

One row per (GPS point, candidate arc). Run as a script to write
    results/candidates_train.parquet
    results/candidates_studtest.parquet
    results/point_labels_<partition>.parquet   (reference arc for EVERY point)
    results/candidate_miss_rate.csv

Why each feature (see also results/feature_notes.md)
----------------------------------------------------
dist          Perpendicular distance point->arc in metres. The strongest single cue:
              the true arc is usually the closest.
dist_rank     Rank of that distance among the point's candidates (0 = nearest). Robust
              to GPS noise level changing between trips (relative, not absolute).
heading_diff  Angle (0..180 deg) between the GPS direction of travel and the arc bearing.
              Separates the two directions of a two-way road and perpendicular side streets.
heading_valid 0 when the vehicle barely moved so the GPS heading is meaningless.
speed         GPS speed (m/s) from consecutive points. Slow speed means heading is
              noisy and the point may sit at a junction; high speed hints at highways.
arc_len       Arc length in metres. Short arcs are junction fragments, long arcs are
              straight road; it changes how much dist/frac should be trusted.
frac          Position of the projection along the arc (0-1). Points near 0/1 are
              ambiguous between this arc and its neighbour.
connects_prev 1 if the candidate is the previous point's nearest arc or starts where it
              ends. Encodes route continuity before the HMM is even applied.
n_cand        Number of candidates for the point (a crowded junction vs a lone road).
dt_prev       Time gap to the previous point (s). Large gaps make speed/continuity weak.

Feature v2 (context features, added after the v1 error analysis: wrong picks were mostly
look-alike arcs that a single point cannot tell apart, e.g. the reverse arc of a two-way road
or a parallel street; neighbouring points can)
dist_to_best  dist minus the smallest dist among the point's candidates (0 for the nearest).
in_prev       1 if the same arc is also a candidate of the previous point.
in_next       1 if the same arc is also a candidate of the next point.
connects_next 1 if the candidate flows into (or is) the nearest arc of the next point.
dist_win5     mean dist of this arc over the 5 points around it (a missing point counts as R).
turn_angle    GPS turning angle (0-180 deg) at the point; high values mark junctions.
No labels are used by any of these, so they are available at inference time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.labels import LABEL_MAX_M, derive_labels  # noqa: E402
from data_processing.load import DATA_ROOT, REPO_ROOT, list_trips, load_trip  # noqa: E402
from features.candidates import (K_DEFAULT, R_DEFAULT, R_MAX, TripGeometry,  # noqa: E402
                                 find_candidates)

FEATURE_COLS_V1 = ["dist", "dist_rank", "heading_diff", "heading_valid", "speed",
                   "arc_len", "frac", "connects_prev", "n_cand", "dt_prev"]
FEATURE_COLS_V2_NEW = ["dist_to_best", "in_prev", "in_next", "connects_next", "dist_win5", "turn_angle"]
FEATURE_COLS = FEATURE_COLS_V1 + FEATURE_COLS_V2_NEW
META_COLS = ["trip_id", "obs_idx", "t", "lon", "lat", "arc_id"]
# extra geometry carried for the HMM; NOT model inputs (qx,qy are local metres)
GEOM_COLS = ["qx", "qy"]

MIN_MOVE_M = 1.0  # below this displacement the GPS heading is treated as undefined
RESULTS = REPO_ROOT / "results"


def gps_motion(track_xy: np.ndarray, t: np.ndarray):
    """Per-point GPS heading (deg), heading validity, speed (m/s), time gap to previous point and turning angle."""
    n = len(track_xy)
    prev_i = np.maximum(np.arange(n) - 1, 0)
    next_i = np.minimum(np.arange(n) + 1, n - 1)
    d = track_xy[next_i] - track_xy[prev_i]                 # central difference
    move = np.hypot(d[:, 0], d[:, 1])
    heading = (np.degrees(np.arctan2(d[:, 0], d[:, 1])) + 360.0) % 360.0
    valid = (move >= MIN_MOVE_M).astype(np.int8)
    step = np.hypot(*(track_xy[1:] - track_xy[:-1]).T)
    dt = np.diff(t)
    speed = np.zeros(n)
    speed[1:] = step / np.maximum(dt, 1e-6)
    speed[0] = speed[1] if n > 1 else 0.0
    dt_prev = np.ones(n)
    dt_prev[1:] = dt
    # turning angle at each point: bearing(i-1 -> i) vs bearing(i -> i+1)
    step_v = np.vstack([np.zeros((1, 2)), track_xy[1:] - track_xy[:-1]])           # step arriving at i
    step_n = np.vstack([track_xy[1:] - track_xy[:-1], np.zeros((1, 2))])           # step leaving i
    b_in = (np.degrees(np.arctan2(step_v[:, 0], step_v[:, 1])) + 360.0) % 360.0
    b_out = (np.degrees(np.arctan2(step_n[:, 0], step_n[:, 1])) + 360.0) % 360.0
    ok = (np.hypot(*step_v.T) >= MIN_MOVE_M) & (np.hypot(*step_n.T) >= MIN_MOVE_M)
    turn = np.where(ok, _angle_diff(b_in, b_out), 0.0)
    return heading, valid, speed, dt_prev, turn


def _angle_diff(a, b):
    """Smallest absolute angle between two bearings, 0..180."""
    d = np.abs(a - b) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)


def candidates_for_trip(track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                        radius: float = R_DEFAULT, k: int = K_DEFAULT,
                        trip_id: str = "", return_geometry: bool = False,
                        _geom: TripGeometry | None = None, _cands: pd.DataFrame | None = None):
    """Feature table for one trip (no labels). The API reuses this function.

    `_geom` / `_cands` let the dataset builder reuse a geometry and a wide candidate
    query that it already computed; callers normally leave them out.
    """
    geom = _geom if _geom is not None else TripGeometry(track, nodes, arcs)
    if _cands is None:
        c = find_candidates(geom, radius, k)
    else:
        c = _cands[(_cands["dist"] <= radius) & (_cands["dist_rank"] < k)].reset_index(drop=True)

    t = track["t"].to_numpy()
    heading, valid, speed, dt_prev, turn = gps_motion(geom.track_xy, t)
    oi = c["obs_idx"].to_numpy()
    ai = c["arc_id"].to_numpy()

    # previous point's nearest arc -> connectivity feature
    nearest = c[c["dist_rank"] == 0].set_index("obs_idx")["arc_id"]
    prev_best = np.full(len(track), -1, dtype=np.int64)
    prev_best[nearest.index.to_numpy()] = nearest.to_numpy()
    pb = np.where(oi > 0, prev_best[np.maximum(oi - 1, 0)], -1)
    has_prev = pb >= 0
    pb_safe = np.where(has_prev, pb, 0)
    connects = has_prev & ((ai == pb) | (geom.from_node[ai] == geom.to_node[pb_safe]))
    # next point's nearest arc -> look-ahead connectivity
    nb = np.where(oi < len(track) - 1, prev_best[np.minimum(oi + 1, len(track) - 1)], -1)
    has_next = nb >= 0
    nb_safe = np.where(has_next, nb, 0)
    connects_next = has_next & ((ai == nb) | (geom.to_node[ai] == geom.from_node[nb_safe]))

    out = pd.DataFrame({
        "trip_id": trip_id,
        "obs_idx": oi,
        "t": t[oi],
        "lon": track["lon"].to_numpy()[oi],
        "lat": track["lat"].to_numpy()[oi],
        "arc_id": ai,
        "dist": c["dist"].to_numpy(),
        "dist_rank": c["dist_rank"].to_numpy(),
        "heading_diff": np.where(valid[oi] == 1, _angle_diff(heading[oi], geom.arc_bearing[ai]), 90.0),
        "heading_valid": valid[oi],
        "speed": speed[oi],
        "arc_len": geom.arc_len[ai],
        "frac": c["frac"].to_numpy(),
        "connects_prev": connects.astype(np.int8),
        "n_cand": c.groupby("obs_idx")["arc_id"].transform("size").to_numpy(),
        "dt_prev": dt_prev[oi],
        "qx": c["qx"].to_numpy(),
        "qy": c["qy"].to_numpy(),
    })
    out["dist_to_best"] = out["dist"] - out.groupby("obs_idx")["dist"].transform("min")
    out["connects_next"] = connects_next.astype(np.int8)
    out["turn_angle"] = turn[oi]
    # same arc among the candidates of neighbouring points (offsets -2..+2)
    base = out[["obs_idx", "arc_id", "dist"]]
    near = {}
    for off in (-2, -1, 1, 2):
        other = base.assign(obs_idx=base["obs_idx"] - off).rename(columns={"dist": "d"})
        near[off] = out[["obs_idx", "arc_id"]].merge(other, on=["obs_idx", "arc_id"], how="left")["d"].to_numpy()
    out["in_prev"] = (~np.isnan(near[-1])).astype(np.int8)
    out["in_next"] = (~np.isnan(near[1])).astype(np.int8)
    win = out["dist"].to_numpy() + sum(np.where(np.isnan(v), radius, v) for v in near.values())
    out["dist_win5"] = win / 5.0
    return (out, geom) if return_geometry else out


def build_trip(trip: dict, radius: float = R_DEFAULT, k: int = K_DEFAULT):
    """Labelled candidate table + per-point labels + miss-rate raw data for one trip."""
    track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    geom = TripGeometry(track, nodes, arcs)
    wide = find_candidates(geom, R_MAX, None)       # one wide query serves every R / k
    cands = candidates_for_trip(track, nodes, arcs, radius, k, trip["trip_id"], _geom=geom, _cands=wide)

    label_arc, route_pos, label_dist = derive_labels(geom.track_xy, trip["route"], geom.arc_a, geom.arc_b)
    label_valid = label_dist <= LABEL_MAX_M        # route covers this point?
    oi = cands["obs_idx"].to_numpy()
    cands["label"] = ((cands["arc_id"].to_numpy() == label_arc[oi]) & label_valid[oi]).astype(np.int8)
    cands["label_valid"] = label_valid[oi].astype(np.int8)

    points = pd.DataFrame({"trip_id": trip["trip_id"], "obs_idx": np.arange(len(track)),
                           "label_arc": label_arc, "route_pos": route_pos, "label_dist": label_dist,
                           "label_valid": label_valid.astype(np.int8)})
    # rank of the label arc inside the wide candidate set (NaN if >R_MAX away)
    lab = pd.DataFrame({"obs_idx": points["obs_idx"], "arc_id": label_arc})
    rk = lab.merge(wide[["obs_idx", "arc_id", "dist_rank"]], on=["obs_idx", "arc_id"], how="left")
    points["label_rank"] = rk["dist_rank"].to_numpy()
    return cands, points


def miss_rate_table(points: pd.DataFrame, radii=(10, 20, 30, 50, 75, 100), ks=(1, 3, 5, 8, 16, None)) -> pd.DataFrame:
    """Candidate miss rate = share of label-valid points whose reference arc is NOT a candidate."""
    n_all = len(points)
    points = points[points["label_valid"] == 1]
    rows = []
    for r in radii:
        for k in ks:
            ok = points["label_rank"].notna() & (points["label_dist"] <= r)
            if k is not None:
                ok &= points["label_rank"] < k
            rows.append({"radius_m": r, "k": "all" if k is None else k,
                         "miss_rate": float(1.0 - ok.mean()), "n_points": len(points),
                         "n_points_all": n_all})
    return pd.DataFrame(rows)


def build_partition(name: str, out_name: str, radius: float = R_DEFAULT, k: int = K_DEFAULT):
    trips = list_trips(DATA_ROOT / name)
    cand_parts, point_parts = [], []
    for i, p in enumerate(trips):
        c, pts = build_trip(load_trip(p), radius, k)
        cand_parts.append(c)
        point_parts.append(pts)
        print(f"[{name}] {i + 1}/{len(trips)} trip {p.name}: {len(pts)} pts, {len(c)} candidate rows", flush=True)
    cands = pd.concat(cand_parts, ignore_index=True)
    points = pd.concat(point_parts, ignore_index=True)
    for col in ["dist", "heading_diff", "speed", "arc_len", "frac", "dt_prev", "qx", "qy",
                "dist_to_best", "dist_win5", "turn_angle"]:
        cands[col] = cands[col].astype("float32")
    RESULTS.mkdir(exist_ok=True, parents=True)
    cands.to_parquet(RESULTS / f"candidates_{out_name}.parquet", index=False)
    points.to_parquet(RESULTS / f"point_labels_{out_name}.parquet", index=False)
    mr = miss_rate_table(points)
    mr.insert(0, "partition", name)
    return cands, points, mr


def main():
    frames = []
    for name, out in (("train", "train"), ("student_test", "studtest")):
        cands, points, mr = build_partition(name, out)
        frames.append(mr)
        m = mr[(mr.radius_m == R_DEFAULT) & (mr.k == K_DEFAULT)].miss_rate.iloc[0]
        print(f"== {name}: {len(points)} points ({int(points.label_valid.sum())} label-valid), {len(cands)} rows, "
              f"positive rate {cands.label.mean():.3f}, miss rate (R={R_DEFAULT:g}, k={K_DEFAULT}) {m:.4f}")
    pd.concat(frames).to_csv(RESULTS / "candidate_miss_rate.csv", index=False)


if __name__ == "__main__":
    main()
