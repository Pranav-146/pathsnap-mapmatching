"""HMM / Viterbi map matching over per-point candidate scores (P3).

Hidden states   the candidate arcs of each GPS point.
Emission        classifier score normalised across the point's candidates, in log space,
                times `emission_weight` (a temperature: >1 trusts the classifier more than the
                road geometry, <1 less). `emission_distance_only` is the stand-in used before a
                model exists and as an ablation: exp(-d^2 / 2 sigma^2), also normalised per point.
Transition      exp(-| straight-line distance - road-network distance | / beta) between the
                projected points of consecutive observations, plus a log-space bonus when the
                arc stays the same or moves to an adjacent arc.
Robustness      * disconnected arc pairs get a large finite penalty (never -inf), so decoding
                  cannot break;
                * points with no candidates are skipped and carry the previous arc forward;
                * a time gap over T seconds splits the trip, each piece is decoded on its own;
                * for gaps of a few seconds beta is widened in proportion to the gap.
The recurrence is written in log space with back-pointers.

`prepare()` computes the parameter-independent part (road distances, adjacency) once per trip so
that decoder parameters can be tuned cheaply; `viterbi()` accepts the prepared object.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmm.graph import RoadGraph  # noqa: E402


@dataclass
class HMMParams:
    sigma: float = 10.0           # metres, distance-only emission std (ablation / stand-in only)
    beta: float = 20.0            # metres, transition scale
    gap_T: float = 30.0           # seconds: split the trip when two usable fixes are further apart
    same_bonus: float = 1.0       # log-space bonus for staying on the same arc
    adjacent_bonus: float = 0.5   # log-space bonus for moving to a directly connected arc
    disconnected_cap: float = 500.0   # metres of mismatch charged for unreachable arc pairs
    straight_mode: str = "projected"  # 'projected' (plan) = between candidate projections, 'gps' = raw fixes
    emission_weight: float = 1.0      # temperature on the classifier log-probabilities

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Prepared:
    """Parameter-independent decoding inputs for one trip."""
    c: pd.DataFrame          # candidates sorted by (obs_idx, arc_id)
    obs_ids: np.ndarray      # observations that have candidates
    bounds: np.ndarray       # row ranges of each observation in c
    steps: list              # transition ingredients between consecutive observations with candidates
    t: np.ndarray            # timestamps of every GPS point
    n_points: int


def _log_normalise(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract per-group logsumexp so that exp(x) sums to 1 within every group."""
    s = pd.Series(x)
    m = s.groupby(groups).transform("max")
    lse = m + np.log(np.exp(s - m).groupby(groups).transform("sum"))
    return (s - lse).to_numpy()


def emission_distance_only(cands: pd.DataFrame, sigma: float) -> np.ndarray:
    d = cands["dist"].to_numpy(dtype=float)
    return _log_normalise(-(d ** 2) / (2.0 * sigma ** 2), cands["obs_idx"].to_numpy())


def emission_from_scores(cands: pd.DataFrame, score_col: str = "score", eps: float = 1e-6) -> np.ndarray:
    s = np.clip(cands[score_col].to_numpy(dtype=float), eps, None)
    return _log_normalise(np.log(s), cands["obs_idx"].to_numpy())


def classifier_only(cands: pd.DataFrame, emission: np.ndarray) -> pd.DataFrame:
    """Highest-emission candidate per point (no HMM). One row per point that has candidates."""
    df = cands[["obs_idx", "arc_id"]].assign(_e=emission)
    best = df.loc[df.groupby("obs_idx")["_e"].idxmax()]
    return best[["obs_idx", "arc_id"]].reset_index(drop=True)


def prepare(cands: pd.DataFrame, rg: RoadGraph, t: np.ndarray, track_xy: np.ndarray, n_points: int | None = None) -> Prepared:
    """Sort candidates and precompute road distances / adjacency between consecutive observations."""
    n_points = len(t) if n_points is None else n_points
    c = cands.sort_values(["obs_idx", "arc_id"], kind="mergesort").reset_index(drop=True)
    obs_ids, starts = np.unique(c["obs_idx"].to_numpy(), return_index=True)
    bounds = np.append(starts, len(c))
    arc = c["arc_id"].to_numpy()
    frac, qx, qy = c["frac"].to_numpy(), c["qx"].to_numpy(), c["qy"].to_numpy()
    steps = [None]
    for k in range(1, len(obs_ids)):
        a0, a1 = bounds[k - 1], bounds[k]
        b1 = bounds[k + 1]
        arcs_a, arcs_b = arc[a0:a1], arc[a1:b1]
        route = rg.route_dist_matrix(arcs_a, frac[a0:a1], arcs_b, frac[a1:b1])
        proj = np.hypot(qx[a1:b1][None, :] - qx[a0:a1][:, None], qy[a1:b1][None, :] - qy[a0:a1][:, None])
        same = (arcs_a[:, None] == arcs_b[None, :]).astype(float)
        adj = rg.adjacency(arcs_a, arcs_b)
        gps = float(np.hypot(*(track_xy[obs_ids[k]] - track_xy[obs_ids[k - 1]])))
        steps.append({"route": route, "proj": proj, "gps": gps, "same": same, "adj": adj,
                      "dt": float(t[obs_ids[k]] - t[obs_ids[k - 1]])})
    return Prepared(c, obs_ids, bounds, steps, np.asarray(t), n_points)


def _transition_logp(step: dict, p: HMMParams) -> np.ndarray:
    route = step["route"]
    straight = step["proj"] if p.straight_mode == "projected" else step["gps"]
    diff = np.abs(straight - route)
    diff = np.where(np.isfinite(diff), np.minimum(diff, p.disconnected_cap), p.disconnected_cap)
    beta = p.beta * max(1.0, step["dt"])                 # wider tolerance over a longer gap
    return -diff / beta + p.same_bonus * step["same"] + p.adjacent_bonus * (step["adj"] - step["same"])


def viterbi(cands: pd.DataFrame | None, emission: np.ndarray, rg: RoadGraph | None, t: np.ndarray,
            params: HMMParams | None = None, track_xy: np.ndarray | None = None,
            n_points: int | None = None, prepared: Prepared | None = None) -> pd.DataFrame:
    """Decode one trip.

    cands    candidate rows of ONE trip (needs obs_idx, arc_id, frac, qx, qy); ignored if `prepared`
    emission log emission probability per candidate row, in the row order of `cands`
             (for `prepared`: in the row order of prepared.c, i.e. sorted by obs_idx, arc_id)
    t        GPS timestamps of every point (index = obs_idx)
    Returns one row per GPS point: obs_idx, arc_id, segment, carried (no candidate: previous arc reused).
    """
    p = params or HMMParams()
    if prepared is None:
        prepared = prepare(cands.assign(_e=emission), rg, t, track_xy if track_xy is not None else np.zeros((len(t), 2)), n_points)
        e_all = prepared.c["_e"].to_numpy(dtype=float)
    else:
        e_all = np.asarray(emission, dtype=float)
    e_all = e_all * p.emission_weight
    c, obs_ids, bounds, steps = prepared.c, prepared.obs_ids, prepared.bounds, prepared.steps
    n_points = prepared.n_points
    arc = c["arc_id"].to_numpy()

    chosen = np.full(n_points, -1, dtype=np.int64)
    segment = np.zeros(n_points, dtype=np.int64)
    seg_id, i, n_obs = 0, 0, len(obs_ids)
    while i < n_obs:
        j = i + 1                                        # a segment ends at a time gap above gap_T
        while j < n_obs and t[obs_ids[j]] - t[obs_ids[j - 1]] <= p.gap_T:
            j += 1
        lp = e_all[bounds[i]:bounds[i + 1]].copy()
        backs = []
        for k in range(i + 1, j):
            tot = lp[:, None] + _transition_logp(steps[k], p)
            back = tot.argmax(axis=0)
            lp = tot[back, np.arange(tot.shape[1])] + e_all[bounds[k]:bounds[k + 1]]
            backs.append(back)
        s = int(lp.argmax())
        for k in range(j - 1, i - 1, -1):
            chosen[obs_ids[k]] = arc[bounds[k]:bounds[k + 1]][s]
            segment[obs_ids[k]] = seg_id
            if k > i:
                s = int(backs[k - i - 1][s])
        seg_id += 1
        i = j

    out = pd.DataFrame({"obs_idx": np.arange(n_points), "arc_id": chosen, "segment": segment})
    out["carried"] = out["arc_id"] < 0
    out["arc_id"] = out["arc_id"].where(~out["carried"]).ffill().bfill()
    out["arc_id"] = out["arc_id"].fillna(-1).astype(np.int64)
    out["segment"] = out["segment"].where(~out["carried"]).ffill().bfill().fillna(0).astype(np.int64)
    return out


def top1_vs_labels(pred_arc: np.ndarray, point_labels: pd.DataFrame) -> float:
    """Pointwise Top-1 over label-valid points. pred_arc is indexed by obs_idx (-1 = no prediction)."""
    v = point_labels[point_labels["label_valid"] == 1]
    return float((pred_arc[v["obs_idx"].to_numpy()] == v["label_arc"].to_numpy()).mean())
