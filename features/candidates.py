"""Candidate arc generation (P1).

For each GPS point we take every arc within radius R metres and keep the k
nearest. All geometry is in a per-trip local metric projection (see
data_processing.load.LocalProjector), so R is in real metres.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from data_processing.load import LocalProjector

R_DEFAULT = 50.0   # metres
K_DEFAULT = 8      # max candidates kept per point
R_MAX = 100.0      # widest radius used when measuring the miss rate


class TripGeometry:
    """Projected road network of one trip plus a spatial index over its arcs."""

    def __init__(self, track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame):
        self.proj = LocalProjector.from_track(track)
        nx_, ny_ = self.proj.project(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
        self.node_xy = np.column_stack([nx_, ny_])
        fn, tn = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
        self.from_node, self.to_node = fn, tn
        self.arc_a = self.node_xy[fn]          # (A, 2) start of every arc
        self.arc_b = self.node_xy[tn]          # (A, 2) end of every arc
        self.arc_len = np.hypot(*(self.arc_b - self.arc_a).T)
        # bearing of each arc, degrees clockwise from north (0..360)
        d = self.arc_b - self.arc_a
        self.arc_bearing = (np.degrees(np.arctan2(d[:, 0], d[:, 1])) + 360.0) % 360.0
        self.tree = STRtree(shapely.linestrings(np.stack([self.arc_a, self.arc_b], axis=1)))
        x, y = self.proj.project(track["lon"].to_numpy(), track["lat"].to_numpy())
        self.track_xy = np.column_stack([x, y])


def point_segment(px, py, a, b):
    """Closest point on segment a-b for point(s) p. Returns (dist, frac in 0..1, qx, qy)."""
    d = b - a
    l2 = (d ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(l2 > 0, ((px - a[:, 0]) * d[:, 0] + (py - a[:, 1]) * d[:, 1]) / l2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    qx, qy = a[:, 0] + t * d[:, 0], a[:, 1] + t * d[:, 1]
    return np.hypot(px - qx, py - qy), t, qx, qy


def find_candidates(geom: TripGeometry, radius: float = R_DEFAULT, k: int | None = K_DEFAULT) -> pd.DataFrame:
    """All (point, arc) pairs within `radius` metres, ranked by distance, at most k per point.

    Columns: obs_idx, arc_id, dist, frac, qx, qy, dist_rank.
    Points with no arc in range simply have no rows (candidate miss).
    """
    pts = shapely.points(geom.track_xy)
    pi, ai = geom.tree.query(pts, predicate="dwithin", distance=radius)
    px, py = geom.track_xy[pi, 0], geom.track_xy[pi, 1]
    dist, frac, qx, qy = point_segment(px, py, geom.arc_a[ai], geom.arc_b[ai])
    df = pd.DataFrame({"obs_idx": pi.astype(np.int64), "arc_id": ai.astype(np.int64),
                       "dist": dist, "frac": frac, "qx": qx, "qy": qy})
    df = df.sort_values(["obs_idx", "dist", "arc_id"], kind="mergesort").reset_index(drop=True)
    df["dist_rank"] = df.groupby("obs_idx").cumcount()
    if k is not None:
        df = df[df["dist_rank"] < k].reset_index(drop=True)
    return df
