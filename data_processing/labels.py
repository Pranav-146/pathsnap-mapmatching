"""Reference-arc labels (P1).

No GPS point comes with its own label. We derive one: the reference arc of a
point is the nearest arc on the ground-truth `.route` sequence, found by moving
FORWARD along the route and never going back.

Algorithm
---------
`derive_labels` (used for the dataset) solves the forward-only assignment exactly.
It picks route positions j_0 <= j_1 <= ... <= j_N that minimise the total
(capped) distance from each point to its assigned route arc, by dynamic
programming with a prefix-minimum. A tiny per-step jump penalty breaks ties in
favour of staying on the earliest arc.

`derive_labels_greedy` is the literal "pointer" version: it only looks WINDOW
positions ahead and takes the nearest. It is kept for comparison. On several
trips (e.g. 00000008, 00000095, 00000085) the greedy pointer jumps ahead on a
look-alike arc, then mislabels everything until the track catches up, so only
~70% of points ended up within 50 m of their label. The DP removes that failure
while keeping the same forward-only rule.

Points whose reference arc is farther than LABEL_MAX_M from the GPS point are
flagged `label_valid = False`: the `.route` simply does not cover them (e.g. the
route of 00000054 covers ~13% of its track). They are excluded from training and
metrics, and counted in the report.

Limitations (also noted in the report)
--------------------------------------
* U-turns / out-and-back routes: the route may revisit the same road. Because labels can
  never go back, once the alignment has moved past an earlier visit, points near it are
  assigned to the later visit if that is the cheaper forward alignment.
* Overlapping / duplicated geometry: two-way roads are two arcs with identical
  geometry. The route picks one direction, and we only search route arcs, so the
  label is the direction actually travelled. Distinct arcs with overlapping
  geometry that are NOT on the route can never be labels.
* GPS noise or a long gap can make the nearest route arc a few positions ahead
  or behind the true one; the per-step jump penalty keeps such skips rare.
* Labels are only as good as the .route file; the label is an arc id, so a point
  between two route arcs may land on either.
"""
from __future__ import annotations

import numpy as np

WINDOW = 25         # greedy pointer only: route positions it may look ahead
TIE_M = 0.5         # greedy pointer only: metres treated as ties
COST_CAP_M = 100.0  # DP: distance cap so off-route stretches do not dominate the cost
JUMP_PEN = 0.05     # DP: metres of cost per route position skipped (tie-breaker)
LABEL_MAX_M = 100.0 # reference arc farther than this => label_valid = False


def _point_seg_dist(px, py, ax, ay, bx, by):
    """Distance from point(s) to segment(s) a-b, all numpy arrays / scalars (metres)."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / l2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def derive_labels_greedy(track_xy: np.ndarray, route: np.ndarray, arc_a: np.ndarray, arc_b: np.ndarray,
                  window: int = WINDOW, tie_m: float = TIE_M):
    """Assign each GPS point a reference arc from the route.

    Parameters
    ----------
    track_xy : (N, 2) projected GPS points in metres
    route    : (M,) arc ids in travel order
    arc_a, arc_b : (A, 2) projected start / end coordinates of every arc (metres)

    Returns
    -------
    label_arc : (N,) arc id of the reference arc
    route_pos : (N,) index into `route` (non-decreasing)
    label_dist: (N,) distance in metres from the point to its reference arc
    """
    n, m = len(track_xy), len(route)
    if m == 0:
        raise ValueError("empty route")
    ra, rb = arc_a[route], arc_b[route]
    pos = np.zeros(n, dtype=np.int64)
    dist = np.zeros(n)
    p = 0
    for i in range(n):
        hi = min(p + window + 1, m)
        d = _point_seg_dist(track_xy[i, 0], track_xy[i, 1],
                            ra[p:hi, 0], ra[p:hi, 1], rb[p:hi, 0], rb[p:hi, 1])
        dmin = d.min()
        j = int(np.argmax(d <= dmin + tie_m))  # earliest among near-ties
        p += j
        pos[i], dist[i] = p, d[j]
    return route[pos], pos, dist


def derive_labels(track_xy: np.ndarray, route: np.ndarray, arc_a: np.ndarray, arc_b: np.ndarray,
                  cap: float = COST_CAP_M, jump_pen: float = JUMP_PEN):
    """Forward-only reference arcs via dynamic programming (see module docstring).

    Same inputs and outputs as `derive_labels_greedy`.
    """
    n, m = len(track_xy), len(route)
    if m == 0:
        raise ValueError("empty route")
    ra, rb = arc_a[route], arc_b[route]
    cost = np.empty((n, m), dtype=np.float32)
    for s in range(0, n, 2000):  # chunked point-to-route-arc distances
        p = track_xy[s:s + 2000]
        cost[s:s + 2000] = np.minimum(
            _point_seg_dist(p[:, 0:1], p[:, 1:2], ra[None, :, 0], ra[None, :, 1],
                            rb[None, :, 0], rb[None, :, 1]), cap)
    idx = np.arange(m, dtype=np.float64)
    back = np.zeros((n, m), dtype=np.int32)
    f = cost[0].astype(np.float64)
    ar = np.arange(m)
    for i in range(1, n):
        g = f - jump_pen * idx                      # f[j'] - lam*j'
        run = np.minimum.accumulate(g)
        back[i] = np.maximum.accumulate(np.where(g == run, ar, 0))
        f = cost[i] + run + jump_pen * idx
    pos = np.zeros(n, dtype=np.int64)
    pos[-1] = int(np.argmin(f))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    dist = _point_seg_dist(track_xy[:, 0], track_xy[:, 1], ra[pos, 0], ra[pos, 1], rb[pos, 0], rb[pos, 1])
    return route[pos], pos, dist
