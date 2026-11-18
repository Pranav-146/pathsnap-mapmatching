# Feature notes (P1)

One row per (GPS point, candidate arc). Candidates: arcs within R = 50 m of the point, at most the
k = 8 nearest. Distances are metres in a per-trip local projection. Model inputs are the first ten
columns below; `qx`, `qy` (projected point on the arc, local metres) are kept only for the HMM.

| Feature | What it is | Why it is there |
|---|---|---|
| `dist` | perpendicular distance point to arc (m) | the strongest single cue; the true arc is usually the closest |
| `dist_rank` | rank of `dist` among the point's candidates (0 = nearest) | relative, so robust when GPS noise differs between trips |
| `heading_diff` | angle between GPS travel direction and arc bearing, 0-180 deg | separates the two directions of a two-way road and perpendicular side streets |
| `heading_valid` | 0 when the vehicle moved < 1 m around the point | heading is meaningless when stationary (heading_diff is set to a neutral 90 deg) |
| `speed` | GPS speed (m/s) from consecutive points | slow speed = noisy heading / junction; high speed = highways |
| `arc_len` | arc length (m) | short arcs are junction fragments, long arcs are straight road |
| `frac` | position of the projection along the arc (0-1) | points near 0 or 1 are ambiguous between an arc and its neighbour |
| `connects_prev` | candidate is the previous point's nearest arc, or starts where it ends | encodes route continuity before the HMM is applied |
| `n_cand` | number of candidates for the point | crowded junction vs a lone road |
| `dt_prev` | time gap to the previous point (s) | large gaps weaken speed and continuity cues |

Extra columns: `label` (1 = the point's reference arc), `label_valid` (0 when the reference arc is more
than 100 m from the point, i.e. the `.route` does not cover that part of the track; fit and evaluate on
`label_valid == 1` only).

## Deviations from the plan, and why

* **Projection.** The plan suggested UTM 32N (Germany). The dataset is worldwide (lon -100 to 137), so each
  trip uses its own local azimuthal-equidistant projection in metres instead.
* **Labels.** The greedy forward pointer mislabelled several trips (it jumps ahead on a look-alike arc), so
  the labels use an exact forward-only alignment (dynamic programming). Same "never go back" rule.
