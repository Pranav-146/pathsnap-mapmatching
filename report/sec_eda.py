"""Report section 2.2: dataset understanding and EDA (P1). All numbers come from results/."""
from __future__ import annotations

import numpy as np
import pandas as pd

from report.common import (BODY, PAGE_W, RESULTS, H1, H2, H3, P, SMALL, bullets, f, img, pb, read, sp, table)


def build(story: list):
    inv = read("eda/invalid_checks.csv")
    cc = read("eda/connectivity.csv")
    ts = read("eda/trip_summary.csv")
    miss = read("candidate_miss_rate.csv")
    abl = read("feature_v2_ablation.csv")
    pl_tr = pd.read_parquet(RESULTS / "point_labels_train.parquet")
    pl_te = pd.read_parquet(RESULTS / "point_labels_studtest.parquet")

    story += [P("2.2 Dataset understanding and exploratory data analysis (P1)", H1)]
    story.append(P("Every analysis below states the partition it used. Only the 80 <b>train</b> trips and the 10 <b>student_test</b> "
                   "trips were looked at; the 10 evaluator trips were never opened, used for fitting, or used for any choice.", SMALL))

    # ---- inventory
    story.append(P("Dataset inventory", H2))
    rows = [["Partition", "Trips", "GPS points (total)", "Points per trip (min / median / max)", "Nodes per trip (median)", "Arcs per trip (median)", "Route arcs per trip (median)"]]
    for part, g in ts.groupby("partition"):
        rows.append([part, str(len(g)), f"{int(g.n_points.sum()):,}", f"{int(g.n_points.min()):,} / {int(g.n_points.median()):,} / {int(g.n_points.max()):,}",
                     f"{int(g.n_nodes.median()):,}", f"{int(g.n_arcs.median()):,}", f"{int(g.n_route_arcs.median()):,}"])
    story.append(table(rows, [2.2 * 28.3, 1.3 * 28.3, 2.6 * 28.3, 4.6 * 28.3, 2.6 * 28.3, 2.6 * 28.3, 2.7 * 28.3]))
    story.append(P("Each trip folder has four whitespace-separated files without headers: <b>.track</b> (lon, lat, t in s), <b>.nodes</b> (lon, lat; the row "
                   "index is the node id), <b>.arcs</b> (from_node, to_node; the row index is the arc id) and <b>.route</b> (the reference arc ids in order of travel). "
                   "<font name='Helvetica-Bold'>load_trip()</font> loads all 90 trips and checks column counts, dtypes, timestamp order and that arc and route ids are in range. "
                   "The reference route is treated as reference information, not as independently verified truth.", BODY))

    # ---- Q1 preprocessing
    story.append(P("Q1. How did you preprocess the data?", H2))
    story += bullets([
        "<b>Loading and validation</b> (partitions: train + student_test): column counts and dtypes, strictly increasing timestamps, arc endpoints inside the node range, route arcs inside the arc range. All 90 trips pass.",
        "<b>Projection to metres</b>: the plan suggested UTM zone 32N (Germany), but the trips are spread over the whole world (figure 2.3), so one UTM zone would distort most of them. "
        "Each trip is projected with its own local azimuthal-equidistant projection centred on its track (pyproj); all distances are in metres, never in degrees.",
        "<b>Reference labels</b>: no GPS point has its own label. The reference arc of a point is the nearest arc on the <font name='Courier'>.route</font> sequence, found by moving forward along the "
        "route and never going back. A greedy forward pointer mislabelled several trips (it jumps ahead onto a look-alike arc), so the labels use an exact forward-only alignment (dynamic programming over "
        "non-decreasing route positions that minimises the total capped distance). Limitations: U-turns and out-and-back routes, duplicated geometry of two-way roads, and the quality of the .route itself.",
        f"<b>Points the route does not cover</b>: if the reference arc is more than 100 m from the point it is flagged <font name='Courier'>label_valid = 0</font> and excluded from fitting and metrics: "
        f"{int((pl_tr.label_valid == 0).sum()):,} of {len(pl_tr):,} train points ({(pl_tr.label_valid == 0).mean():.1%}), almost all in two trips whose .route covers only a small part of the track (figure 2.6), and "
        f"{int((pl_te.label_valid == 0).sum())} of {len(pl_te):,} student_test points.",
        "<b>Candidate generation</b>: for each GPS point the arcs within R = 50 m, at most the k = 8 nearest (shapely STRtree on the projected arcs). Every point-arc pair is one row of the candidate table with a 0/1 label.",
        "<b>Scaling</b>: tree models use the raw features; scale-sensitive models (logistic regression, KNN, MLP) use a StandardScaler fitted on the training trips only, inside a scikit-learn pipeline.",
    ])

    # ---- Q2 patterns
    story.append(P("Q2. What patterns did you observe, and which features did you extract and why?", H2))
    g_tr, g_te = ts[ts.partition == "train"], ts[ts.partition == "student_test"]
    gap_share = lambda g: g.n_gaps_gt1s.sum() / (g.n_points.sum() - len(g))
    story += bullets([
        f"<b>Sampling</b> (train + student_test): GPS is nominally 1 Hz; {1 - gap_share(g_tr):.1%} of train steps and {1 - gap_share(g_te):.1%} of student_test steps are exactly 1 s. "
        f"The largest gap is {g_tr.max_gap_s.max():,.0f} s in train (trip {g_tr.loc[g_tr.max_gap_s.idxmax(), 'trip_id']}) and {g_te.max_gap_s.max():,.0f} s in student_test, so the decoder must split trips at large gaps (figure 2.4).",
        f"<b>Geography</b>: track centres run from longitude {ts[['lon_min']].min().iloc[0]:.0f} to {ts[['lon_max']].max().iloc[0]:.0f} and latitude {ts.lat_min.min():.0f} to {ts.lat_max.max():.0f}. Trip length varies from {int(ts.n_points.min())} to {int(ts.n_points.max()):,} points.",
        f"<b>Road networks</b>: between {int(ts.n_nodes.min()):,} and {int(ts.n_nodes.max()):,} nodes per trip. They are fragmented: the median number of weakly connected components is "
        f"{cc[cc.partition == 'train'].weak_components.median():.0f} in train and {cc[cc.partition == 'student_test'].weak_components.median():.0f} in student_test (maximum {int(cc.weak_components.max())}), with "
        f"{int(cc.isolated_nodes.sum())} isolated nodes. A route between two candidate arcs is therefore not always reachable, which the HMM handles with a finite penalty.",
        f"<b>Data quality</b> (table below): no missing, out-of-range or (0, 0) coordinates and no duplicate timestamps; the networks contain {int(inv.arcs_self_loops.sum())} self-loop arcs and {int(inv.arcs_duplicates.sum()):,} duplicate arcs "
        "(two-way roads are two arcs with the same geometry, which is why distance alone cannot tell the direction).",
        f"<b>Noise</b>: the median distance from a point to its reference arc is {pl_tr[pl_tr.label_valid == 1].label_dist.median():.1f} m (train) and {pl_te[pl_te.label_valid == 1].label_dist.median():.1f} m (student_test); the miss rate for R = 50 m, k = 8 is "
        f"{miss[(miss.partition == 'train') & (miss.radius_m == 50) & (miss.k.astype(str) == '8')].miss_rate.iloc[0]:.2%} (train) and {miss[(miss.partition == 'student_test') & (miss.radius_m == 50) & (miss.k.astype(str) == '8')].miss_rate.iloc[0]:.2%} (student_test); a 10 m radius would miss about "
        f"{miss[(miss.partition == 'train') & (miss.radius_m == 10) & (miss.k.astype(str) == '8')].miss_rate.iloc[0]:.0%}, and a radius above 50 m adds little (figure 2.5).",
    ])
    # missing-value analysis table
    s = inv.drop(columns=["partition", "trip_id"]).sum()
    mv = [["Check (summed over the 90 trips)", "Count"],
          ["NaN values in .track / .nodes", f"{int(s.track_nan)} / {int(s.nodes_nan)}"],
          ["Longitude/latitude out of range (track / nodes)", f"{int(s.track_bad_range)} / {int(s.nodes_bad_range)}"],
          ["Points at (0, 0)", f"{int(s.track_zero_zero)}"], ["Duplicate timestamps", f"{int(s.dup_timestamps)}"],
          ["Self-loop arcs / duplicate arcs", f"{int(s.arcs_self_loops)} / {int(s.arcs_duplicates):,}"],
          ["Route arc ids not in the network", f"{int(s.route_arcs_not_in_network)}"]]
    story += [P("Missing-value and validity analysis (partitions: train + student_test)", H3), table(mv, [PAGE_W * 0.7, PAGE_W * 0.3]), sp(6)]

    story.append(P("Features (one row per GPS point and candidate arc)", H3))
    feat = [["Feature", "Meaning", "Why it was chosen"],
            ["dist, dist_rank", "perpendicular distance to the arc (m); rank among the point's candidates", "the true arc is usually the nearest; the rank is relative, so it survives changes in GPS noise between trips"],
            ["heading_diff, heading_valid", "angle between GPS travel direction and arc bearing (0-180 deg); 0 when the vehicle did not move", "separates the two directions of a two-way road and perpendicular side streets"],
            ["speed, arc_len", "GPS speed (m/s); arc length (m)", "slow speed means a noisy heading or a junction; short arcs are junction fragments"],
            ["frac", "position of the projection along the arc (0-1)", "points near 0 or 1 are ambiguous between an arc and its neighbour"],
            ["connects_prev, n_cand, dt_prev", "candidate follows the previous point's nearest arc; number of candidates; time gap", "continuity before the HMM, crowded vs lone road, reliability of speed"],
            ["<b>v2:</b> dist_to_best, dist_win5", "distance minus the smallest candidate distance; mean distance of the same arc over 5 points", "look-alike arcs (parallel streets, the reverse arc) look the same for one point but not over a window"],
            ["<b>v2:</b> in_prev, in_next, connects_next", "same arc is a candidate of the previous / next point; candidate flows into the next point's nearest arc", "adds look-ahead consistency that a single point cannot see; no label is used"],
            ["<b>v2:</b> turn_angle", "GPS turning angle at the point", "junction indicator that tells the model when distance is least reliable"]]
    story += [table(feat, [PAGE_W * 0.2, PAGE_W * 0.38, PAGE_W * 0.42]), sp(6)]
    a1, a2 = abl.iloc[0], abl.iloc[1]
    story.append(P(f"<b>Feature v2 vs v1</b> (same HistGradientBoosting, 5-fold GroupKFold on train, then student_test): CV Macro F1 {a1.cv_macro_f1_mean:.4f} to {a2.cv_macro_f1_mean:.4f}, "
                   f"CV Top-1 {a1.cv_top1_mean:.4f} to {a2.cv_top1_mean:.4f}, student_test Top-1 {a1.test_top1:.4f} to {a2.test_top1:.4f}, student_test log loss {a1.test_log_loss:.4f} to {a2.test_log_loss:.4f} "
                   "(results/feature_v2_ablation.csv). Every later experiment uses the 16-feature v2 set.", BODY))

    # ---- figures
    story.append(P("Visualisations", H2))
    E = RESULTS / "eda"
    story += img(E / "gps_trajectories.png", "Figure 2.1 GPS trajectories (blue) and reference routes (green) of six trips in a local metric projection. Partitions: train (top row), student_test (bottom row).", PAGE_W * 0.97)
    story += img(E / "road_network_trip.png", "Figure 2.2 Road network (grey), reference route (green) and GPS points (blue) around student_test trip 00000052.", PAGE_W * 0.62)
    story += img(E / "world_extent.png", "Figure 2.3 Trip locations, partitions train and student_test: the dataset covers the whole world, so one UTM zone does not fit.", PAGE_W * 0.85)
    story += img(E / "time_gap_hist.png", "Figure 2.4 Time gap between consecutive fixes (log counts); the red line marks gaps above 1 s. Partitions: train, student_test.", PAGE_W * 0.9)
    story += img(E / "miss_rate_vs_radius.png", "Figure 2.5 Candidate miss rate against search radius R for k = 8 and no cap on k (label-valid points; partitions train and student_test).", PAGE_W * 0.62)
    story += img(E / "label_coverage.png", "Figure 2.6 Share of each train trip's points that the .route covers; the two red trips are excluded from most of the training signal.", PAGE_W * 0.85)
    story += img(E / "dist_to_route.png", "Figure 2.7 Distance from each GPS point to its reference arc (label-valid points). Partitions: train (left), student_test (right).", PAGE_W * 0.9)
    story += img(E / "points_per_trip.png", "Figure 2.8 GPS points per trip (log axis). Partitions: train, student_test.", PAGE_W * 0.7)
    story += img(E / "feature_distributions.png", "Figure 2.9 Feature distributions for the reference arc and the other candidates (train partition, 200,000 random candidate rows).", PAGE_W)
    shot = RESULTS / "screenshots" / "folium_eda_00000052.png"
    story += img(shot, "Figure 2.10 Folium map (screenshot of results/maps/eda_student_test_00000052.html, partition student_test): GPS points (blue), road network (grey), reference route (green), with legend and layer control. "
                       "The interactive HTML is in the repository.", PAGE_W * 0.95)
    story.append(pb())
