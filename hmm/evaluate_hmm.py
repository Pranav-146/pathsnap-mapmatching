"""Classifier-only vs classifier + HMM on the 10 student_test trips (P3).

    python hmm/evaluate_hmm.py

Runs the real pipeline (hmm.pipeline.run_trip) with the tuned decoder parameters
(artifacts/hmm_config.json) and compares, per trip and pooled:
  * pointwise Top-1 against the reference arcs
  * route continuity (% of consecutive predicted arcs that are the same arc or connected)
  * sequence-level: precision / recall / F1 of the predicted arc SET against the .route arcs
Two more variants are included to show where the gain comes from: the nearest-arc baseline and
the HMM with distance-only emission. Writes results/hmm_comparison.csv, results/hmm_notes.md and a
Folium map per trip in results/maps/ (GPS, network, reference, classifier-only, HMM).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.load import DATA_ROOT, REPO_ROOT, list_trips  # noqa: E402
from hmm.pipeline import compute_metrics, load_hmm_params, per_point_arcs, run_trip, warm_up  # noqa: E402
from hmm.viterbi import classifier_only, emission_distance_only, viterbi  # noqa: E402
from visualisations.folium_maps import MAPS_DIR, make_map  # noqa: E402


def main():
    params = warm_up()
    sigma = params.sigma
    rows = []
    for p in list_trips(DATA_ROOT / "student_test"):
        r = run_trip(p, params, keep_internal=True)
        it = r["internals"]
        n, prep, rg, trip, geom = r["n_points"], it["prep"], it["rg"], r["trip"], it["geom"]
        # extra variants (distance-only emission)
        em_d = emission_distance_only(prep.c, sigma)
        nearest = per_point_arcs(classifier_only(prep.c, em_d), n)
        dist_hmm = viterbi(None, em_d, rg, trip["track"]["t"].to_numpy(), params, prepared=prep)["arc_id"].to_numpy()
        m_near = compute_metrics(trip, geom, nearest, dist_hmm)      # 'classifier' slot = nearest, 'hmm' slot = distance-only HMM
        m = r["metrics"]
        row = {
            "trip_id": r["trip_id"], "n_points": n, "n_label_valid": m["n_label_valid"],
            "top1_nearest_arc": m_near["top1_classifier"], "top1_hmm_distance_only": m_near["top1_hmm"],
            "top1_classifier_only": m["top1_classifier"], "top1_classifier_hmm": m["top1_hmm"],
            "continuity_classifier_only_pct": m["continuity_classifier_pct"], "continuity_classifier_hmm_pct": m["continuity_hmm_pct"],
            "arc_precision_classifier_only": m["arc_precision_classifier"], "arc_precision_classifier_hmm": m["arc_precision_hmm"],
            "arc_recall_classifier_only": m["arc_recall_classifier"], "arc_recall_classifier_hmm": m["arc_recall_hmm"],
            "arc_f1_classifier_only": m["arc_f1_classifier"], "arc_f1_classifier_hmm": m["arc_f1_hmm"],
            "hmm_segments": r["n_hmm_segments"], "inference_total_s": r["timing"]["inference_total_s"],
        }
        row["hmm_gain_top1"] = row["top1_classifier_hmm"] - row["top1_classifier_only"]
        row["hmm_helped"] = bool(row["hmm_gain_top1"] > 0)
        rows.append(row)
        make_map(trip, clf_arcs=r["clf_arcs"], hmm_arcs=r["hmm_arcs"], out_path=MAPS_DIR / f"trip_{r['trip_id']}_hmm.html")
        print(f"{r['trip_id']}: classifier {row['top1_classifier_only']:.4f} -> +HMM {row['top1_classifier_hmm']:.4f}  "
              f"(continuity {row['continuity_classifier_only_pct']:.1f} -> {row['continuity_classifier_hmm_pct']:.1f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results" / "hmm_comparison.csv", index=False)

    w = df["n_label_valid"]
    pooled = {c: float(np.average(df[c], weights=w)) for c in
              ("top1_nearest_arc", "top1_hmm_distance_only", "top1_classifier_only", "top1_classifier_hmm")}
    worse = df[df["hmm_gain_top1"] <= 0]
    lines = ["# HMM vs classifier-only on student_test (auto-generated from results/hmm_comparison.csv)\n",
             f"Decoder parameters: `{json.dumps(params.to_dict())}`\n",
             "Pooled pointwise Top-1 (weighted by label-valid points):\n"]
    lines += [f"* {k}: {v:.4f}" for k, v in pooled.items()]
    lines.append(f"\nHMM improved Top-1 on {int(df.hmm_helped.sum())} of {len(df)} trips.")
    if len(worse):
        lines.append("\nTrips where the HMM did not help (gain <= 0):\n")
        lines += [f"* {r.trip_id}: {r.top1_classifier_only:.4f} -> {r.top1_classifier_hmm:.4f} ({r.hmm_gain_top1:+.4f}), "
                  f"{r.n_points} points, {r.hmm_segments} decoded segment(s)" for r in worse.itertuples()]
    (REPO_ROOT / "results" / "hmm_notes.md").write_text("\n".join(lines) + "\n")
    print("\npooled:", {k: round(v, 4) for k, v in pooled.items()})
    print("HMM did not help on:", worse["trip_id"].tolist())


if __name__ == "__main__":
    main()
