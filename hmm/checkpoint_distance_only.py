"""11:30 checkpoint (P3): road graph + Viterbi end to end with DISTANCE-ONLY emission.

For each student_test trip: nearest-arc baseline ("classifier-only" stand-in) vs HMM,
measured by pointwise Top-1 against the reference arcs. Writes
results/hmm_checkpoint_distance_only.csv and a Folium map for a few trips.
Replaced by hmm/pipeline.py + hmm/evaluate_hmm.py once the trained model is wired in.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.load import DATA_ROOT, REPO_ROOT, list_trips, load_trip  # noqa: E402
from features.build_features import candidates_for_trip  # noqa: E402
from hmm.graph import RoadGraph  # noqa: E402
from hmm.viterbi import (HMMParams, classifier_only, emission_distance_only, top1_vs_labels,  # noqa: E402
                         viterbi)
from visualisations.folium_maps import MAPS_DIR, make_map  # noqa: E402

MAP_TRIPS = {"00000052", "00000091"}


def main():
    pts_all = pd.read_parquet(REPO_ROOT / "results" / "point_labels_studtest.parquet")
    params = HMMParams()
    rows = []
    for p in list_trips(DATA_ROOT / "student_test"):
        trip = load_trip(p)
        t0 = time.perf_counter()
        cands, geom = candidates_for_trip(trip["track"], trip["nodes"], trip["arcs"],
                                          trip_id=trip["trip_id"], return_geometry=True)
        rg = RoadGraph(geom)
        emis = emission_distance_only(cands, params.sigma)
        base = classifier_only(cands, emis)
        n = len(trip["track"])
        base_arc = np.full(n, -1)
        base_arc[base["obs_idx"].to_numpy()] = base["arc_id"].to_numpy()
        hmm_df = viterbi(cands, emis, rg, trip["track"]["t"].to_numpy(), params, geom.track_xy)
        hmm_arc = hmm_df["arc_id"].to_numpy()
        secs = time.perf_counter() - t0

        pl = pts_all[pts_all.trip_id == trip["trip_id"]]
        b, h = top1_vs_labels(base_arc, pl), top1_vs_labels(hmm_arc, pl)
        rows.append({"trip_id": trip["trip_id"], "n_points": n, "n_segments": int(hmm_df.segment.nunique()),
                     "top1_nearest_arc": b, "top1_hmm": h, "hmm_gain": h - b,
                     "carried_points": int(hmm_df.carried.sum()), "seconds": secs})
        print(f"{trip['trip_id']}: nearest-arc {b:.4f} -> HMM {h:.4f}  ({secs:.1f}s)", flush=True)
        if trip["trip_id"] in MAP_TRIPS:
            make_map(trip, clf_arcs=base_arc, hmm_arcs=hmm_arc,
                     out_path=MAPS_DIR / f"hmm_distance_only_{trip['trip_id']}.html")
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results" / "hmm_checkpoint_distance_only.csv", index=False)
    w = df.n_points
    print(f"pooled over {int(w.sum())} points: nearest-arc {np.average(df.top1_nearest_arc, weights=w):.4f}, "
          f"HMM {np.average(df.top1_hmm, weights=w):.4f}")


if __name__ == "__main__":
    main()
