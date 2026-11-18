"""Extra EDA figures for the report (P1). Each title names the partition used.

    python data_processing/eda_plots.py        # writes results/eda/*.png

gps_trajectories.png      GPS tracks and reference routes of six trips (train and student_test)
road_network_trip.png     road network around one trip with the reference route and GPS points
world_extent.png          where the 90 trips are in the world (why a fixed UTM zone is wrong)
miss_rate_vs_radius.png   candidate miss rate against the search radius R
label_coverage.png        share of each train trip's points that the .route covers
feature_distributions.png distance / heading difference / speed for true vs other candidates (train)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.load import DATA_ROOT, REPO_ROOT, LocalProjector, list_trips, load_trip  # noqa: E402

OUT = REPO_ROOT / "results" / "eda"
RES = REPO_ROOT / "results"
SHOW = [("train", "00000014"), ("train", "00000008"), ("train", "00000037"),
        ("student_test", "00000052"), ("student_test", "00000009"), ("student_test", "00000091")]


def _xy(trip, proj):
    return proj.project(trip["track"]["lon"], trip["track"]["lat"])


def _route_lines(trip, proj):
    r = np.unique(trip["route"])
    arcs, nodes = trip["arcs"], trip["nodes"]
    nx_, ny_ = proj.project(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    f, t = arcs["from_node"].to_numpy()[trip["route"]], arcs["to_node"].to_numpy()[trip["route"]]
    return nx_, ny_, f, t


def gps_trajectories():
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (part, tid) in zip(axs.ravel(), SHOW):
        trip = load_trip(DATA_ROOT / part / tid)
        proj = LocalProjector.from_track(trip["track"])
        x, y = _xy(trip, proj)
        nx_, ny_, f, t = _route_lines(trip, proj)
        for a, b in zip(f, t):
            ax.plot([nx_[a], nx_[b]], [ny_[a], ny_[b]], color="green", lw=2.2, solid_capstyle="round",
                    label="reference route" if a == f[0] else None)
        ax.scatter(x, y, s=3, color="blue", alpha=.6, label="GPS points", zorder=3)
        ax.set_aspect("equal"); ax.set_title(f"{part}  trip {tid}  ({len(x):,} points)", fontsize=10)
        ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
        if ax is axs[0, 0]:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle("GPS trajectories and reference routes in a local metric projection (partitions as titled)")
    fig.tight_layout(); fig.savefig(OUT / "gps_trajectories.png", dpi=130); plt.close(fig)


def road_network_trip(part="student_test", tid="00000052"):
    trip = load_trip(DATA_ROOT / part / tid)
    proj = LocalProjector.from_track(trip["track"])
    x, y = _xy(trip, proj)
    nx_, ny_ = proj.project(trip["nodes"]["lon"].to_numpy(), trip["nodes"]["lat"].to_numpy())
    pad = 300
    a, b = trip["arcs"]["from_node"].to_numpy(), trip["arcs"]["to_node"].to_numpy()
    inview = (np.minimum(nx_[a], nx_[b]) > x.min() - pad) & (np.maximum(nx_[a], nx_[b]) < x.max() + pad) & \
             (np.minimum(ny_[a], ny_[b]) > y.min() - pad) & (np.maximum(ny_[a], ny_[b]) < y.max() + pad)
    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.plot(np.array([nx_[a[inview]], nx_[b[inview]]]), np.array([ny_[a[inview]], ny_[b[inview]]]), color="#9a9a9a", lw=.8)
    ra, rb = a[trip["route"]], b[trip["route"]]
    ax.plot(np.array([nx_[ra], nx_[rb]]), np.array([ny_[ra], ny_[rb]]), color="green", lw=3)
    ax.scatter(x, y, s=6, color="blue", zorder=3)
    ax.plot([], [], color="#9a9a9a", label=f"road network ({int(inview.sum()):,} arcs in view)")
    ax.plot([], [], color="green", lw=3, label="reference route"); ax.scatter([], [], s=10, color="blue", label="GPS points")
    ax.set_aspect("equal"); ax.legend(loc="best", fontsize=9)
    ax.set_title(f"Road network, route and GPS points - {part} trip {tid}"); ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    fig.tight_layout(); fig.savefig(OUT / "road_network_trip.png", dpi=130); plt.close(fig)


def world_extent():
    rows = pd.read_csv(OUT / "trip_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for part, c in (("train", "#4c78a8"), ("student_test", "#e07b39")):
        d = rows[rows.partition == part]
        ax.scatter((d.lon_min + d.lon_max) / 2, (d.lat_min + d.lat_max) / 2, s=26, color=c, label=f"{part} ({len(d)} trips)", alpha=.85)
    ax.axvline(8.6, color="grey", ls=":", lw=1); ax.annotate("UTM zone 32N centre (8.6 E)", (9.5, -40), fontsize=8, color="grey")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.grid(alpha=.3); ax.legend()
    ax.set_title("Trip locations (centre of each GPS track): the data covers the whole world")
    fig.tight_layout(); fig.savefig(OUT / "world_extent.png", dpi=130); plt.close(fig)


def miss_rate_vs_radius():
    m = pd.read_csv(RES / "candidate_miss_rate.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for part, c in (("train", "#4c78a8"), ("student_test", "#e07b39")):
        for k, ls in (("8", "-"), ("all", "--")):
            d = m[(m.partition == part) & (m.k.astype(str) == k)]
            ax.plot(d.radius_m, d.miss_rate * 100, ls, marker="o", color=c, label=f"{part}, k={k}")
    ax.set_xlabel("search radius R (m)"); ax.set_ylabel("candidate miss rate (%)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Candidate miss rate vs radius (label-valid points)")
    fig.tight_layout(); fig.savefig(OUT / "miss_rate_vs_radius.png", dpi=130); plt.close(fig)


def label_coverage():
    p = pd.read_parquet(RES / "point_labels_train.parquet").groupby("trip_id").label_valid.mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(range(len(p)), p.values * 100, color=["#d62728" if v < .9 else "#4c78a8" for v in p.values])
    ax.set_xlabel("train trips (sorted)"); ax.set_ylabel("% of points covered by the .route")
    for i, (tid, v) in enumerate(p.items()):
        if v < .9:
            ax.annotate(tid, (i, v * 100 + 2), fontsize=8, ha="left")
    ax.set_title("Label coverage per train trip: two trips have a .route that covers little of the track")
    fig.tight_layout(); fig.savefig(OUT / "label_coverage.png", dpi=130); plt.close(fig)


def feature_distributions():
    c = pd.read_parquet(RES / "candidates_train.parquet")
    c = c[c.label_valid == 1].sample(200_000, random_state=42)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, col, rng, lab in ((axs[0], "dist", (0, 50), "distance to arc (m)"), (axs[1], "heading_diff", (0, 180), "heading difference (deg)"),
                              (axs[2], "dist_win5", (0, 50), "mean distance over 5 points (m)"), (axs[3], "speed", (0, 30), "GPS speed (m/s)")):
        for v, name, col_ in ((1, "reference arc", "green"), (0, "other candidates", "#999999")):
            ax.hist(c.loc[c.label == v, col], bins=40, range=rng, density=True, alpha=.6, color=col_, label=name)
        ax.set_xlabel(lab)
    axs[0].legend(fontsize=8)
    fig.suptitle("Feature distributions, true arc vs other candidates (train partition, 200k random candidate rows)")
    fig.tight_layout(); fig.savefig(OUT / "feature_distributions.png", dpi=130); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (gps_trajectories, road_network_trip, world_extent, miss_rate_vs_radius, label_coverage, feature_distributions):
        fn()
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
