"""Folium maps (P3): GPS points, road network, reference route, classifier-only and HMM routes."""
from __future__ import annotations

import sys
from pathlib import Path

import folium
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.load import REPO_ROOT  # noqa: E402

MAX_NETWORK_ARCS = 40000


def _lines(nodes, arcs, arc_ids):
    """List of [(lat, lon), (lat, lon)] segments for the given arc ids."""
    f = arcs["from_node"].to_numpy()[arc_ids]
    t = arcs["to_node"].to_numpy()[arc_ids]
    la, lo = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    return [[(float(la[a]), float(lo[a])), (float(la[b]), float(lo[b]))] for a, b in zip(f, t)]


def _route_arcs(per_point_arcs: np.ndarray) -> np.ndarray:
    """Arc ids in order of appearance with consecutive repeats removed (-1 ignored)."""
    a = np.asarray(per_point_arcs)
    a = a[a >= 0]
    if len(a) == 0:
        return a
    return a[np.concatenate([[True], a[1:] != a[:-1]])]


def make_map(trip: dict, clf_arcs: np.ndarray | None = None, hmm_arcs: np.ndarray | None = None,
             show_reference: bool = True, out_path: str | Path | None = None) -> folium.Map:
    """Draw one trip.

    trip      dict from data_processing.load.load_trip
    clf_arcs  predicted arc per GPS point from the classifier alone (index = obs_idx)
    hmm_arcs  predicted arc per GPS point from classifier + HMM
    Layers: GPS points (blue), road network (grey), reference route (green),
    classifier-only route (orange), HMM route (red); legend and layer control.
    """
    t, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    m = folium.Map(location=[float(t.lat.mean()), float(t.lon.mean())], zoom_start=14, tiles="cartodbpositron")
    m.fit_bounds([[float(t.lat.min()), float(t.lon.min())], [float(t.lat.max()), float(t.lon.max())]])

    pad = 0.003
    near = (nodes.lon.between(t.lon.min() - pad, t.lon.max() + pad)
            & nodes.lat.between(t.lat.min() - pad, t.lat.max() + pad)).to_numpy()
    in_view = np.flatnonzero(near[arcs.from_node.to_numpy()] & near[arcs.to_node.to_numpy()])
    if len(in_view) > MAX_NETWORK_ARCS:
        in_view = in_view[:: int(np.ceil(len(in_view) / MAX_NETWORK_ARCS))]
    g = folium.FeatureGroup(name=f"Road network ({len(in_view)} arcs in view)")
    folium.PolyLine(_lines(nodes, arcs, in_view), color="#9a9a9a", weight=2, opacity=.7).add_to(g)
    g.add_to(m)

    legend = [("blue", "GPS points"), ("#9a9a9a", "Road network")]
    if show_reference and len(trip["route"]):
        ref = _route_arcs(trip["route"])
        g = folium.FeatureGroup(name=f"Reference route ({len(ref)} arcs)")
        folium.PolyLine(_lines(nodes, arcs, ref), color="green", weight=6, opacity=.85).add_to(g)
        g.add_to(m)
        legend.append(("green", "Reference route"))
    for arr, name, col, w in ((clf_arcs, "Classifier-only route", "orange", 4), (hmm_arcs, "HMM route", "red", 3)):
        if arr is None:
            continue
        ids = _route_arcs(arr)
        g = folium.FeatureGroup(name=f"{name} ({len(ids)} arcs)")
        folium.PolyLine(_lines(nodes, arcs, ids), color=col, weight=w, opacity=.9).add_to(g)
        g.add_to(m)
        legend.append((col, name))

    g = folium.FeatureGroup(name=f"GPS points ({len(t)})")
    for la, lo in zip(t.lat.to_numpy(), t.lon.to_numpy()):
        folium.CircleMarker((float(la), float(lo)), radius=2, color="blue", weight=1, fill=True, fill_opacity=.8).add_to(g)
    g.add_to(m)

    rows = "".join(f'<span style="color:{c}">&#9632;</span> {n}<br>' for c, n in legend)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 12px;'
        f'border:1px solid #999;font-size:13px"><b>Trip {trip["trip_id"]}</b><br>{rows}</div>'))
    folium.LayerControl(collapsed=False).add_to(m)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_path))
    return m


MAPS_DIR = REPO_ROOT / "results" / "maps"
