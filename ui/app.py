"""Streamlit UI (P4), runs on Laptop 2.

    streamlit run ui/app.py

Enter Laptop 1's API URL (e.g. http://192.168.1.20:8000), pick a trip, press Run.
The map shows the GPS points, the HMM route (red) and the classifier-only route (orange) returned by
the API; the cards show Top-1, route continuity and inference time. Ground truth never reaches the UI.

For screenshots the page can open a result directly: ui/app.py?trip=00000052&run=1
"""
from __future__ import annotations

import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap map matching", layout="wide")
st.title("PathSnap - GPS map matching")

params = st.query_params
api_url = st.sidebar.text_input("API URL (Laptop 1)", params.get("api", os.environ.get("PATHSNAP_API", "http://localhost:8000"))).rstrip("/")


@st.cache_data(ttl=5, show_spinner=False)
def fetch_trips(url: str):
    r = requests.get(f"{url}/trips", timeout=10)
    r.raise_for_status()
    return r.json()["trips"]


try:
    trips = fetch_trips(api_url)
except Exception as e:  # network down, wrong IP, firewall...
    st.error(f"Cannot reach the API at {api_url}: {e}")
    st.stop()

if not trips:
    st.warning("The API returned no trips. Check EVAL_DIR on Laptop 1.")
    st.stop()

default_idx = trips.index(params["trip"]) if params.get("trip") in trips else 0
trip_id = st.sidebar.selectbox(f"Trip ({len(trips)} available)", trips, index=default_idx)


def run(trip):
    try:
        with st.spinner(f"Running the pipeline on trip {trip} ..."):
            r = requests.post(f"{api_url}/predict", json={"trip_id": trip}, timeout=600)
            r.raise_for_status()
        st.session_state["result"] = r.json()
    except Exception as e:
        st.session_state.pop("result", None)
        st.error(f"Prediction failed: {e}")


if st.sidebar.button("Run", type="primary"):
    run(trip_id)
elif params.get("run") == "1" and "result" not in st.session_state:
    run(trip_id)

res = st.session_state.get("result")
if res:
    if res.get("stub"):
        st.info("Stub response: the API is running the placeholder backend, the route is not a real prediction.")
    m = res["metrics"]
    fmt = lambda v, f: "n/a" if v is None else f.format(v)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top-1 accuracy (HMM)", fmt(m.get("top1"), "{:.2%}"), None if m.get("top1_classifier_only") is None else
              f"{(m['top1'] - m['top1_classifier_only']) * 100:+.1f} pts vs classifier only")
    c2.metric("Route continuity", fmt(m.get("route_continuity_pct"), "{:.1f}%"))
    c3.metric("Inference time", fmt(m.get("inference_time_s"), "{:.2f} s"))
    c4.metric("GPS points", fmt(m.get("n_points"), "{:,}"))
    st.caption(f"Trip {res['trip_id']} | classifier-only Top-1 {fmt(m.get('top1_classifier_only'), '{:.2%}')} | "
               f"arc precision {fmt(m.get('arc_precision'), '{:.2%}')} | arc recall {fmt(m.get('arc_recall'), '{:.2%}')} "
               f"| server time {fmt(m.get('server_time_s'), '{:.2f} s')}")

    # Development-stage benchmarks, fetched from the API (which reads them from results/ on Laptop 1).
    # Nothing here is typed in by hand.
    with st.expander("Baseline and model comparison (student_test development results)", expanded=False):
        try:
            bm = requests.get(f"{api_url}/benchmarks", timeout=10).json()
        except Exception as e:
            bm = None
            st.warning(f"Could not load benchmarks from the API: {e}")
        if bm and bm.get("pooled_top1"):
            p = bm["pooled_top1"]
            st.markdown("**Pooled pointwise Top-1 on the 10 student_test trips** (source: results/hmm_comparison.csv)")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Nearest arc", fmt(p.get("nearest_arc"), "{:.2%}"))
            b2.metric("Classifier only", fmt(p.get("classifier_only"), "{:.2%}"))
            b3.metric("Classifier + HMM (final)", fmt(p.get("classifier_hmm"), "{:.2%}"))
            b4.metric("HMM, distance-only emission", fmt(p.get("hmm_distance_only"), "{:.2%}"))
        if bm and bm.get("models"):
            st.markdown("**Classifier families** (5-fold GroupKFold CV on train; source: results/model_comparison.csv)")
            st.dataframe(bm["models"], use_container_width=True, hide_index=True)

    gps = res["gps_points"]
    fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=14, tiles="OpenStreetMap")
    fmap.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)], [max(p[0] for p in gps), max(p[1] for p in gps)]])
    g = folium.FeatureGroup(name=f"GPS points ({len(gps)})")
    for lat, lon in gps[:: max(1, len(gps) // 1500)]:      # thin very long traces for browser speed
        folium.CircleMarker((lat, lon), radius=2, color="blue", weight=1, fill=True, fill_opacity=.8).add_to(g)
    g.add_to(fmap)
    for key, name, color, w in (("classifier_route_segments", "Classifier-only route", "orange", 5),
                                ("route_segments", "Predicted route (classifier + HMM)", "red", 3)):
        segs = res.get(key) or []
        if segs:
            grp = folium.FeatureGroup(name=name)
            for s in segs:
                folium.PolyLine(s, color=color, weight=w, opacity=.9).add_to(grp)
            grp.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    st_folium(fmap, height=620, use_container_width=True, returned_objects=[])
else:
    st.write("Pick a trip in the sidebar and press **Run**.")
