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

    # Baseline comparison summary expander
    with st.expander("📊 Baseline & Model Comparison Benchmarks (Overall Evaluation)", expanded=False):
        st.markdown("#### Baseline vs Model Pipeline Comparison on `student_test` (10 Trips Pooled)")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Nearest Arc Baseline", "53.44%", help="Geometric Euclidean projection to nearest road arc")
        b2.metric("Classifier-Only Baseline", "86.26%", help="Gradient boosted trees alone, no sequence decoder")
        b3.metric("Classifier + HMM (Final)", "93.88%", "+7.62 pts vs Classifier", help="Viterbi HMM decoder with topological bonuses")
        b4.metric("Distance-Only HMM", "94.07%", help="HMM using pure spatial emission without classifier features")
        
        st.markdown("---")
        st.markdown("#### Classifier Family Benchmarks (5-Fold GroupKFold Cross-Validation)")
        st.dataframe(
            [
                {"Rank": 1, "Model": "HistGradientBoosting (Selected)", "CV Macro F1": "90.48% (±2.4%)", "Test Top-1": "86.27%", "Inference Latency": "0.09 s", "Why Chosen": "Optimal balance of accuracy, low ECE (0.005), and 3x faster inference than RF"},
                {"Rank": 2, "Model": "Random Forest", "CV Macro F1": "90.59% (±2.4%)", "Test Top-1": "86.63%", "Inference Latency": "0.26 s", "Why Chosen": "Slightly higher raw F1 but 3x slower inference and much larger artifact size"},
                {"Rank": 3, "Model": "Multilayer Perceptron (MLP)", "CV Macro F1": "90.28% (±2.5%)", "Test Top-1": "86.44%", "Inference Latency": "0.08 s", "Why Chosen": "Competitive but required 275s train time and complex scaling"},
                {"Rank": 4, "Model": "KNN (k=15)", "CV Macro F1": "89.18% (±2.2%)", "Test Top-1": "85.16%", "Inference Latency": "7.72 s", "Why Chosen": "Severely unviable inference latency (85x slower than HGB)"},
                {"Rank": 5, "Model": "Logistic Regression", "CV Macro F1": "88.51% (±2.2%)", "Test Top-1": "85.11%", "Inference Latency": "0.02 s", "Why Chosen": "Fast linear baseline but underfits non-linear spatial interactions"},
                {"Rank": 6, "Model": "Gaussian Naive Bayes", "CV Macro F1": "75.93% (±1.3%)", "Test Top-1": "82.20%", "Inference Latency": "0.05 s", "Why Chosen": "Violates independence assumptions among spatial/heading features"},
            ],
            use_container_width=True,
            hide_index=True
        )

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
