"""Streamlit UI (P4), runs on Laptop 2.

    streamlit run ui/app.py

Enter Laptop 1's API URL (e.g. http://192.168.1.20:8000), pick a trip, press Run.
The map shows the GPS points and the predicted route returned by the API; the metric
cards show Top-1, route continuity and inference time. Ground truth never reaches the UI.
"""
from __future__ import annotations

import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap map matching", layout="wide")
st.title("PathSnap - GPS map matching")

api_url = st.sidebar.text_input("API URL (Laptop 1)", os.environ.get("PATHSNAP_API", "http://localhost:8000")).rstrip("/")


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

trip_id = st.sidebar.selectbox(f"Trip ({len(trips)} available)", trips)
if st.sidebar.button("Run", type="primary"):
    try:
        r = requests.post(f"{api_url}/predict", json={"trip_id": trip_id}, timeout=600)
        r.raise_for_status()
        st.session_state["result"] = r.json()
    except Exception as e:
        st.session_state.pop("result", None)
        st.error(f"Prediction failed: {e}")

res = st.session_state.get("result")
if res:
    if res.get("stub"):
        st.info("Stub response: the API is not running the real pipeline yet, the route is a placeholder.")
    m = res["metrics"]
    c1, c2, c3 = st.columns(3)
    fmt = lambda v, f: "n/a" if v is None else f.format(v)
    c1.metric("Top-1 accuracy", fmt(m.get("top1"), "{:.2%}"))
    c2.metric("Route continuity", fmt(m.get("route_continuity_pct"), "{:.1f}%"))
    c3.metric("Inference time", fmt(m.get("inference_time_s", m.get("server_time_s")), "{:.2f} s"))

    gps, route = res["gps_points"], res["route_coords"]
    fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=14, tiles="cartodbpositron")
    fmap.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)], [max(p[0] for p in gps), max(p[1] for p in gps)]])
    g = folium.FeatureGroup(name=f"GPS points ({len(gps)})")
    for lat, lon in gps[:: max(1, len(gps) // 1500)]:      # thin very long traces for browser speed
        folium.CircleMarker((lat, lon), radius=2, color="blue", weight=1, fill=True, fill_opacity=.8).add_to(g)
    g.add_to(fmap)
    folium.PolyLine(route, color="red", weight=4, opacity=.9, tooltip="Predicted route").add_to(
        folium.FeatureGroup(name="Predicted route").add_to(fmap))
    folium.LayerControl().add_to(fmap)
    st_folium(fmap, height=620, use_container_width=True, returned_objects=[])
else:
    st.write("Pick a trip in the sidebar and press **Run**.")
