"""Report section 7: API, UI and deployment (P4). The request table is parsed from the captured API console log."""
from __future__ import annotations

import re

from report.common import PAGE_W, RESULTS, H1, H2, P, SMALL, bullets, img, pb, sp, table

LINE = re.compile(r"POST /predict trip=(\d+) points=(\d+) inference=([\d.]+)s total=([\d.]+)s top1=([\d.]+) \(classifier-only ([\d.]+)\) continuity=([\d.]+)% arcF1=([\d.]+)")


def parse_log():
    p = RESULTS / "api_console_log.txt"
    if not p.exists():
        return []
    rows = []
    for m in LINE.finditer(p.read_text(encoding="utf-8", errors="replace")):
        rows.append(m.groups())
    return rows


def build(story: list):
    story.append(P("7. API, UI and deployment (P4)", H1))
    story.append(P("The system follows the two-laptop architecture of the brief. <b>Laptop 1</b> runs the FastAPI backend (<font name='Courier'>uvicorn api.main:app --host 0.0.0.0 --port 8000</font>): it loads the "
                   "trained classifier once at startup, runs the whole pipeline (candidate generation, features, classifier scores, HMM/Viterbi) on request, and computes accuracy on the server from the trip's own .route. "
                   "<b>Laptop 2</b> runs a Streamlit UI that lists the trips, sends the chosen trip id to the API and shows the returned route and metrics.", BODY))
    story.append(P("Endpoints", H2))
    story.append(table([["Endpoint", "Purpose"],
                        ["GET /trips", "lists the trip ids found in the folder EVAL_DIR (re-scanned on each call, so evaluator trips dropped into the folder are picked up)"],
                        ["POST /predict {trip_id}", "runs run_trip() on the server; returns the GPS points, the predicted route (connected polylines), the classifier-only route and metric numbers"],
                        ["GET /health", "liveness, backend and model summary"]], [PAGE_W * 0.25, PAGE_W * 0.75]))
    story.append(sp(4))
    story += bullets([
        "<b>Ground truth stays on the server.</b> The .route file is read only inside the server-side metric computation; only the numbers (Top-1, route continuity, arc precision/recall) are returned. It never reaches the model, the decoder or the UI.",
        "<b>Logging.</b> Each request is printed on the API console with the trip id, number of points, inference time, total time, Top-1 (HMM and classifier-only), route continuity and arc F1 (figure 7.1, table below).",
        "<b>Robustness.</b> Unknown trip ids return HTTP 404; CORS is enabled; the model is loaded once, not per request.",
    ])
    rows = parse_log()
    if rows:
        story.append(P("API requests captured during the run in which the UI evaluated all 10 student_test trips", H2))
        t = [["Trip", "Points", "Inference time (s)", "Total request time (s)", "Top-1 (classifier + HMM)", "Top-1 (classifier only)", "Route continuity", "Arc F1"]]
        for r in rows:
            t.append([r[0], f"{int(r[1]):,}", r[2], r[3], r[4], r[5], f"{r[6]}%", r[7]])
        story.append(table(t, [PAGE_W / 8] * 8))
        story.append(P("Source: results/api_console_log.txt, the raw console output of the API process. Each of these requests was issued by the UI (headless Edge opening the Streamlit page for each trip).", SMALL))
    S = RESULTS / "screenshots"
    story.append(P("Screenshots", H2))
    story += img(S / "ui_00000052.png", "Figure 7.2 The Streamlit UI after running trip 00000052: metric cards, GPS points (blue), classifier-only route (orange) and predicted HMM route (red), with layer control.", PAGE_W * 0.97)
    story += img(S / "ui_00000009.png", "Figure 7.3 The UI after running trip 00000009 (the trip where the classifier alone is weakest).", PAGE_W * 0.97)
    story += img(S / "api_console.png", "Figure 7.1 The API console on Laptop 1 (rendered from the captured console text in results/api_console_log.txt): one line per request with trip, timing and accuracy.", PAGE_W)
    story.append(P("Two-laptop setup", H2))
    story += bullets([
        "Both laptops on the same Wi-Fi or hotspot; on Laptop 1 allow port 8000 through the Windows firewall and read the IP with <font name='Courier'>ipconfig</font>.",
        "From Laptop 2, <font name='Courier'>curl http://&lt;ip&gt;:8000/trips</font> must list the trips; then <font name='Courier'>streamlit run ui/app.py</font> and enter <font name='Courier'>http://&lt;ip&gt;:8000</font> as the API URL.",
        "The screenshots above were produced on a single machine (API and UI on localhost) with the same code; the cross-machine path differs only in the API URL.",
    ])
    story.append(pb())
