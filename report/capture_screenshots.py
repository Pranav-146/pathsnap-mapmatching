"""Capture screenshots for report section 7 and section 6 (P4).

    python report/capture_screenshots.py

1. Starts the API (uvicorn, EVAL_DIR = student_test) and the Streamlit UI on this machine.
2. For every one of the 10 trips opens the UI in headless Microsoft Edge/Chrome with
   ?trip=<id>&run=1, so the UI itself calls POST /predict; saves results/screenshots/ui_<trip>.png.
3. Saves the API's console output to results/api_console_log.txt and renders it as an image
   (results/screenshots/api_console.png). The text is the real captured output.
4. Takes headless screenshots of exported Folium maps (results/screenshots/folium_*.png).
A browser that is not installed is skipped with a message; nothing is drawn by hand.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import requests

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data_processing.load import DATA_ROOT  # noqa: E402

SHOTS = REPO / "results" / "screenshots"
API_PORT, UI_PORT = 8000, 8501
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]


def browser() -> str | None:
    return next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("msedge") or shutil.which("chrome")


def shot(url: str, out: Path, size="1500,1000", budget=40000):
    b = browser()
    if not b:
        print("no Edge/Chrome found, skipping", out.name)
        return False
    cmd = [b, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={size}",
           f"--virtual-time-budget={budget}", f"--screenshot={out}", url]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    return out.exists()


def wait(url, tries=60):
    for _ in range(tries):
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


def render_console(text: str, out: Path, max_lines=46):
    lines = [l for l in text.splitlines() if "POST /predict" in l or "GET /trips" in l or "model loaded" in l or "starting" in l]
    lines = [l[:170] for l in lines][-max_lines:]
    fig = plt.figure(figsize=(15, 0.26 * len(lines) + 0.8), facecolor="#0c0c0c")
    fig.text(0.01, 0.99, "C:\\> uvicorn api.main:app --host 0.0.0.0 --port 8000   (API console on Laptop 1)", color="#9cdcfe",
             family="monospace", fontsize=8, va="top")
    for i, l in enumerate(lines):
        fig.text(0.01, 0.93 - i * (0.9 / max(len(lines), 1)), l, color="#cccccc", family="monospace", fontsize=7.2, va="top")
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor()); plt.close(fig)


def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    env = {**os.environ, "EVAL_DIR": str(DATA_ROOT / "student_test"), "PYTHONUNBUFFERED": "1"}
    log_path = REPO / "results" / "api_console_log.txt"
    with open(log_path, "w", encoding="utf-8") as logf:
        api = subprocess.Popen([py, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", str(API_PORT)],
                               cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)
        ui = subprocess.Popen([py, "-m", "streamlit", "run", "ui/app.py", "--server.port", str(UI_PORT), "--server.headless", "true",
                               "--browser.gatherUsageStats", "false"], cwd=REPO, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            assert wait(f"http://localhost:{API_PORT}/health", 120), "API did not start"
            assert wait(f"http://localhost:{UI_PORT}/_stcore/health", 60), "UI did not start"
            trips = requests.get(f"http://localhost:{API_PORT}/trips").json()["trips"]
            for t in trips:
                out = SHOTS / f"ui_{t}.png"
                ok = shot(f"http://localhost:{UI_PORT}/?trip={t}&run=1&api=http://localhost:{API_PORT}", out)
                print("UI screenshot", t, "ok" if ok else "FAILED", flush=True)
        finally:
            for p in (ui, api):
                p.terminate()
            time.sleep(2)
    render_console(log_path.read_text(encoding="utf-8", errors="replace"), SHOTS / "api_console.png")
    maps = [("eda_student_test_00000052.html", "folium_eda_00000052.png")]
    maps += [(f"trip_{t}_hmm.html", f"folium_{t}.png") for t in ("00000052", "00000009")]
    for html, name in maps:
        m = REPO / "results" / "maps" / html
        if m.exists():
            print("folium screenshot", name, "ok" if shot(m.as_uri(), SHOTS / name, "1400,950", 25000) else "FAILED")


if __name__ == "__main__":
    main()
