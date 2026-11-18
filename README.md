# PathSnap map matching

Match noisy GPS points to the road network. Pipeline:

1. generate candidate road arcs for every GPS point (`features/`)
2. score the candidates with an ML classifier (`models/`)
3. run an HMM with Viterbi over the scores to get one connected route (`hmm/`)
4. serve it from a FastAPI backend (`api/`) with a Streamlit UI on a second laptop (`ui/`)

> Status: the project is built up to the mid-way checkpoint (data, labels, candidates, first three
> models, distance-only HMM, API stub and UI). Sections marked *(later)* are not built yet.

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The trips are **not** in the repo. Put them in `data/` (so `data/train/<id>/` and `data/student_test/<id>/`
exist) or point to them with `PATHSNAP_DATA`:

```bash
export PATHSNAP_DATA="/path/to/Student folder/Student folder"
```

Each trip folder holds `<id>.track` (lon, lat, t), `<id>.nodes` (lon, lat; row = node id),
`<id>.arcs` (from_node, to_node; row = arc id) and `<id>.route` (ground-truth arc ids in order).
The data covers the whole world, so distances are measured in metres in a per-trip local projection
(`data_processing/load.py: LocalProjector`), never in degrees.

## Build the candidate tables

```bash
python features/build_features.py        # ~1 min; writes results/candidates_{train,studtest}.parquet,
                                          # results/point_labels_*.parquet, results/candidate_miss_rate.csv
jupyter nbconvert --to notebook --execute --inplace data_processing/eda.ipynb   # EDA, results/eda/
```

The parquet files are gitignored; everyone regenerates them with the command above. Candidate table
columns: `trip_id, obs_idx, t, lon, lat, arc_id, <features>, label, label_valid` (see `results/feature_notes.md`).
Fit and evaluate on `label_valid == 1` rows only.

## Train and compare models

```bash
python models/train_all.py --models logreg rf hgb      # add gnb knn mlp for the rest
```

Writes `results/model_comparison.csv` (5-fold GroupKFold CV grouped by trip, student-test Top-1 / Macro F1 /
log loss, timings). `random_state=42` everywhere. *(later: `hpo/optuna_search.py`, final model export)*

## HMM

```bash
python hmm/checkpoint_distance_only.py   # distance-only emission, nearest-arc baseline vs HMM on the 10 test trips
```

Writes `results/hmm_checkpoint_distance_only.csv` and Folium maps in `results/maps/`.
*(later: `hmm/pipeline.py`, `hmm/evaluate_hmm.py`)*

## API

```bash
# EVAL_DIR holds the trip folders; the evaluator trips are dropped in there on the day
EVAL_DIR="path/to/student_test" uvicorn api.main:app --host 0.0.0.0 --port 8000
```

* `GET /trips` list of trip ids in `EVAL_DIR`
* `POST /predict {"trip_id": "..."}` GPS points, predicted route coordinates and server-computed metrics
* `GET /health`

The backend is `stub` until the pipeline is wired in (`PATHSNAP_BACKEND`); stub responses are flagged
`"stub": true`. Ground-truth arcs are never returned. Every request is logged to the console.

## UI

```bash
streamlit run ui/app.py
```

Enter the API URL in the sidebar, pick a trip, press **Run**.

## Two-laptop setup

1. Put both laptops on the same Wi-Fi or hotspot.
2. Laptop 1 (API): allow port 8000 through the Windows firewall
   (`netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000`,
   in an administrator terminal), find the IP with `ipconfig`, start the API as above.
3. Laptop 2 (UI): check `curl http://<laptop1-ip>:8000/trips`, then `streamlit run ui/app.py` and set the API URL
   to `http://<laptop1-ip>:8000`.

## Layout

```
data_processing/  loader, projection, EDA notebook, labels
features/         candidate generation and feature table
models/           metrics, data helpers, training harness
hpo/              Bayesian tuning (later)
hmm/              road graph, Viterbi, pipeline
api/  ui/         FastAPI backend, Streamlit UI
visualisations/   Folium maps (plots later)
results/          CSVs, logs, plots, maps     artifacts/  model files and configs     report/
```
