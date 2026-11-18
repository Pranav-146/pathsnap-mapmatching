# PathSnap: remaining work split (4 people, 4 machines)

Status at hand-off: data layer, labels, 16 features, all 6 models trained and compared, HMM/Viterbi decoder,
pipeline, API, UI and the report builder exist in the repo. What is left is **running** the long jobs, fixing
whatever breaks on the first run, and assembling the final PDF. Each person below has a task that can start
**right now** on their own laptop. Nobody needs to wait for anybody until the sync point (section 3).

> **Honest status.** These scripts were written and unit-checked, but have **not** been run end-to-end yet:
> `hpo/optuna_search.py`, `visualisations/model_plots.py`, `hmm/tune_hmm.py`, `hmm/evaluate_hmm.py`,
> `visualisations/hmm_plots.py`, `report/capture_screenshots.py`, and the section modules in `report/`.
> The first run may hit a small bug. If it does: copy the full traceback and send it to the coordinator
> (Pranav, who has the AI assistant that wrote them). Do not rewrite the scripts yourself unless the fix is obvious.

---

## 1. Setup (everyone, about 10 minutes)

Python 3.10 to 3.12. Windows PowerShell commands (use `export` instead of `$env:` on Mac/Linux).

```powershell
git clone https://github.com/Pranav-146/pathsnap-mapmatching.git
cd pathsnap-mapmatching
git checkout develop
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# the folder that CONTAINS train\ and student_test\ (the raw data is not in the repo)
$env:PATHSNAP_DATA = "C:\path\to\Student folder\Student folder"

# identify yourself: use YOUR GitHub username and the email on your GitHub account
git config user.name  "your-github-username"
git config user.email "your-github-email"

python features/build_features.py        # about 1.5 min, writes results/*.parquet (gitignored)
```

`build_features.py` must print exactly these lines (they are deterministic); if yours differ, stop and tell the coordinator:

```
== train: 208417 points (187865 label-valid), 1084832 rows, positive rate 0.169, miss rate (R=50, k=8) 0.0228
== student_test: 19160 points (19143 label-valid), 113639 rows, positive rate 0.168, miss rate (R=50, k=8) 0.0031
```

## 2. Rules (all of them are graded)

1. **Only edit your own files** (listed in your task). This is what keeps merges conflict-free.
2. Work on **your own branch**, never directly on `develop` or `main`.
3. **Every number in the report must come from a file in `results/`.** Never type a result by hand, never fake a plot.
4. The 10 **evaluator trips are never used** for training, tuning, feature choices or thresholds. You only have the
   `train` and `student_test` partitions anyway; keep it that way.
5. Do not commit: raw data, `*.parquet`, `artifacts/all_models/`, passwords or keys (`.gitignore` already covers them).
6. Every member must be able to explain the whole pipeline in the viva. Read section 7 and spend the last 15 minutes
   walking each other through your parts.

Branch workflow for every task:

```powershell
git checkout develop ; git pull
git checkout -b <your-branch>
# ... do the work ...
git add <only your files>
git commit -m "short message about what you did"
git push -u origin <your-branch>
```

## 3. Who waits for whom

```
now ──────────────────────────────────────────────────────────────► final
P1  verify data stage + EDA + reproducibility ────────────────────────────►(viva, QA)
P2  Optuna ► export model ► plots ► push ── SYNC S1 ──┐
P3  decoder tuning + evaluation (interim model) ──────┼► re-run (final model) ► push
P4  API/UI/screenshots (interim model) ───────────────┴► re-run ► build PDF ► final push
```

* **Interim model.** `artifacts/model.pkl` currently holds the untuned HistGradientBoosting, so P3 and P4 can run their
  full stages today. P2's Optuna run **replaces** it. If the tuned model beats the default on CV Macro F1 (the rule in
  `hpo/optuna_search.py`) it is exported, otherwise the default stays.
* **Sync point S1:** P2 pushes `p2/hpo-export` and posts in the group chat. The coordinator merges the branches into
  `develop` (section 6). After that P3 and P4 pull `develop` and **re-run** their commands once with the final model
  (about 20 minutes each). Their first run is not wasted: it finds the bugs.

---

## 4. Tasks

### Task P1 (Pranav-146): data stage verification, EDA, reproducibility, QA

Goal: prove the data stage reproduces from a fresh clone, check report section 2.2, and be ready to explain the data.
Branch: `p1/qa-eda`. Files you own: `data_processing/*`, `features/*`, `report/sec_eda.py`, `results/eda/*`, `results/feature_*`.

1. Do section 1 and confirm the two printed lines.
2. Feature ablation (2 to 10 minutes): `python features/ablation_v1_v2.py`. Expect v1 CV Macro F1 about 0.8990 and v2 about 0.9048
   (`results/feature_v2_ablation.csv`).
3. EDA figures (40 s): `python data_processing/eda_plots.py` writes `results/eda/*.png`.
4. Notebook (35 s): `jupyter nbconvert --to notebook --execute --inplace data_processing/eda.ipynb`, then open it and check that the
   **charts appear inline** (points per trip, time gaps, distance to route) and the Folium map cell finishes.
5. Section 2.2: `python report/test_section.py sec_eda`, open `report/_test_sec_eda.pdf` and read all of it. Every sentence must be
   something you can defend. Fix wording in `report/sec_eda.py` if needed (numbers fill in automatically).
6. **Fresh-clone test of the README** (required in the final dry run): follow `README.md` literally from an empty folder up to the end of
   the data stage. Write down anything that is missing or wrong and fix `README.md` (P4 owns it; make the fix on your branch and tell P4).
7. Be **Laptop 2** for P4's two-laptop network test when P4 asks (section 5).

Commit only what actually changed (probably `README.md`/`report/sec_eda.py`; re-running the scripts should reproduce the committed results).
Done when: numbers match, notebook charts show, `sec_eda` PDF reads correctly, README works from scratch.

### Task P2 (srijan-nanda): Bayesian tuning, final model export, model plots, report sections 3 and 4

Goal: Optuna search, export the final model, make the comparison plots. This is the critical path, start first.
Branch: `p2/hpo-export`. Files you own: `models/*`, `hpo/*`, `visualisations/model_plots.py`, `report/sec_models.py`, `artifacts/model.pkl`,
`artifacts/feature_config.json`, `results/model_*`, `results/cv_folds.csv`, `results/hpo_*`, `results/plots/*` (except `hmm_comparison.png`).

1. Do section 1.
2. Run the search: `python hpo/optuna_search.py --model hgb --trials 40`
   * It prints one line per trial (`trial 7 COMPLETE value=0.9051 best=0.9055`). A trial took about 40 seconds on a 28-core machine,
     so 40 trials is roughly 25 to 30 minutes there; **on a laptop expect 45 to 90 minutes**:
     if the first trial takes more than 90 seconds use `--trials 30` (the plan allows 30 to 50).
   * Smoke-tested already with 2 trials: the script runs end to end (trial log, four Optuna plots, tuned-vs-untuned table, model export).
   * Outputs: `results/hpo_trials.csv`, `results/hpo_plots/*.png`, `results/hpo_tuned_vs_untuned.csv`,
     `artifacts/model.pkl`, `artifacts/feature_config.json` (its `"variant"` says `tuned` or `untuned`).
3. `python models/select_final.py` (instant) and `python visualisations/model_plots.py`, then open every file in `results/plots/` and `results/hpo_plots/`:
   bar charts, confusion matrices, the MLP loss curve, tuned vs untuned.
4. `python report/test_section.py sec_models`, open `report/_test_sec_models.pdf`, read sections 3 and 4. The per-model answers cite real numbers;
   check each claim against the table (for example, that a model "overfits" only where train F1 is clearly above CV F1). Edit `report/sec_models.py` if a sentence is wrong.
5. Commit and push (list below), then **post in the group chat: "S1: model exported, variant = tuned/untuned"**.

```powershell
git add hpo results/hpo_trials.csv results/hpo_tuned_vs_untuned.csv results/hpo_plots
git add artifacts/model.pkl artifacts/feature_config.json
git add results/model_selection.csv results/plots report/sec_models.py
git commit -m "Run Optuna search, export final model and add model comparison plots"
git push -u origin p2/hpo-export
```

Done when: `hpo_trials.csv` has 30 to 50 rows, `artifacts/feature_config.json` exists, all plots open and look right, and you can explain why this model was chosen.

### Task P3 (skandanadig): decoder tuning, HMM evaluation, Folium maps, report sections 5 and 6

Goal: tune the HMM parameters, prove with numbers whether the HMM helps, make a map for every student_test trip.
Branch: `p3/hmm-run`. Files you own: `hmm/*`, `visualisations/folium_maps.py`, `visualisations/hmm_plots.py`, `report/sec_hmm.py`,
`artifacts/hmm_config.json`, `results/hmm_*`, `results/maps/trip_*`, `results/plots/hmm_comparison.png`.

**Step 1: today, with the interim model that is already in the repo.**

1. Do section 1.
2. `python hmm/tune_hmm.py`  (10 to 25 minutes). It computes out-of-fold classifier scores for the train trips, tunes beta, emission weight,
   bonuses, gap threshold and distance mode on 12 train trips + the 10 student_test trips, and checks on 12 other train trips. It prints the best
   parameters and writes `artifacts/hmm_config.json`, `results/hmm_tuning_log.csv`, `results/hmm_holdout_train_trips.csv`.
3. `python hmm/evaluate_hmm.py`  (3 to 6 minutes). Per trip: classifier-only vs classifier + HMM (Top-1, route continuity, arc precision/recall/F1).
   Writes `results/hmm_comparison.csv`, `results/hmm_notes.md` and 10 maps `results/maps/trip_<id>_hmm.html`.
4. `python visualisations/hmm_plots.py`, then open two or three of the maps in a browser: check the blue, grey, green, orange and red layers and the legend.
5. `python report/test_section.py sec_hmm`. Figures that need screenshots (`results/screenshots/folium_*.png`) show a "missing figure" note until P4 captures them; that is fine.
6. Look at `results/hmm_notes.md`: it lists the trips where the HMM did **not** help. Be ready to explain why.

**What the smoke test already showed (so you know what to expect).** Run with untuned default decoder parameters and the interim model, the pooled student_test Top-1
was 0.863 for the classifier alone and 0.903 for classifier + HMM (better on 9 of 10 trips; trip 00000007 got worse, 0.919 to 0.864), while the HMM with **distance-only**
emission reached 0.934. In other words the default decoder trusts the classifier too much: the v2 features already contain neighbour-consistency information, so the
HMM partly counts it twice. The tuning grid now includes low emission weights (0.1, 0.25, 0.5) for exactly this reason. After tuning, look at whether classifier + HMM
catches up with or passes the distance-only HMM; whichever way it ends, **report the real numbers** (the final system must still use the classifier, but the comparison table
may honestly show the distance-only variant being close or better, and you should explain why).

**Step 2: after sync point S1.** `git checkout develop ; git pull`, then rerun steps 2 to 4 with the final model, and re-commit the changed result files.

```powershell
git add hmm artifacts/hmm_config.json results/hmm_tuning_log.csv results/hmm_holdout_train_trips.csv
git add results/hmm_comparison.csv results/hmm_notes.md results/maps results/plots/hmm_comparison.png
git add visualisations report/sec_hmm.py
git commit -m "Tune decoder parameters, evaluate classifier-only vs HMM and generate maps"
git push -u origin p3/hmm-run
```

Done when: HMM improves pooled Top-1 over classifier-only (if it does not, tell the coordinator before doing anything else), 10 maps open correctly,
and you can explain emission, transition, gap and disconnected-arc handling.

### Task P4 (sneha1a408-stud): API, UI, screenshots, two-laptop test, final report PDF

Goal: a demo that works across two laptops for all 10 trips, the screenshots the report needs, and the final merged PDF.
Branch: `p4/api-ui-final`. Files you own: `api/*`, `ui/*`, `README.md`, `requirements.txt`, `.gitignore`, `report/common.py`, `report/sec_api.py`,
`report/build_report.py`, `report/capture_screenshots.py`, `report/test_section.py`, `docs/*`, `results/screenshots/*`, `results/api_console_log.txt`.

**Step 1: today, with the interim model.**

1. Do section 1. API in one terminal (EVAL_DIR is the student_test folder):

   ```powershell
   $env:EVAL_DIR = "$env:PATHSNAP_DATA\student_test"
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   The console must print `model loaded once ...`. In a second terminal: `curl http://localhost:8000/health`, `curl http://localhost:8000/trips`.
2. UI: `streamlit run ui/app.py`, pick a trip, press **Run**. Check: map shows GPS (blue), predicted route (red) and classifier-only route (orange); the cards show Top-1,
   continuity and inference time; the API console prints one line per request with trip, timing and accuracy. Run **all 10 trips** and note any that fail.
3. Screenshots (8 to 12 minutes, needs Microsoft Edge or Chrome installed): stop the API/UI first (the script starts its own), then
   `python report/capture_screenshots.py`. It writes `results/screenshots/*.png` and `results/api_console_log.txt`; open a few of the PNGs.
4. **Two-laptop test (required deliverable):** both laptops on the same Wi-Fi or hotspot. Laptop 1 (API): allow port 8000
   (`netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000` in an administrator terminal), run `ipconfig` and note the IPv4 address,
   start the API. Laptop 2 (P1's laptop): `curl http://<ip>:8000/trips`, then `streamlit run ui/app.py` and put `http://<ip>:8000` in the sidebar. Run all 10 trips; take a photo/screenshot of both screens.
5. `python report/test_section.py sec_api` and read it (it parses the real console log). If you captured the two-laptop screenshots, add them to `report/sec_api.py` and update the sentence that says screenshots were taken on one machine.

**Step 2: after S1 and after P2, P3 pushed.** `git checkout develop ; git pull` (develop must contain the final model and the HMM results), then re-run `python report/capture_screenshots.py`,
then **`python report/build_report.py`** and read every page of `report/PathSnap_Report.pdf`: no "missing figure" notes, no `N/A` where a number should be, every table filled.
Commit the PDF and push.

```powershell
git add api ui README.md requirements.txt .gitignore docs report/common.py report/sec_api.py report/build_report.py report/capture_screenshots.py report/test_section.py
git add results/screenshots results/api_console_log.txt report/PathSnap_Report.pdf
git commit -m "Finish API, UI, screenshots and assemble the technical report"
git push -u origin p4/api-ui-final
```

Done when: all 10 trips run through the UI with logs on the API console, the two-laptop test worked, and the final PDF builds with no missing parts.

---

## 5. Evaluation-day checklist (P4 drives, everyone helps)

1. Laptop 1: `$env:EVAL_DIR = "<folder where the evaluator trips are dropped>"`, `uvicorn api.main:app --host 0.0.0.0 --port 8000`. Keep this console visible: evaluators check the request logs.
2. Laptop 2: `streamlit run ui/app.py`, API URL = `http://<laptop-1-ip>:8000`. The dropdown must show the 10 trips (the API re-scans the folder on every call).
3. Evaluator runs any trip: the map and metric cards appear, Laptop 1 logs the request with Top-1.
4. Nothing may be hard-coded; ground truth stays on the server.

## 6. Merging (coordinator = Pranav)

```powershell
git checkout develop ; git pull
git merge --no-ff origin/p1/qa-eda   -m "Merge p1/qa-eda into develop"
git merge --no-ff origin/p2/hpo-export -m "Merge p2/hpo-export into develop"
git merge --no-ff origin/p3/hmm-run  -m "Merge p3/hmm-run into develop"
git merge --no-ff origin/p4/api-ui-final -m "Merge p4/api-ui-final into develop"
git push origin develop
git checkout main ; git merge --no-ff develop -m "Merge develop into main" ; git push origin main
```

Run the first merges (p2) at S1 so P3 and P4 can pull the final model; the last merge is after P4's PDF is pushed. Because everybody edits different files there should be no conflicts;
if Git reports one, stop and ask.

## 7. What each person must be able to explain in the viva

* **P1:** how the data is organised (4 files per trip); why a per-trip local projection and not UTM 32N (the data is worldwide); how labels are derived (forward-only DP along the `.route`, why it
  replaced a greedy pointer, what `label_valid` means); candidate generation (R = 50 m, k = 8, STRtree) and the miss rate; the 16 features and why; what feature v2 added and the ablation result.
* **P2:** the metrics (Top-1 counts a candidate miss as wrong; Macro F1 on binary candidate labels; log loss; continuity); GroupKFold by trip and why; the six models and their families;
  overfitting evidence (train vs CV vs test F1); the selection rule and why HGB beat RF; Optuna TPE, search space, objective, tuned vs untuned; what is in `artifacts/`.
* **P3:** hidden states; emission (normalised classifier score, in log space, temperature); transition (|straight - road distance| / beta plus same/adjacent bonus); missing candidates, large gaps,
  disconnected arcs (finite penalty); why precomputing road distances makes tuning cheap; out-of-fold scores for tuning; the hold-out check; where the HMM did not help.
* **P4:** the API contract and the pipeline it calls; why ground truth stays server-side; the metrics the server computes; logging; the two-laptop setup (firewall, `ipconfig`, `curl`); how the report builder reads every number from `results/`.

## 8. Troubleshooting

* **`ModuleNotFoundError`**: the virtual environment is not active (`.\.venv\Scripts\Activate.ps1`) or `pip install -r requirements.txt` failed.
* **`FileNotFoundError ... candidates_train.parquet`**: run `python features/build_features.py` first (the parquet files are not in git).
* **Cannot find the trips**: `$env:PATHSNAP_DATA` must point at the folder that contains `train` and `student_test`.
* **Optuna is too slow**: lower the trial count (`--trials 30`); do not lower the folds.
* **Port 8000 not reachable from Laptop 2**: firewall rule missing, wrong IP (use the Wi-Fi adapter's IPv4), or the laptops are on different networks.
* **Screenshot script finds no browser**: install Microsoft Edge or Chrome, or take the UI and Folium screenshots by hand and save them under the names the report expects (`results/screenshots/ui_<trip>.png`, `folium_<trip>.png`, `api_console.png`).
* **Anything else**: send the full traceback and the command you ran to the coordinator.
