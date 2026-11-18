import sys
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable

doc = SimpleDocTemplate(
    "docs/PathSnap_Team_Study_Guide.pdf",
    pagesize=A4,
    leftMargin=36,
    rightMargin=36,
    topMargin=36,
    bottomMargin=36
)

styles = getSampleStyleSheet()

# Custom styles
primary_color = colors.HexColor("#0f2b48")
secondary_color = colors.HexColor("#1b6ca8")
accent_color = colors.HexColor("#d9534f")
text_dark = colors.HexColor("#222222")
bg_light = colors.HexColor("#f8f9fa")
border_color = colors.HexColor("#e2e8f0")

title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=22,
    leading=26,
    textColor=primary_color,
    alignment=1,
    spaceAfter=6
)

subtitle_style = ParagraphStyle(
    'DocSubTitle',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=11,
    leading=15,
    textColor=colors.HexColor("#555555"),
    alignment=1,
    spaceAfter=15
)

h1_style = ParagraphStyle(
    'Heading1_Custom',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=15,
    leading=19,
    textColor=primary_color,
    spaceBefore=14,
    spaceAfter=8,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    'Heading2_Custom',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=12,
    leading=16,
    textColor=secondary_color,
    spaceBefore=10,
    spaceAfter=4,
    keepWithNext=True
)

body_style = ParagraphStyle(
    'Body_Custom',
    parent=styles['BodyText'],
    fontName='Helvetica',
    fontSize=9.2,
    leading=13.5,
    textColor=text_dark,
    spaceAfter=6
)

bullet_style = ParagraphStyle(
    'Bullet_Custom',
    parent=body_style,
    leftIndent=14,
    firstLineIndent=-10,
    spaceAfter=4
)

code_style = ParagraphStyle(
    'Code_Custom',
    parent=body_style,
    fontName='Courier',
    fontSize=8.2,
    leading=11,
    textColor=colors.HexColor("#1a365d")
)

callout_style = ParagraphStyle(
    'Callout_Text',
    parent=body_style,
    fontSize=9,
    leading=13,
    textColor=colors.HexColor("#1e293b")
)

table_header_style = ParagraphStyle(
    'TH',
    parent=body_style,
    fontName='Helvetica-Bold',
    fontSize=8.5,
    leading=11,
    textColor=colors.white
)

table_cell_style = ParagraphStyle(
    'TD',
    parent=body_style,
    fontSize=8,
    leading=11
)

story = []

# Title & Overview
story.append(Paragraph("PathSnap: Map Matching Project", title_style))
story.append(Paragraph("Comprehensive Team Viva Preparation & Role Study Guide", subtitle_style))
story.append(Paragraph("<b>Repository:</b> <font color='#1b6ca8'>https://github.com/Pranav-146/pathsnap-mapmatching</font> &nbsp;|&nbsp; <b>Evaluation Architecture:</b> Client-Server Distributed 2-Tier Pipeline", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=12))

# Executive Summary Table
summary_data = [
    [Paragraph("Member", table_header_style), Paragraph("Assigned Role", table_header_style), Paragraph("Key Modules & Codebase Files", table_header_style), Paragraph("Primary Viva Accountability", table_header_style)],
    [Paragraph("<b>P1: Pranav</b><br/>(Pranav-146)", table_cell_style),
     Paragraph("Data Pipeline, EDA & Reproducibility", table_cell_style),
     Paragraph("data_processing/*<br/>features/*<br/>report/sec_eda.py", code_style),
     Paragraph("Spatial projection (WGS84 -> metric), DP ground-truth label derivation, STRtree candidate filtering (R=50m, k=8), feature v1/v2 ablation.", table_cell_style)],
    [Paragraph("<b>P2: Srijan</b><br/>(srijan-nanda)", table_cell_style),
     Paragraph("Classification, Optuna HPO & Model Selection", table_cell_style),
     Paragraph("models/*<br/>hpo/*<br/>visualisations/model_plots.py<br/>report/sec_models.py", code_style),
     Paragraph("GroupKFold (no trip leakage), comparison of 6 model families, HistGradientBoosting selection vs RF, Bayesian Optuna search space & objective.", table_cell_style)],
    [Paragraph("<b>P3: Skanda</b><br/>(skandanadig)", table_cell_style),
     Paragraph("HMM Decoder Tuning, Viterbi & Folium Maps", table_cell_style),
     Paragraph("hmm/*<br/>visualisations/folium_maps.py<br/>report/sec_hmm.py", code_style),
     Paragraph("Emission probability (calibrated log probabilities), transition decay (beta), topological bonuses, handling 30s gaps and disconnected arcs.", table_cell_style)],
    [Paragraph("<b>P4: Sneha</b><br/>(sneha1a408-stud)", table_cell_style),
     Paragraph("FastAPI Backend, UI & Technical Report", table_cell_style),
     Paragraph("api/*<br/>ui/*<br/>report/build_report.py<br/>report/capture_screenshots.py", code_style),
     Paragraph("Client-server isolation (ground truth strictly server-side), real-time latency logging, two-laptop network setup, 24-page report assembly.", table_cell_style)]
]

t_summary = Table(summary_data, colWidths=[100, 110, 140, 170])
t_summary.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), primary_color),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('GRID', (0, 0), (-1, -1), 0.5, border_color),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
]))
story.append(t_summary)
story.append(Spacer(1, 15))

# Overall System Concept (All Members Must Know)
story.append(Paragraph("High-Level Pipeline (Everyone Must Know This Flow)", h1_style))
story.append(Paragraph("PathSnap performs map matching on GPS traces recorded under noisy urban conditions with sampling intervals up to 30 seconds. The pipeline has four sequential phases:", body_style))
story.append(Paragraph("<b>1. Candidate Generation (STRtree):</b> For each raw GPS point (lat, lon), query spatial R-tree index for road arcs within radius R = 50m, retaining top k = 8 candidates.", bullet_style))
story.append(Paragraph("<b>2. Feature Extraction (16 features):</b> Computes spatial distance, heading discrepancy, speed consistency, topological connectivity with previous/next candidate sets, and window metrics.", bullet_style))
story.append(Paragraph("<b>3. Candidate Classifier (HistGradientBoosting):</b> Scores each candidate arc with binary probability P(is_true_arc), normalized across candidates of that ping to form emission likelihoods.", bullet_style))
story.append(Paragraph("<b>4. Sequence Decoder (Hidden Markov Model + Viterbi):</b> Solves for the global sequence of arcs maximizing joint emission and road transition probabilities across the entire trip.", bullet_style))
story.append(Spacer(1, 15))

# MEMBER 1
story.append(PageBreak())
story.append(Paragraph("Person 1 (P1 - Pranav): Data Stage, EDA & Feature Engineering", h1_style))
story.append(Paragraph("<b>Owned Modules:</b> <font name='Courier'>data_processing/*</font>, <font name='Courier'>features/*</font>, <font name='Courier'>report/sec_eda.py</font>", callout_style))
story.append(Spacer(1, 6))

story.append(Paragraph("1. Data Structure & Coordinate Systems", h2_style))
story.append(Paragraph("• <b>Four raw files per trip:</b> <font name='Courier'>.track</font> (raw GPS pings with timestamp, lat, lon), <font name='Courier'>.nodes</font> (road intersections with lat, lon), <font name='Courier'>.arcs</font> (directed road segments connecting nodes), and <font name='Courier'>.route</font> (ground-truth sequence of traversed road arcs).", bullet_style))
story.append(Paragraph("• <b>Local Tangent Plane vs Global UTM:</b> Why not standard UTM 32N? The dataset spans worldwide cities (Paris, Seattle, Melbourne, etc.). A single global UTM zone introduces severe projection distortion away from its central meridian. PathSnap projects coordinates into a local Euclidean metric plane centred dynamically at the trip's origin <font name='Courier'>(lat0, lon0)</font> using PyProj azimuthal equidistant or transverse projection. All spatial computations (distances, heading, radius queries) occur in real metric meters.", bullet_style))

story.append(Paragraph("2. Ground-Truth Label Derivation", h2_style))
story.append(Paragraph("• <b>Why Forward-Only Dynamic Programming (not greedy nearest-point):</b> A greedy matcher fails when GPS drift pulls a ping closer to an opposite-direction lane or parallel overpass. PathSnap uses a forward-only DP that matches pings along the ordered ground-truth <font name='Courier'>.route</font>, guaranteeing monotonic progression along the driven path.", bullet_style))
story.append(Paragraph("• <b>What <font name='Courier'>label_valid</font> means:</b> Points occurring before route entry, after route exit, or during extreme GPS dropouts where projection is degenerate are flagged as <font name='Courier'>label_valid = 0</font>. Models are evaluated strictly on <font name='Courier'>label_valid = 1</font> points.", bullet_style))

story.append(Paragraph("3. Spatial Indexing & Candidate Generation", h2_style))
story.append(Paragraph("• <b>STRtree (Sort-Tile-Recursive):</b> An R-tree index over road arcs allows sub-millisecond bounding-box nearest candidate queries instead of comparing against thousands of arcs in the city.", bullet_style))
story.append(Paragraph("• <b>Radius R = 50 m & k = 8 candidates:</b> Tradeoff between recall and inference efficiency. With R=50m and k=8, candidate miss rate is only <b>0.31%</b> on test trips and <b>2.28%</b> on training trips, keeping the candidate set compact while capturing >99.6% of true arcs.", bullet_style))

story.append(Paragraph("4. Feature Engineering & Ablation (v1 vs v2)", h2_style))
story.append(Paragraph("• <b>16 Features:</b> Spatial metrics (<font name='Courier'>dist</font>, <font name='Courier'>dist_rank</font>, <font name='Courier'>dist_to_best</font>), kinematic metrics (<font name='Courier'>heading_diff</font>, <font name='Courier'>speed</font>, <font name='Courier'>turn_angle</font>), geometric (<font name='Courier'>arc_len</font>, <font name='Courier'>frac</font>), and topological connectivity (<font name='Courier'>connects_prev</font>, <font name='Courier'>connects_next</font>, <font name='Courier'>in_prev</font>, <font name='Courier'>in_next</font>, <font name='Courier'>dist_win5</font>).", bullet_style))
story.append(Paragraph("• <b>v1 vs v2 Ablation:</b> v1 features relied solely on point-to-arc static geometry. v2 added temporal window metrics and multi-candidate topology (<font name='Courier'>connects_next</font>, <font name='Courier'>dist_to_best</font>), boosting 5-fold CV Macro F1 from <b>0.8990 to 0.9048</b>.", bullet_style))

# MEMBER 2
story.append(PageBreak())
story.append(Paragraph("Person 2 (P2 - Srijan): Model Benchmarks, HPO & Final Model Selection", h1_style))
story.append(Paragraph("<b>Owned Modules:</b> <font name='Courier'>models/*</font>, <font name='Courier'>hpo/*</font>, <font name='Courier'>visualisations/model_plots.py</font>, <font name='Courier'>report/sec_models.py</font>", callout_style))
story.append(Spacer(1, 6))

story.append(Paragraph("1. Evaluation Metrics & Data Splitting", h2_style))
story.append(Paragraph("• <b>Why GroupKFold (grouped by trip_id):</b> Standard K-Fold randomly splits points, causing spatial leakage because sequential points from the same vehicle share identical vehicle dynamics and road network context. GroupKFold ensures an entire trip belongs to either train or validation folds.", bullet_style))
story.append(Paragraph("• <b>Metrics Defined:</b>", bullet_style))
story.append(Paragraph("&nbsp;&nbsp;a) <b>Pointwise Top-1:</b> Accuracy of picking the correct ground-truth arc. <i>Crucially: candidate miss is penalized as 0.</i>", bullet_style))
story.append(Paragraph("&nbsp;&nbsp;b) <b>Candidate-level Macro F1:</b> Extreme class imbalance exists (1 true arc vs ~5-7 negative candidate arcs, ~16.8% positive rate). Macro F1 treats both classes equally without being skewed by majority negative candidates.", bullet_style))
story.append(Paragraph("&nbsp;&nbsp;c) <b>Log Loss & Expected Calibration Error (ECE):</b> Probabilities must reflect true confidence for proper weighting in the HMM emission step.", bullet_style))

story.append(Paragraph("2. The Six Model Families & Benchmark Comparison", h2_style))
story.append(Paragraph("Evaluated 6 algorithms across distinct machine learning paradigms on 1,066,564 training candidate rows:", body_style))

model_tbl_data = [
    [Paragraph("Model", table_header_style), Paragraph("Algorithm Family", table_header_style), Paragraph("CV Macro F1", table_header_style), Paragraph("Test Top-1", table_header_style), Paragraph("Inference Time", table_header_style), Paragraph("Verdict / Viva Defense", table_header_style)],
    [Paragraph("<b>HGB (Winner)</b>", table_cell_style), Paragraph("Gradient Boosted Trees", table_cell_style), Paragraph("90.48% (±2.4%)", table_cell_style), Paragraph("86.27%", table_cell_style), Paragraph("0.09 s", table_cell_style), Paragraph("Selected: Superior speed, low ECE (0.005), native handling of candidate structures.", table_cell_style)],
    [Paragraph("<b>Random Forest</b>", table_cell_style), Paragraph("Bagging Ensemble", table_cell_style), Paragraph("90.59% (±2.4%)", table_cell_style), Paragraph("86.63%", table_cell_style), Paragraph("0.26 s", table_cell_style), Paragraph("Near-identical F1, but 3x slower inference and overfits train set (train F1 95.6%).", table_cell_style)],
    [Paragraph("<b>MLP</b>", table_cell_style), Paragraph("Neural Network (64, 32)", table_cell_style), Paragraph("90.28% (±2.5%)", table_cell_style), Paragraph("86.44%", table_cell_style), Paragraph("0.08 s", table_cell_style), Paragraph("Competitive accuracy but 275s train time and sensitive to scaling.", table_cell_style)],
    [Paragraph("<b>KNN (k=15)</b>", table_cell_style), Paragraph("Instance-based", table_cell_style), Paragraph("89.18% (±2.2%)", table_cell_style), Paragraph("85.16%", table_cell_style), Paragraph("7.72 s", table_cell_style), Paragraph("Rejected: 85x slower inference than HGB, impractical for real-time API.", table_cell_style)],
    [Paragraph("<b>Logistic Reg</b>", table_cell_style), Paragraph("Linear / Probabilistic", table_cell_style), Paragraph("88.51% (±2.2%)", table_cell_style), Paragraph("85.11%", table_cell_style), Paragraph("0.02 s", table_cell_style), Paragraph("Fastest, but underfits non-linear kinematic interactions (heading + speed).", table_cell_style)],
    [Paragraph("<b>Gaussian NB</b>", table_cell_style), Paragraph("Generative Probabilistic", table_cell_style), Paragraph("75.93% (±1.3%)", table_cell_style), Paragraph("82.20%", table_cell_style), Paragraph("0.05 s", table_cell_style), Paragraph("Poor calibration (ECE 0.168); violates feature conditional independence.", table_cell_style)],
]
t_models = Table(model_tbl_data, colWidths=[75, 95, 75, 60, 60, 155])
t_models.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), primary_color),
    ('GRID', (0, 0), (-1, -1), 0.5, border_color),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
]))
story.append(t_models)
story.append(Spacer(1, 6))

story.append(Paragraph("3. Bayesian Hyperparameter Optimization (Optuna TPE)", h2_style))
story.append(Paragraph("• <b>Search Space:</b> <font name='Courier'>learning_rate</font> [0.02, 0.3], <font name='Courier'>max_iter</font> [100, 500], <font name='Courier'>max_leaf_nodes</font> [15, 127], <font name='Courier'>min_samples_leaf</font> [20, 300], <font name='Courier'>l2_regularization</font> [1e-3, 10.0], <font name='Courier'>max_depth</font> [None, 6, 10, 16].", bullet_style))
story.append(Paragraph("• <b>Objective & Pruner:</b> Median pruner halts unpromising trials after fold 2; objective optimizes mean 5-fold CV Macro F1 strictly on training data (test set never seen during tuning).", bullet_style))
story.append(Paragraph("• <b>Artifacts Exported:</b> <font name='Courier'>artifacts/model.pkl</font> and <font name='Courier'>artifacts/feature_config.json</font> containing exact hyperparameters.", bullet_style))

# MEMBER 3
story.append(PageBreak())
story.append(Paragraph("Person 3 (P3 - Skanda): Sequence Decoding, HMM Formulation & Folium Maps", h1_style))
story.append(Paragraph("<b>Owned Modules:</b> <font name='Courier'>hmm/*</font>, <font name='Courier'>visualisations/folium_maps.py</font>, <font name='Courier'>visualisations/hmm_plots.py</font>, <font name='Courier'>report/sec_hmm.py</font>", callout_style))
story.append(Spacer(1, 6))

story.append(Paragraph("1. Hidden Markov Model Architecture", h2_style))
story.append(Paragraph("• <b>States:</b> Hidden state at time step t is the true road arc candidate <font name='Courier'>c_i,t</font> chosen from the k candidates queried by STRtree.", bullet_style))
story.append(Paragraph("• <b>Emission Probability:</b> Normalized classifier score in log space: <br/>"
                       "&nbsp;&nbsp;&nbsp;&nbsp;<i>ln P(z_t | c_i,t) = (emission_weight / temperature) * ln( P_clf(c_i,t) / sum(P_clf) )</i><br/>"
                       "<b>Key Viva Insight:</b> Why was emission_weight tuned down to <b>0.25</b>? Feature engineering v2 already included neighbor-consistency features (<font name='Courier'>connects_prev</font>, <font name='Courier'>dist_win5</font>). If emission weight is 1.0, the classifier and HMM transition doubly penalize disconnections, making the decoder over-trust noisy pings.", bullet_style))
story.append(Paragraph("• <b>Transition Probability:</b> Exponential decay based on distance discrepancy between straight-line projection and network shortest path: <br/>"
                       "&nbsp;&nbsp;&nbsp;&nbsp;<i>ln P(c_j,t | c_i,t-1) = - | d_straight - d_network | / beta + bonus(same) + bonus(adjacent)</i><br/>"
                       "Tuned parameter <b>beta = 5.0</b> governs tolerance to GPS jitter. If two consecutive steps choose the same arc, +1.0 bonus is added; if adjacent, +0.5 bonus is added.", bullet_style))

story.append(Paragraph("2. Edge Cases: Missing Candidates, Gaps & Disconnected Arcs", h2_style))
story.append(Paragraph("• <b>Large Time Gaps (gap_T = 30.0 s):</b> If dt between consecutive pings > 30 seconds (e.g. tunnel, device pause), the Markov transition assumption breaks down. The transition cost is neutralized (reset), treating the ping as a fresh sub-trajectory start.", bullet_style))
story.append(Paragraph("• <b>Disconnected Arcs (<font name='Courier'>disconnected_cap = 500m</font>):</b> When NetworkX shortest path finds no route (directed one-way street violations or network graph dead-ends), assigning infinity crashes the trellis. PathSnap assigns a finite capped penalty of 500m.", bullet_style))
story.append(Paragraph("• <b>Computational Efficiency:</b> Precomputing Dijkstra shortest paths between all candidate endpoints once per trip reduces tuning and inference time from minutes to milliseconds.", bullet_style))

story.append(Paragraph("3. Empirical Evaluation Numbers (10 Test Trips)", h2_style))
story.append(Paragraph("• <b>Nearest-arc Baseline:</b> 53.44% Top-1", bullet_style))
story.append(Paragraph("• <b>Classifier-only:</b> 86.26% Top-1", bullet_style))
story.append(Paragraph("• <b>Classifier + HMM:</b> <b>93.88% Top-1</b> (Improves Top-1 on <b>10 out of 10 trips</b>, 0 regressions).", bullet_style))
story.append(Paragraph("• <b>Trip 00000077 (Noisy):</b> Classifier achieved only 69.2% Top-1 due to parallel road drift; HMM corrected it to <b>84.3%</b> (+15.1 points).", bullet_style))
story.append(Paragraph("• <b>Folium Maps Generated:</b> 10 interactive HTML maps in <font name='Courier'>results/maps/</font> with OpenStreetMap base layer, blue GPS markers, orange classifier path, and red connected HMM route.", bullet_style))

# MEMBER 4
story.append(PageBreak())
story.append(Paragraph("Person 4 (P4 - Sneha): API Contract, UI, Multi-Machine Setup & Report PDF", h1_style))
story.append(Paragraph("<b>Owned Modules:</b> <font name='Courier'>api/*</font>, <font name='Courier'>ui/*</font>, <font name='Courier'>report/build_report.py</font>, <font name='Courier'>report/capture_screenshots.py</font>, <font name='Courier'>README.md</font>", callout_style))
story.append(Spacer(1, 6))

story.append(Paragraph("1. API Architecture & Data Integrity Contract", h2_style))
story.append(Paragraph("• <b>FastAPI Backend (<font name='Courier'>api/main.py</font>):</b> Serves predictions on port 8000.", bullet_style))
story.append(Paragraph("• <b>Zero-Leakage Principle (Ground Truth Stays on Server):</b> The client UI sends only <font name='Courier'>{\"trip_id\": \"00000052\"}</font>. The server loads the trip's <font name='Courier'>.track</font>, runs candidates -> features -> model -> HMM, and compares against <font name='Courier'>.route</font> <i>strictly on the backend</i> to compute metrics. The ground truth route geometry never leaves the server.", bullet_style))
story.append(Paragraph("• <b>Endpoints:</b>", bullet_style))
story.append(Paragraph("&nbsp;&nbsp;• <font name='Courier'>GET /health</font>: System health, loaded model variant, feature list.", code_style))
story.append(Paragraph("&nbsp;&nbsp;• <font name='Courier'>GET /trips</font>: Dynamic directory scanner; automatically picks up evaluator test trips.", code_style))
story.append(Paragraph("&nbsp;&nbsp;• <font name='Courier'>POST /predict</font>: Executes full pipeline, returns GPS coordinates, predicted polylines, and metric scores.", code_style))

story.append(Paragraph("2. Streamlit UI (<font name='Courier'>ui/app.py</font>)", h2_style))
story.append(Paragraph("• Runs on port 8501. Features dynamic trip dropdown, metric cards (Top-1, gain vs classifier, continuity %, latency), benchmark comparison view (all 6 models and baselines), and an interactive OpenStreetMap canvas with layer control.", bullet_style))

story.append(Paragraph("3. Multi-Laptop Hardware Deployment (The Two-Laptop Test)", h2_style))
story.append(Paragraph("• <b>Laptop 1 (API Server):</b> Connected to local Wi-Fi / hotspot. Runs <font name='Courier'>uvicorn api.main:app --host 0.0.0.0 --port 8000</font>. Port 8000 opened on OS firewall. IP found via <font name='Courier'>ipconfig</font> / <font name='Courier'>ifconfig</font> (e.g. 192.168.1.15).", bullet_style))
story.append(Paragraph("• <b>Laptop 2 (Client UI):</b> Enters <font name='Courier'>http://192.168.1.15:8000</font> in the Streamlit sidebar. Tests connectivity via <font name='Courier'>curl http://192.168.1.15:8000/trips</font>. Demonstrates true distributed map-matching over the local network.", bullet_style))

story.append(Paragraph("4. Technical Report Builder (<font name='Courier'>report/build_report.py</font>)", h2_style))
story.append(Paragraph("• Generates the comprehensive 24-page report: <font name='Courier'>report/PathSnap_Report.pdf</font>.", bullet_style))
story.append(Paragraph("• Automated programmatic compilation: Every number in the report is parsed directly from files in <font name='Courier'>results/</font> and <font name='Courier'>artifacts/</font> (no hardcoded numbers).", bullet_style))
story.append(Paragraph("• Headless screenshot automation (<font name='Courier'>capture_screenshots.py</font>) launches the API and UI in the background, iterates through trips using headless browser engine, and injects rendered screenshots into the PDF.", bullet_style))

# Quick Viva Checklist
story.append(Spacer(1, 15))
story.append(Paragraph("Quick Viva Survival Cheat-Sheet (Last 5 Minutes Before Viva)", h1_style))
cheat_data = [
    [Paragraph("If the Examiner Asks...", table_header_style), Paragraph("Who Answers", table_header_style), Paragraph("The 15-Second Perfect Answer", table_header_style)],
    [Paragraph("Why not use standard UTM projection?", table_cell_style), Paragraph("<b>P1</b>", table_cell_style), Paragraph("Data covers cities worldwide. UTM is only valid within 6° longitude bands; a local Euclidean tangent plane at the trip origin prevents global metric distortion.", table_cell_style)],
    [Paragraph("Why use GroupKFold instead of random KFold?", table_cell_style), Paragraph("<b>P2</b>", table_cell_style), Paragraph("Points within the same trip share vehicle speed, heading, and driver behavior. Random splitting leaks sequential context between train and test.", table_cell_style)],
    [Paragraph("Why did you pick HistGradientBoosting over Random Forest?", table_cell_style), Paragraph("<b>P2</b>", table_cell_style), Paragraph("HGB runs inference in 0.09s vs 0.26s for RF (3x faster), has identical Macro F1 (90.5%), lower ECE calibration error, and avoids severe train overfitting.", table_cell_style)],
    [Paragraph("Why did HMM distance-only beat classifier on some trips?", table_cell_style), Paragraph("<b>P3</b>", table_cell_style), Paragraph("Classifier v2 features already contained spatial connectivity info. Tuning emission weight down to 0.25 prevents double-counting and reached 93.9% Top-1.", table_cell_style)],
    [Paragraph("How does your decoder handle tunnels or GPS outages?", table_cell_style), Paragraph("<b>P3</b>", table_cell_style), Paragraph("Any time gap > 30 seconds triggers a gap reset, neutralizing transition penalties to avoid forcing disconnected arcs across unobserved intervals.", table_cell_style)],
    [Paragraph("Does the client ever see ground truth?", table_cell_style), Paragraph("<b>P4</b>", table_cell_style), Paragraph("Never. Ground truth .route stays on the server. The client only sends the trip ID; accuracy metrics are calculated server-side and returned as numbers.", table_cell_style)],
]
t_cheat = Table(cheat_data, colWidths=[120, 60, 340])
t_cheat.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), primary_color),
    ('GRID', (0, 0), (-1, -1), 0.5, border_color),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(t_cheat)

doc.build(story)
print("PathSnap Team Study Guide PDF successfully created at docs/PathSnap_Team_Study_Guide.pdf")
