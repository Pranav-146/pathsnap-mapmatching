"""Report sections 5 (HMM / Viterbi) and 6 (Folium maps) (P3). Numbers come from results/ and artifacts/."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from report.common import ARTIFACTS, PAGE_W, RESULTS, H1, H2, H3, BODY, SMALL, P, bullets, f, img, pb, read, sp, table


def build(story: list):
    hc = read("hmm_comparison.csv")
    hold = read("hmm_holdout_train_trips.csv")
    log = read("hmm_tuning_log.csv")
    cfg = json.loads((ARTIFACTS / "hmm_config.json").read_text())
    p = cfg["params"]
    w = hc["n_label_valid"]
    pooled = lambda c, d=hc, ww=w: float((d[c] * ww).sum() / ww.sum())
    wh = hold["n_valid_points"]

    story.append(P("5. HMM and Viterbi route reconstruction (P3)", H1))
    story.append(P("Independent per-point predictions can form an impossible route (consecutive points on arcs that are not connected). The decoder adds the sequence: it chooses, for the whole trip, the "
                   "candidate sequence that maximises the product of how well each candidate fits its own GPS point (emission) and how plausible each move between consecutive points is (transition).", BODY))

    story.append(P("1. Hidden states", H3))
    story.append(P("The hidden state at GPS point i is one of that point's candidate arcs (at most k = 8 within 50 m). The state keeps its link to the road network (arc id, position along the arc, projected point), "
                   "so the decoded state sequence is directly a sequence of road segments, one per GPS point; no state is shared between points.", BODY))
    story.append(P("2. Emission probabilities", H3))
    story.append(P(f"The emission of a candidate is the classifier score P(arc is the reference arc), normalised across the candidates of its point so the emissions of one point sum to 1, and used in log space. "
                   f"A temperature (emission weight {p['emission_weight']:g}) multiplies the log emission, so values above 1 trust the classifier more than the road-network geometry. "
                   "Before a trained model existed, and as an ablation, a distance-only emission exp(-d<super>2</super> / 2 sigma<super>2</super>) was used, with sigma tuned separately "
                   f"(sigma = {p['sigma']:g} m).", BODY))
    story.append(P("3. Transition probabilities", H3))
    story.append(P(f"Between consecutive points the transition score is exp(-|straight-line distance - road-network distance| / beta), in log space -|d<sub>straight</sub> - d<sub>road</sub>| / beta, plus a bonus when the arc stays the same "
                   f"(+{p['same_bonus']:g}) or moves to an adjacent arc (+{p['adjacent_bonus']:g}). The road distance is the shortest-path distance on the trip's road graph (networkx, arc lengths in metres, cached per source node) between the "
                   f"projected positions on the two arcs. The straight-line distance is measured {'between the two candidates' + chr(39) + ' projected points' if p['straight_mode'] == 'projected' else 'between the two raw GPS fixes'}; "
                   f"beta = {p['beta']:g} m, widened in proportion to the time gap when two fixes are more than 1 s apart. The recurrence is written in log space with back-pointers."
                   + (" The tuning set both bonuses to 0: with the final classifier the same-arc and adjacency information is already carried by its continuity features "
                      "(connects_prev, connects_next, in_prev, in_next), so an extra bonus only double-counts it." if p["same_bonus"] == 0 and p["adjacent_bonus"] == 0 else ""), BODY))
    story.append(P("4. Missing candidates and large GPS gaps", H3))
    seg = hc.hmm_segments
    story.append(P(f"A point with no candidate in range is skipped by the decoder and takes the previous decoded arc (a point can only be right if the road is near it, so no state is invented). "
                   f"When two usable fixes are more than T = {p['gap_T']:g} s apart the trip is split and each piece is decoded on its own, so a long gap cannot force an unrealistic path; "
                   f"on the student_test trips this gave {int(seg.sum())} decoded segments over {len(hc)} trips (between {int(seg.min())} and {int(seg.max())} per trip).", BODY))
    story.append(P("5. Disconnected arcs", H3))
    story.append(P(f"If no road path exists within 1,500 m between two candidates (the road networks are fragmented into many components) the transition is not -infinity: the mismatch is capped at {p['disconnected_cap']:g} m, "
                   f"which costs {p['disconnected_cap'] / p['beta']:.0f} log units at beta = {p['beta']:g} m. The move is allowed but very unlikely, so decoding can never fail or become undefined and the decoder prefers any connected alternative.", BODY))

    story.append(P("Decoder parameters: tuned on train and student_test data only", H3))
    t = [["Parameter", "Value", "Search values"], ["beta (m)", f"{p['beta']:g}", "5, 10, 20, 40, 80, 160"], ["emission weight", f"{p['emission_weight']:g}", "0.1, 0.25, 0.5, 1, 2, 3"],
         ["same-arc bonus", f"{p['same_bonus']:g}", "0, 0.5, 1, 2"], ["adjacent-arc bonus", f"{p['adjacent_bonus']:g}", "0, 0.25, 0.5, 1"], ["gap threshold T (s)", f"{p['gap_T']:g}", "10, 30, 60, 120"],
         ["straight-line distance", p["straight_mode"], "projected, gps"], ["sigma for distance-only emission (m)", f"{p['sigma']:g}", "3, 5, 8, 12, 20, 30"]]
    story.append(table(t, [PAGE_W * 0.4, PAGE_W * 0.2, PAGE_W * 0.4]))
    first, tuned = log[log["round"] == 0].tuning_top1.iloc[0], cfg["tuning_top1_hmm_tuned"]
    story.append(P(f"Two rounds of coordinate descent maximised pooled pointwise Top-1 over a tuning set of {len(cfg['tuned_on']['train_trips (out-of-fold scores)'])} train trips (scored with out-of-fold classifier scores, so the classifier never saw the trip it scores) "
                   f"and the {len(cfg['tuned_on']['student_test_trips'])} student_test trips: Top-1 {first:.4f} with the default parameters, {tuned:.4f} after tuning, against {cfg['tuning_top1_classifier_only']:.4f} for the classifier alone on the same trips. "
                   f"The evaluator trips were not used. All {len(log)} evaluations are in results/hmm_tuning_log.csv.", BODY))

    # ---- 6 comparison
    story.append(P("6. Did HMM decoding improve the final route?", H3))
    story.append(P("Per-trip comparison on the 10 student_test trips with the final classifier (pointwise Top-1 over label-valid points, route continuity, and sequence-level precision / recall / F1 of the predicted arc set against the .route arcs):", BODY))
    t = [["Trip", "Points", "Top-1 classifier only", "Top-1 classifier + HMM", "Gain", "Continuity classifier only", "Continuity + HMM", "Arc F1 classifier only", "Arc F1 + HMM", "Arc precision", "Arc recall"]]
    for r in hc.itertuples():
        t.append([r.trip_id, f"{r.n_points:,}", f"{r.top1_classifier_only:.4f}", f"{r.top1_classifier_hmm:.4f}", f"{r.hmm_gain_top1:+.4f}", f"{r.continuity_classifier_only_pct:.2f}%",
                  f"{r.continuity_classifier_hmm_pct:.2f}%", f"{r.arc_f1_classifier_only:.3f}", f"{r.arc_f1_classifier_hmm:.3f}",
                  f"{r.arc_precision_classifier_only:.3f} -> {r.arc_precision_classifier_hmm:.3f}", f"{r.arc_recall_classifier_only:.3f} -> {r.arc_recall_classifier_hmm:.3f}"])
    t.append(["<b>pooled</b>", f"{int(hc.n_points.sum()):,}", f"{pooled('top1_classifier_only'):.4f}", f"{pooled('top1_classifier_hmm'):.4f}", f"{pooled('top1_classifier_hmm') - pooled('top1_classifier_only'):+.4f}",
              f"{pooled('continuity_classifier_only_pct'):.2f}%", f"{pooled('continuity_classifier_hmm_pct'):.2f}%", f"{hc.arc_f1_classifier_only.mean():.3f}", f"{hc.arc_f1_classifier_hmm.mean():.3f}", "", ""])
    story.append(table(t, [PAGE_W * x for x in (0.08, 0.06, 0.09, 0.09, 0.07, 0.09, 0.08, 0.08, 0.08, 0.14, 0.14)], font=6.4))
    story.append(P("Pooled values are weighted by label-valid points (continuity unweighted mean of trips for arc F1). Source: results/hmm_comparison.csv.", SMALL))

    story.append(P("Where the gain comes from (pooled Top-1 on student_test)", H3))
    t = [["Method", "Top-1"], ["Nearest arc (distance-only, no HMM)", f"{pooled('top1_nearest_arc'):.4f}"], ["HMM with distance-only emission", f"{pooled('top1_hmm_distance_only'):.4f}"],
         ["Classifier only (final model)", f"{pooled('top1_classifier_only'):.4f}"], ["<b>Classifier + HMM (final system)</b>", f"<b>{pooled('top1_classifier_hmm'):.4f}</b>"]]
    story.append(table(t, [PAGE_W * 0.6, PAGE_W * 0.2]))
    helped, worse = hc[hc.hmm_gain_top1 > 0], hc[hc.hmm_gain_top1 <= 0]
    best, small = hc.loc[hc.hmm_gain_top1.idxmax()], hc.loc[hc.hmm_gain_top1.idxmin()]
    txt = (f"<b>Yes, on the whole.</b> Pooled Top-1 rises from {pooled('top1_classifier_only'):.4f} to {pooled('top1_classifier_hmm'):.4f} "
           f"({pooled('top1_classifier_hmm') - pooled('top1_classifier_only'):+.4f}), and route continuity from {pooled('continuity_classifier_only_pct'):.2f}% to {pooled('continuity_classifier_hmm_pct'):.2f}%. "
           f"The HMM improved Top-1 on {len(helped)} of {len(hc)} trips; the largest gain is on trip {best.trip_id} ({best.top1_classifier_only:.4f} to {best.top1_classifier_hmm:.4f}).")
    if len(worse):
        txt += " <b>Where it did not help:</b> " + "; ".join(f"trip {r.trip_id} ({r.top1_classifier_only:.4f} to {r.top1_classifier_hmm:.4f}, {r.hmm_gain_top1:+.4f})" for r in worse.itertuples()) + \
               ". On these trips the classifier alone was already strong, so the sequence constraint has little to correct, and a wrong junction choice that the transition model finds plausible can be locked in for several points."
    else:
        txt += f" The smallest gain is on trip {small.trip_id} ({small.hmm_gain_top1:+.4f}); there the classifier alone was already strong, so the sequence constraint has little to correct."
    story.append(P(txt, BODY))
    d_only, c_hmm = pooled("top1_hmm_distance_only"), pooled("top1_classifier_hmm")
    h_d = float((hold["top1_hmm_distance_only"] * wh).sum() / wh.sum())
    h_c = float((hold["top1_hmm_tuned_params"] * wh).sum() / wh.sum())
    if d_only > c_hmm:
        story.append(P(f"<b>Classifier emission vs distance-only emission.</b> On student_test the HMM with a distance-only emission scores {d_only:.4f}, above the final classifier + HMM "
                       f"({c_hmm:.4f}). On the 12 held-out train trips the order is reversed ({h_c:.4f} for classifier + HMM against {h_d:.4f} for distance-only). "
                       "So neither emission dominates on every partition. The classifier's context features (in_prev, in_next, dist_win5) already encode some of the sequence information "
                       "the HMM adds, which is why the tuning chose a low emission weight. We keep the classifier emission because the brief asks the decoder to use the selected model's outputs, "
                       "and because it is the better variant on trips that were not used to tune the decoder.", BODY))
    # sequence-level
    story.append(P(f"<b>Sequence level.</b> Treating the predicted arcs as a set and comparing it with the arcs of the .route, mean arc precision rises from {hc.arc_precision_classifier_only.mean():.3f} to {hc.arc_precision_classifier_hmm.mean():.3f}: "
                   f"the classifier alone scatters wrong arcs (look-alike streets, the reverse direction) that do not belong to the route and the decoder removes most of them; mean arc recall changes from "
                   f"{hc.arc_recall_classifier_only.mean():.3f} to {hc.arc_recall_classifier_hmm.mean():.3f}.", BODY))
    # hold out
    t = [["Method (12 held-out train trips, out-of-fold scores)", "Pooled Top-1"]]
    for c, lab in (("top1_classifier_only", "Classifier only"), ("top1_hmm_default_params", "Classifier + HMM, default parameters"), ("top1_hmm_tuned_params", "Classifier + HMM, tuned parameters"), ("top1_hmm_distance_only", "HMM with distance-only emission")):
        t.append([lab, f"{float((hold[c] * wh).sum() / wh.sum()):.4f}"])
    story += [P("Check on trips that were not used for tuning", H3),
              P("The decoder parameters were tuned using the student_test trips, so the student_test comparison above is slightly optimistic for the HMM. To check this, 12 other train trips that were never used for tuning "
                "were decoded with out-of-fold classifier scores:", BODY), table(t, [PAGE_W * 0.7, PAGE_W * 0.2]),
              P("Source: results/hmm_holdout_train_trips.csv.", SMALL)]
    story += img(RESULTS / "plots" / "hmm_comparison.png", "Figure 5.1 Pointwise Top-1 per student_test trip: nearest arc, classifier only, classifier + HMM.", PAGE_W * 0.9)
    story.append(pb())

    # ---- 6 Folium
    story.append(P("6. Geographical visualisation with Folium (P3)", H1))
    story.append(P("One interactive Folium map is generated for every student_test trip (visualisations/folium_maps.py, files results/maps/trip_&lt;id&gt;_hmm.html; the HTML files are committed). Layers, each with its own colour and a layer-control entry:", BODY))
    story += bullets(["<font color='#1f1fff'><b>blue</b></font> original GPS points (every n-th point is drawn on very long trips to keep the file light);",
                      "<font color='#777777'><b>grey</b></font> road-network geometry around the trip;",
                      "<font color='#008000'><b>green</b></font> reference route;",
                      "<font color='#ff8c00'><b>orange</b></font> classifier-only predicted route;",
                      "<font color='#ff0000'><b>red</b></font> classifier + HMM predicted route; a legend names every layer and the map is fitted to the trajectory."])
    S = RESULTS / "screenshots"
    story += img(S / "folium_00000052.png", "Figure 6.1 Folium map of student_test trip 00000052 (screenshot of results/maps/trip_00000052_hmm.html).", PAGE_W)
    story += img(S / "folium_00000009.png", "Figure 6.2 Folium map of student_test trip 00000009: the red HMM route versus the orange classifier-only route and the green reference route.", PAGE_W)
    story.append(P("The maps are produced by the same code the API uses (hmm.pipeline.run_trip), from the real predictions; none is edited by hand.", SMALL))
    story.append(pb())
