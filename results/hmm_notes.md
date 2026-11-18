# HMM vs classifier-only on student_test (auto-generated from results/hmm_comparison.csv)

Decoder parameters: `{"sigma": 3.0, "beta": 5.0, "gap_T": 30.0, "same_bonus": 1.0, "adjacent_bonus": 0.5, "disconnected_cap": 500.0, "straight_mode": "projected", "emission_weight": 0.25}`

Pooled pointwise Top-1 (weighted by label-valid points):

* top1_nearest_arc: 0.5344
* top1_hmm_distance_only: 0.9407
* top1_classifier_only: 0.8626
* top1_classifier_hmm: 0.9388

HMM improved Top-1 on 10 of 10 trips.
