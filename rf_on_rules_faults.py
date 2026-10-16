"""
Random Forest classifier run on the SAME hand-placed faults that the rule-based
classifier (layer1_classifier.py / layer2_integrity.py) was scored on.

The 17 faults (11 spikes, 3 missing-value blocks, 3 offset blocks) come from
layer2_integrity.inject_typed().  The forest is NOT trained on them: it is trained
on randomly generated ringdowns and faults (rf_classifier.make_scenario) and then
simply applied to this data, so the result is a fair like-for-like test.

Needs layer1_classifier.py, layer2_integrity.py and rf_classifier.py in the same folder.

Usage:
    python3 rf_on_rules_faults.py --data Real_data_malin_round_4.xlsx
"""
import argparse
import numpy as np
import pandas as pd

import layer1_classifier as L1
import layer2_integrity as L2
import rf_classifier as R

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data.csv")
ap.add_argument("--trees", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--train-scenarios", type=int, default=6)
a = ap.parse_args()

t, x = L1.load(a.data)
ix = lambda s: int(np.searchsorted(t, s))
seg0, info0 = L1.classify(t, x)          # only used to mark the 4 real ringdowns for training
real_rd = info0["rd"]

# ------------------------------------------------------------------ 1. train on RANDOM faults
print(f"Training on {a.train_scenarios} random scenarios (t < {R.TRAIN_END:.0f} s) ...")
k_end = ix(R.TRAIN_END) // R.M
Xs, ys_state, ys_type = [], [], []
for s in range(a.train_scenarios):
    y, rd, fault = R.make_scenario(t, x, real_rd, seed=a.seed * 100 + s + 1, t0=30, t1=R.TRAIN_END - 10)
    F = R.segment_features(y).iloc[:k_end]
    st, ty = R.segment_truth(rd, fault)
    Xs.append(F); ys_state.append(st[:k_end]); ys_type.append(ty[:k_end])
X = pd.concat(Xs, ignore_index=True)
cols = list(X.columns)
y_state = np.concatenate(ys_state); y_type = np.concatenate(ys_type)
A = R.new_forest(a.trees, a.seed).fit(X, y_state)                        # Forest A: NA/NR/DA/DR
B = R.new_forest(a.trees, a.seed).fit(X[y_type != ""], y_type[y_type != ""])  # Forest B: MV/OL/FDI
print("Done.\n")

# ------------------------------------------------------------------ 2. the SAME faults as the rules
y_old, truth_old = L2.inject_typed(t, x)          # identical injection used for the rule-based results
tab_rf = R.predict_table(A, B, t, y_old, cols)    # Random Forest labels
tab_ru, _ = L2.classify_integrity(t, y_old)       # rule-based labels (for comparison)
tab_rf.to_csv("labels_rf_handplaced.csv", index=False)

print("Layer 1 state counts on the hand-placed-fault data")
print("  Random Forest:", tab_rf.state.value_counts().to_dict())
print("  Rules        :", tab_ru.state.value_counts().to_dict())

# ------------------------------------------------------------------ 3. event table (same as layer 2)
ev, fa = L2.evaluate(tab_rf, truth_old)
ev_ru, fa_ru = L2.evaluate(tab_ru, truth_old)
ev = ev.rename(columns={"predicted": "RF_predicted", "correct": "RF_correct"})
ev["Rules_predicted"] = ev_ru["predicted"].to_numpy()
ev["Rules_correct"] = ev_ru["correct"].to_numpy()
print("\nPer injected event (+/-1 s tolerance):")
print(ev.to_string(index=False))
print(f"\nRandom Forest: events typed correctly {int(ev.RF_correct.sum())} of {len(ev)}, false-alarm segments: {fa}")
print(f"Rules        : events typed correctly {int(ev.Rules_correct.sum())} of {len(ev)}, false-alarm segments: {fa_ru}")

# ------------------------------------------------------------------ 4. plot (same window as the rule-based figure)
L2.plot(t, y_old, tab_rf, "result_rf_handplaced_200_1050.png", (200, 1050),
        "Random Forest on the SAME hand-placed faults used for the rule-based results")
print("\nSaved: labels_rf_handplaced.csv, result_rf_handplaced_200_1050.png")
print("Compare with: result_layer2_injected_200_1050.png (rule-based)")
