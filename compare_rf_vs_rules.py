"""
Side-by-side comparison of the rule-based classifier (layer1_classifier.py +
layer2_integrity.py) and the Random Forest classifier (rf_classifier.py) on the
SAME signal.

Three tests:
  1. Your real data, no faults            -> do they agree, any false damage?
  2. Hand-placed faults (17 events)        -> the exact data the rules were first scored on
  3. 5 random hold-out scenarios           -> random ringdowns + faults, unseen by the forest

Needs layer1_classifier.py, layer2_integrity.py and rf_classifier.py in the same folder.

Usage:
    python3 compare_rf_vs_rules.py --data Real_data_malin_round_4.xlsx
"""
import argparse
import numpy as np
import pandas as pd

import layer1_classifier as L1
import layer2_integrity as L2
import rf_classifier as R

STATES = R.STATES


def train_forests(t, x, real_rd, trees, seed, n_scen=6):
    """Train both forests on random scenarios from the first part of the record."""
    ix = lambda s: int(np.searchsorted(t, s))
    k_end = ix(R.TRAIN_END) // R.M
    Xs, ys_state, ys_type = [], [], []
    for s in range(n_scen):
        y, rd, fault = R.make_scenario(t, x, real_rd, seed=seed * 100 + s + 1, t0=30, t1=R.TRAIN_END - 10)
        F = R.segment_features(y).iloc[:k_end]
        st, ty = R.segment_truth(rd, fault)
        Xs.append(F); ys_state.append(st[:k_end]); ys_type.append(ty[:k_end])
    X = pd.concat(Xs, ignore_index=True)
    y_state = np.concatenate(ys_state); y_type = np.concatenate(ys_type)
    A = R.new_forest(trees, seed).fit(X, y_state)
    B = R.new_forest(trees, seed).fit(X[y_type != ""], y_type[y_type != ""])
    return A, B, list(X.columns)


def event_scores(table, truth_codes):
    """events typed correctly per type + false-alarm segments (+/-1 s tolerance)"""
    ev, fa = L2.evaluate(table, truth_codes)
    out = {}
    for k in ["MV", "OL", "FDI"]:
        sub = ev[ev.truth == k]
        out[k] = (int(sub.correct.sum()), len(sub))
    return out, fa


def fmt_events(d):
    return ", ".join(f"{k} {a}/{b}" for k, (a, b) in d.items())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t, x = L1.load(a.data)
    ix = lambda s: int(np.searchsorted(t, s))
    seg0, info0 = L1.classify(t, x)
    real_rd = info0["rd"]

    print("Training the Random Forest (about 15 s) ...")
    A, B, cols = train_forests(t, x, real_rd, a.trees, a.seed)
    rows = []

    # ------------------------------------------------------------ test 1: real data
    rf_raw = R.predict_table(A, B, t, x, cols)
    ru_raw, _ = L2.classify_integrity(t, x)
    agree = 100 * (rf_raw.state.to_numpy() == ru_raw.state.to_numpy()).mean()
    print("\n1) YOUR REAL DATA (no faults)")
    print("   Rules        :", ru_raw.state.value_counts().to_dict(),
          "| damage found:", int((ru_raw.defect != "").sum()))
    print("   Random Forest:", rf_raw.state.value_counts().to_dict(),
          "| damage found:", int((rf_raw.defect != "").sum()))
    print(f"   agreement between the two: {agree:.2f} %")
    rows += [("Real data: NA / NR segments (rules)", f"{(ru_raw.state=='NA').sum()} / {(ru_raw.state=='NR').sum()}"),
             ("Real data: NA / NR segments (RF)", f"{(rf_raw.state=='NA').sum()} / {(rf_raw.state=='NR').sum()}"),
             ("Real data: damage segments found (rules / RF)", f"{(ru_raw.defect!='').sum()} / {(rf_raw.defect!='').sum()}"),
             ("Real data: agreement", f"{agree:.2f} %")]

    # ------------------------------------------------------------ test 2: hand-placed faults
    y_old, truth_old = L2.inject_typed(t, x)
    rf_old = R.predict_table(A, B, t, y_old, cols)
    ru_old, _ = L2.classify_integrity(t, y_old)
    ev_rf, fa_rf = event_scores(rf_old, truth_old)
    ev_ru, fa_ru = event_scores(ru_old, truth_old)
    print("\n2) HAND-PLACED FAULTS (17 events: 11 spikes, 3 missing blocks, 3 offsets)")
    print(f"   Rules        : {fmt_events(ev_ru)} | false-alarm segments: {fa_ru}")
    print(f"   Random Forest: {fmt_events(ev_rf)} | false-alarm segments: {fa_rf}")
    rows += [("Hand-placed faults: events typed correctly (rules)", fmt_events(ev_ru)),
             ("Hand-placed faults: events typed correctly (RF)", fmt_events(ev_rf)),
             ("Hand-placed faults: false-alarm segments (rules / RF)", f"{fa_ru} / {fa_rf}")]

    # ------------------------------------------------------------ test 3: random hold-out
    N_TEST = 5
    k0 = ix(R.TEST_START) // R.M
    truth_all, rf_all, ru_all = [], [], []
    tot = {"Rules": {}, "Random Forest": {}}
    fa_tot = {"Rules": 0, "Random Forest": 0}
    for ts in range(N_TEST):
        y_, rd_, f_ = R.make_scenario(t, x, real_rd, seed=99000 + ts,
                                      t0=R.TEST_START + 10, t1=t[-1] - 10, n_rd=5)
        st_, _ = R.segment_truth(rd_, f_)
        tb_rf = R.predict_table(A, B, t, y_, cols).iloc[k0:].reset_index(drop=True)
        tb_ru = L2.classify_integrity(t, y_)[0].iloc[k0:].reset_index(drop=True)
        truth_all.append(st_[k0:]); rf_all.append(tb_rf.state.to_numpy()); ru_all.append(tb_ru.state.to_numpy())
        for name, tb in [("Rules", tb_ru), ("Random Forest", tb_rf)]:
            ev, fa = event_scores(tb, f_[k0 * R.M:])
            fa_tot[name] += fa
            for k, (s_, c_) in ev.items():
                p, q = tot[name].get(k, (0, 0))
                tot[name][k] = (p + s_, q + c_)
    truth_all = np.concatenate(truth_all)
    sc_rf = R.strict_scores(np.concatenate(rf_all), truth_all)
    sc_ru = R.strict_scores(np.concatenate(ru_all), truth_all)
    print(f"\n3) RANDOM HOLD-OUT ({N_TEST} scenarios, t >= {R.TEST_START:.0f} s, unseen by the forest)")
    print(pd.DataFrame({"Rules": sc_ru, "Random Forest": sc_rf}).round(3).to_string())
    for name in ["Rules", "Random Forest"]:
        print(f"   {name:13s}: {fmt_events(tot[name])} | false-alarm segments: {fa_tot[name]}")
    rows += [("Random hold-out: state accuracy (rules / RF)", f"{sc_ru['state accuracy']:.3f} / {sc_rf['state accuracy']:.3f}"),
             ("Random hold-out: ringdown (NR) recall (rules / RF)", f"{sc_ru['recall NR']:.3f} / {sc_rf['recall NR']:.3f}"),
             ("Random hold-out: DR recall (rules / RF)", f"{sc_ru['recall DR']:.3f} / {sc_rf['recall DR']:.3f}"),
             ("Random hold-out: events typed correctly (rules)", fmt_events(tot["Rules"])),
             ("Random hold-out: events typed correctly (RF)", fmt_events(tot["Random Forest"])),
             ("Random hold-out: false-alarm segments (rules / RF)", f"{fa_tot['Rules']} / {fa_tot['Random Forest']}")]

    pd.DataFrame(rows, columns=["metric", "value"]).to_csv("comparison_results.csv", index=False)
    print("\nSaved: comparison_results.csv")
