"""
Random-Forest version of layers 1 and 2  (machine-learning alternative to the
rule-based scripts, standing in for the two TabPFN blocks of the diagram).

   Forest A  (layer 1): segment  ->  NA / NR / DA / DR
   Forest B  (layer 2): segment  ->  MV / OL / FDI      (only for DA / DR)

Idea:  every 1-second segment is turned into a row of numbers (features).
The forests LEARN the decision thresholds from labelled examples instead of me
choosing them by hand.

Where do labelled examples come from?  Your data has no faults, so they are made
here: damped-oscillation "ringdowns" and faults (spikes, missing values, offsets)
are injected at random places, and the exact position/type of each is recorded.
Training uses only the first part of the record (t < 1850 s); testing uses the
later part (t >= 1950 s) with different random faults the forests never saw.

Needs layer1_classifier.py and layer2_integrity.py in the same folder.

Usage:
    python3 rf_classifier.py --data Real_data_malin_round_4.xlsx
"""
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import RandomForestClassifier

import layer1_classifier as L1
import layer2_integrity as L2

FS = L1.FS
M = int(L1.SEG * FS)                   # samples per 1-s segment
STATES = ["NA", "NR", "DA", "DR"]
TYPES = {1: "MV", 2: "OL", 3: "FDI"}
TRAIN_END, TEST_START = 1850.0, 1950.0  # time split of the record (s)


# ==========================================================================
# 1. FEATURES  (numbers that describe each 1-second segment)
# ==========================================================================
def sample_series(x):
    """Continuous 'evidence' signals, one value per sample (no thresholds)."""
    n = len(x)
    nan = ~np.isfinite(x)
    xf = pd.Series(x).interpolate(limit_direction="both").to_numpy().copy()
    d = np.r_[0.0, np.diff(xf)]
    dsig = np.maximum(1.4826 * L1._roll_med(np.abs(d), int(30 * FS)), 1e-3)   # normal noise size

    # flat runs: length of the run of identical values each sample belongs to
    flat = np.r_[False, d[1:] == 0]
    flen = np.zeros(n)
    for a, b in zip(*L1._runs(flat)):
        flen[max(a - 1, 0):b] = b - max(a - 1, 0)

    # spike evidence: stands out from the samples 3 steps before AND after
    g = 3
    rl = np.zeros(n); rr = np.zeros(n)
    rl[g:] = xf[g:] - xf[:-g]
    rr[:-g] = xf[:-g] - xf[g:]
    res = np.where(np.sign(rl) == np.sign(rr),
                   np.sign(rl) * np.minimum(np.abs(rl), np.abs(rr)), 0.0)
    spike_abs = np.abs(res)

    # ringdown evidence: band-limited energy (flat blocks bridged, 5-pt median)
    xb = xf.copy()
    for a, b in zip(*L1._runs(flat)):
        if b - a >= L1.FLAT_MIN:
            xb[max(a - 1, 0):b] = np.nan
    xb = pd.Series(xb).interpolate(limit_direction="both").to_numpy()
    xm = L1._roll_med(xb, 5)
    sos = butter(4, L1.BAND, btype="bandpass", fs=FS, output="sos")
    b = sosfiltfilt(sos, xm - np.median(xm))
    roll = lambda a_, w, f: getattr(pd.Series(a_).rolling(int(w * FS), center=True, min_periods=1), f)().to_numpy()
    sta = np.sqrt(roll(b ** 2, L1.STA_WIN, "mean"))
    lta = roll(sta, L1.LTA_WIN, "median")
    ratio = sta / np.maximum(lta, 1e-6)
    env = roll(np.abs(b), L1.STA_WIN, "max")
    sg = np.signbit(b)
    cross = np.r_[0, (sg[1:] != sg[:-1])].astype(float)
    zc = roll(cross, 10, "sum")                      # zero crossings in +/-5 s: oscillation vs step

    # level evidence: how far is this second from its surroundings?
    xs = L1._roll_med(xf, 5)
    m1 = L1._roll_med(xs, int(1 * FS))
    dev60 = np.abs(m1 - L1._roll_med(xs, int(60 * FS)))
    dev120 = np.abs(m1 - L1._roll_med(xs, int(120 * FS)))
    w = int(2 * FS)
    ser = pd.Series(xs)
    before = ser.rolling(w, min_periods=w // 2).median().shift(1).to_numpy()
    after = ser[::-1].rolling(w, min_periods=w // 2).median().shift(1)[::-1].to_numpy()
    step = np.nan_to_num(np.abs(after - before))
    nrm = lambda v: v / max(1.4826 * np.median(v), 1e-3)

    return dict(nan=nan.astype(float), zero=(np.nan_to_num(x, nan=1.0) == 0).astype(float),
                flen=flen, spike_abs=spike_abs, spike_rel=spike_abs / dsig,
                spike_env=spike_abs / (env + 1.0), dn=np.abs(d) / dsig,
                ratio=ratio, sta=sta, env=env, zc=zc,
                dev60=nrm(dev60), dev120=nrm(dev120), step=nrm(step))


CTX_KEYS = ["nanfrac", "zerofrac", "flat_max", "spike_rel", "ratio_max",
            "zc_mean", "dev60_mean", "dev120_mean", "step_max"]
SHIFTS = (-4, -2, -1, 1, 2, 4)           # neighbouring seconds the model may look at


def segment_features(x):
    S = sample_series(x)
    nseg = len(x) // M
    ag = lambda a, f: f(a[:nseg * M].reshape(nseg, M), axis=1)
    F = pd.DataFrame({
        "nanfrac": ag(S["nan"], np.mean), "zerofrac": ag(S["zero"], np.mean),
        "flat_max": ag(S["flen"], np.max),
        "spike_abs": ag(S["spike_abs"], np.max), "spike_rel": ag(S["spike_rel"], np.max),
        "spike_env": ag(S["spike_env"], np.max),
        "dn_mean": ag(S["dn"], np.mean), "dn_max": ag(S["dn"], np.max),
        "ratio_max": ag(S["ratio"], np.max), "ratio_mean": ag(S["ratio"], np.mean),
        "sta_mean": ag(S["sta"], np.mean), "env_max": ag(S["env"], np.max),
        "zc_mean": ag(S["zc"], np.mean),
        "dev60_mean": ag(S["dev60"], np.mean), "dev120_mean": ag(S["dev120"], np.mean),
        "step_max": ag(S["step"], np.max),
    })
    ctx = {}
    for k in CTX_KEYS:                    # add the same numbers from neighbouring seconds
        v = F[k].to_numpy()
        for s in SHIFTS:
            idx = np.clip(np.arange(nseg) - s, 0, nseg - 1)
            ctx[f"{k}@{s:+d}"] = v[idx]
    return pd.concat([F, pd.DataFrame(ctx)], axis=1)


# ==========================================================================
# 2. TRAINING DATA: random ringdowns + faults with known truth
# ==========================================================================
def make_scenario(t, x, real_rd, seed, t0, t1, n_rd=5):
    """Copy of the record with random synthetic ringdowns and faults inside
    [t0, t1].  Returns signal y, per-sample ringdown truth, per-sample fault
    type (0 clean, 1 MV, 2 OL, 3 FDI)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    y = x.copy()
    rd = real_rd.copy()
    fault = np.zeros(n, int)
    ix = lambda s: int(np.searchsorted(t, s))

    # --- synthetic ringdowns: damped sinusoid  A e^{-sigma t} sin(2 pi f t)
    real_on = t[np.flatnonzero(np.diff(real_rd.astype(int)) == 1)]
    used = list(real_on)
    for _ in range(n_rd):
        for _try in range(50):
            on = rng.uniform(t0 + 20, t1 - 45)
            if all(abs(on - u) > 60 for u in used):
                break
        used.append(on)
        A, f, sg = rng.uniform(30, 160), rng.uniform(0.3, 1.2), rng.uniform(0.1, 0.3)
        T = min(np.log(A / 5) / sg, 25.0)        # until the swing falls to ~5 MW
        i, j = ix(on), ix(on + T + 8)
        tau = t[i:j] - t[i]
        y[i:j] += A * np.exp(-sg * tau) * np.sin(2 * np.pi * f * tau)
        rd[i:ix(on + T)] = True
        # damage INSIDE the ringdown too, so the models see DR examples
        if rng.random() < 0.7:                                       # a spike
            k = ix(on + rng.uniform(1.0, max(T - 1.0, 2.0)))
            w_ = rng.integers(1, 3)
            y[k:k + w_] += rng.choice([-1, 1]) * np.exp(rng.uniform(np.log(25), np.log(450)))
            fault[k:k + w_] = 2
        if rng.random() < 0.3:                                       # a short gap
            k = ix(on + rng.uniform(2.0, max(T - 3.0, 3.0)))
            gap = int(rng.uniform(1, 3) * FS)
            y[k:k + gap] = np.nan
            fault[k:k + gap] = 1

    # --- faults
    def place(dur):
        for _try in range(40):
            s = rng.uniform(t0, t1 - dur)
            i, j = ix(s), ix(s + dur)
            if not fault[max(i - 90, 0):j + 90].any():
                return i, max(j, i + 1)
        return None

    for _ in range(25):                                              # OL spikes
        p = place(0.2)
        if p:
            i, _j = p
            w = rng.integers(1, 3)
            amp = rng.choice([-1, 1]) * np.exp(rng.uniform(np.log(15), np.log(450)))
            y[i:i + w] += amp; fault[i:i + w] = 2
    for kind, cnt, lo, hi in [("nan", 5, 1, 8), ("zero", 4, 1, 5), ("stuck", 4, 2, 10)]:   # MV
        for _ in range(cnt):
            p = place(rng.uniform(lo, hi))
            if p:
                i, j = p
                if kind == "nan":
                    y[i:j] = np.nan
                elif kind == "zero":
                    y[i:j] = 0.0
                else:
                    y[i:j] = y[i]
                fault[i:j] = 1
    for _ in range(8):                                               # FDI offsets
        p = place(rng.uniform(8, 25))
        if p:
            i, j = p
            y[i:j] += rng.choice([-1, 1]) * rng.uniform(15, 60); fault[i:j] = 3
    return y, rd, fault


def segment_truth(rd, fault):
    nseg = len(rd) // M
    rdseg = rd[:nseg * M].reshape(nseg, M).mean(1) > 0.5
    ft = fault[:nseg * M].reshape(nseg, M)
    deg = (ft > 0).any(1)
    state = np.array(STATES)[rdseg.astype(int) + 2 * deg.astype(int)]
    typ = np.where((ft == 1).any(1), "MV", np.where((ft == 3).any(1), "FDI",
                   np.where((ft == 2).any(1), "OL", "")))
    return state, typ


# ==========================================================================
# 3. MODELS
# ==========================================================================
def new_forest(trees, seed):
    return RandomForestClassifier(n_estimators=trees, min_samples_leaf=2, n_jobs=-1,
                                  class_weight="balanced_subsample", random_state=seed)


def predict_table(modelA, modelB, t, x, cols):
    F = segment_features(x)[cols]
    state = modelA.predict(F)
    defect = np.array([""] * len(F), dtype=object)
    deg = np.isin(state, ["DA", "DR"])
    if deg.any():
        defect[deg] = modelB.predict(F[deg])
    return pd.DataFrame({"t": t[:len(F) * M:M], "state": state, "defect": defect})


def strict_scores(pred_state, true_state):
    out = {}
    out["state accuracy"] = float((pred_state == true_state).mean())
    for c in STATES:
        tc = true_state == c
        out[f"recall {c}"] = float((pred_state[tc] == c).mean()) if tc.any() else np.nan
    return out


# ==========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--train-scenarios", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t, x = L1.load(a.data)
    seg0, info0 = L1.classify(t, x)           # only used to mark the 4 REAL ringdowns
    real_rd = info0["rd"]
    ix = lambda s: int(np.searchsorted(t, s))

    # ---------------- build training set (first part of record only)
    print(f"Building {a.train_scenarios} random training scenarios (t < {TRAIN_END:.0f} s) ...")
    Xs, ys_state, ys_type = [], [], []
    k_end = ix(TRAIN_END) // M
    for s in range(a.train_scenarios):
        y, rd, fault = make_scenario(t, x, real_rd, seed=a.seed * 100 + s + 1, t0=30, t1=TRAIN_END - 10)
        F = segment_features(y).iloc[:k_end]
        st, ty = segment_truth(rd, fault)
        Xs.append(F); ys_state.append(st[:k_end]); ys_type.append(ty[:k_end])
    X = pd.concat(Xs, ignore_index=True)
    cols = list(X.columns)
    y_state = np.concatenate(ys_state); y_type = np.concatenate(ys_type)
    print("  training rows:", len(X), " class counts:", pd.Series(y_state).value_counts().to_dict())

    # ---------------- fit the two forests
    A = new_forest(a.trees, a.seed).fit(X, y_state)
    deg = y_type != ""
    B = new_forest(a.trees, a.seed).fit(X[deg], y_type[deg])
    print("  Forest A (state) and Forest B (damage type) trained.")

    # ---------------- hold-out test: later part of the record, unseen faults
    # five independent hold-out scenarios (different random ringdowns and faults);
    # scoring on several avoids drawing conclusions from one lucky / unlucky draw
    N_TEST = 5
    k0 = ix(TEST_START) // M
    all_truth, all_rf, all_rules = [], [], []
    ev_tot = {"Random Forest": {}, "Rules": {}}
    fa_tot = {"Random Forest": 0, "Rules": 0}
    for ts in range(N_TEST):
        yt_, rdt_, ft_ = make_scenario(t, x, real_rd, seed=99000 + ts, t0=TEST_START + 10, t1=t[-1] - 10, n_rd=5)
        tab_ = predict_table(A, B, t, yt_, cols)
        st_, _ = segment_truth(rdt_, ft_)
        tabT_ = tab_.iloc[k0:].reset_index(drop=True)
        tabR_, _ = L2.classify_integrity(t, yt_)
        tabR_ = tabR_.iloc[k0:].reset_index(drop=True)
        all_truth.append(st_[k0:]); all_rf.append(tabT_.state.to_numpy()); all_rules.append(tabR_.state.to_numpy())
        for name, tb in [("Random Forest", tabT_), ("Rules", tabR_)]:
            ev, fa = L2.evaluate(tb, ft_[k0 * M:])
            fa_tot[name] += fa
            for k, g_ in ev.groupby("truth").correct:
                s_, c_ = ev_tot[name].get(k, (0, 0))
                ev_tot[name][k] = (s_ + int(g_.sum()), c_ + len(g_))
        if ts == 0:                              # keep the first one for the plot
            yt, tab = yt_, tab_
    stT = np.concatenate(all_truth); rfT = np.concatenate(all_rf); ruT = np.concatenate(all_rules)
    print("\n=== HOLD-OUT TEST: %d scenarios, t >= %.0f s, faults / ringdowns never seen in training ===" % (N_TEST, TEST_START))
    print("Layer 1 – truth (rows) vs Random Forest (columns), 1-s segments summed over scenarios:")
    print(pd.crosstab(pd.Series(stT, name="truth"), pd.Series(rfT, name="RF"))
          .reindex(index=STATES, columns=STATES, fill_value=0).to_string())
    cmp = pd.DataFrame({"Random Forest": strict_scores(rfT, stT), "Rules": strict_scores(ruT, stT)}).round(3)
    print("\nRandom Forest vs rule-based on the SAME hold-out data:")
    print(cmp.to_string())
    for name in ["Random Forest", "Rules"]:
        print(f"\nLayer 2 – {name}: injected damage events found AND typed correctly")
        for k in ["MV", "OL", "FDI"]:
            s_, c_ = ev_tot[name].get(k, (0, 0))
            print(f"   {k:3s}: {s_} of {c_}")
        print(f"   false-alarm segments: {fa_tot[name]}")

    # ---------------- your REAL data (no faults expected)
    tabRaw = predict_table(A, B, t, x, cols)
    tabRaw.to_csv("labels_rf_raw.csv", index=False)
    rules_raw = seg0.label.to_numpy()
    print("\n=== YOUR REAL DATA (no faults expected) ===")
    print("Random Forest state counts:", tabRaw.state.value_counts().to_dict())
    print("Rule-based    state counts:", pd.Series(rules_raw).value_counts().to_dict())
    print("Defect types found by RF   :", tabRaw.defect.replace("", "none").value_counts().to_dict())
    print("Segments where RF and rules agree: %.2f %%" % (100 * (tabRaw.state.to_numpy() == rules_raw).mean()))
    tabT_full = tab.copy()
    tabT_full.to_csv("labels_rf_test_injected.csv", index=False)

    # ---------------- feature importance
    imp = pd.DataFrame({"feature": cols, "forest_A_state": A.feature_importances_,
                        "forest_B_type": B.feature_importances_})
    imp.sort_values("forest_A_state", ascending=False).to_csv("rf_feature_importance.csv", index=False)

    # ---------------- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L2.plot(t, yt, tab, "result_rf_test_injected.png", (TEST_START, TEST_START + 1000),
            "Random Forest on HOLD-OUT data (random synthetic ringdowns and faults)")
    L2.plot(t, x, tabRaw, "result_rf_raw_200_1050.png", (200, 1050),
            "Random Forest on your data as given (no faults expected)")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for axx, col, ttl in [(ax[0], "forest_A_state", "Forest A – NA/NR/DA/DR"),
                          (ax[1], "forest_B_type", "Forest B – MV/OL/FDI")]:
        top = imp.sort_values(col, ascending=False).head(12)[::-1]
        axx.barh(top.feature, top[col], color="#3b6ea5"); axx.set_title(ttl, fontsize=10)
        axx.set_xlabel("importance")
    fig.tight_layout(); fig.savefig("result_rf_feature_importance.png", dpi=120); plt.close(fig)
    print("\nSaved: labels_rf_raw.csv, labels_rf_test_injected.csv, rf_feature_importance.csv,")
    print("       result_rf_test_injected.png, result_rf_raw_200_1050.png, result_rf_feature_importance.png")
