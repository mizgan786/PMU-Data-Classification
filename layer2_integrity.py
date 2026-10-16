"""
Layer 2 of the workflow: integrity classification of DEGRADED data.

Takes the segments that layer 1 called degraded (DA / DR) and says WHAT is
wrong with them:

    MV  = Missing Values         (gaps / NaN / dropouts / stuck-at-constant data)
    OL  = Outliers               (isolated spikes)
    FDI = False Data Injection   (level offset / step: values shifted, not random)

This is the "simpler algorithm" replacement for the second TabPFN block.
It reuses the detectors from layer1_classifier.py, which already finds each
kind of damage separately; here we just keep them apart and label by type.

Needs layer1_classifier.py in the same folder.

Usage:
    python3 layer2_integrity.py --data Real_data_malin_round_4.xlsx
"""
import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

import layer1_classifier as L1

FS = L1.FS
TYPES = ["MV", "OL", "FDI"]
# If one 1-s segment contains several kinds of damage, the more "structural"
# one wins:  missing data  >  injected offset  >  outlier.
PRIORITY = ["MV", "FDI", "OL"]


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def classify_integrity(t, x):
    """Returns (table, info).  table has one row per 1-s segment with
       t, state (NA/NR/DA/DR from layer 1) and defect ('' / MV / OL / FDI)."""
    seg1, info = L1.classify(t, x)

    masks = {
        "MV": info["missing"] | info["stuck"],
        "OL": info["spike"],
        "FDI": info["offset"],
    }
    m = int(L1.SEG * FS)
    nseg = len(seg1)
    defect = []
    for i in range(nseg):
        sl = slice(i * m, (i + 1) * m)
        found = [k for k in PRIORITY if masks[k][sl].any()]
        # only segments that layer 1 already called degraded get a type
        defect.append(found[0] if (found and seg1.label[i] in ("DA", "DR")) else "")
    out = seg1.rename(columns={"label": "state"})
    out["defect"] = defect
    return out, dict(info, masks=masks)


# --------------------------------------------------------------------------
# Synthetic corruption with a TYPE for every damaged sample (ground truth)
# --------------------------------------------------------------------------
def inject_typed(t, x):
    y = x.copy()
    truth = np.zeros(len(x), int)                 # 0 clean, 1 MV, 2 OL, 3 FDI
    ix = lambda s: int(np.searchsorted(t, s))

    # OL: spikes (2 samples wide). 904 s and 2686 s fall inside ringdowns.
    for s, amp in [(240, -80), (340, 300), (362, 350), (540, 450), (590, 60),
                   (785, 250), (880, -120), (904.0, 200), (1500, 150),
                   (2100, -200), (2686, 120)]:
        i = ix(s); y[i:i + 2] += amp; truth[i:i + 2] = 2

    # MV: NaN gap, zero-filled dropout, stuck-at-constant
    i, j = ix(425), ix(432); y[i:j] = np.nan; truth[i:j] = 1
    i, j = ix(660), ix(664); y[i:j] = 0.0; truth[i:j] = 1
    i, j = ix(1200), ix(1210); y[i:j] = y[i]; truth[i:j] = 1

    # FDI: constant offsets added for a while
    for a, b, off in [(480, 495, 40), (740, 760, -30), (1000, 1015, 40)]:
        i, j = ix(a), ix(b); y[i:j] += off; truth[i:j] = 3
    return y, truth


# --------------------------------------------------------------------------
# Evaluation: event level (one row per injected damage event)
# --------------------------------------------------------------------------
def evaluate(table, truth):
    m = int(L1.SEG * FS)
    n = len(table)
    # per-segment truth type (priority as above)
    tr_seg = np.zeros((n, 4), bool)
    tr = truth[:n * m].reshape(n, m)
    for k in (1, 2, 3):
        tr_seg[:, k] = (tr == k).any(1)
    pred = table["defect"].to_numpy()
    st = table["t"].to_numpy()

    rows = []
    name = {1: "MV", 2: "OL", 3: "FDI"}
    for k in (1, 2, 3):
        a, b = L1._runs(tr_seg[:, k])
        for s, e in zip(a, b):
            lo, hi = max(0, s - 1), min(n, e + 1)            # +/- 1 s tolerance
            got = sorted({p for p in pred[lo:hi] if p})
            rows.append(dict(t_start=round(st[s], 1), seconds=int(e - s), truth=name[k],
                             predicted="/".join(got) if got else "-missed-",
                             correct=name[k] in got and len(got) == 1))
    ev = pd.DataFrame(rows).sort_values("t_start").reset_index(drop=True)

    # false alarms: labelled segments with no true damage within +/-1 s
    any_truth = binary_dilation(tr_seg[:, 1:].any(1), iterations=1)
    fa = int(((pred != "") & ~any_truth).sum())
    return ev, fa


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
DCOL = {"MV": "#1f77b4", "OL": "#ff7f0e", "FDI": "#9467bd"}
SCOL = {"NA": "#33dd22", "NR": "#ffee22", "DA": "#a0522d", "DR": "#ee2222"}


def _spans(ax, st, lab, colors, xlim, ymin=0, ymax=1):
    k = 0
    while k < len(lab):
        j = k
        while j + 1 < len(lab) and lab[j + 1] == lab[k]:
            j += 1
        if lab[k] in colors and st[j] + 1 >= xlim[0] and st[k] <= xlim[1]:
            ax.axvspan(st[k], st[j] + 1.0, ymin=ymin, ymax=ymax, color=colors[lab[k]], lw=0, alpha=0.85)
        k = j + 1


def plot(t, x, table, fname, xlim, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 1]))
    st = table["t"].to_numpy()
    for ax, labcol, labs, ttl in [
            (a1, SCOL, table["state"].to_numpy(), "Layer 1 – operational state"),
            (a2, DCOL, table["defect"].to_numpy(), "Layer 2 – type of degradation")]:
        _spans(ax, st, labs, labcol, xlim)
        ax.plot(t, x, "k", lw=0.7)
        ax.set_xlim(*xlim)
        finite = x[np.isfinite(x) & (t >= xlim[0]) & (t <= xlim[1])]
        ax.set_ylim(min(finite.min(), 300) - 10, max(finite.max(), 600) + 20)
        ax.set_ylabel("Power (MW)"); ax.set_title(ttl, fontsize=10, loc="left")
    a1.legend(handles=[Patch(color=SCOL[k], label=n) for k, n in
                       [("NA", "NA ambient clean"), ("NR", "NR ringdown clean"),
                        ("DA", "DA ambient degraded"), ("DR", "DR ringdown degraded")]],
              loc="upper right", fontsize=7, ncol=2, framealpha=1)
    a2.legend(handles=[Patch(color=DCOL[k], label=n) for k, n in
                       [("MV", "MV missing values"), ("OL", "OL outliers"), ("FDI", "FDI false data injection")]],
              loc="upper right", fontsize=7, ncol=3, framealpha=1)
    a2.set_xlabel("Time (Sec)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv")
    a = ap.parse_args()

    t, x = L1.load(a.data)

    # 1) data exactly as given
    tab, _ = classify_integrity(t, x)
    tab.to_csv("labels_layer2_raw.csv", index=False)
    print("RAW DATA – defect counts:", tab["defect"].replace("", "none").value_counts().to_dict())

    # 2) synthetic corruption with known types
    y, truth = inject_typed(t, x)
    tab2, _ = classify_integrity(t, y)
    tab2.to_csv("labels_layer2_injected.csv", index=False)
    print("INJECTED DATA – layer 1 states:", tab2["state"].value_counts().to_dict())
    print("INJECTED DATA – defect counts :", tab2["defect"].replace("", "none").value_counts().to_dict())

    ev, fa = evaluate(tab2, truth)
    print("\nPer injected event (+/-1 s tolerance):")
    print(ev.to_string(index=False))
    print(f"\nEvents typed correctly: {int(ev.correct.sum())} of {len(ev)}")
    print(f"False-alarm segments  : {fa}")

    plot(t, y, tab2, "result_layer2_injected_200_1050.png", (200, 1050),
         "Layer 1 + Layer 2 on data with synthetic corruption")
    plot(t, x, tab, "result_layer2_raw_200_1050.png", (200, 1050),
         "Layer 1 + Layer 2 on data as given (no corruption expected)")
