"""
Layer 1 of the workflow: operational-state classification of PMU data
into  NA (Normal-Ambient), NR (Normal-Ringdown), DA (Degraded-Ambient),
DR (Degraded-Ringdown).

This is the "simpler algorithm" version of the TabPFN block: two rule/feature
based decisions on 1-second segments.

  1. Ringdown vs Ambient  -> STA/LTA energy detector in the modal band
  2. Normal  vs Degraded  -> integrity checks (missing/flat, spikes, level jump)

Usage:
    python3 layer1_classifier.py                 # run on the raw file as given
    python3 layer1_classifier.py --inject        # also run on a synthetically
                                                 # corrupted copy (with ground truth)
"""
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import binary_dilation, binary_closing

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
FS = 30.0                      # PMU reporting rate (Hz)  (dt = 0.033 s in file)
SEG = 1.0                      # classification segment length (s)

# ringdown detector
BAND = (0.2, 2.0)              # modal band (Hz)
STA_WIN = 2.0                  # short-term RMS window (s)
LTA_WIN = 120.0                # long-term (median) window (s)
RD_ON = 5.0                    # STA/LTA to declare a ringdown
RD_OFF = 4.0                   # STA/LTA at which the ringdown has decayed
RD_MIN_ABS = 3.0               # min band-RMS (MW) so tiny noise never triggers
RD_MIN_XINGS = 6               # min zero-crossings of the band signal in an event
RD_MAX_LEN = 40.0              # safety cap on one event (s)

# integrity checks
SPIKE_RD_FRAC = 0.5            # inside ringdown: spike must exceed this x local swing
SPIKE_K = 7.0                  # threshold in robust sigmas
SPIKE_MIN_ABS = 3.0            # ... and at least this many MW
FLAT_MIN = 8                   # samples of identical value => stuck / missing
JUMP_K = 5.0                   # level-jump threshold in robust sigmas
JUMP_PAIR_MAX = 120.0          # max length (s) of an offset block
JUMP_WIN = 2.0                 # window (s) for level estimates
LABELS = ["NA", "NR", "DA", "DR"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load(path):
    d = np.loadtxt(path, delimiter=",") if path.endswith(".csv") else \
        pd.read_excel(path, header=None).to_numpy(float)
    return d[:, 0], d[:, 1]


def _roll_med(x, n):
    return pd.Series(x).rolling(n, center=True, min_periods=1).median().to_numpy()


def _runs(mask):
    """start, stop (exclusive) of each True run."""
    m = np.r_[0, mask.astype(int), 0]
    e = np.flatnonzero(np.diff(m))
    return e[::2], e[1::2]


# --------------------------------------------------------------------------
# Ringdown detection  -> boolean mask per sample
#   Works on a 5-point-median copy of the signal, which removes 1-2 sample
#   spikes but keeps genuine oscillations, so corruption cannot fake (or hide)
#   a ringdown.
# --------------------------------------------------------------------------
def ringdown_mask(x):
    n = len(x)
    xf = pd.Series(x).interpolate(limit_direction="both").to_numpy().copy()
    # stuck / zero-filled blocks are dropouts, not oscillations: bridge over them,
    # otherwise the sudden drop (e.g. to 0) rings the band-pass like a ringdown
    for a_, b_ in zip(*_runs(np.r_[False, np.diff(xf) == 0])):
        if b_ - a_ >= FLAT_MIN:
            xf[a_ - 1:b_] = np.nan
    xf = pd.Series(xf).interpolate(limit_direction="both").to_numpy()
    xm = _roll_med(xf, 5)
    sos = butter(4, BAND, btype="bandpass", fs=FS, output="sos")
    b = sosfiltfilt(sos, xm - np.median(xm))
    sta = np.sqrt(pd.Series(b ** 2).rolling(int(STA_WIN * FS), center=True, min_periods=1).mean().to_numpy())
    lta = pd.Series(sta).rolling(int(LTA_WIN * FS), center=True, min_periods=1).median().to_numpy()
    ratio = sta / np.maximum(lta, 1e-6)
    env = pd.Series(np.abs(b)).rolling(int(STA_WIN * FS), center=True, min_periods=1).max().to_numpy()

    trig = (ratio > RD_ON) & (sta > RD_MIN_ABS)
    mask = np.zeros(n, bool)
    s, e = _runs(trig)
    for a, _ in zip(s, e):
        j = a                                   # extend until the response has decayed
        while j < n - 1 and ratio[j] > RD_OFF and (j - a) < RD_MAX_LEN * FS:
            j += 1
        a0 = max(0, a - int(1.0 * FS))          # catch the onset (filter / STA smear)
        # a ringdown OSCILLATES: a step / level change rings the band-pass only once or twice
        seg_b = b[a0:j + 1]
        if np.count_nonzero(np.diff(np.signbit(seg_b))) < RD_MIN_XINGS:
            continue
        mask[a0:j + 1] = True
    return mask, ratio, env


# --------------------------------------------------------------------------
# Integrity (degraded-data) detection  -> boolean mask per sample
# --------------------------------------------------------------------------
def integrity_mask(x, rd, env):
    n = len(x)
    bad = np.zeros(n, bool)

    # a) missing values / stuck (flat) data
    nan = ~np.isfinite(x)
    bad |= nan
    xf = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    flat = np.r_[False, np.diff(xf) == 0]
    stuck = np.zeros(n, bool)                  # kept separately for layer 2
    s, e = _runs(flat)
    for a, b in zip(s, e):
        if b - a >= FLAT_MIN:
            stuck[a - 1:b] = True
    bad |= stuck

    # b) spikes / outliers: deviation from the mean of the samples 3 steps
    #    either side (immune to spikes up to 2 samples wide).
    g = 3
    rl = np.zeros(n); rr = np.zeros(n)
    rl[g:] = xf[g:] - xf[:-g]                  # difference to the sample g before
    rr[:-g] = xf[:-g] - xf[g:]                 # difference to the sample g after
    # a spike stands out from BOTH sides in the same direction; sign-agreeing min
    res = np.where(np.sign(rl) == np.sign(rr), np.sign(rl) * np.minimum(np.abs(rl), np.abs(rr)), 0.0)
    sig = np.maximum(1.4826 * _roll_med(np.abs(res), int(30 * FS)), 1e-3)
    thr = np.maximum(SPIKE_K * sig, SPIKE_MIN_ABS)
    # inside a ringdown the curvature of the oscillation itself produces
    # residuals, so the threshold is raised to a fraction of the local swing
    thr = np.where(rd, np.maximum(thr, SPIKE_RD_FRAC * env), thr)
    spike = np.abs(res) > thr
    bad |= spike

    # c) abrupt level jump / offset (false-data-injection style steps) --
    #    compare 2-s medians just before/after each sample; not applied inside
    #    a ringdown (swings look like steps there).
    xc = xf.copy(); xc[bad] = np.nan
    w = int(JUMP_WIN * FS)
    ser = pd.Series(xc)
    before = ser.rolling(w, min_periods=w // 2).median().shift(1).to_numpy()
    after = ser[::-1].rolling(w, min_periods=w // 2).median().shift(1)[::-1].to_numpy()
    step = np.abs(after - before)
    ok = np.isfinite(step)
    sc = 1.4826 * np.median(step[ok])
    jump = np.zeros(n, bool)
    jump[ok] = step[ok] > JUMP_K * sc
    jump &= ~binary_dilation(rd, iterations=w)
    offset = binary_dilation(jump, iterations=int(JUMP_WIN * FS / 2))   # kept separately for layer 2
    # an offset block = an up-step followed by a matching down-step: flag in between
    signed = after - before
    ev = []
    for a_, b_ in zip(*_runs(jump)):
        k = a_ + int(np.argmax(np.abs(signed[a_:b_])))
        ev.append((k, signed[k]))
    for (k1, v1), (k2, v2) in zip(ev[:-1], ev[1:]):
        if np.sign(v1) != np.sign(v2) and 0.5 < abs(v1 / v2) < 2 and (k2 - k1) < JUMP_PAIR_MAX * FS:
            offset[k1:k2 + 1] = True
    bad |= offset
    return bad, dict(missing=nan, stuck=stuck, spike=spike, jump=jump, offset=offset)


# --------------------------------------------------------------------------
# Classify 1-s segments
# --------------------------------------------------------------------------
def classify(t, x):
    rd, ratio, env = ringdown_mask(x)
    bad, parts = integrity_mask(x, rd, env)

    m = int(SEG * FS)
    nseg = len(x) // m
    seg_t = t[:nseg * m:m]
    lab = np.zeros(nseg, int)             # 0 NA, 1 NR, 2 DA, 3 DR
    for i in range(nseg):
        sl = slice(i * m, (i + 1) * m)
        lab[i] = (1 if rd[sl].mean() > 0.5 else 0) + (2 if bad[sl].any() else 0)
    return pd.DataFrame({"t": seg_t, "label": [LABELS[k] for k in lab]}), \
        dict(bad=bad, rd=rd, ratio=ratio, **parts)


# --------------------------------------------------------------------------
# Synthetic corruption (ground truth) – only for validation / reproducing the
# "degraded" look of the reference figure.
# --------------------------------------------------------------------------
def inject(t, x, seed=1):
    rng = np.random.default_rng(seed)
    y = x.copy()
    truth = np.zeros(len(x), bool)
    ix = lambda s: int(np.searchsorted(t, s))     # time axis in the file is 0.033 s/sample

    # outlier spikes (single / few samples, large amplitude)
    for s, amp in [(240, -80), (340, 300), (362, 350), (540, 450), (590, 60),
                   (785, 250), (880, -120), (904.0, 200), (1500, 150), (2100, -200)]:
        i = ix(s); y[i:i + 2] += amp; truth[i:i + 2] = True
    # missing block (NaN)
    i, j = ix(425), ix(432); y[i:j] = np.nan; truth[i:j] = True
    # false-data-injection: constant offset block
    i, j = ix(1000), ix(1015); y[i:j] += 40; truth[i - 1:j + 1] = True
    # stuck value
    i, j = ix(1200), ix(1210); y[i:j] = y[i]; truth[i - 1:j + 1] = True
    # spike inside a ringdown (DR)
    i = ix(2686); y[i:i + 2] += 120; truth[i:i + 2] = True
    return y, truth


# --------------------------------------------------------------------------
# Plot in the style of the reference figure
# --------------------------------------------------------------------------
COL = {"NA": "#33dd22", "NR": "#ffee22", "DA": "#a0522d", "DR": "#ee2222"}
NAME = {"NA": "Ambient Clean", "NR": "Ringdown Clean",
        "DA": "Ambient Corrupted", "DR": "Ringdown Corrupted"}


def plot(t, x, seg, fname, xlim=(200, 950), ylim=(300, 950), title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(13, 4.6))
    # merge consecutive identical labels into spans
    lab = seg["label"].to_numpy(); st = seg["t"].to_numpy()
    k = 0
    while k < len(lab):
        j = k
        while j + 1 < len(lab) and lab[j + 1] == lab[k]:
            j += 1
        if st[j] + SEG >= xlim[0] and st[k] <= xlim[1]:
            ax.axvspan(st[k], st[j] + SEG, color=COL[lab[k]], lw=0)
        k = j + 1
    ax.plot(t, x, "k", lw=0.8)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("Time (Sec)"); ax.set_ylabel("Power (MW)")
    h = [Line2D([0], [0], color="k", lw=1, label="Signal")] + \
        [Patch(color=COL[c], label=NAME[c]) for c in ["NA", "DA", "NR", "DR"]]
    ax.legend(handles=h, loc="upper right", fontsize=8, framealpha=1)
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv")
    ap.add_argument("--inject", action="store_true")
    a = ap.parse_args()

    t, x = load(a.data)
    seg, info = classify(t, x)
    seg.to_csv("labels_raw.csv", index=False)
    print("RAW DATA – segment counts:\n", seg["label"].value_counts().to_string())
    plot(t, x, seg, "result_raw_200_950.png", title="Layer-1 classification – data as given")
    plot(t, x, seg, "result_raw_full.png", xlim=(0, t[-1]), ylim=(240, 600),
         title="Layer-1 classification – full record, data as given")

    if a.inject:
        y, truth = inject(t, x)
        seg2, info2 = classify(t, y)
        seg2.to_csv("labels_injected.csv", index=False)
        print("\nINJECTED DATA – segment counts:\n", seg2["label"].value_counts().to_string())
        plot(t, y, seg2, "result_injected_200_950.png",
             title="Layer-1 classification – synthetic corruption injected")
        # accuracy of the degraded/normal decision vs ground truth (per segment)
        m = int(SEG * FS); n = len(seg2)
        tr = truth[:n * m].reshape(n, m).any(1)
        pr = seg2["label"].isin(["DA", "DR"]).to_numpy()
        near = lambda v: binary_dilation(v, iterations=1)      # +/- 1 s tolerance
        print("\nDegraded detection vs ground truth (per 1-s segment, +/-1 s tolerance):")
        print("  truth segments detected :", int((tr & near(pr)).sum()), "of", int(tr.sum()))
        print("  false alarms            :", int((pr & ~near(tr)).sum()))
