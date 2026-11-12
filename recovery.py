"""
Recovery of degraded PMU data (the 'condition-specific recovery' block).

Takes the signal and the masks found by the rule-based classifier
(layer1_classifier.classify -> info) and repairs each kind of damage:

    FDI  constant offset      -> estimate the offset and subtract it
    OL   isolated spikes      -> cubic-spline interpolation over the spike
    MV   gaps / zeros / stuck -> short gap: cubic spline
                                 long gap : autoregressive (AR) model that
                                 predicts into the gap from both sides and
                                 blends the two predictions

Only samples that the classifier flagged are changed; everything else is
returned untouched.

Usage:
    python3 recovery.py --data data.csv
"""
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_dilation

import layer1_classifier as L1

FS = L1.FS

SHORT_GAP = 15            # samples (0.5 s): up to this length use the spline
SPLINE_PTS = 6            # clean samples used on each side of a spline gap
MERGE = 3                 # runs of bad samples closer than this are merged
AR_ORDER = 8              # AR order (a damped sine is AR(2); the rest models the noise)
AR_CTX = 150              # context used for fitting, samples each side (5 s)
STEP_WIN = int(L1.JUMP_WIN * FS)
PAIR_TOL = 0.35           # an up-step and a down-step form a block if their sizes differ by < 35 %
OFFSET_EST = "interior"   # "interior": interior median minus outside level; "edges": mean of the two steps


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _runs(mask):
    """start (inclusive) and stop (exclusive) of each True run."""
    return L1._runs(mask)


def _merge_runs(starts, stops, gap):
    out = []
    for a, b in zip(starts, stops):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = b
        else:
            out.append([int(a), int(b)])
    return out


def _spline_fill(x, c, e):
    fin = np.isfinite(x)
    left = np.flatnonzero(fin[:c])[-SPLINE_PTS:]
    right = e + np.flatnonzero(fin[e:])[:SPLINE_PTS]
    idx = np.r_[left, right]
    if len(left) == 0 or len(right) == 0:                 # at the record edge
        x[c:e] = x[idx[0]] if len(idx) else 0.0
        return
    if len(idx) < 4:
        x[c:e] = np.interp(np.arange(c, e), idx, x[idx])
        return
    cs = CubicSpline(idx, x[idx])
    x[c:e] = cs(np.arange(c, e))


def _ar_equations(seg, p):
    """Forward + backward linear-prediction equations from one clean segment.
    Equations touching a NaN are dropped."""
    rows, ys = [], []
    for s in (seg, seg[::-1]):
        if len(s) <= p + 1:
            continue
        # windows [s[i-1], ..., s[i-p]]  ->  s[i]
        W = np.lib.stride_tricks.sliding_window_view(s, p + 1)      # (n-p, p+1)
        Xw = W[:, :p][:, ::-1]
        yw = W[:, p]
        ok = np.isfinite(W).all(1)
        rows.append(Xw[ok]); ys.append(yw[ok])
    if not rows:
        return np.zeros((0, p)), np.zeros(0)
    return np.vstack(rows), np.concatenate(ys)


def _stabilise(a):
    """Shrink AR coefficients (bandwidth expansion) if any pole is on/outside
    the unit circle, so a long prediction cannot blow up."""
    p = len(a)
    comp = np.zeros((p, p))
    comp[0, :] = a
    comp[1:, :-1] = np.eye(p - 1)
    r = np.max(np.abs(np.linalg.eigvals(comp)))
    if r >= 0.995:
        g = 0.995 / r
        a = a * g ** np.arange(1, p + 1)
    return a


def _predict(tail, a, n):
    """Recursive prediction of n samples after `tail` (oldest first)."""
    p = len(a)
    buf = list(tail[-p:])
    out = np.empty(n)
    for k in range(n):
        v = float(np.dot(a, buf[::-1][:p]))
        out[k] = v
        buf.append(v)
        buf.pop(0)
    return out


def _ar_fill(x, c, e, rd):
    n = len(x)
    L = e - c
    in_rd = rd[max(c - 30, 0):min(e + 30, n)].any()
    p, ctx = AR_ORDER, AR_CTX
    lo, hi = max(0, c - ctx), min(n, e + ctx)
    if in_rd:                                           # stay inside this ringdown
        a_, b_ = _runs(rd)
        k = np.flatnonzero((a_ <= c) & (b_ >= e))
        if len(k):
            lo2, hi2 = max(lo, int(a_[k[0]])), min(hi, int(b_[k[0]]))
            if c - lo2 >= 60 and hi2 - e >= 60:
                lo, hi = lo2, hi2
    left, right = x[lo:c].copy(), x[e:hi].copy()
    fin = np.concatenate([left[np.isfinite(left)], right[np.isfinite(right)]])
    if len(fin) < 4 * p:
        _spline_fill(x, c, e)
        return
    mu = fin.mean()
    left -= mu; right -= mu

    X1, y1 = _ar_equations(left, p)
    X2, y2 = _ar_equations(right, p)
    X = np.vstack([X1, X2]); y = np.concatenate([y1, y2])
    if len(y) < 3 * p:
        _spline_fill(x, c, e)
        return
    lam = 1e-3 * np.trace(X.T @ X) / p
    a = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
    a = _stabilise(a)

    have_l = len(left) >= p and np.isfinite(left[-p:]).all()
    have_r = len(right) >= p and np.isfinite(right[:p]).all()
    if not (have_l or have_r):
        _spline_fill(x, c, e)
        return
    xf = _predict(left, a, L) if have_l else None
    xb = _predict(right[::-1], a, L)[::-1] if have_r else None
    if xf is None:
        est = xb
    elif xb is None:
        est = xf
    else:
        w = (np.arange(L) + 1.0) / (L + 1.0)
        est = (1 - w) * xf + w * xb
    x[c:e] = est + mu


# --------------------------------------------------------------------------
# FDI: estimate and remove constant offsets
# --------------------------------------------------------------------------
def _signed_step(x):
    """after-median minus before-median (2 s each), NaN-aware."""
    w = STEP_WIN
    ser = pd.Series(x)
    before = ser.rolling(w, min_periods=w // 2).median().shift(1).to_numpy()
    after = ser[::-1].rolling(w, min_periods=w // 2).median().shift(1)[::-1].to_numpy()
    return after - before


def _refine_edge(x, k0, half=45, w=10):
    """Sample-accurate position of a level step near k0.  For every k the median of
    the next w samples is compared with the median of the previous w samples; this
    difference is flat for about w samples around the true edge, so the centre of
    that flat top is taken as the edge."""
    n = len(x)
    ks = np.arange(max(w, k0 - half), min(n - w, k0 + half))
    v = np.full(len(ks), -1.0)
    for i, k in enumerate(ks):
        l, r = x[k - w:k], x[k:k + w]
        if np.isfinite(l).sum() >= w // 2 and np.isfinite(r).sum() >= w // 2:
            v[i] = abs(np.nanmedian(r) - np.nanmedian(l))
    if v.max() < 0:
        return k0
    top = np.flatnonzero(v >= 0.9 * v.max())
    return int(ks[top].mean().round())


def find_offset_blocks(x, jump_mask):
    """Pair level steps into offset blocks.
    A step event is the largest |2-s median step| inside each flagged jump run.
    Adjacent events with opposite sign and (nearly) equal size form a block;
    the best-matching pairs are taken first, so two neighbouring blocks are not
    mistaken for one long block.  Returns [(start, stop, offset), ...]."""
    s = _signed_step(x)
    ev = []
    for a, b in zip(*_runs(jump_mask)):
        w = s[a:b]
        if not np.isfinite(w).any():
            continue
        # centre of the plateau of the largest step (the 2-s medians give a 2-s plateau)
        top = np.nanmax(np.abs(w))
        idx = np.flatnonzero(np.abs(w) >= 0.9 * top)
        k = a + int(idx.mean().round())
        ev.append((_refine_edge(x, k), float(s[k])))
    ev.sort()
    cand = []
    for i in range(len(ev) - 1):
        (k1, v1), (k2, v2) = ev[i], ev[i + 1]
        if np.sign(v1) == np.sign(v2) or not (STEP_WIN <= k2 - k1 <= L1.JUMP_PAIR_MAX * FS):
            continue
        mis = abs(v1 + v2) / max(abs(v1), abs(v2))
        if mis <= PAIR_TOL:
            cand.append((mis, i))
    cand.sort()
    used, blocks = set(), []
    for mis, i in cand:
        if i in used or i + 1 in used:
            continue
        used.update((i, i + 1))
        (k1, v1), (k2, v2) = ev[i], ev[i + 1]
        blocks.append((k1, k2, 0.5 * (v1 - v2)))
    return sorted(blocks)


def remove_offsets(x, jump_mask):
    """Subtract the offset of every detected block.  x is modified in place.
    Returns the list of (start, stop, offset) that were corrected."""
    blocks = find_offset_blocks(x, jump_mask)
    out = []
    for k1, k2, d in blocks:
        # offset = interior level minus the level just outside (average of both sides)
        pad = 5
        inner = np.nanmedian(x[k1 + pad:k2 - pad])
        lo = np.nanmedian(x[max(0, k1 - 60):max(1, k1 - pad)])
        hi = np.nanmedian(x[k2 + pad:k2 + 60])
        d2 = inner - 0.5 * (lo + hi) if np.isfinite(inner + lo + hi) else d
        d_use = d2 if OFFSET_EST == "interior" else d
        x[k1:k2] -= d_use
        out.append((k1, k2, float(d_use)))
    return out


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def recover(x, info, method="ar"):
    """x    : measured signal (may contain NaN)
       info : dict from layer1_classifier.classify (masks 'missing', 'stuck',
              'spike', 'offset', 'rd')
       method: 'ar'     = spline for short gaps, AR for long ones  (proposed)
               'linear' = straight-line fill for every flagged sample (baseline)
       Returns (recovered signal, report dict)."""
    y = np.asarray(x, float).copy()
    bad_mv = info["missing"] | info["stuck"]
    bad_ol = binary_dilation(info["spike"], iterations=1)

    # 1) FDI first, using the signal with MV / OL samples blanked
    work = y.copy()
    work[bad_mv | bad_ol] = np.nan
    blocks = remove_offsets(work, info["jump"])

    # 2) fill MV and OL samples (blank in `work`, and merged if close together)
    fin = np.isfinite(work)
    s, e = _runs(~fin)
    runs = _merge_runs(s, e, MERGE)
    n_spline = n_ar = 0
    for c, ee in runs:
        if method == "linear":
            idx = np.flatnonzero(np.isfinite(work))
            work[c:ee] = np.interp(np.arange(c, ee), idx, work[idx])
        elif ee - c <= SHORT_GAP:
            _spline_fill(work, c, ee)
            n_spline += 1
        else:
            _ar_fill(work, c, ee, info["rd"])
            n_ar += 1
    # anything still NaN (record edges) -> nearest value
    if not np.isfinite(work).all():
        work = pd.Series(work).interpolate(limit_direction="both").to_numpy()
    return work, dict(offset_blocks=blocks, gaps_spline=n_spline, gaps_ar=n_ar,
                      n_runs=len(runs))


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import layer2_integrity as L2
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv")
    a = ap.parse_args()
    t, x = L1.load(a.data)

    y, truth = L2.inject_typed(t, x)           # 17 hand-placed faults, true signal = x
    _, info = L1.classify(t, y)
    z, rep = recover(y, info)
    zl, _ = recover(y, info, method="linear")

    def rmse(u, m):
        return float(np.sqrt(np.mean((u[m] - x[m]) ** 2)))

    print("offset blocks corrected:", [(round(t[a_], 1), round(t[b_], 1), round(d, 1))
                                       for a_, b_, d in rep["offset_blocks"]])
    print(f"gaps filled: {rep['gaps_spline']} with spline, {rep['gaps_ar']} with AR")
    print(f"{'fault type':12s} {'samples':>8s} {'corrupted':>10s} {'linear':>8s} {'proposed':>9s}   (RMSE in MW)")
    for k, nm in [(1, "MV"), (2, "OL"), (3, "FDI")]:
        m = truth == k
        fin = m & np.isfinite(y)
        print(f"{nm:12s} {m.sum():8d} {rmse(y, fin):10.2f} {rmse(zl, m):8.2f} {rmse(z, m):9.2f}")
    m = truth == 0
    print(f"{'untouched':12s} {m.sum():8d} {'':>10s} {rmse(zl, m):8.2f} {rmse(z, m):9.2f}")
