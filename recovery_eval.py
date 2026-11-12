"""
Evaluation of the recovery block on random fault-injection scenarios.

For every scenario we know the TRUE signal (record + synthetic ringdowns, no
faults), so we can measure how close the recovered signal gets to it.

    python3 recovery_eval.py --data data.csv
"""
import argparse
import numpy as np
import pandas as pd

import layer1_classifier as L1
import recovery as R

FS = L1.FS


def make_scenario(t, x, real_rd, seed, t0, t1, n_rd=5):
    """Same fault generator as rf_classifier.make_scenario, but it also returns
    the clean (true) signal.  Returns y (corrupted), yc (true), rd, fault."""
    rng = np.random.default_rng(seed)
    n = len(x)
    y = x.copy(); yc = x.copy()
    rd = real_rd.copy()
    fault = np.zeros(n, int)
    ix = lambda s: int(np.searchsorted(t, s))

    real_on = t[np.flatnonzero(np.diff(real_rd.astype(int)) == 1)]
    used = list(real_on)
    for _ in range(n_rd):
        for _try in range(50):
            on = rng.uniform(t0 + 20, t1 - 45)
            if all(abs(on - u) > 60 for u in used):
                break
        used.append(on)
        A, f, sg = rng.uniform(30, 160), rng.uniform(0.3, 1.2), rng.uniform(0.1, 0.3)
        T = min(np.log(A / 5) / sg, 25.0)
        i, j = ix(on), ix(on + T + 8)
        tau = t[i:j] - t[i]
        add = A * np.exp(-sg * tau) * np.sin(2 * np.pi * f * tau)
        y[i:j] += add; yc[i:j] += add
        rd[i:ix(on + T)] = True
        if rng.random() < 0.7:
            k = ix(on + rng.uniform(1.0, max(T - 1.0, 2.0)))
            w_ = rng.integers(1, 3)
            y[k:k + w_] += rng.choice([-1, 1]) * np.exp(rng.uniform(np.log(25), np.log(450)))
            fault[k:k + w_] = 2
        if rng.random() < 0.3:
            k = ix(on + rng.uniform(2.0, max(T - 3.0, 3.0)))
            gap = int(rng.uniform(1, 3) * FS)
            y[k:k + gap] = np.nan
            fault[k:k + gap] = 1

    def place(dur):
        for _try in range(40):
            s = rng.uniform(t0, t1 - dur)
            i, j = ix(s), ix(s + dur)
            if not fault[max(i - 90, 0):j + 90].any():
                return i, max(j, i + 1)
        return None

    for _ in range(25):
        p = place(0.2)
        if p:
            i, _j = p
            w = rng.integers(1, 3)
            amp = rng.choice([-1, 1]) * np.exp(rng.uniform(np.log(15), np.log(450)))
            y[i:i + w] += amp; fault[i:i + w] = 2
    for kind, cnt, lo, hi in [("nan", 5, 1, 8), ("zero", 4, 1, 5), ("stuck", 4, 2, 10)]:
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
    for _ in range(8):
        p = place(rng.uniform(8, 25))
        if p:
            i, j = p
            y[i:j] += rng.choice([-1, 1]) * rng.uniform(15, 60); fault[i:j] = 3
    return y, yc, rd, fault


def run(t, x, seeds, t0, t1):
    rd0, _, _ = L1.ringdown_mask(x)
    rows = []
    for sd in seeds:
        y, yc, rd, fault = make_scenario(t, x, rd0, sd, t0, t1)
        _, info = L1.classify(t, y)
        z, _ = R.recover(y, info)
        zl, _ = R.recover(y, info, method="linear")
        # gap length for every MV sample
        glen = np.zeros(len(y), int)
        for a, b in zip(*L1._runs(fault == 1)):
            glen[a:b] = b - a
        for i in range(len(y)):
            pass
        sel = np.arange(len(y)) >= int(np.searchsorted(t, t0))
        for k, nm in [(1, "MV"), (2, "OL"), (3, "FDI")]:
            for where, wm in [("ambient", ~rd), ("ringdown", rd)]:
                m = (fault == k) & wm & sel
                if k == 1:
                    for lo, hi, lab in [(0, 15, "<=0.5 s"), (16, 90, "0.5-3 s"), (91, 10**9, ">3 s")]:
                        mm = m & (glen >= lo) & (glen <= hi)
                        if mm.any():
                            rows.append(dict(seed=sd, type=nm, where=where, size=lab, n=int(mm.sum()),
                                             se_lin=float(((zl - yc)[mm] ** 2).sum()),
                                             se_rec=float(((z - yc)[mm] ** 2).sum()),
                                             se_raw=float(((y - yc)[mm & np.isfinite(y)] ** 2).sum()),
                                             n_raw=int((mm & np.isfinite(y)).sum())))
                elif m.any():
                    rows.append(dict(seed=sd, type=nm, where=where, size="all", n=int(m.sum()),
                                     se_lin=float(((zl - yc)[m] ** 2).sum()),
                                     se_rec=float(((z - yc)[m] ** 2).sum()),
                                     se_raw=float(((y - yc)[m] ** 2).sum()), n_raw=int(m.sum())))
        m = (fault == 0) & sel
        rows.append(dict(seed=sd, type="clean", where="all", size="all", n=int(m.sum()),
                         se_lin=float(((zl - yc)[m] ** 2).sum()),
                         se_rec=float(((z - yc)[m] ** 2).sum()), se_raw=0.0, n_raw=int(m.sum())))
    d = pd.DataFrame(rows)
    g = d.groupby(["type", "where", "size"], sort=False).sum(numeric_only=True)
    out = pd.DataFrame({"samples": g.n,
                        "corrupted": np.sqrt(g.se_raw / g.n_raw.clip(lower=1)),
                        "linear": np.sqrt(g.se_lin / g.n),
                        "proposed": np.sqrt(g.se_rec / g.n)}).round(2)
    f = d[d.type != "clean"].sum(numeric_only=True)
    out.loc[("ALL FAULTS", "", ""), :] = [int(f.n), round(float(np.sqrt(f.se_raw / f.n_raw)), 2),
                                          round(float(np.sqrt(f.se_lin / f.n)), 2),
                                          round(float(np.sqrt(f.se_rec / f.n)), 2)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    ap.add_argument("--t0", type=float, default=1950)
    a = ap.parse_args()
    t, x = L1.load(a.data)
    out = run(t, x, a.seeds, a.t0, float(t[-1]))
    print("RMSE in MW against the true signal (seeds %s, t >= %g s)" % (a.seeds, a.t0))
    print(out.to_string())
    out.to_csv("recovery_results.csv")
