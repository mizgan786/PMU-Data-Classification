import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import layer1_classifier as L1, layer2_integrity as L2, recovery as R

t, x = L1.load("data/data.csv")
y, truth = L2.inject_typed(t, x)
_, info = L1.classify(t, y)
z, rep = R.recover(y, info)
panels = [("Outlier (OL)", 336, 346), ("Missing values (MV), 7 s gap", 420, 440),
          ("False data injection (FDI), +40 MW", 470, 505), ("Outlier inside a ringdown (OL)", 898, 912)]
fig, axs = plt.subplots(2, 2, figsize=(11, 6.2))
for ax, (ttl, a, b) in zip(axs.ravel(), panels):
    m = (t >= a) & (t <= b)
    ax.plot(t[m], y[m], color="#d62728", lw=0.9, label="corrupted")
    ax.plot(t[m], x[m], color="k", lw=0.9, label="true")
    ax.plot(t[m], z[m], color="#1f77b4", lw=1.3, ls="--", label="recovered")
    ys = [x[m], z[m]] + ([y[m]] if "FDI" in ttl else [])
    lo = min(np.nanmin(u) for u in ys); hi = max(np.nanmax(u) for u in ys)
    ax.set_ylim(lo - 0.25 * (hi - lo) - 5, hi + 0.35 * (hi - lo) + 5)
    ax.set_title(ttl, fontsize=10); ax.set_xlabel("Time (s)"); ax.set_ylabel("Power (MW)")
axs[0, 0].legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig("result_recovery_examples.png", dpi=140)
