"""Figures of the paper, from committed result files only."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
c = json.load(open("runs/gpu/main/capabilities.json"))

# Figure 1: forecast fans, ETTh1 test, oil temperature (standardised units)
F = c["ett_fans_OT"][:6]
fig, axs = plt.subplots(2, 3, figsize=(8.6, 4.0), dpi=200, sharey=False)
for ax, f in zip(axs.ravel(), F):
    h = np.array(f["history"]); y = np.array(f["truth"])
    xh = np.arange(-len(h), 0); xf = np.arange(len(y))
    ax.plot(xh, h, color="#333", lw=0.8)
    ax.fill_between(xf, f["q10"], f["q90"], color="#2f5d8a", alpha=0.25, lw=0)
    ax.plot(xf, f["q50"], color="#2f5d8a", lw=1.2)
    ax.plot(xf, y, color="#c0392b", lw=0.8)
    ax.axvline(0, color="#aaa", lw=0.6)
    ax.set_title(f"window at index {f['start']}: 80% band covers {100 * f['coverage_80']:.0f}%", fontsize=7.5)
fig.suptitle("Held-out ETTh1 oil temperature: history (grey), 80% band and median (blue), truth (red); 96 h ahead", fontsize=9)
fig.tight_layout(); fig.savefig("paper/fig_fans.png"); plt.close(fig)

# Figure 2: calibration (PIT histograms) on held-out real data
keys = [("calibration_v4_test_weather", "weather (Jena)"), ("calibration_v4_test_grid", "grid (ETTh1)"),
        ("calibration_v4_test_audio", "speech audio"), ("calibration_series_heldout", "held-out series")]
fig, axs = plt.subplots(1, 4, figsize=(8.6, 1.9), dpi=200)
for ax, (k, name) in zip(axs, keys):
    hst = np.array(c[k]["pit_hist"], float); hst = hst / hst.mean()
    ax.bar(np.arange(10) + 0.5, hst, width=0.9, color="#2f5d8a")
    ax.axhline(1, color="#c0392b", lw=0.8)
    ax.set_title(f"{name}\nKS {c[k]['pit_ks']:.3f}, CRPS {c[k]['crps']:.3f} (ref {c[k]['crps_reference']:.3f})", fontsize=7)
    ax.set_xticks([]); ax.set_ylim(0, 2.2)
fig.tight_layout(); fig.savefig("paper/fig_calibration.png"); plt.close(fig)
print("ok")
