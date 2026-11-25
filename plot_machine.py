"""Figures for the machine: probabilistic 16-step forecasts per signal family, and the predicted density on the bimodal task."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from machine import build, path, WIN
from pfm.world import TASKS, batch

fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
for ax, task in zip(axes.ravel()[:6], TASKS):
    d = np.load(f"runs/machine/rollout_{task}.npz")
    i = 0
    ctx, tr, P = d["context"][i][-24:], d["truth"][i], d["paths"][i]
    tc = np.arange(-len(ctx), 0)
    th = np.arange(len(tr))
    ax.plot(tc, ctx, "k.-", lw=1, label="geçmiş (girdi)")
    lo, hi = np.quantile(P, [0.05, 0.95], 0)
    q1, q3 = np.quantile(P, [0.25, 0.75], 0)
    ax.fill_between(th, lo, hi, color="C0", alpha=0.18, label="%90 aralık")
    ax.fill_between(th, q1, q3, color="C0", alpha=0.35, label="%50 aralık")
    for k in range(4):
        ax.plot(th, P[k], color="C0", lw=0.6, alpha=0.6)
    ax.plot(th, tr, "r.-", lw=1.4, label="gerçek (görülmedi)")
    ax.axvline(-0.5, color="gray", lw=0.6)
    ax.set_title(task)
    ax.set_xlabel("örnek indeksi (zaman aralıkları düzensiz)")
axes[0, 0].legend(fontsize=7, loc="lower left")

# predicted next-value density on the bimodal task (machine, seed 0)
m = build("machine", 0)
m.load_state_dict(torch.load(path("machine", 0) + ".pt"))
m.eval()
x, dt, q, y, _ = batch(["bimodal"], np.random.default_rng(7), 3, 64)
xs = torch.linspace(*WIN, 2001, dtype=torch.float64)
with torch.no_grad():
    p = m.head.pdf_grid(m.trunk(x, dt, q)[:, -1], xs).numpy()
ax = axes[1, 2]
for i in range(3):
    ax.plot(xs.numpy() / 4 - 1, p[i] * 4, color=f"C{i}", lw=1.2)
    cur = x[i, -1].item() / 4 - 1
    ax.axvline(cur, color=f"C{i}", ls=":", lw=0.8)
    ax.plot([cur - 0.25, cur + 0.25], [0, 0], "v", color=f"C{i}", ms=8)
ax.set_xlim(-1.3, 1.3)
ax.set_title("bimodal: tahmin edilen yoğunluk\n(noktalı = şimdiki değer, ▼ = gerçek iki olasılık x±0.25)", fontsize=9)

# training curves
import json, glob
ax = axes[1, 3]
for f in sorted(glob.glob("runs/machine/*_train.json")):
    r = json.load(open(f))
    c = r["curve"]
    ax.plot([e["step"] for e in c], [e["nll_x"] for e in c], label=f"{r['kind']} s{r['seed']}", lw=1)
ax.set_ylim(-2.2, 1.0)
ax.set_xlabel("adım"); ax.set_ylabel("eğitim NLL (nat)")
ax.legend(fontsize=6)
ax.set_title("eğitim eğrileri")
plt.tight_layout()
plt.savefig("runs/machine/machine.png", dpi=110)
print("saved runs/machine/machine.png")
