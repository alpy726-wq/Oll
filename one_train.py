"""Train OneMachine on the mixed event world (one model, all tasks at once)."""
import argparse, json, math, os, time
import numpy as np
import torch
from pfm.one import OneMachine
from pfm.oneworld import STREAMS, batch

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--B", type=int, default=24)
ap.add_argument("--T", type=int, default=64)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--R", type=int, default=128)
ap.add_argument("--L", type=int, default=3)
ap.add_argument("--family", default="sqrt_p")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--out", default="runs/one/one_s0")
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
m = OneMachine(STREAMS, R=a.R, L=a.L, family=a.family, seed=a.seed)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
log, t0 = [], time.time()
groups = {"signal": lambda k: k not in ("fusion", "text") and not k.startswith("recall"), "fusion": lambda k: k == "fusion",
          "recall": lambda k: k.startswith("recall"), "text": lambda k: k == "text"}
ema = {}
for step in range(a.steps):
    for g in opt.param_groups:
        g["lr"] = a.lr * min(1, (step + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
    b = batch(rng, a.B, a.T)
    l = m.nll(m(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"]) * b["mask"]
    loss = l.sum() / b["mask"].sum()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    with torch.no_grad():
        per = l.sum(1) / b["mask"].sum(1).clamp(min=1)
        for g, f in groups.items():
            idx = [i for i, k in enumerate(b["kinds"]) if f(k)]
            if idx:
                v = float(per[idx].mean())
                ema[g] = v if g not in ema else 0.97 * ema[g] + 0.03 * v
    if step % 100 == 0 or step == a.steps - 1:
        log.append(dict(step=step, sec=time.time() - t0, mem_beta=[round(v, 3) for v in m.mem_beta.tolist()], **{k: round(v, 4) for k, v in ema.items()}))
        print(log[-1], flush=True)
torch.save(m.state_dict(), a.out + ".pt")
json.dump(dict(args=vars(a), params=sum(p.numel() for p in m.parameters()), curve=log), open(a.out + "_train.json", "w"), indent=1)
