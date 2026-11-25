"""Main run: OneMachine (all components) on the large real corpus, fixed wall-clock budget, checkpoint/resume on the
volume, then the full evaluation. Runs inside Modal (called from gpu/main_app.py)."""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/oll"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pfm.one import OneMachine
from pfm.bigworld import STREAMS_BIG, BigData, make_batch_big, DOMAINS_BIG
from pfm.realworld import SID, ETT_VARS, fixed_eval_set, DOMAINS
from pfm.ett_eval import evaluate as ett_evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--R", type=int, default=512)
ap.add_argument("--L", type=int, default=6)
ap.add_argument("--M", type=int, default=16)
ap.add_argument("--RM", type=int, default=1024)
ap.add_argument("--B", type=int, default=32)
ap.add_argument("--T", type=int, default=512)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--minutes", type=float, default=240)
ap.add_argument("--est_steps", type=int, default=20000, help="for the cosine schedule; training stops at the time budget")
ap.add_argument("--mix", default="series:0.35,web:0.25,weather:0.08,grid:0.08,market:0.06,audio:0.1,text:0.08")
ap.add_argument("--ckpt", default="/vol/ckpt/main")
ap.add_argument("--out", default="/vol/results/main.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda")
ap.add_argument("--big", default="/vol/big")
ap.add_argument("--v4", default="/oll/data/v4")
ap.add_argument("--ett_L", type=int, default=512)
ap.add_argument("--ett_stride_mult", type=int, default=1)
a = ap.parse_args()
dev = a.device
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True
os.makedirs(os.path.dirname(a.out), exist_ok=True)
mix = {k: float(v) for k, v in (x.split(":") for x in a.mix.split(","))}
doms, probs = list(mix), np.array(list(mix.values())) / sum(mix.values())

m = OneMachine(STREAMS_BIG, R=a.R, L=a.L, M=a.M, RM=a.RM, seed=a.seed).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
step, log, spent = 0, [], 0.0
os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
if os.path.exists(a.ckpt + ".pt"):  # resume
    ck = torch.load(a.ckpt + ".pt", map_location=dev)
    m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step, log, spent = ck["step"], ck["log"], ck["spent"]
    rng = np.random.default_rng(a.seed + step)
data = BigData(a.big, "train")
params = sum(p.numel() for p in m.parameters())
print("params", params, "resume step", step, flush=True)
t0 = time.time() - spent * 60
ema = {}
while (time.time() - t0) / 60 < a.minutes:
    frac = min(1.0, step / a.est_steps)
    for g in opt.param_groups:
        g["lr"] = a.lr * min(1, (step + 1) / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
    d = doms[int(rng.choice(len(doms), p=probs))]
    b = make_batch_big(data, rng, a.B, a.T, d, dev)
    l = m.nll(m(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"])
    loss = (l * b["mask"]).sum() / b["mask"].sum()
    if not torch.isfinite(loss):  # never update on a non-finite loss; counted and reported
        bad = res_bad = globals().get("n_bad", 0) + 1; globals()["n_bad"] = bad
        print("non-finite loss skipped", d, bad, flush=True); step += 1; continue
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    if not torch.isfinite(gn):
        globals()["n_bad"] = globals().get("n_bad", 0) + 1; print("non-finite grad skipped", d, flush=True); step += 1; continue
    opt.step()
    ema[d] = float(loss) if d not in ema else 0.98 * ema[d] + 0.02 * float(loss)
    step += 1
    if step % 200 == 0:
        log.append(dict(step=step, minutes=(time.time() - t0) / 60, **{k: round(v, 4) for k, v in ema.items()}))
        print(log[-1], flush=True)
    if step % 2000 == 0:
        torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), step=step, log=log, spent=(time.time() - t0) / 60), a.ckpt + ".pt")
torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), step=step, log=log, spent=(time.time() - t0) / 60), a.ckpt + ".pt")

# ---------------------------------------------------------------- evaluation
m.eval()
res = dict(args=vars(a), params=params, steps=step, skipped_nonfinite=globals().get("n_bad", 0), train_minutes=(time.time() - t0) / 60, curve=log)
with torch.no_grad():
    for split in ("val", "test"):  # frozen v4 windows (same as every CPU experiment)
        ev = fixed_eval_set(split)
        r = {}
        for dname, b in ev.items():
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            l = m.nll(m(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"])
            v = float((l * b["mask"]).sum() / b["mask"].sum())
            r[dname] = v / math.log(2) if dname == "text" else v
        res["v4_" + split] = r
    # held-out web text (bits per byte)
    vd = BigData(a.big, "val"); vr = np.random.default_rng(999)
    tot = []
    for _ in range(16):
        b = make_batch_big(vd, vr, 32, a.T, "web", dev)
        l = m.nll(m(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"])
        tot.append(float((l * b["mask"]).sum() / b["mask"].sum()))
    res["fineweb_val_bits_per_byte"] = float(np.mean(tot) / math.log(2))
    # held-out tail of every big series collection
    tot = []
    for _ in range(16):
        b = make_batch_big(vd, vr, 32, a.T, "series", dev)
        l = m.nll(m(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"])
        tot.append(float((l * b["mask"]).sum() / b["mask"].sum()))
    res["series_val_nll"] = float(np.mean(tot))
json.dump(res, open(a.out, "w"), indent=1)
# standard ETT long-horizon protocol (ETTh1 was trained on through its train split; ETTh2 and ETTm1 never seen)
sids = [SID["ett:" + v] for v in ETT_VARS]
for name, stride in (("ETTh1", 24), ("ETTh2", 24), ("ETTm1", 96)):
    res["ett_" + name] = ett_evaluate(m, sids, name, a.v4, dev, L=a.ett_L, stride=stride * a.ett_stride_mult)
    json.dump(res, open(a.out, "w"), indent=1)
    print(name, json.dumps({h: res["ett_" + name][h] for h in (96, 720)}), flush=True)
print("DONE", json.dumps({k: v for k, v in res.items() if k not in ("curve", "args")})[:3000])
