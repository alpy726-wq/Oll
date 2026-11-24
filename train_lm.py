"""Byte-level LM training on the prime basis (or an ablation basis)."""
import argparse, json, math, os, time
import torch
from pfm.model import PrimeSeqModel, param_breakdown
from pfm.data import ByteStream, eval_all, DOMAINS

ap = argparse.ArgumentParser()
ap.add_argument("--family", default="sqrt_p")
ap.add_argument("--R", type=int, default=256)
ap.add_argument("--L", type=int, default=8)
ap.add_argument("--K", type=int, default=256)
ap.add_argument("--B", type=int, default=16)
ap.add_argument("--T", type=int, default=512)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--minutes", type=float, default=1e9, help="wall-clock budget; LR schedule follows steps")
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--phase_lr_mult", type=float, default=5.0)
ap.add_argument("--domains", default=",".join(DOMAINS))
ap.add_argument("--init", default="")
ap.add_argument("--eval_domains", default=",".join(DOMAINS))
ap.add_argument("--train", default="all", choices=["all", "phase", "dense"])
ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--eval_n", type=int, default=48)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

torch.manual_seed(a.seed)
torch.set_num_threads(int(os.environ.get("THREADS", 4)))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
m = PrimeSeqModel(R=a.R, L=a.L, K=a.K, family=a.family, seed=a.seed)
if a.init:
    m.load_state_dict(torch.load(a.init + ".pt"))

PHASE = ("theta", "sigma", "alpha")
for n, p in m.named_parameters():
    leaf = n.split(".")[-1]
    p.requires_grad = a.train == "all" or (a.train == "phase" and leaf in PHASE) or (a.train == "dense" and leaf == "V")
phase_p = [p for n, p in m.named_parameters() if p.requires_grad and n.split(".")[-1] in PHASE + ("temp", "out_gain")]
other_p = [p for n, p in m.named_parameters() if p.requires_grad and n.split(".")[-1] not in PHASE + ("temp", "out_gain")]
groups = [g for g in [dict(params=other_p, lr=a.lr, weight_decay=0.0), dict(params=phase_p, lr=a.lr * a.phase_lr_mult, weight_decay=0.0)] if g["params"]]
for g in groups:
    g["base_lr"] = g["lr"]
opt = torch.optim.AdamW(groups, betas=(0.9, 0.98))
n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)

data = ByteStream(a.domains.split(","), seed=a.seed)
log = dict(args=vars(a), params_total=sum(p.numel() for p in m.parameters()), params_trained=n_train,
           breakdown=param_breakdown(m), curve=[])
print(json.dumps({k: log[k] for k in ("params_total", "params_trained", "breakdown")}))
if a.init:
    log["eval_init"] = eval_all(m, a.T, a.eval_n, a.eval_domains.split(","))
    print("init", log["eval_init"])
t0, step, warm = time.time(), 0, min(200, a.steps // 10)
ema = None
while step < a.steps and (time.time() - t0) / 60 < a.minutes:
    lrf = min(1, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * lrf
    x, y = data.batch(a.B, a.T)
    logits = m(x if a.family == "table" else x.double())
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
    opt.step()
    step += 1
    bits = loss.item() / math.log(2)
    ema = bits if ema is None else 0.98 * ema + 0.02 * bits
    if step % 50 == 0:
        print(f"step {step} train_bpb {ema:.3f} {(time.time()-t0)/60:.1f}min", flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        ev = eval_all(m, a.T, a.eval_n, a.eval_domains.split(","))
        log["curve"].append(dict(step=step, minutes=(time.time() - t0) / 60, train_bpb_ema=ema, **ev))
        print("eval", step, json.dumps(ev), flush=True)
        torch.save(m.state_dict(), a.out + ".pt")
if not log["curve"] or log["curve"][-1]["step"] != step:
    ev = eval_all(m, a.T, a.eval_n, a.eval_domains.split(","))
    log["curve"].append(dict(step=step, minutes=(time.time() - t0) / 60, train_bpb_ema=ema, **ev))
torch.save(m.state_dict(), a.out + ".pt")
log["final"] = log["curve"][-1]
log["steps_done"] = step
json.dump(log, open(a.out + ".json", "w"), indent=1)
print("final", json.dumps(log["final"]))
