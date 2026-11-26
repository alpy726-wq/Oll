"""OneMachine on real data only. Joint vs staged curricula, ablations, seeds; evaluation on frozen val/test windows.

Each step draws one real domain (a homogeneous batch). Schedules:
  joint : all domains from the first step
  staged: perception first (weather, grid, market, audio), then language added, same total steps
Reported per domain on the frozen windows: NLL (continuous, nats in standardised units) or bits per byte (text),
next to a no-learning reference (persistence for continuous streams, unigram for text)."""
import argparse, json, math, os, time
import numpy as np
import torch
from pfm.one import OneMachine
from pfm.realworld import STREAMS_REAL, RealData, make_batch, fixed_eval_set, DOMAINS
from pfm.eventgpt import EventGPT

ap = argparse.ArgumentParser()
ap.add_argument("--schedule", default="joint", choices=["joint", "staged"])
ap.add_argument("--ablate", default="", help="comma list: memory, phaselinear, kuramoto, flatreadout")
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--R", type=int, default=96)
ap.add_argument("--L", type=int, default=3)
ap.add_argument("--B", type=int, default=16)
ap.add_argument("--T", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", 4)))
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", required=True)
ap.add_argument("--arch", default="machine", choices=["machine", "gpt"])
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--mem_orders", default="1", help="comma list, e.g. 1,4,8 (context-keyed memory)")
ap.add_argument("--RM", type=int, default=512)
ap.add_argument("--clock", action="store_true", help="calendar clock (day/week/year phases of absolute time)")
ap.add_argument("--anticipate", action="store_true", help="also learn when and which sense comes next")
ap.add_argument("--query", action="store_true", help="query head: trunk state independent of the asked time; trained to answer far moments directly")
ap.add_argument("--ett", action="store_true", help="after training: ETT long-horizon protocol (30 windows, with adaptation)")
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
abl = set(x for x in a.ablate.split(",") if x)
if a.arch == "gpt":
    m = EventGPT(STREAMS_REAL, d=a.d, L=a.L, T=a.T).to(a.device)
else:
    m = OneMachine(STREAMS_REAL, R=a.R, L=a.L, M=8, RM=a.RM, seed=a.seed, readout_amp="flatreadout" in abl,
                   mem_orders=tuple(int(v) for v in a.mem_orders.split(",")), clock=a.clock, anticipate=a.anticipate, query_head=a.query).to(a.device)
with torch.no_grad():
    if "memory" in abl:
        m.mem_beta.zero_(); m.mem_beta.requires_grad = False
    if "phaselinear" in abl:
        for P in m.P:
            P.pl.requires_grad = False
    if "kuramoto" in abl:
        for br in list(m.F) + list(m.G):
            br.sync.kp.zero_(); br.sync.kp.requires_grad = False; br.sync.kq.requires_grad = False
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr if a.arch == "machine" else 1e-3)
train = RealData("train")
PERCEPTION = ["weather", "grid", "market", "audio"]


def domains_at(step):
    if a.schedule == "joint" or step >= a.steps // 2:
        return list(DOMAINS)
    return PERCEPTION


def far_targets(b, rng):
    """half of the positions ask about the next event, half about a random later event of the window (target value,
    its stream, the time until it, its absolute time); anticipation keeps the next-event targets"""
    X = torch.cat([b["x"], b["y"][:, -1:]], 1); S = torch.cat([b["s"], b["s_out"][:, -1:]], 1)
    D = torch.cat([b["dt"], b["q"][:, -1:]], 1); TA = torch.cat([b["t_in"], b["t_out"][:, -1:]], 1)
    Bn, T = b["x"].shape
    tau = D.double().cumsum(1)
    i = torch.arange(T, device=X.device)[None].expand(Bn, T)
    far = torch.from_numpy(rng.random((Bn, T)) < 0.5).to(X.device)
    jr = i + 1 + (torch.from_numpy(rng.random((Bn, T))).to(X.device) * (T - i).double()).long().clamp(max=T - 1 - i + 0)
    j = torch.where(far, jr.clamp(max=T), i + 1)
    g = lambda A: A.gather(1, j)
    out = dict(b)
    out.update(y=g(X), s_out=g(S), q=(g(tau) - tau[:, :T]).float(), t_out=g(TA), q_next=b["q"], s_next=b["s_out"])
    return out


t0, log = time.time(), []
for step in range(a.steps):
    for g in opt.param_groups:
        g["lr"] = (a.lr if a.arch == "machine" else 1e-3) * min(1, (step + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
    doms = domains_at(step)
    b = make_batch(train, rng, a.B, a.T, [doms[int(rng.integers(len(doms)))]], a.device)
    if a.arch == "machine" and a.query:
        b = far_targets(b, rng)
    if a.arch == "machine" and a.anticipate:
        h_, ha_ = m.forward_all(b["x"], b["s"], b["dt"], b["q"], t_in=b["t_in"], t_out=b["t_out"])
        w_, c_ = m.anticipation_nll(ha_, b.get("s_next", b["s_out"]), b.get("q_next", b["q"]))
        l = m.nll(h_, b["s_out"], b["y"]) + w_ + c_
    elif a.arch == "machine":
        l = m.nll(m(b["x"], b["s"], b["dt"], b["q"], t_in=b["t_in"], t_out=b["t_out"]), b["s_out"], b["y"])
    else:
        l = m.nll(m(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"])
    loss = (l * b["mask"]).sum() / b["mask"].sum()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if step % 100 == 0:
        log.append(dict(step=step, domain=b["domains"][0], loss=float(loss), sec=time.time() - t0)); print(log[-1], flush=True)


def evaluate(split):
    ev = fixed_eval_set(split)
    out = {}
    m.eval()
    with torch.no_grad():
        for d, b in ev.items():
            b = {k: (v.to(a.device) if torch.is_tensor(v) else v) for k, v in b.items()}
            kw = dict(t_in=b["t_in"], t_out=b["t_out"]) if a.arch == "machine" else {}
            l = m.nll(m(b["x"], b["s"], b["dt"], b["q"], **kw), b["s_out"], b["y"])
            v = float((l * b["mask"]).sum() / b["mask"].sum())
            out[d] = v / math.log(2) if d == "text" else v
    m.train()
    return out


res = dict(args=vars(a), params=sum(p.numel() for p in m.parameters()), curve=log, train_seconds=time.time() - t0,
           val=evaluate("val"), test=evaluate("test"))
if a.ett:
    from pfm.ett_eval import evaluate as ett_evaluate
    from pfm.realworld import SID, ETT_VARS
    m.eval()
    with torch.no_grad():
        for name, stride in (("ETTh1", 24), ("ETTh2", 24), ("ETTm1", 96)):
            r = ett_evaluate(m, [SID["ett:" + v] for v in ETT_VARS], name, "data/v4", a.device, L=512, stride=stride, max_windows=30, adapt=True,
                             direct=a.query)
            res["ett_" + name] = {h: r[h] for h in (96, 336)}
            print(name, json.dumps(res["ett_" + name]), flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save(m.state_dict(), a.out + ".pt")
json.dump(res, open(a.out + ".json", "w"), indent=1)
print("VAL", json.dumps(res["val"]), "\nTEST", json.dumps(res["test"]))
