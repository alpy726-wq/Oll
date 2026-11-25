"""Issue 9: learning real domains one after another without forgetting.

Order: weather -> grid -> market -> audio -> text. After each stage, every domain is evaluated on frozen val windows.
Methods (same machine, same steps per stage):
  finetune : all weights trained on each new domain (the usual way; expected to forget)
  adapters : base trained on the first domain, then for each new domain only its own phase adapter is trained and
             stored; at test time the adapter of the queried domain is applied (the machine's §5.6 mechanism)
  replay   : all weights, with 20% of each batch drawn from earlier domains (a standard strong reference)
Metrics: final per-domain score, average forgetting = mean over domains of (best score ever - final score)."""
import argparse, copy, json, math, os, time
import numpy as np
import torch
from pfm.one import OneMachine, PHASE_PARAMS
from pfm.realworld import STREAMS_REAL, RealData, make_batch, fixed_eval_set

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["finetune", "adapters", "replay"], required=True)
ap.add_argument("--steps", type=int, default=400, help="steps per domain")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", 4)))
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
ORDER = ["weather", "grid", "market", "audio", "text"]
train = RealData("train")
ev = fixed_eval_set("val", n_per_domain=48)
m = OneMachine(STREAMS_REAL, R=96, L=3, M=8, RM=512, seed=a.seed)


def score(model, d):
    b = ev[d]
    with torch.no_grad():
        l = model.nll(model(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"])
    v = float((l * b["mask"]).sum() / b["mask"].sum())
    return v / math.log(2) if d == "text" else v  # lower is better for every domain


def train_on(model, domains_fn, steps, only_phase=False):
    ps = [p for n, p in model.named_parameters() if (not only_phase) or n.split(".")[-1] in PHASE_PARAMS]
    opt = torch.optim.AdamW(ps, lr=2e-3)
    for p in model.parameters():
        p.requires_grad = any(p is q for q in ps)
    for k in range(steps):
        for g in opt.param_groups:
            g["lr"] = 2e-3 * min(1, (k + 1) / 50) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * k / steps)))
        d = domains_fn()
        b = make_batch(train, rng, 16, 128, [d])
        l = model.nll(model(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"])
        loss = (l * b["mask"]).sum() / b["mask"].sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ps, 1.0); opt.step()
    for p in model.parameters():
        p.requires_grad = True


history = []
adapters, base = {}, None
for i, d in enumerate(ORDER):
    seen = ORDER[: i + 1]
    if a.method == "finetune" or (a.method == "adapters" and i == 0):
        train_on(m, lambda: d, a.steps)
    elif a.method == "replay":
        train_on(m, lambda: d if rng.random() < 0.8 or i == 0 else seen[int(rng.integers(i))], a.steps)
    if a.method == "adapters":
        if i == 0:
            base = copy.deepcopy(m.state_dict())
            adapters[d] = {}
        else:
            mm = copy.deepcopy(m); mm.load_state_dict(base)
            train_on(mm, lambda: d, a.steps, only_phase=True)
            adapters[d] = mm.adapter(base)
    row = {}
    for e in seen:
        if a.method == "adapters":
            mm = copy.deepcopy(m); mm.apply_adapters(base, adapters[e]); row[e] = score(mm, e)
        else:
            row[e] = score(m, e)
    history.append(dict(after=d, scores=row)); print(history[-1], flush=True)

composed = None
if a.method == "adapters":  # issue 6c: all domain adapters applied at once
    mm = copy.deepcopy(m); mm.apply_adapters(base, *[adapters[e] for e in ORDER if adapters[e]])
    composed = {e: score(mm, e) for e in ORDER}
final = history[-1]["scores"]
best = {e: min(h["scores"][e] for h in history if e in h["scores"]) for e in ORDER}
res = dict(method=a.method, seed=a.seed, history=history, final=final, all_adapters_composed=composed,
           avg_forgetting=float(np.mean([final[e] - best[e] for e in ORDER[:-1]])),
           adapter_params=(sum(v.numel() for v in adapters[ORDER[1]].values()) if a.method == "adapters" else None))
os.makedirs("runs/real", exist_ok=True)
json.dump(res, open(f"runs/real/continual_{a.method}_s{a.seed}.json", "w"), indent=1)
print("FINAL", json.dumps(final), "forgetting", res["avg_forgetting"])
