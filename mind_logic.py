"""Does the machine form logic?  In-context rule induction: from a few (x, y) examples of a hidden rule, answer y for a
new x. Trained on many random rules/constants; tested on fresh constants, on every number of examples, and on rules it
never saw (compositions)."""
import json, math, sys, time
import numpy as np
import torch
from pfm.one import OneMachine
from pfm.mind import MIND_STREAMS, rule_batch, RULES

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
torch.manual_seed(0)
rng = np.random.default_rng(0)
m = OneMachine(MIND_STREAMS, R=128, L=4, M=8, RM=512)
opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
t0, log = time.time(), []
for step in range(STEPS):
    for g in opt.param_groups:
        g["lr"] = 2e-3 * min(1, (step + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
    b = rule_batch(rng, 32, int(rng.integers(1, 9)))
    l = m.nll(m(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"])
    loss = (l * b["mask"]).sum() / b["mask"].sum()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if step % 200 == 0:
        log.append(dict(step=step, nll=float(loss), sec=time.time() - t0)); print(log[-1], flush=True)
torch.save(m.state_dict(), "runs/mind_logic%s.pt" % (sys.argv[3] if len(sys.argv) > 3 else ""))

XS = torch.linspace(-1.5, 1.5, 1201, dtype=torch.float64)


def evaluate(rule, n, B=96, seed=0, custom=None):
    r = np.random.default_rng(1000 + seed)
    if custom is None:
        b = rule_batch(r, B, n, rule)
    else:  # an unseen rule: given as a function of x
        xs = r.uniform(-1, 1, (B, n + 1)); ys = np.clip(custom(xs, r), -1.2, 1.2)
        x = torch.tensor(np.stack([xs, ys], 2).reshape(B, -1)); s = torch.tensor(np.tile([0, 1], n + 1))[None].repeat(B, 1)
        one = torch.ones(x.shape)
        b = dict(x=x[:, :-1], s=s[:, :-1], dt=one[:, :-1], q=one[:, 1:], y=x[:, 1:])
    with torch.no_grad():
        h = m(b["x"], b["s"], b["dt"], b["q"])[:, -1]
        d = m.density(h, 1, XS).double()
        cdf = (d / d.sum(-1, keepdim=True)).cumsum(-1)
        med = XS[torch.searchsorted(cdf.contiguous(), torch.full((B, 1), 0.5, dtype=torch.float64)).clamp(max=len(XS) - 1).squeeze(1)]
    err = (med - b["y"][:, -1]).abs()
    return dict(acc_within_0p1=float((err < 0.1).float().mean()), mean_abs_err=float(err.mean()))


res = {"train_curve": log, "by_examples": {}, "per_rule_8_examples": {}, "unseen_rules_8_examples": {}}
for n in (0, 1, 2, 4, 8):
    vals = [evaluate(rule, n) for rule in RULES]
    res["by_examples"][n] = dict(acc_within_0p1=float(np.mean([v["acc_within_0p1"] for v in vals])),
                                 mean_abs_err=float(np.mean([v["mean_abs_err"] for v in vals])))
for rule in RULES:
    res["per_rule_8_examples"][rule] = evaluate(rule, 8)
unseen = {
    "2x - 0.3 (scale then add)": lambda x, r: 2 * x - 0.3,
    "0.5 - |x| (abs mirrored)": lambda x, r: 0.5 - np.abs(x),
    "x^3": lambda x, r: x ** 3,
}
for k, f in unseen.items():
    res["unseen_rules_8_examples"][k] = evaluate(None, 8, custom=f)
json.dump(res, open("runs/mind_logic%s.json" % (sys.argv[3] if len(sys.argv) > 3 else ""), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "train_curve"}, indent=1))
