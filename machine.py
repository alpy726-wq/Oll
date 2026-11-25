"""The machine: one continuous-in / continuous-time / continuous-out model on the frozen prime basis.

  input  : a real value x (no tokenizer, no table)           -> b(x) prime-frequency phases
  time   : any gap dt between samples (irregular, mixed rate)  -> complex-exponential state, advanced by e^{a dt}
  output : a full probability density over the next value     -> q(x) = |<b(x),h>|^2 / Z, Z in closed form

  python3 machine.py train --kind machine --seed 0      (kinds: machine, ct_gauss, ct_mdn, gru_mdn)
  python3 machine.py eval  --kind machine --seed 0
  python3 machine.py report
"""
import argparse, glob, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
from pfm.continuous import CTSeqModel, MDNHead
from pfm.world import TASKS, batch, gen, to_model

WIN = (-2.0, 10.0)
LOG4 = math.log(4.0)  # x' = 4x + 4  ->  densities in x units: nll_x = nll_x' - log 4


class GRUMDN(nn.Module):
    """Standard baseline: GRU over [x, log dt, log q] + 5-component mixture density head."""

    def __init__(self, H=160):
        super().__init__()
        self.H = H
        self.cell = nn.GRUCell(3, H)
        self.head = MDNHead(H)

    def feats(self, x, dt, q):
        return torch.stack([(x.float() - 4) / 4, dt.clamp(min=1e-3).log(), q.clamp(min=1e-3).log()], -1)

    def trunk(self, x, dt, q):
        f = self.feats(x, dt, q)
        h = torch.zeros(x.shape[0], self.H)
        hs = []
        for t in range(x.shape[1]):
            h = self.cell(f[:, t], h)
            hs.append(h)
        return torch.stack(hs, 1)

    def nll(self, x, dt, q, y):
        return self.head.nll(self.trunk(x, dt, q), y)

    def init_state(self, B):
        return torch.zeros(B, self.H)

    def step(self, x, dt, q, h):
        h = self.cell(self.feats(x, dt, q), h)
        return h, h


def build(kind, seed):
    if kind == "machine":
        return CTSeqModel(R=128, L=3, M=4, mode="ct", head="phase", seed=seed)
    if kind == "ct_gauss":
        return CTSeqModel(R=128, L=3, M=4, mode="ct", head="gauss", seed=seed)
    if kind == "ct_mdn":
        return CTSeqModel(R=128, L=3, M=4, mode="ct", head="mdn", seed=seed)
    if kind == "gru_mdn":
        return GRUMDN()
    raise ValueError(kind)


def path(kind, seed):
    return f"runs/machine/{kind}_s{seed}"


def train(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    m = build(a.kind, a.seed)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    t0, log = time.time(), []
    for s in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1, (s + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
        x, dt, q, y, _ = batch(TASKS, rng, a.B, a.T)
        loss = m.nll(x, dt, q, y)[:, 8:].mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 100 == 0 or s == a.steps - 1:
            log.append(dict(step=s, nll_x=loss.item() - LOG4, sec=time.time() - t0))
            print(a.kind, a.seed, log[-1], flush=True)
    os.makedirs("runs/machine", exist_ok=True)
    torch.save(m.state_dict(), path(a.kind, a.seed) + ".pt")
    json.dump(dict(kind=a.kind, seed=a.seed, steps=a.steps, params=sum(p.numel() for p in m.parameters()),
                   train_seconds=time.time() - t0, curve=log), open(path(a.kind, a.seed) + "_train.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------- evaluation
def pdf_on_grid(m, h, xs, chunk=256):
    out = []
    for i in range(0, h.shape[0], chunk):
        out.append(m.head.pdf_grid(h[i:i + chunk], xs))
    p = torch.cat(out).clamp(min=0)
    return p / torch.trapezoid(p, xs.float(), dim=-1).unsqueeze(-1)  # renormalise on the window (no-op for phase head)


def grid_metrics(p, xs, y):
    """CRPS and PIT from a density on a grid. y: (N,)"""
    dx = (xs[1] - xs[0]).item()
    F = torch.cumsum(p, -1) * dx
    F = F / F[:, -1:]
    ind = (xs.float()[None, :] >= y.float()[:, None]).float()
    crps = ((F - ind) ** 2).sum(-1) * dx
    k = torch.searchsorted(xs.float().contiguous(), y.float().contiguous()).clamp(0, len(xs) - 1)
    pit = F.gather(1, k[:, None]).squeeze(1)
    return crps / 4, pit  # CRPS in x units


def sample_grid(p, xs, gen_):
    F = torch.cumsum(p, -1)
    F = F / F[:, -1:]
    u = torch.rand(p.shape[0], 1, generator=gen_)
    k = torch.searchsorted(F.contiguous(), u).clamp(0, len(xs) - 1).squeeze(1)
    return xs[k] + (torch.rand(p.shape[0], generator=gen_).double() - 0.5) * (xs[1] - xs[0])


@torch.no_grad()
def evaluate(a):
    torch.set_num_threads(a.threads)
    m = build(a.kind, a.seed)
    m.load_state_dict(torch.load(path(a.kind, a.seed) + ".pt"))
    m.eval()
    xs = torch.linspace(*WIN, 1201, dtype=torch.float64)
    res = dict(kind=a.kind, seed=a.seed, one_step={}, rollout={})
    g = torch.Generator().manual_seed(1234)
    for ti, task in enumerate(TASKS):
        rng = np.random.default_rng(10_000 + ti)  # identical held-out sequences for every model
        x, dt, q, y, _ = batch([task], rng, 96, 64)
        nll = m.nll(x, dt, q, y)[:, 8:]
        h = m.trunk(x, dt, q)[:, 8:].reshape(-1, h_dim(m))
        p = pdf_on_grid(m, h, xs)
        crps, pit = grid_metrics(p, xs, y[:, 8:].reshape(-1))
        ps = torch.sort(pit).values
        ks = float((ps - torch.arange(1, len(ps) + 1) / len(ps)).abs().max())
        res["one_step"][task] = dict(nll=float(nll.mean()) - LOG4, crps=float(crps.mean()), pit_ks=ks,
                                     pit_hist=torch.histc(pit, 10, 0, 1).div(len(pit)).tolist())
        # 16-step probabilistic forecast from a 48-sample context: 32 sample paths per sequence (streaming mode)
        n, S, C, Hh = 48, 32, 48, 16
        X, DT, Q, Y = x[:n], dt[:n], q[:n], torch.cat([x[:n], y[:n, -1:]], 1)  # full value sequence 0..64
        Xr, DTr, Qr = X.repeat_interleave(S, 0), DT.repeat_interleave(S, 0), Q.repeat_interleave(S, 0)
        st = m.init_state(Xr.shape[0])
        for t in range(C):
            hcur, st = m.step(Xr[:, t], DTr[:, t], Qr[:, t], st)
        paths = []
        for k in range(Hh):
            pk = pdf_on_grid(m, hcur, xs)
            v = sample_grid(pk, xs, g)
            paths.append(v)
            t = C + k  # v is a sample of x_t; feed it with its own gap dt_t and the next query gap q_t
            if k < Hh - 1:
                hcur, st = m.step(v, DTr[:, t], Qr[:, t], st)
        P = torch.stack(paths, 1).reshape(n, S, Hh) / 4 - 1  # back to x units
        truth = Y[:, C: C + Hh] / 4 - 1
        e1 = (P - truth[:, None]).abs().mean(1)
        e2 = (P[:, :, None] - P[:, None]).abs().mean((1, 2))
        crps_h = (e1 - 0.5 * e2).mean(0)
        rmse_h = ((P.median(1).values - truth) ** 2).mean(0).sqrt()
        res["rollout"][task] = dict(crps_by_h=crps_h.tolist(), rmse_median_by_h=rmse_h.tolist(),
                                    crps_mean=float(crps_h.mean()), rmse_mean=float(rmse_h.mean()))
        if a.kind == "machine" and a.seed == 0:
            np.savez(f"runs/machine/rollout_{task}.npz", paths=P[:4].numpy(), truth=truth[:4].numpy(),
                     context=(Y[:4, :C] / 4 - 1).numpy())
        print(a.kind, a.seed, task, json.dumps({k: round(v, 4) for k, v in res["one_step"][task].items() if k != "pit_hist"}),
              "rollout crps", round(res["rollout"][task]["crps_mean"], 4), flush=True)
    json.dump(res, open(path(a.kind, a.seed) + "_eval.json", "w"), indent=1)


def h_dim(m):
    return m.H if isinstance(m, GRUMDN) else 2 * m.R


def persistence_baseline():
    """No-learning reference: next ~ N(x_t, sigma_task^2), sigma fitted on training sequences, evaluated identically."""
    res = dict(kind="persistence", one_step={})
    for ti, task in enumerate(TASKS):
        tr = np.random.default_rng(1)
        d = np.concatenate([np.diff(gen(task, tr, 64)[0]) for _ in range(200)])
        sig = d.std()
        rng = np.random.default_rng(10_000 + ti)
        x, dt, q, y, _ = batch([task], rng, 96, 64)
        xv, yv = (x[:, 8:] / 4 - 1).numpy(), (y[:, 8:] / 4 - 1).numpy()
        z = (yv - xv) / sig
        nll = 0.5 * z ** 2 + math.log(sig) + 0.5 * math.log(2 * math.pi)
        from math import erf, sqrt, pi
        Phi = 0.5 * (1 + np.vectorize(erf)(z / sqrt(2)))
        phi = np.exp(-0.5 * z ** 2) / sqrt(2 * pi)
        crps = sig * (z * (2 * Phi - 1) + 2 * phi - 1 / sqrt(pi))
        res["one_step"][task] = dict(nll=float(nll.mean()), crps=float(crps.mean()))
    json.dump(res, open("runs/machine/persistence_eval.json", "w"), indent=1)
    return res


def report():
    rows = {}
    for f in sorted(glob.glob("runs/machine/*_eval.json")):
        r = json.load(open(f))
        rows.setdefault(r["kind"], []).append(r)
    out = {}
    for kind, rs in rows.items():
        o = {}
        for task in TASKS:
            for met in ("nll", "crps", "pit_ks"):
                v = [r["one_step"][task][met] for r in rs if met in r["one_step"][task]]
                if v:
                    o[f"{task}/{met}"] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
            v = [r["rollout"][task]["crps_mean"] for r in rs if "rollout" in r]
            if v:
                o[f"{task}/rollout_crps"] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
        for met in ("nll", "crps", "pit_ks", "rollout_crps"):
            per_seed = [np.mean([r["one_step"][t][met] if met != "rollout_crps" else r["rollout"][t]["crps_mean"] for t in TASKS])
                        for r in rs if (met in r["one_step"][TASKS[0]]) or (met == "rollout_crps" and "rollout" in r)]
            if per_seed:
                o[f"ALL/{met}"] = (float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0, len(per_seed))
        out[kind] = o
    json.dump(out, open("runs/machine/summary.json", "w"), indent=1)
    keys = [f"{t}/{m}" for t in TASKS + ("ALL",) for m in ("nll", "crps", "pit_ks", "rollout_crps")]
    print("metric".ljust(26) + "".join(k.ljust(22) for k in out))
    for k in keys:
        if any(k in o for o in out.values()):
            print(k.ljust(26) + "".join((f"{o[k][0]:.4f}±{o[k][1]:.4f}" if k in o else "-").ljust(22) for o in out.values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "report", "persistence"])
    ap.add_argument("--kind", default="machine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--B", type=int, default=24)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    {"train": train, "eval": evaluate, "report": lambda a: report(), "persistence": lambda a: persistence_baseline()}[a.cmd](a)
