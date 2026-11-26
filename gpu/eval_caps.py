"""Capabilities of the trained main-run OneMachine, measured on held-out REAL data only.
  1. streaming (one event at a time, constant memory) == full pass
  2. invertibility of the trained trunk (float64)
  3. output densities integrate to 1 (closed-form normaliser)
  4. calibration of the predictive distributions: PIT uniformity (KS distance) and CRPS, held-out series tails and
     the frozen v4 test windows; reference: a Gaussian with the empirical step-change width (no learning)
  5. missing data: the same held-out windows with 50% / 75% of the events removed (time gaps grow), NLL per event
  6. forecast fans on ETTh1 test (median and 80% band), saved as numbers for plotting
Writes /vol/results/capabilities.json"""
import argparse, copy, json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, "/oll"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pfm.one import OneMachine
from pfm.bigworld import STREAMS_BIG, BigData, make_batch_big
from pfm.realworld import fixed_eval_set, SID, ETT_VARS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/vol/ckpt/main.pt")
ap.add_argument("--big", default="/vol/big")
ap.add_argument("--v4", default="/oll/data/v4")
ap.add_argument("--out", default="/vol/results/capabilities.json")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
dev = a.device
ck = torch.load(a.ckpt, map_location=dev)
args = json.load(open(os.path.join(os.path.dirname(a.out), "main.json")))["args"] if os.path.exists(os.path.join(os.path.dirname(a.out), "main.json")) else None
cfg = args or dict(R=1024, L=8, M=8, RM=512)
m = OneMachine(STREAMS_BIG, R=cfg["R"], L=cfg["L"], M=cfg["M"], RM=cfg["RM"]).to(dev)
m.load_state_dict(ck["model"]); m.eval()
R = {"trained_steps": ck["step"]}
GRID = torch.linspace(-1.5, 1.5, 3001, dtype=torch.float64, device=dev)


def densities(h, s_out):
    out = torch.empty(h.shape[0], len(GRID), dtype=torch.float64, device=dev)
    for sid in s_out.unique().tolist():
        sel = s_out == sid
        idx = sel.nonzero().squeeze(1)
        for j in range(0, len(idx), 32):  # chunks: the grid x prime-bank product is large
            out[idx[j:j + 32]] = m.density(h[idx[j:j + 32]], sid, GRID).double()
    return out


def calib(b):
    with torch.no_grad():
        h = m(b["x"], b["s"], b["dt"], b["q"])
    mk = b["mask"]
    cont = torch.tensor([STREAMS_BIG[i].kind == "continuous" for i in range(len(STREAMS_BIG))], device=dev)[b["s_out"]]
    sel = mk & cont
    hh, so, y = h[sel], b["s_out"][sel], b["y"][sel]
    d = densities(hh, so)
    dx = float(GRID[1] - GRID[0])
    F = torch.cumsum(d, -1) * dx; F = F / F[:, -1:]
    k = torch.searchsorted(GRID.contiguous(), y.double().contiguous()).clamp(0, len(GRID) - 1)
    pit = F.gather(1, k[:, None]).squeeze(1).cpu().numpy()
    ind = (GRID[None, :] >= y.double()[:, None]).double()
    crps = (((F - ind) ** 2).sum(-1) * dx).cpu().numpy()
    integ = torch.trapezoid(d, GRID, dim=-1).cpu().numpy()
    # no-learning reference: Gaussian centred on the previous value of the same stream, width = empirical change std
    x = b["x"][sel].double().cpu().numpy(); yy = y.double().cpu().numpy()
    sd = max(np.std(yy - x), 1e-3)
    from math import erf, sqrt, pi
    z = (yy - x) / sd
    Phi = 0.5 * (1 + np.vectorize(erf)(z / sqrt(2))); phi = np.exp(-0.5 * z * z) / sqrt(2 * pi)
    crps_ref = sd * (z * (2 * Phi - 1) + 2 * phi - 1 / sqrt(pi))
    ps = np.sort(pit)
    return dict(n=int(len(pit)), pit_ks=float(np.max(np.abs(ps - (np.arange(1, len(ps) + 1) / len(ps))))),
                pit_hist=np.histogram(pit, 10, (0, 1))[0].tolist(), crps=float(crps.mean()), crps_reference=float(crps_ref.mean()),
                density_integral_min_max=[float(integ.min()), float(integ.max())])


torch.set_grad_enabled(False)
vd = BigData(a.big, "val"); rng = np.random.default_rng(2024)
R["calibration_series_heldout"] = calib(make_batch_big(vd, rng, 32, 256, "series", dev))
ev = fixed_eval_set("test")
for dname in ("weather", "grid", "market", "audio"):
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ev[dname].items()}
    R[f"calibration_v4_test_{dname}"] = calib(b)

# streaming == full pass, invertibility
b = make_batch_big(vd, rng, 4, 128, "series", dev)
with torch.no_grad():
    full = m(b["x"], b["s"], b["dt"], b["q"]); st = m.init_state(4); outs = []
    for t in range(128):
        f, st = m.step(b["x"][:, t], b["s"][:, t], b["dt"][:, t], b["q"][:, t], st); outs.append(f)
R["streaming_vs_full_rel_diff"] = float((torch.stack(outs, 1) - full).norm() / full.norm())
md = copy.deepcopy(m).double()
with torch.no_grad():
    e = md.embed(b["x"], b["s"], b["dt"]).double(); hh = md.trunk_from(e, b["dt"].double(), b["q"].double())
    R["inverse_rel_err_fp64"] = float((md.inverse(hh, b["dt"].double(), b["q"].double()) - e).norm() / e.norm())


# missing data: drop events from held-out windows (real timestamps make the gaps exact)
def nll_with_drop(domain_batch, drop, seed=0):
    r = np.random.default_rng(seed)
    x, s, dt, mk = [v.cpu().numpy() for v in (domain_batch["x"], domain_batch["s"], domain_batch["dt"], domain_batch["mask"])]
    t = np.cumsum(dt, 1); tot, cnt = 0.0, 0
    for i in range(x.shape[0]):
        keep = r.random(x.shape[1]) >= drop; keep[0] = True
        xi, si, ti = x[i][keep], s[i][keep], t[i][keep]
        if len(xi) < 20:
            continue
        di = np.diff(ti, prepend=ti[0] - (ti[1] - ti[0]))
        X = torch.tensor(xi, device=dev)[None]; S = torch.tensor(si, device=dev)[None]; D = torch.tensor(di, dtype=torch.float32, device=dev)[None]
        with torch.no_grad():
            l = m.nll(m(X[:, :-1], S[:, :-1], D[:, :-1], D[:, 1:]), S[:, 1:], X[:, 1:])[0, 8:]
        tot += float(l.sum()); cnt += l.numel()
    return tot / max(cnt, 1)


b = make_batch_big(vd, np.random.default_rng(7), 32, 256, "series", dev)
R["missing_data_series_nll_per_event"] = {f"{int(p * 100)}%_removed": nll_with_drop(b, p) for p in (0.0, 0.5, 0.75)}
bw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ev["weather"].items()}
R["missing_data_weather_nll_per_event"] = {f"{int(p * 100)}%_removed": nll_with_drop(bw, p) for p in (0.0, 0.5, 0.75)}

# forecast fans on ETTh1 test (oil temperature channel), 6 windows, 96 steps, 32 sampled paths each
from pfm.ett_eval import load_ett
t, Z, ts_, te_, pm = load_ett("ETTh1", a.v4)
sid = SID["ett:OT"]; starts = list(range(ts_, te_ - 96, (te_ - ts_) // 6))[:6]
fans = []
g = torch.Generator(device=dev).manual_seed(0)
with torch.no_grad():
    for s0 in starts:
        n = 32
        x = torch.tensor(np.clip(Z[s0 - 336:s0, -1] / 3, -1.2, 1.2), device=dev)[None].repeat(n, 1)
        st = m.init_state(n); one = torch.ones(n, device=dev); sids = torch.full((n,), sid, device=dev)
        for k in range(336):
            h, st = m.step(x[:, k], sids, one, one, st)
        paths = []
        for k in range(96):
            d = m.density(h, sid, GRID).double(); cdf = (d / d.sum(-1, keepdim=True)).cumsum(-1)
            u = torch.rand(n, 1, generator=g, device=dev, dtype=torch.float64)
            v = GRID[torch.searchsorted(cdf.contiguous(), u).clamp(max=len(GRID) - 1).squeeze(1)]
            paths.append((v * 3).cpu().numpy()); h, st = m.step(v, sids, one, one, st)
        P = np.stack(paths, 1)
        truth = Z[s0:s0 + 96, -1]
        q10, q50, q90 = np.quantile(P, [0.1, 0.5, 0.9], 0)
        fans.append(dict(start=int(s0), history=Z[s0 - 96:s0, -1].tolist(), truth=truth.tolist(), q10=q10.tolist(), q50=q50.tolist(),
                         q90=q90.tolist(), coverage_80=float(((truth >= q10) & (truth <= q90)).mean())))
R["ett_fans_OT"] = fans
R["ett_fans_mean_coverage_80"] = float(np.mean([f["coverage_80"] for f in fans]))
json.dump(R, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "ett_fans_OT"}, indent=1)[:4000])
