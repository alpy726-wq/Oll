"""Exploration of the trained OneMachine (v2): abilities that come from its structure, without any further training."""
import json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pfm.one import OneMachine
from pfm.oneworld import STREAMS, seq_recall, seq_fusion, batch
from pfm.world import gen

torch.set_num_threads(4)
CK = "runs/one/one_v2"
args = json.load(open(CK + "_train.json"))["args"]
m = OneMachine(STREAMS, R=args["R"], L=args["L"], family=args["family"], seed=args["seed"])
m.load_state_dict(torch.load(CK + ".pt")); m.eval()
OUT = {}
XS = torch.linspace(-1.5, 1.5, 1501, dtype=torch.float64)


def run(x, s, dt, q, st=None):
    """stream events through the machine; returns last features and state."""
    st = st or m.init_state(x.shape[0])
    with torch.no_grad():
        for t in range(x.shape[1]):
            h, st = m.step(x[:, t], s[:, t], dt[:, t], q[:, t], st)
    return h, st


def sym_pred(h):
    with torch.no_grad():
        return m.discrete_logits(m.coeffs(h, torch.full(h.shape[:-1], 4)), STREAMS[4]).argmax(-1)


def T(a, dtype=torch.float64):
    return torch.tensor(np.asarray(a), dtype=dtype)


# ------------------------------------------------------------- A. one-shot learning, surgical forgetting, editing
rng = np.random.default_rng(1)
n = 12
keys = rng.uniform(-1, 1, n); vals = rng.integers(0, 64, n)
xs = np.ravel(np.stack([keys, vals], 1)); ss = np.tile([3, 4], n)
h, st = run(T(xs)[None], torch.tensor(ss)[None], torch.ones(1, 2 * n), torch.ones(1, 2 * n))


def ask(st, k):
    h, _ = m.step(T([k]), torch.tensor([3]), torch.ones(1), torch.ones(1), {kk: (v if kk != "conv" else [c.clone() for c in v]) for kk, v in st.items()})
    return int(sym_pred(h)[0])


def code_pair(k, v):
    kk = torch.polar(torch.ones(1, dtype=torch.float64), m.mem_phases(T([k]), torch.tensor([3]))).to(torch.complex64)
    _, ph = m.code(T([float(v)]), torch.tensor([4]))
    vv = torch.polar(torch.ones_like(ph), ph).to(torch.complex64)
    return kk.conj()[:, :, None] * vv[:, None, :]


with torch.no_grad():
    before = [ask(st, k) == v for k, v in zip(keys, vals)]
    st_forget = dict(st); st_forget["M"] = st["M"] - code_pair(keys[4], vals[4])  # erase one fact exactly
    after_forget = [ask(st_forget, k) == v for k, v in zip(keys, vals)]
    new_val = (int(vals[7]) + 17) % 64
    st_edit = dict(st); st_edit["M"] = st["M"] - code_pair(keys[7], vals[7]) + code_pair(keys[7], new_val)  # edit a fact
    edit_ok = ask(st_edit, keys[7]) == new_val
    others_ok = [ask(st_edit, k) == v for i, (k, v) in enumerate(zip(keys, vals)) if i != 7]
OUT["A_one_shot_recall_of_12_facts_seen_once"] = float(np.mean(before))
OUT["A_after_erasing_fact_4"] = dict(fact_4_still_recalled=bool(after_forget[4]),
                                     other_11_facts_recalled=float(np.mean([a for i, a in enumerate(after_forget) if i != 4])))
OUT["A_after_editing_fact_7"] = dict(new_value_returned=bool(edit_ok), other_11_facts_recalled=float(np.mean(others_ok)))

# ------------------------------------------------------------- B. virtual sensor: infer B where only A is observed
rng = np.random.default_rng(21)
err_m, err_last, err_mean = [], [], []
for _ in range(40):
    x, s, dt, _, _ = seq_fusion(rng, 96)
    t = np.cumsum(dt)
    cut = 48  # after event 48, sensor B goes dark; A keeps reporting
    keep = np.arange(len(x)) < cut
    keep |= (s == 1)
    xo, so, to = x[keep], s[keep], t[keep]
    hidden = np.where((~keep) & (s == 2))[0]
    if len(hidden) == 0:
        continue
    dto = np.diff(to, prepend=to[0] - 0.5)
    st = m.init_state(1)
    last_b = x[:cut][s[:cut] == 2][-1]
    hb = iter(hidden)
    nxt = next(hb, None)
    with torch.no_grad():
        for i in range(len(xo)):
            # before feeding observed event i, query B at every hidden B time that falls between t_{i-1} and t_i
            while nxt is not None and i > 0 and to[i - 1] <= t[nxt] < to[i]:
                q = torch.tensor([t[nxt] - to[i - 1]], dtype=torch.float32)
                hq, _ = m.step(T([xo[i - 1]]), torch.tensor([so[i - 1]]), torch.tensor([float(dto[i - 1])]), q,
                               dict(conv=[c.clone() for c in prev_st["conv"]], M=prev_st["M"], prev=prev_st["prev"]))
                dens = m.density(hq, 2, XS)[0]
                est = float((dens * XS).sum() / dens.sum())
                if to[i - 1] > t[cut]:
                    err_m.append(abs(est - x[nxt])); err_last.append(abs(last_b - x[nxt]))
                nxt = next(hb, None)
            prev_st = dict(conv=[c.clone() for c in st["conv"]], M=st["M"], prev=st["prev"])
            _, st = m.step(T([xo[i]]), torch.tensor([so[i]]), torch.tensor([float(dto[i])]),
                           torch.tensor([float(dto[i + 1]) if i + 1 < len(xo) else 0.5]), st)
OUT["B_virtual_sensor_mean_abs_error"] = dict(machine=float(np.mean(err_m)), last_known_value=float(np.mean(err_last)), n=len(err_m))

# ------------------------------------------------------------- C/D. time elasticity and missing data
def nll_family(fam, gap_scale=1.0, drop=0.0, seed=5):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(48):
        x, dt = gen(fam, rng, 140)
        t = np.cumsum(dt)
        idx = np.arange(len(x))
        if drop > 0:
            idx = idx[rng.random(len(x)) > drop]
        if gap_scale != 1.0:  # resample the same trajectory family at a different rate: stretch the clock
            t = t * gap_scale
        idx = idx[:65]
        xx, tt = x[idx], t[idx]
        d = np.diff(tt, prepend=tt[0] - np.median(np.diff(tt)))
        X = T(xx)[None]; D = torch.tensor(d, dtype=torch.float32)[None]; S = torch.zeros_like(X, dtype=torch.long)
        with torch.no_grad():
            l = m.nll(m(X[:, :-1], S[:, :-1], D[:, :-1], D[:, 1:]), S[:, 1:], X[:, 1:])[0, 8:]
        out.append(float(l.mean()))
    return float(np.mean(out))


OUT["C_oscillator_nll"] = {"trained_rate": nll_family("oscillator"), "50%_of_samples_deleted": nll_family("oscillator", drop=0.5),
                           "75%_deleted": nll_family("oscillator", drop=0.75)}
OUT["C_pendulum_nll"] = {"trained_rate": nll_family("pendulum"), "50%_of_samples_deleted": nll_family("pendulum", drop=0.5),
                         "75%_deleted": nll_family("pendulum", drop=0.75)}

# ------------------------------------------------------------- E. dreaming: free-running futures from one context
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
g = torch.Generator().manual_seed(0)
for a, fam in zip(ax, ("bimodal", "pendulum", "lorenz")):
    rng = np.random.default_rng(3)
    x, dt = gen(fam, rng, 100)
    ctx = 48
    S_ = 24
    X = T(x[:ctx])[None].repeat(S_, 1); D = torch.tensor(dt[:ctx], dtype=torch.float32)[None].repeat(S_, 1)
    Q = torch.tensor(dt[1:ctx + 1], dtype=torch.float32)[None].repeat(S_, 1)
    h, st = run(X, torch.zeros_like(X, dtype=torch.long), D, Q)
    paths = []
    with torch.no_grad():
        for k in range(30):
            p = m.density(h, 0, XS).double()
            cdf = (p / p.sum(-1, keepdim=True)).cumsum(-1)
            u = torch.rand(S_, 1, generator=g, dtype=torch.float64)
            v = XS[torch.searchsorted(cdf.contiguous(), u).clamp(max=len(XS) - 1).squeeze(1)]
            paths.append(v.numpy())
            t_ = ctx + k
            h, st = m.step(v, torch.zeros(S_, dtype=torch.long), torch.tensor(dt[t_], dtype=torch.float32).repeat(S_),
                           torch.tensor(dt[t_ + 1], dtype=torch.float32).repeat(S_), st)
    P = np.stack(paths, 1)
    tc = np.cumsum(dt)
    a.plot(tc[:ctx], x[:ctx], "k.-", lw=1, label="seen")
    for i in range(S_):
        a.plot(tc[ctx:ctx + 30], P[i], color="C0", alpha=0.25, lw=0.8)
    a.plot(tc[ctx:ctx + 30], x[ctx:ctx + 30], "r.-", lw=1.2, label="what actually happened")
    a.set_title(f"{fam}: 24 imagined futures"); a.set_xlabel("time")
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("runs/one/dreams.png", dpi=130)

json.dump(OUT, open("runs/one/explore.json", "w"), indent=1)
print(json.dumps(OUT, indent=1))
