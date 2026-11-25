"""Every capability of the paper, measured on ONE trained OneMachine (runs/one/one_s0.pt)."""
import copy, json, math, sys
import numpy as np
import torch
from pfm.one import OneMachine, PHASE_PARAMS
from pfm.oneworld import STREAMS, batch, make, seq_recall, seq_fusion
from pfm.world import TASKS, _rk4

torch.set_num_threads(4)
CK = sys.argv[1] if len(sys.argv) > 1 else "runs/one/one_s0"
args = json.load(open(CK + "_train.json"))["args"]
m = OneMachine(STREAMS, R=args["R"], L=args["L"], family=args["family"], seed=args["seed"])
base = torch.load(CK + ".pt")
m.load_state_dict(base)
m.eval()
R = {}


def masked_nll(model, b):
    with torch.no_grad():
        l = model.nll(model(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"])
    return l, b["mask"]


# ---------------------------------------------------------------- 1. continuous world (6 families, irregular time)
fam = {}
for i, f in enumerate(TASKS):
    rng = np.random.default_rng(10_000 + i)
    b = batch(rng, 64, 64, kinds=["signal"] * 64, family=f)
    l, mk = masked_nll(m, b)
    fam[f] = float(l[:, 8:].mean())
R["1_continuous_world_nll_by_family"] = fam
R["1_continuous_world_nll_mean"] = float(np.mean(list(fam.values())))

# bimodal: predicted density has two peaks at x +/- 0.25
rng = np.random.default_rng(77)
b = batch(rng, 16, 64, kinds=["signal"] * 16, family="bimodal")
with torch.no_grad():
    h = m(b["x"], b["s"], b["dt"], b["q"])[:, -1]
    xs = torch.linspace(-1.5, 1.5, 3001, dtype=torch.float64)
    d = m.density(h, 0, xs)
dx = float(xs[1] - xs[0])
cur = b["x"][:, -1]
near = lambda c: float(np.mean([(d[i] * ((xs - c[i]).abs() < 0.08)).sum().item() * dx for i in range(16)]))
inside = (cur.abs() < 0.7)
R["2_multimodal_density"] = dict(
    integral=float(torch.trapezoid(d, xs.float(), dim=-1).mean()),
    mass_near_x_plus_0p25=near(cur + 0.25), mass_near_x_minus_0p25=near(cur - 0.25), mass_near_x=near(cur))

# ---------------------------------------------------------------- 3. text (discrete symbols through the same head)
rng = np.random.default_rng(5)
b = batch(rng, 64, 64, kinds=["text"] * 64, split="val")
l, mk = masked_nll(m, b)
R["3_text_bits_per_byte"] = float(l[:, 8:].mean() / math.log(2))

# ---------------------------------------------------------------- 4. associative recall via the phase memory
rec, rec_off = {}, {}
m_off = copy.deepcopy(m)
with torch.no_grad():
    m_off.mem_beta.zero_()
for n in (2, 4, 8, 16):
    rng = np.random.default_rng(100 + n)
    seqs = [seq_recall(rng, 64, n=n) for _ in range(64)]
    x = torch.tensor(np.stack([q[0] for q in seqs])); s = torch.tensor(np.stack([q[1] for q in seqs]))
    dt = torch.ones(64, 65); mk = torch.tensor(np.stack([q[3] for q in seqs]))
    for model, out in ((m, rec), (m_off, rec_off)):
        with torch.no_grad():
            h = model(x[:, :-1], s[:, :-1], dt[:, :-1], dt[:, 1:])
            c = model.coeffs(h[mk], torch.full((int(mk.sum()),), 4))
            pred = model.discrete_logits(c, STREAMS[4]).argmax(-1)  # temperature does not change argmax
        out[n] = float((pred == x[:, 1:][mk].long()).float().mean())
R["4_recall_accuracy"] = rec
R["4_recall_accuracy_memory_off"] = rec_off
R["4_recall_chance"] = 1 / 64

# ---------------------------------------------------------------- 5. multi-sensor fusion (no fusion layer)
rng = np.random.default_rng(55)
with_a, without_a = [], []
for _ in range(48):
    x, s, dt, _, _ = seq_fusion(rng, 64)
    t = np.cumsum(dt)
    xb, sb, tb = x[s == 2], s[s == 2], t[s == 2]
    for xs_, ss_, ts_ in ((x, s, t), (xb, sb, tb)):
        dts = np.diff(ts_, prepend=ts_[0] - 0.5)
        X = torch.tensor(xs_)[None]; S = torch.tensor(ss_)[None]; D = torch.tensor(dts, dtype=torch.float32)[None]
        with torch.no_grad():
            l = m.nll(m(X[:, :-1], S[:, :-1], D[:, :-1], D[:, 1:]), S[:, 1:], X[:, 1:])[0]
        tgt_b = (S[0, 1:] == 2)
        tgt_b[:8] = False
        (with_a if len(xs_) == len(x) else without_a).append(float(l[tgt_b].mean()))
R["5_fusion_sensorB_nll"] = dict(with_sensorA=float(np.mean(with_a)), without_sensorA=float(np.mean(without_a)))

# ---------------------------------------------------------------- 6. streaming == full pass, constant state (incl. memory)
rng = np.random.default_rng(9)
b = batch(rng, 4, 64)
with torch.no_grad():
    full = m(b["x"], b["s"], b["dt"], b["q"])
    st = m.init_state(4)
    outs = []
    for t in range(64):
        f_, st = m.step(b["x"][:, t], b["s"][:, t], b["dt"][:, t], b["q"][:, t], st)
        outs.append(f_)
    sizes = [m.state_numel(st)]
    for t in range(500):
        _, st = m.step(b["x"][:, t % 64], b["s"][:, t % 64], b["dt"][:, t % 64], b["q"][:, t % 64], st)
    sizes.append(m.state_numel(st))
R["6_streaming_vs_full_rel_diff"] = float((torch.stack(outs, 1) - full).norm() / full.norm())
R["6_state_size_after_64_and_564_events"] = sizes

# ---------------------------------------------------------------- 7. invertibility (no information loss through depth)
md = copy.deepcopy(m).double()
with torch.no_grad():
    e = md.embed(b["x"], b["s"], b["dt"]).double()
    hh = md.trunk_from(e, b["dt"].double(), b["q"].double())
    R["7_inverse_rel_err_fp64"] = float((md.inverse(hh, b["dt"].double(), b["q"].double()) - e).norm() / e.norm())

# ---------------------------------------------------------------- 8. arithmetic and typed regions inside the representation
with torch.no_grad():
    a_ = torch.rand(200, dtype=torch.float64) * 10; c_ = torch.rand(200, dtype=torch.float64) * 10
    s0 = torch.zeros(200, dtype=torch.long)
    pa, pc, pac = m.phases(a_, s0), m.phases(c_, s0), m.phases(a_ + c_, s0)
    psi = m.psi[0]
    dev = torch.angle(torch.exp(1j * ((pa - psi) + (pc - psi) - (pac - psi)))).abs().max()
    R["8_group_law_max_phase_dev"] = float(dev)
    # typed regions: the same value in two streams differs exactly by the role phase
    p0, p1 = m.phases(a_, s0), m.phases(a_, torch.ones(200, dtype=torch.long))
    R["8_role_binding_max_dev"] = float(torch.angle(torch.exp(1j * (p1 - p0 - (m.psi[1] - m.psi[0])))).abs().max())
    # multiplicative integer channel (zeta variant): phase(n*k) = phase(n) + phase(k)
    n1 = torch.tensor([6., 10., 12., 15., 21., 35.]); n2 = torch.tensor([7., 9., 11., 4., 25., 2.])
    ip = lambda n: m.integer_phase(n)
    R["8_integer_channel_multiplicative_max_dev"] = float((ip(n1 * n2) - ip(n1) - ip(n2)).abs().max())

# ---------------------------------------------------------------- 9. continuous time on the trained machine
with torch.no_grad():
    conv = m.F[0].conv
    a, cc = conv.a_cc()
    s0_ = torch.randn(2, m.R, a.shape[1], dtype=torch.complex64)
    R["9_time_semigroup_max_dev"] = float((torch.exp(1.91 * a) * (torch.exp(0.37 * a) * s0_) - torch.exp(2.28 * a) * s0_).abs().max())
    rng = np.random.default_rng(3)
    b = batch(rng, 8, 64, kinds=["signal"] * 8, family="oscillator")
    h1 = m(b["x"], b["s"], b["dt"], b["q"]); h2 = m(b["x"], b["s"], b["dt"] * 2, b["q"] * 2)
    R["9_same_values_double_time_rel_change"] = float((h1 - h2).norm() / h1.norm())

# ---------------------------------------------------------------- 10. synchrony (Kuramoto coupling, Sec. 5.5)
coup = [float((br.sync.kp @ br.sync.kq.T).norm()) for br in list(m.F) + list(m.G)]
m_ns = copy.deepcopy(m)
with torch.no_grad():
    for br in list(m_ns.F) + list(m_ns.G):
        br.sync.kp.zero_()
rng = np.random.default_rng(12)
b = batch(rng, 96, 64)
l1, mk = masked_nll(m, b); l2, _ = masked_nll(m_ns, b)
R["10_kuramoto_coupling_norms"] = coup
R["10_nll_with_sync_vs_without"] = [float((l1 * mk).sum() / mk.sum()), float((l2 * mk).sum() / mk.sum())]


# ---------------------------------------------------------------- 11. phase adapters: new skills, composition, removal
def vdp(rng, T):  # Van der Pol oscillator (never seen in training)
    mu = rng.uniform(0.5, 2.5)
    tr = _rk4(lambda s: np.array([s[1], mu * (1 - s[0] ** 2) * s[1] - s[0]]), rng.uniform(-2, 2, 2), 0.05, 4 * (T + 1) * 6)[::6]
    x = tr[: T + 1, 0] / 2.5
    dt = np.full(T + 1, 0.3)
    return np.clip(x, -1.2, 1.2), np.zeros(T + 1, int), dt, np.ones(T, bool), "vdp"


def duffing(rng, T):  # driven Duffing oscillator (never seen in training)
    w = rng.uniform(1.0, 1.4)
    f = lambda s: np.array([s[1], -0.2 * s[1] + s[0] - s[0] ** 3 + 0.3 * np.cos(w * s[2]), 1.0])
    tr = _rk4(f, np.array([rng.uniform(-1, 1), 0.0, 0.0]), 0.05, 6 * (T + 1) + 10)[::6]
    return np.clip(tr[: T + 1, 0] / 1.6, -1.2, 1.2), np.zeros(T + 1, int), np.full(T + 1, 0.3), np.ones(T, bool), "duffing"


def mk_batch(fn, rng, B, T=64):
    seqs = [fn(rng, T) for _ in range(B)]
    x = torch.tensor(np.stack([q[0] for q in seqs])); s = torch.tensor(np.stack([q[1] for q in seqs]))
    dt = torch.tensor(np.stack([q[2] for q in seqs]), dtype=torch.float32)
    return dict(x=x[:, :-1], s=s[:, :-1], dt=dt[:, :-1], q=dt[:, 1:], y=x[:, 1:], s_out=s[:, 1:], mask=torch.ones(B, T, dtype=torch.bool))


def ev(model, fn, seed=4242):
    l, _ = masked_nll(model, mk_batch(fn, np.random.default_rng(seed), 64))
    return float(l[:, 8:].mean())


def band_mask(name, p, band, nb):
    """adapter restricted to one frequency band of channels (Sec. 5.3 bandwidth allocation); kq stays shared-frozen."""
    leaf = name.split(".")[-1]
    if leaf == "kq":
        return torch.zeros_like(p)
    n = p.shape[0]
    mask = torch.zeros_like(p)
    mask[band * n // nb:(band + 1) * n // nb] = 1
    return mask


def train_adapter(fn, steps=300, seed=1, band=None, nb=2):
    mm = copy.deepcopy(m); mm.load_state_dict(base); mm.train()
    ps = [p for n, p in mm.named_parameters() if n.split(".")[-1] in PHASE_PARAMS]
    for n, p in mm.named_parameters():
        p.requires_grad = n.split(".")[-1] in PHASE_PARAMS
        if p.requires_grad and band is not None:
            msk = band_mask(n, p, band, nb)
            p.register_hook(lambda g, msk=msk: g * msk)
    opt = torch.optim.Adam(ps, lr=float(__import__("os").environ.get("ADAPTER_LR", "3e-3")))
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        bb = mk_batch(fn, rng, 24)
        loss = mm.nll(mm(bb["x"], bb["s"], bb["dt"], bb["q"]), bb["s_out"], bb["y"])[:, 4:].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    mm.eval()
    return mm.adapter(base), sum(p.numel() for p in ps)


ad_v, n_ad = train_adapter(vdp)
ad_d, _ = train_adapter(duffing, seed=2)
mv, md_, mc = copy.deepcopy(m), copy.deepcopy(m), copy.deepcopy(m)
mv.apply_adapters(base, ad_v); md_.apply_adapters(base, ad_d); mc.apply_adapters(base, ad_v, ad_d)
mback = copy.deepcopy(m); mback.apply_adapters(base, ad_v, ad_d, {k: -v for k, v in ad_v.items()}, {k: -v for k, v in ad_d.items()})
R["11_adapter_params"] = n_ad
R["11_adapter_fraction_of_model"] = n_ad / sum(p.numel() for p in m.parameters())
R["11_vdp_nll"] = dict(base=ev(m, vdp), vdp_adapter=ev(mv, vdp), duffing_adapter=ev(md_, vdp), both=ev(mc, vdp))
R["11_duffing_nll"] = dict(base=ev(m, duffing), vdp_adapter=ev(mv, duffing), duffing_adapter=ev(md_, duffing), both=ev(mc, duffing))
bv, _ = train_adapter(vdp, band=0)
bd, _ = train_adapter(duffing, seed=2, band=1)
bv = {k: v for k, v in bv.items() if v.abs().max() > 0}; bd = {k: v for k, v in bd.items() if v.abs().max() > 0}
mbv, mbd, mbc = copy.deepcopy(m), copy.deepcopy(m), copy.deepcopy(m)
mbv.apply_adapters(base, bv); mbd.apply_adapters(base, bd); mbc.apply_adapters(base, bv, bd)
R["11b_band_adapters_vdp_nll"] = dict(vdp_adapter=ev(mbv, vdp), both=ev(mbc, vdp))
R["11b_band_adapters_duffing_nll"] = dict(duffing_adapter=ev(mbd, duffing), both=ev(mbc, duffing))
R["11_remove_adapters_restores_base_max_dev"] = max(float((mback.state_dict()[k] - base[k]).abs().max()) for k in base)

json.dump(R, open(CK + "_capabilities.json", "w"), indent=1)
print(json.dumps(R, indent=1))
