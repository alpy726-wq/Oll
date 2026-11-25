"""Independent verification for the audit. Every check prints PASS/FAIL with the measured number; nothing is assumed.
Run: python3 audit/verify.py  -> audit/verify_results.json"""
import hashlib, json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.manual_seed(0)
R = {}


def check(name, ok, **vals):
    R[name] = dict(status="PASS" if ok else "FAIL", **{k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v) for k, v in vals.items()})
    print(("PASS " if ok else "FAIL ") + name, vals, flush=True)


# ------------------------------------------------------------------ 1. basis propositions (re-derived, float64)
from pfm.basis import first_primes
p = first_primes(2048); f = np.sqrt(p); A = p ** -0.5
b = lambda x: A * np.exp(2j * np.pi * ((np.outer(np.atleast_1d(x), f)) % 1.0))
rng = np.random.default_rng(1)
x, y = rng.uniform(-1e5, 1e5, 300), rng.uniform(-1e5, 1e5, 300)
dev = np.abs(b(x) * b(y) - A * b(x + y)).max()
check("P1 homomorphism b(x)b(y)=A b(x+y)", dev < 1e-8, max_abs_dev=dev)
n2 = (np.abs(b(rng.uniform(-1e4, 1e4, 300))) ** 2).sum(1)
check("P2 constant norm = sum 1/p", np.ptp(n2) < 1e-9 and abs(n2.mean() - (1 / p).sum()) < 1e-9, spread=np.ptp(n2))
g = np.array([np.vdot(b(xx + 7.0)[0], b(xx)[0]).real for xx in rng.uniform(-1e4, 1e4, 100)])
closed = (np.cos(2 * np.pi * 7 * f) / p).sum()
check("P3 Toeplitz Gram <b(x),b(x+7)> = closed form", np.abs(g - closed).max() < 1e-8, max_dev=np.abs(g - closed).max())
r1 = 1 - np.real(np.vdot(b(3.3)[0], b(3.3 + 1e-4)[0])) / n2.mean()
r2 = 1 - np.real(np.vdot(b(3.3)[0], b(3.3 + 1e-5)[0])) / n2.mean()
check("P4 continuity 1-cos ~ delta^2 (ratio for 10x delta = 100)", abs(r1 / r2 - 100) < 1, ratio=r1 / r2)

# ------------------------------------------------------------------ 2. closed-form normaliser for every continuous stream
from pfm.one import OneMachine
from pfm.bigworld import STREAMS_BIG
m = OneMachine(STREAMS_BIG, R=64, L=2, M=4, RM=128).double().eval()
for p_ in m.parameters():
    p_.data += 0.05 * torch.randn_like(p_)
xs = torch.linspace(-1.5, 1.5, 60001, dtype=torch.float64)
worst = 0.0
with torch.no_grad():
    h = torch.randn(4, 128, dtype=torch.float64)
    for i, st in enumerate(STREAMS_BIG):
        if st.kind == "continuous":
            d = m.density(h, i, xs).double()
            worst = max(worst, float((torch.trapezoid(d, xs, dim=-1) - 1).abs().max()))
check("Z closed form = numerical integral (all continuous streams)", worst < 1e-4, worst_abs_error=worst)

# ------------------------------------------------------------------ 3. continuous-time operator: three implementations agree
from pfm.continuous import CTConv, DT_MAX
c = CTConv(16, 4).double()
for p_ in c.parameters():
    p_.data += 0.1 * torch.randn_like(p_)
u = torch.randn(3, 40, 16, dtype=torch.float64)
a, cc = c.a_cc()
dt_reg = torch.full((3, 40), 0.7, dtype=torch.float64); q_reg = torch.full((3, 40), 0.3, dtype=torch.float64)
dt_irr = torch.tensor(rng.exponential(3.0, (3, 40))).clamp(max=DT_MAX); q_irr = torch.tensor(rng.uniform(0, 2, (3, 40)))
with torch.no_grad():
    for name, dt, q in (("regular (FFT path)", dt_reg, q_reg), ("irregular incl. long gaps (chunked scan)", dt_irr, q_irr)):
        ccq = cc[None, None] * torch.exp(q[..., None, None] * a)
        ref = c.scan_reference(u.float(), dt.float(), a, ccq.to(torch.complex64)).double()
        out = c(u.float(), dt.float(), q.float()).double()
        err = float((out - ref).norm() / ref.norm())
        check(f"CT operator == sequential recurrence, {name}", err < 1e-4, rel_err=err)
dt_gap = torch.full((2, 64), 72.0)
u2 = torch.randn(2, 64, 16, requires_grad=True)
cf = CTConv(16, 4)
out = cf(u2, dt_gap, torch.zeros(2, 64)); out.sum().backward()
check("CT operator finite values and gradients with 72-unit gaps (the NaN found by the smoke test)",
      bool(torch.isfinite(out).all() and torch.isfinite(u2.grad).all()))

# ------------------------------------------------------------------ 4. gradients: autograd vs finite differences on the whole machine
from pfm.realworld import STREAMS_REAL
mm = OneMachine(STREAMS_REAL, R=16, L=1, M=2, RM=32).double()
for p_ in mm.parameters():
    p_.data += 0.05 * torch.randn_like(p_)
B_, T_ = 2, 10
xx = torch.rand(B_, T_ + 1, dtype=torch.float64) * 2 - 1
ss = torch.zeros(B_, T_ + 1, dtype=torch.long); dd = torch.full((B_, T_ + 1), 0.5)
def lossf():
    hh = mm(xx[:, :-1], ss[:, :-1], dd[:, :-1], dd[:, 1:])
    return mm.nll(hh, ss[:, 1:], xx[:, 1:]).double().mean()
worst = 0.0
for name in ("P.0.pl", "F.0.alpha", "G.0.W.V", "F.0.conv.c_re", "F.0.sync.kp", "gain"):
    prm = dict(mm.named_parameters())[name]
    mm.zero_grad(); lossf().backward(); g_ = prm.grad.detach().clone()
    dvec = torch.randn_like(prm); dvec /= dvec.norm()
    eps = 1e-4
    with torch.no_grad():
        prm.add_(eps * dvec); lp = lossf().item(); prm.sub_(2 * eps * dvec); lm = lossf().item(); prm.add_(eps * dvec)
    fd = (lp - lm) / (2 * eps); an = float((g_ * dvec).sum())
    rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-8)
    worst = max(worst, rel)
    print("   grad", name, "autograd", round(an, 6), "finite-diff", round(fd, 6))
check("Backprop matches finite differences (6 parameter groups, float32 internals)", worst < 2e-2, worst_rel_err=worst)

# ------------------------------------------------------------------ 5. memory: write / read / exact erase on the current code
mem = OneMachine(STREAMS_REAL, R=64, L=1, M=2, RM=512).eval()
from pfm.realworld import SID
keys = torch.tensor(rng.uniform(-1, 1, 20)); vals = torch.tensor(rng.integers(0, 256, 20)).double()
xseq = torch.stack([keys, vals], 1).reshape(1, -1); sseq = torch.tensor([SID["vix"], SID["text"]] * 20)[None]
with torch.no_grad():
    st = mem.init_state(1)
    for t_ in range(xseq.shape[1]):
        _, st = mem.step(xseq[:, t_], sseq[:, t_], torch.ones(1), torch.ones(1), st)
    def recall(state, k):
        kk = torch.polar(torch.ones(1, dtype=torch.float64), mem.mem_phases(torch.tensor([k]), torch.tensor([SID["vix"]]))).to(torch.complex64)
        r = torch.einsum("bm,bmr->br", kk, state["M"][0]) / mem.RM
        cf_ = r.to(torch.complex128) * torch.exp(-1j * mem.psi[SID["text"]])
        return int(mem.discrete_logits(cf_, STREAMS_REAL[SID["text"]]).argmax())
    before_pred = [recall(st, float(k)) for k in keys]
    acc = np.mean([bp == int(v) for bp, v in zip(before_pred, vals)])
    # exact erase of fact 3
    k3 = torch.polar(torch.ones(1, dtype=torch.float64), mem.mem_phases(keys[3:4], torch.tensor([SID["vix"]]))).to(torch.complex64)
    _, ph = mem.code(vals[3:4], torch.tensor([SID["text"]])); v3 = torch.polar(torch.ones_like(ph), ph).to(torch.complex64)
    st2 = dict(st); st2["M"] = st["M"] - (k3.conj()[:, :, None] * v3[:, None, :])[None]
    after = [recall(st2, float(k)) == int(v) for k, v in zip(keys, vals)]
    after_pred = [recall(st2, float(k)) for k in keys]
check("Memory: 20 key->symbol facts written once, recalled (untrained machine, mechanism only)", acc >= 0.95, accuracy=acc)
check("Memory: erasing one fact removes it and leaves every other answer exactly as before", (not after[3]) and all(before_pred[i] == after_pred[i] for i in range(20) if i != 3),
      erased_still_recalled=bool(after[3]), other_answers_changed=sum(before_pred[i] != after_pred[i] for i in range(20) if i != 3))
# multi-event facts: keys are single events (bigram binding). A fact like "France:" -> "Paris" needs a context key.
txt = b"France:Paris\nItaly:Rome\nSpain:Madrid\n"
with torch.no_grad():
    xs_ = torch.tensor(list(txt), dtype=torch.float64)[None]; ss_ = torch.full(xs_.shape, SID["text"])
    st = mem.init_state(1)
    for t_ in range(xs_.shape[1]):
        _, st = mem.step(xs_[:, t_], ss_[:, t_], torch.ones(1), torch.ones(1), st)
    out = recall_txt = None
    k = torch.polar(torch.ones(1, dtype=torch.float64), mem.mem_phases(torch.tensor([float(ord(":"))]), torch.tensor([SID["text"]]))).to(torch.complex64)
    r = torch.einsum("bm,bmr->br", k, st["M"][0]) / mem.RM
    lg = mem.discrete_logits(r.to(torch.complex128) * torch.exp(-1j * mem.psi[SID["text"]]), STREAMS_REAL[SID["text"]])[0]
    top3 = [chr(int(i)) for i in lg.topk(3).indices]
check("Memory LIMITATION: key is the single previous event, so 'X:' retrieves a superposition of all capitals' first letters",
      set(top3) >= {"P", "R", "M"} or len(set(top3) & {"P", "R", "M"}) >= 2, top3_after_colon=top3)

# ------------------------------------------------------------------ 6. evaluation protocol and data hygiene
from pfm.ett_eval import load_ett
borders = {}
for name, exp in (("ETTh1", (11520, 14400)), ("ETTh2", (11520, 14400)), ("ETTm1", (46080, 57600))):
    t, Z, a_, b_, pm = load_ett(name, "data/v4")
    borders[name] = (a_, b_, len(Z))
    check(f"ETT borders {name} = standard (test {exp[0]}..{exp[1]})", (a_, b_) == exp and len(Z) >= b_, test_start=a_, test_end=b_, n=len(Z))
man = json.load(open("data/v4/MANIFEST.json")); bad = []
for k_, v in man.items():
    fpath = f"data/v4/{k_}.npz" if os.path.exists(f"data/v4/{k_}.npz") else f"data/v4/{k_}.json"
    if hashlib.sha256(open(fpath, "rb").read()).hexdigest() != v["sha256"]:
        bad.append(k_)
check("Frozen corpus: every file matches its SHA-256", not bad, mismatched=bad)
# text test/train overlap: fraction of 64-byte test windows that occur verbatim in train
tr = np.load("data/v4/text_train.npz")["bytes"]; te = np.load("data/v4/text_test.npz")["bytes"]
def shingles(a, k=32, step=1):
    a = a.astype(np.uint64); h = np.zeros(len(a) - k + 1, dtype=np.uint64)
    for i in range(k):
        h = h * np.uint64(1099511628211) + a[i:len(a) - k + 1 + i]
    return h[::step]
htr = np.unique(shingles(tr)); hte = shingles(te, step=97)
ov = np.isin(hte, htr).mean()
check("Text leakage: share of 32-byte test windows found verbatim in train (Gutenberg boilerplate expected)", ov < 0.05, overlap=ov)
fs_tr = np.load("data/v4/fsdd_train.npz"); fs_te = np.load("data/v4/fsdd_test.npz")
check("Audio split: standard FSDD split (same speakers in train and test -> speaker generalisation NOT tested)",
      True, shared_speakers=sorted(set(fs_tr["speaker"]) & set(fs_te["speaker"])))

os.makedirs("audit", exist_ok=True)
json.dump(R, open("audit/verify_results.json", "w"), indent=1)
print("\nSUMMARY", sum(v["status"] == "PASS" for v in R.values()), "PASS /", len(R))
