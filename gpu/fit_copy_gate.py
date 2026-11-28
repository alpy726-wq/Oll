"""Fit the copy gate of a trained Oll-Chat (trunk frozen) on cached outputs, then evaluate generation.
1. Run the frozen model over real training windows (dialog + general), cache per byte: log p_trunk(true), log p_copy(true)
   for a few copy temperatures, and the context-familiarity features km_n (orders 4, 8, 12).
2. Fit lam = sigmoid(b + sum_n a_n km_n) (and the copy temperature) by maximum likelihood of the mixture.
3. Evaluate on held-out data: bits per byte (general, dialog), copy probes, SQuAD validation generation (EM / F1)."""
import argparse, json, math, os, re, string, sys, time
from collections import Counter
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pfm.one import OneMachine
from pfm.bigworld import STREAMS_BIG
from pfm.realworld import SID

TEXT = SID["text"]
ap = argparse.ArgumentParser()
ap.add_argument("--init", default="models/oll-chat.pt")
ap.add_argument("--data", required=True)
ap.add_argument("--RM", type=int, default=4096)
ap.add_argument("--orders", default="1,4,8,12")
ap.add_argument("--n_fit", type=int, default=24)
ap.add_argument("--T", type=int, default=1024)
ap.add_argument("--n_squad", type=int, default=64)
ap.add_argument("--max_new", type=int, default=30)
ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="runs/chat/copy_gate_fit.json")
ap.add_argument("--save", default="models/oll-chat-copy.pt")
a = ap.parse_args()
torch.set_num_threads(a.threads); dev = a.device
orders = tuple(int(v) for v in a.orders.split(","))
ck = torch.load(a.init, map_location=dev); cfg = ck["config"]
m = OneMachine(STREAMS_BIG, R=cfg["R"], L=cfg["L"], M=cfg["M"], RM=a.RM, mem_orders=orders, copy_out=True).to(dev)
m.load_state_dict({k: v.float() if v.dtype == torch.float16 else v for k, v in ck["model"].items()}, strict=False); m.eval()
with torch.no_grad():  # input path unchanged (order 1 as trained); copy path from orders >= 4 only
    m.mem_order_logw.copy_(torch.tensor([math.log(len(orders))] + [-30.0] * (len(orders) - 1)))
    m.copy_gain.copy_(torch.tensor([0.0 if n == 1 else 3.0 for n in orders]))
gen = np.load(f"{a.data}/general.npy", mmap_mode="r"); dia = np.load(f"{a.data}/dialog.npy", mmap_mode="r")
gcut, dcut = int(0.99 * len(gen)), int(0.99 * len(dia))
squad = json.load(open(f"{a.data}/squad_val.json"))[: a.n_squad]
TAUS = (-1.0, -0.5, 0.0, 0.5)


@torch.no_grad()
def cache(seq):
    X = torch.tensor([seq], dtype=torch.float64, device=dev); s = torch.full(X.shape, TEXT, device=dev); d = torch.ones(X.shape, device=dev)
    h = m(X[:, :-1], s[:, :-1], d[:, :-1], d[:, 1:])[0]
    R = m.R; y = X[0, 1:].long(); sid = torch.full(h.shape[:-1], TEXT, device=dev); st = m.streams[TEXT]
    lt = torch.log_softmax(m.discrete_logits(m.coeffs(h[:, :2 * R], sid), st) * m.sym_logtemp[TEXT].exp(), -1).gather(1, y[:, None])[:, 0]
    raw = m.discrete_logits(m.coeffs(h[:, 2 * R:4 * R], sid), st)
    lc = torch.stack([torch.log_softmax(raw * math.exp(t), -1).gather(1, y[:, None])[:, 0] for t in TAUS], -1)
    return lt.double(), lc.double(), m._copy[1][0].double()


def windows(arr, lo, hi, n, r):
    return [list(arr[i:i + a.T + 1]) for i in r.integers(lo, hi - a.T - 2, n)]


t0 = time.time(); r = np.random.default_rng(0)
fit = [cache(q) for q in windows(dia, 0, dcut, a.n_fit // 2, r) + windows(gen, 0, gcut, a.n_fit // 2, r)]
print("cached", len(fit), "windows in", round(time.time() - t0), "s", flush=True)
LT = torch.cat([f[0][16:] for f in fit]); LC = torch.cat([f[1][16:] for f in fit]); KM = torch.cat([f[2][16:] for f in fit])
use = [i for i, n in enumerate(orders) if n > 1]
best = None
for ti, tau in enumerate(TAUS):
    wa = torch.zeros(len(use), dtype=torch.float64, requires_grad=True); wb = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([wa, wb], max_iter=200, line_search_fn="strong_wolfe")
    def nllf():
        lam = torch.sigmoid(wb + KM[:, use] @ wa)
        return -torch.logaddexp(torch.log1p(-lam + 1e-9) + LT, torch.log(lam + 1e-9) + LC[:, ti]).mean()
    def closure():
        opt.zero_grad(); l = nllf(); l.backward(); return l
    opt.step(closure)
    l = float(nllf()) / math.log(2)
    print(f"tau {tau}: fitted bits/byte {l:.4f} (trunk alone {float(-LT.mean()) / math.log(2):.4f}) a={wa.detach().numpy().round(2)} b={float(wb):.2f}", flush=True)
    if best is None or l < best[0]:
        best = (l, tau, wa.detach().clone(), float(wb))
with torch.no_grad():
    m.copy_logtemp.fill_(best[1]); m.gate_b.fill_(best[3]); m.gate_km.zero_()
    for j, i in enumerate(use):
        m.gate_km[i] = best[2][j]
res = dict(args=vars(a), fit=dict(bits_mixture=best[0], bits_trunk=float(-LT.mean()) / math.log(2), copy_logtemp=best[1],
                                   gate_km=[float(v) for v in m.gate_km], gate_b=best[3]))


@torch.no_grad()
def bits(seq, copy=True):
    X = torch.tensor([seq], dtype=torch.float64, device=dev); s = torch.full(X.shape, TEXT, device=dev); d = torch.ones(X.shape, device=dev)
    h = m(X[:, :-1], s[:, :-1], d[:, :-1], d[:, 1:])
    if not copy:
        h = h[..., :2 * m.R]
    return (m.nll(h, s[:, 1:], X[:, 1:])[0] / math.log(2)).cpu().numpy()


vr = np.random.default_rng(123)
for k, arr, lo, hi in (("general", gen, gcut, len(gen)), ("dialog", dia, dcut, len(dia))):
    ws = windows(arr, lo, hi, 6, vr)
    res[f"heldout_{k}"] = dict(copy=float(np.mean([bits(w)[16:].mean() for w in ws])), trunk_only=float(np.mean([bits(w, False)[16:].mean() for w in ws])))
pr = np.random.default_rng(7); L = 40
for gap in (0, 400):
    sq = []
    for _ in range(10):
        i, j = pr.integers(gcut, len(gen) - 2000, 2); sp = list(gen[i:i + L]); sq.append(sp + list(gen[j:j + gap]) + sp)
    res[f"copy_gap{gap}"] = {c: dict(first=float(np.mean([bits(x, c)[:L - 1].mean() for x in sq])), repeat=float(np.mean([bits(x, c)[-L + 5:].mean() for x in sq])))
                             for c in (True, False)}
print(json.dumps({k: v for k, v in res.items() if k != "args"}), flush=True)


def norm(t):
    t = t.lower(); t = "".join(ch for ch in t if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", t).split())


def f1(p, g):
    pt, gt = norm(p).split(), norm(g).split(); ns = sum((Counter(pt) & Counter(gt)).values())
    return 0.0 if ns == 0 else 2 * (ns / len(pt)) * (ns / len(gt)) / (ns / len(pt) + ns / len(gt))


@torch.no_grad()
def answer(prompts, max_new):
    outs, confs = [], []
    for c0 in range(0, len(prompts), a.chunk):
        P = [p.encode("utf-8")[-1800:] for p in prompts[c0:c0 + a.chunk]]
        Lm = max(len(p) for p in P); B = len(P)
        X = torch.tensor([[10] * (Lm - len(p)) + list(p) for p in P], dtype=torch.float64, device=dev)
        s = torch.full((B,), TEXT, device=dev); one = torch.ones(B, device=dev); st = m.init_state(B)
        for k in range(Lm):
            h, st = m.step(X[:, k], s, one, one, st)
        done = torch.zeros(B, dtype=torch.bool, device=dev); txt = [[] for _ in range(B)]; lp = torch.zeros(B, device=dev); n = torch.zeros(B, device=dev)
        for _ in range(max_new):
            logp = m.symbol_logp(h, TEXT).float(); c = logp.argmax(-1); stop = (c == 10) | done
            lp = lp + torch.where(stop, torch.zeros_like(lp), logp.gather(1, c[:, None])[:, 0]); n = n + (~stop).float()
            for i in range(B):
                if not stop[i]:
                    txt[i].append(int(c[i]))
            done = stop
            if bool(done.all()):
                break
            h, st = m.step(c.double(), s, one, one, st)
        outs += [bytes(t).decode("utf-8", "replace").strip() for t in txt]; confs += (lp / n.clamp(min=1)).tolist()
        print("answered", len(outs), flush=True)
    return outs, confs


t0 = time.time()
ans, conf = answer([f"Document: {q['context'].strip()}\nUser: {q['question'].strip()}\nOll: " for q in squad], a.max_new)
em = [max(float(norm(p) == norm(g)) for g in q["answers"]) for p, q in zip(ans, squad)]
fs = [max(f1(p, g) for g in q["answers"]) for p, q in zip(ans, squad)]
res["squad"] = dict(n=len(squad), exact_match=float(np.mean(em)), f1=float(np.mean(fs)), seconds=time.time() - t0,
                    examples=[dict(q=q["question"], gold=q["answers"][0], answer=p, conf=round(c, 3)) for p, q, c in zip(ans, squad, conf)])
print("squad", res["squad"]["exact_match"], res["squad"]["f1"], flush=True)
prompts = ["Document: Maria lives in Istanbul and studies prime numbers.\nUser: Where does Maria live?\nOll: ",
           "Document: The meeting was moved to Thursday at 3 pm in room 204.\nUser: When is the meeting?\nOll: ",
           "User: What is the capital of France?\nOll: "]
outs, confs = answer(prompts, 60)
res["samples"] = [dict(prompt=p, answer=o) for p, o in zip(prompts, outs)]
for x in res["samples"]:
    print(json.dumps(x, ensure_ascii=False), flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
cfg2 = dict(cfg, RM=a.RM, mem_orders=list(orders), copy_out=True)
torch.save(dict(config=cfg2, model={k: (v.half() if v.is_floating_point() and not k.startswith(("copy_", "gate_", "mem_order")) else v)
                                    for k, v in m.state_dict().items()}), a.save)
print("DONE", flush=True)
