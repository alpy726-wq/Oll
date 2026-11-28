"""Oll-Reader v0 (CPU only): answer a question from a document.
  1. search: phase-memory retrieval of the sentence by the meaning of the question (gpu/reader_retrieval.py math:
     words -> phase codes from the frozen prime basis, sentence = idf-weighted superposition; no training)
  2. read: every word span (1..L words) of the top sentences is scored by a linear model over hashed features
     (question type x span shape, closeness to the question's words, neighbours, overlap with the question,
     retrieval rank); trained on SQuAD v1.1 train (real, human-written) with a softmax over the spans, on CPU
  3. know what it does not know: the probability of the chosen span is its confidence; below a threshold it says so.
Evaluated on SQuAD v1.1 validation (never trained on): EM / F1, and accuracy by confidence."""
import argparse, json, math, re, string, sys, time, zlib
from collections import Counter
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--RM", type=int, default=4096)
ap.add_argument("--L", type=int, default=6)
ap.add_argument("--top", type=int, default=2)
ap.add_argument("--D", type=int, default=1 << 21)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--n_train", type=int, default=0)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--out", default="runs/chat/reader.json")
ap.add_argument("--save", default="models/oll-reader.pt")
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)

# ---------------------------------------------------------------- phase-memory search (frozen prime basis)
RM = a.RM
primes, n = [], 2
while len(primes) < RM:
    if all(n % p for p in primes if p * p <= n):
        primes.append(n)
    n += 1
fq = np.sqrt(np.array(primes, dtype=np.float64))
prng = np.random.default_rng(4242)
perms = [np.arange(RM)] + [prng.permutation(RM) for _ in range(31)]
_codes = {}


def code(w):
    if w not in _codes:
        ph = np.zeros(RM)
        for k, b in enumerate(w.encode()[:32]):
            ph += (2 * np.pi * np.modf(b * fq / 7.0)[0])[perms[k]]
        _codes[w] = np.exp(1j * ph).astype(np.complex64)
    return _codes[w]


WORD = re.compile(r"[A-Za-z0-9]+(?:[\-'][A-Za-z0-9]+)*|[^\sA-Za-z0-9]")


def words(t):
    return [(m.group(), m.start(), m.end()) for m in WORD.finditer(t)]


def sent_spans(c):
    out, st = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+", c):
        out.append((st, m.start())); st = m.end()
    out.append((st, len(c)))
    return [(s, e) for s, e in out if e > s]


def norm(t):
    t = t.lower(); t = "".join(ch for ch in t if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", t).split())


def f1(p, g):
    pt, gt = norm(p).split(), norm(g).split(); ns = sum((Counter(pt) & Counter(gt)).values())
    return 0.0 if ns == 0 else 2 * (ns / len(pt)) * (ns / len(gt)) / (ns / len(pt) + ns / len(gt))


WH = ("how many", "how much", "how long", "how old", "how far", "how big", "what year", "which year", "what percentage",
      "who", "whom", "whose", "when", "where", "why", "which", "what", "how")


def qtype(q):
    ql = q.lower()
    for w in WH:
        if re.search(r"\b" + w + r"\b", ql):
            nxt = re.search(r"\b" + w + r"\b\s+([a-z0-9]+)", ql)
            return w, (nxt.group(1) if nxt else "")
    return "none", ""


def shape(w):
    if w.isdigit():
        return "d" if len(w) != 4 else "dddd"
    if any(ch.isdigit() for ch in w):
        return "xd"
    if w[:1].isupper():
        return "X"
    if w.isalpha():
        return "x"
    return "p" + w


def H(s):
    return zlib.crc32(s.encode()) % a.D


class Ex:
    """candidate spans of one (document, question) and their hashed features"""

    def __init__(self, ctx, q, idf):
        qw = [w.lower() for w, _, _ in words(q) if w[0].isalnum()]
        qs = set(qw)
        qv = sum((idf.get(w, 0) * code(w) for w in qs), np.zeros(RM, np.complex64))
        ss = sent_spans(ctx)
        sc = []
        for s, e in ss:
            sw = {w.lower() for w, _, _ in words(ctx[s:e]) if w[0].isalnum()}
            kv = sum((idf.get(w, 0) * code(w) for w in sw), np.zeros(RM, np.complex64))
            sc.append(float(np.real(np.vdot(kv, qv))) / RM)
        order = np.argsort(sc)[::-1][: a.top]
        wt, nx = qtype(q)
        self.cands, self.feats = [], []
        for rank, si in enumerate(order):
            s0, e0 = ss[si]
            ws = words(ctx[s0:e0]); lw = [w.lower() for w, _, _ in ws]
            inq = [w in qs and w[0].isalnum() for w in lw]
            qpos = [i for i, b in enumerate(inq) if b]
            for i in range(len(ws)):
                if not ws[i][0][0].isalnum():
                    continue
                for j in range(i, min(len(ws), i + a.L)):
                    if not ws[j][0][0].isalnum():
                        break
                    span = lw[i:j + 1]
                    ln = j - i + 1
                    ovl = sum(inq[i:j + 1]) / ln
                    dist = min([abs(p - i) if p < i else abs(p - j) for p in qpos if p < i or p > j] or [99])
                    near = sum(1 for p in qpos if i - 4 <= p < i or j < p <= j + 4)
                    shp = "".join(sorted(set(shape(ws[k][0]) for k in range(i, j + 1))))
                    left = lw[i - 1] if i > 0 else "<s>"; right = lw[j + 1] if j + 1 < len(lw) else "</s>"
                    fs = [f"b", f"rank{rank}", f"len{ln}", f"ovl{round(ovl * 4)}", f"dist{min(dist, 8)}", f"near{min(near, 4)}",
                          f"wt{wt}|shp{shp}", f"wt{wt}|len{ln}", f"wt{wt}|nx{nx}|shp{shp}", f"wt{wt}|first{shape(ws[i][0])}",
                          f"wt{wt}|last{shape(ws[j][0])}", f"L{left}", f"R{right}", f"wt{wt}|L{left}", f"wt{wt}|R{right}",
                          f"nx{nx}|R{right}", f"nx{nx}|L{left}", f"dist{min(dist, 8)}|ovl{round(ovl * 4)}",
                          f"Lq{int(i > 0 and inq[i - 1])}|Rq{int(j + 1 < len(inq) and inq[j + 1])}", f"wt{wt}|dist{min(dist, 8)}",
                          f"rank{rank}|near{min(near, 4)}", f"sc{min(int(sc[si] / (max(sc) + 1e-9) * 4), 4)}"]
                    fs += [f"wt{wt}|w{w}" for w in span[:3]] + [f"nx{nx}|w{w}" for w in span[:2]]
                    fs += [f"inq{int(w in qs)}|{shape(ws[i + k][0])}" for k, w in enumerate(span)]
                    self.cands.append(ctx[s0 + ws[i][1]: s0 + ws[j][2]]); self.feats.append([H(x) for x in fs])


def build(rows, idf, gold=True):
    exs = []
    for c, q, ans in rows:
        e = Ex(c, q, idf)
        if not e.cands:
            continue
        e.ans = ans
        e.y = next((k for k, s in enumerate(e.cands) if norm(s) in {norm(x) for x in ans}), -1)
        if gold and e.y < 0:
            continue
        exs.append(e)
    return exs


def batch(exs):
    ids, off, seg = [], [], []
    for b, e in enumerate(exs):
        for fs in e.feats:
            off.append(len(ids)); ids += fs; seg.append(b)
    return torch.tensor(ids), torch.tensor(off), torch.tensor(seg)


W = torch.nn.EmbeddingBag(a.D, 1, mode="sum"); torch.nn.init.zeros_(W.weight)


def scores(exs):
    ids, off, seg = batch(exs)
    return W(ids, off)[:, 0], seg


def logsoftmax_seg(s, seg, nb):
    mx = torch.full((nb,), -1e9).scatter_reduce(0, seg, s, "amax")
    z = torch.zeros(nb).index_add(0, seg, (s - mx[seg]).exp()).log() + mx
    return s - z[seg]


import pyarrow.parquet as pq
t0 = time.time()
tr = pq.read_table(f"{a.data}/raw/squad_train.parquet").to_pydict()
train_rows = [(c, q, x["text"]) for c, q, x in zip(tr["context"], tr["question"], tr["answers"])]
if a.n_train:
    train_rows = train_rows[: a.n_train]
val = json.load(open(f"{a.data}/squad_val.json"))
docs = {c for c, _, _ in train_rows}
df = Counter(w for c in docs for w in {x.lower() for x, _, _ in words(c) if x[0].isalnum()})
idf = {w: math.log(len(docs) / d) for w, d in df.items()}
opt = torch.optim.Adagrad(W.parameters(), lr=0.2)
res = dict(args=vars(a), curve=[])
rng = np.random.default_rng(0)
for ep in range(a.epochs):
    order = rng.permutation(len(train_rows)); tot, nb_, kept = 0.0, 0, 0
    for b0 in range(0, len(order), 64):
        exs = build([train_rows[i] for i in order[b0:b0 + 64]], idf)
        if not exs:
            continue
        s, seg = scores(exs)
        ls = logsoftmax_seg(s, seg, len(exs))
        starts = torch.tensor(np.cumsum([0] + [len(e.cands) for e in exs[:-1]]))
        loss = -ls[starts + torch.tensor([e.y for e in exs])].mean() + 1e-6 * 0
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss); nb_ += 1; kept += len(exs)
        if nb_ % 200 == 0:
            print(f"epoch {ep} batch {nb_} loss {tot / nb_:.3f} kept {kept} {time.time() - t0:.0f}s", flush=True)
    res["curve"].append(dict(epoch=ep, loss=tot / max(nb_, 1), trainable_questions=kept, seconds=time.time() - t0))
    print(res["curve"][-1], flush=True)

# ---------------------------------------------------------------- held-out evaluation
val_rows = [(q["context"], q["question"], q["answers"]) for q in val]
docs_v = {c for c, _, _ in val_rows}
preds, confs, ems, f1s, ex = [], [], [], [], []
with torch.no_grad():
    for b0 in range(0, len(val_rows), 128):
        rows = val_rows[b0:b0 + 128]
        exs = build(rows, idf, gold=False)
        s, seg = scores(exs)
        p = logsoftmax_seg(s, seg, len(exs)).exp()
        k0 = 0
        for e in exs:
            pe = p[k0:k0 + len(e.cands)]; k0 += len(e.cands)
            j = int(pe.argmax()); ans = e.cands[j]
            preds.append(ans); confs.append(float(pe[j]))
            ems.append(max(float(norm(ans) == norm(g)) for g in e.ans)); f1s.append(max(f1(ans, g) for g in e.ans))
            if len(ex) < 30:
                ex.append(dict(gold=e.ans[0], answer=ans, confidence=round(float(pe[j]), 3)))
ems, f1s, confs = np.array(ems), np.array(f1s), np.array(confs)
o = np.argsort(confs); qn = len(o) // 4
res["squad_validation"] = dict(n=len(ems), exact_match=float(ems.mean()), f1=float(f1s.mean()),
                               em_most_confident_quarter=float(ems[o[-qn:]].mean()), em_least_confident_quarter=float(ems[o[:qn]].mean()),
                               em_by_confidence={f">={t}": dict(fraction_answered=float((confs >= t).mean()), em=float(ems[confs >= t].mean()) if (confs >= t).any() else None)
                                                 for t in (0.0, 0.2, 0.4, 0.6, 0.8)},
                               examples=ex, seconds=time.time() - t0)
print(json.dumps({k: v for k, v in res["squad_validation"].items() if k != "examples"}), flush=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
torch.save(dict(W=W.weight.detach().half(), args=vars(a), idf=idf), a.save)
print("DONE", flush=True)
