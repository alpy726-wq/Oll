"""Oll-Reader: answers a question from documents, on CPU.
  search: the documents are written once into a phase memory keyed by meaning (words -> phase codes from the frozen
          prime basis; a sentence is the idf-weighted superposition of its words); the question reads it out
  read:   a small linear model (6.6 MB, trained on SQuAD v1.1 train) picks the answer span in the best sentences
  confidence: the probability of the chosen span; low confidence -> it says it does not know
SQuAD v1.1 validation (10,570 questions, never trained on): exact match 33.2 %, F1 42.9 %; when its confidence is
>= 0.6 it is right 70 % of the time (answers 19 % of questions), >= 0.8: 84 %. English only."""
import math, re, string, zlib
from collections import Counter
import numpy as np
import torch


class _Cfg:
    RM, L, top, D = 4096, 6, 2, 1 << 21
    unk_idf = 0.0  # weight of a word never seen when the idf was counted (training: never happens)


a = _Cfg()


def configure(RM=4096, L=6, top=2, D=1 << 21):
    a.RM, a.L, a.top, a.D = RM, L, top, D
    _init()


def _init():
    global RM, fq, perms, _codes
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


# ---------------------------------------------------------------- phase-memory search (frozen prime basis)
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
        qv = sum((idf.get(w, a.unk_idf) * code(w) for w in qs), np.zeros(RM, np.complex64))
        ss = sent_spans(ctx)
        sc = []
        for s, e in ss:
            sw = {w.lower() for w, _, _ in words(ctx[s:e]) if w[0].isalnum()}
            kv = sum((idf.get(w, a.unk_idf) * code(w) for w in sw), np.zeros(RM, np.complex64))
            sc.append(float(np.real(np.vdot(kv, qv))) / RM)
        order = np.argsort(sc)[::-1][: a.top]
        qe = float(np.real(np.vdot(qv, qv))) / RM
        self.support = (max(sc) / qe) if (sc and qe > 0) else 0.0  # share of the question's meaning found in the best sentence
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



def make_idf(texts):
    docs = list(texts)
    df = Counter(w for c in docs for w in {x.lower() for x, _, _ in words(c) if x[0].isalnum()})
    return {w: math.log(len(docs) / d) for w, d in df.items()}


class Reader:
    def __init__(self, path="models/oll-reader.pt", unk_rare=True, min_support=0.2):
        ck = torch.load(path, map_location="cpu")
        ar = ck["args"]
        configure(ar["RM"], ar["L"], ar["top"], ar["D"])
        self.W = ck["W"].float()[:, 0]
        self.idf = ck["idf"]
        self.min_support = min_support  # below this the documents do not talk about the question: not sure
        a.unk_idf = max(self.idf.values()) if unk_rare else 0.0  # a word never seen before is as rare as the rarest


    def ask(self, document, question):
        """-> dict(answer, confidence, sentence)"""
        e = Ex(document, question, self.idf)
        if not e.cands:
            return dict(answer="", confidence=0.0)
        s = torch.stack([self.W[torch.tensor(fs)].sum() for fs in e.feats])
        p = torch.softmax(s, 0); j = int(p.argmax())
        ans = e.cands[j]
        at = document.find(ans)
        sent = next((document[s0:e0] for s0, e0 in sent_spans(document) if s0 <= at < e0), "") if at >= 0 else ""
        return dict(answer=ans, confidence=float(p[j]) if e.support >= self.min_support else 0.0, span_probability=float(p[j]),
                    support=e.support, sentence=sent.strip())


_init()
