"""Content-addressed phase memory, no training: can the document be searched by the meaning of the question?
Every word gets a phase code from the frozen prime basis (its bytes bound in order: phase sum under fixed permutations,
the same binding as the context keys). A sentence is written into memory as the superposition of its word codes
weighted by rarity (idf, counted from the documents themselves); the question is read out the same way.
<query, key>/RM = sum over shared words of idf^2 (+ crosstalk ~ 1/sqrt(RM)). Measured on SQuAD v1.1 validation:
how often the sentence with the highest score contains the gold answer (chance and exact word-overlap for reference)."""
import json, math, re, sys
from collections import Counter
import numpy as np

RM = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
val = json.load(open(sys.argv[1]))
rng = np.random.default_rng(4242)
primes = []
n = 2
while len(primes) < RM:
    if all(n % p for p in primes if p * p <= n):
        primes.append(n)
    n += 1
f = np.sqrt(np.array(primes, dtype=np.float64))
perms = [np.arange(RM)] + [rng.permutation(RM) for _ in range(31)]
cache = {}


def code(w):
    """unit phase vector of a word: sum_k perm_k(2 pi frac(byte_k * sqrt(p)))"""
    if w not in cache:
        ph = np.zeros(RM)
        for k, b in enumerate(w.encode()[:32]):
            ph += (2 * np.pi * np.modf(b * f / 7.0)[0])[perms[k]]
        cache[w] = np.exp(1j * ph)
    return cache[w]


tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())
sents = lambda c: [s for s in re.split(r"(?<=[.!?])\s+", c) if s.strip()]
docs = {q["context"] for q in val}
df = Counter(w for c in docs for w in set(tok(c)))
idf = {w: math.log(len(docs) / d) for w, d in df.items()}
hit_phase = hit_exact = chance = n_q = 0
for q in val:
    S = sents(q["context"])
    gold = [i for i, s in enumerate(S) if any(a in s for a in q["answers"])]
    if not gold or len(S) < 2:
        continue
    qw = set(tok(q["question"]))
    qv = sum((idf.get(w, 0) * code(w) for w in qw), np.zeros(RM, complex))
    sc_phase, sc_exact = [], []
    for s in S:
        sw = set(tok(s))
        kv = sum((idf.get(w, 0) * code(w) for w in sw), np.zeros(RM, complex))
        sc_phase.append(float(np.real(np.vdot(kv, qv))) / RM)
        sc_exact.append(sum(idf.get(w, 0) ** 2 for w in sw & qw))
    hit_phase += int(np.argmax(sc_phase)) in gold; hit_exact += int(np.argmax(sc_exact)) in gold
    chance += len(gold) / len(S); n_q += 1
res = dict(RM=RM, questions=n_q, sentence_found_phase_memory=hit_phase / n_q, sentence_found_exact_overlap=hit_exact / n_q,
           chance=chance / n_q)
print(json.dumps(res))
json.dump(res, open(f"runs/chat/reader_retrieval_RM{RM}.json", "w"), indent=1)
