"""E8: multi-query associative recall (MQAR-style) — the known weak point of attention-free convolutional mixers.
Sequence: k1 v1 ... kn vn | q1 v(q1) ... qn v(qn). Loss/accuracy only on the answers.
Prime model vs a small causal transformer with a similar parameter count, same data and steps."""
import json, math, sys, time
import numpy as np
import torch
import torch.nn as nn
from pfm.model import PrimeSeqModel

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
which = sys.argv[3] if len(sys.argv) > 3 else "prime,transformer"


def make(B, n, rng):
    X = np.zeros((B, 4 * n), dtype=np.int64)
    for b in range(B):
        keys = rng.choice(np.arange(64, 128), n, replace=False)
        vals = rng.integers(128, 192, n)
        X[b, 0:2 * n:2], X[b, 1:2 * n:2] = keys, vals
        perm = rng.permutation(n)
        X[b, 2 * n::2], X[b, 2 * n + 1::2] = keys[perm], vals[perm]
    x = torch.from_numpy(X)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, 2 * n:-1:2] = True  # predicting the token after each query key
    return x[:, :-1], x[:, 1:], mask[:, :-1]


class TinyTransformer(nn.Module):
    def __init__(self, d=64, L=4, H=4, T=512):
        super().__init__()
        self.family = "table"
        self.emb = nn.Embedding(256, d)
        self.pos = nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, H, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L)
        self.out = nn.Linear(d, 256)

    def forward(self, x):
        T = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(T))
        h = self.enc(h, mask=nn.Transformer.generate_square_subsequent_mask(T), is_causal=True)
        return self.out(h)


def run(name):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    m = PrimeSeqModel(R=64, L=4, K=256) if name == "prime" else TinyTransformer()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3 if name == "prime" else 1e-3)
    t0 = time.time()
    for s in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = g.setdefault("base", g["lr"]) * min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / STEPS))
        n = int(rng.integers(4, 33))
        x, y, mk = make(32, n, rng)
        lo = m(x if name != "prime" else x.double())
        loss = nn.functional.cross_entropy(lo[mk], y[mk])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 250 == 0:
            print(name, s, round(loss.item(), 3), f"{time.time()-t0:.0f}s", flush=True)
    res = dict(params=sum(p.numel() for p in m.parameters()), train_seconds=time.time() - t0, acc={})
    erng = np.random.default_rng(1)
    with torch.no_grad():
        for n in (4, 8, 16, 32, 64):
            x, y, mk = make(64, n, erng)
            lo = m(x if name != "prime" else x.double())
            res["acc"][n] = float((lo[mk].argmax(-1) == y[mk]).float().mean())
    print(name, json.dumps(res), flush=True)
    return res


out = {w: run(w) for w in which.split(",")}
out["chance"] = 1 / 64
json.dump(out, open(f"runs/E8_recall{STEPS}{'_' + which if ',' not in which else ''}.json", "w"), indent=1)
