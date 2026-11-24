import os
import numpy as np
import torch

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DOMAINS = ("prose", "code", "json")


def load(domain, split):
    return np.frombuffer(open(os.path.join(DATA, f"{domain}.{split}"), "rb").read(), dtype=np.uint8)


class ByteStream:
    def __init__(self, domains=DOMAINS, split="train", seed=0):
        self.arrs = [load(d, split) for d in domains]
        self.rng = np.random.default_rng(seed)

    def batch(self, B, T):
        xs = []
        for i in range(B):
            a = self.arrs[i % len(self.arrs)]
            s = self.rng.integers(0, len(a) - T - 1)
            xs.append(a[s: s + T + 1])
        x = torch.from_numpy(np.stack(xs).astype(np.int64))
        return x[:, :-1], x[:, 1:]


def eval_windows(domain, n=48, T=512, split="val", seed=123):
    a = load(domain, split)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(a) - T - 1, n)
    x = torch.from_numpy(np.stack([a[s: s + T + 1] for s in starts]).astype(np.int64))
    return x[:, :-1], x[:, 1:]


@torch.no_grad()
def bpb(model, x, y, ctx_skip=0, bs=16):
    """bits per byte, scored on positions >= ctx_skip."""
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(x), bs):
        xi, yi = x[i: i + bs], y[i: i + bs]
        inp = xi if model.family == "table" else xi.double()
        lp = torch.log_softmax(model(inp), -1)
        nll = -lp.gather(-1, yi.unsqueeze(-1)).squeeze(-1)[:, ctx_skip:]
        tot += nll.sum().item()
        cnt += nll.numel()
    model.train()
    return tot / cnt / np.log(2)


def eval_all(model, T=512, n=48, domains=DOMAINS):
    r = {d: bpb(model, *eval_windows(d, n, T)) for d in domains}
    r["avg3"] = float(np.mean([r[d] for d in DOMAINS])) if set(DOMAINS) <= set(r) else float(np.mean([r[d] for d in domains]))
    g = torch.Generator().manual_seed(7)
    u = torch.randint(0, 256, (16, T + 1), generator=g)
    r["uniform_random"] = bpb(model, u[:, :-1], u[:, 1:])
    return r
