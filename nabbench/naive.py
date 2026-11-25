"""Naive streaming baseline with the same post-processing: surprise under 'next = current + Gaussian noise' (online sigma)."""
import sys, numpy as np, math
sys.argv = ["x", "primeMachine_v1", "skip"]
exec(open("explore3.py").read().split("V = {")[0])
def naive_surprise(rel):
    v = np.array(corpus[rel]["values"], dtype=float); n = len(v)
    prob = int(min(math.floor(0.15 * n), 750)); lo, hi = v[:prob].min(), v[:prob].max(); half = max((hi - lo) / 2, 1e-9)
    z = (v - (lo + hi) / 2) / half * 0.6 * 4   # same units as the machine (x')
    d = np.diff(z, prepend=z[0]); s = np.full(n, np.nan)
    for i in range(prob, n):
        sd = d[max(1, i - 2000):i].std() + 1e-3
        s[i] = 0.5 * (d[i] / sd) ** 2 + math.log(sd) + 0.919
    return s
N = {rel: naive_surprise(rel) for rel in corpus}
for k, fn in {"naive surprise only": lambda rel: sq(np.nan_to_num(N[rel], nan=-50)),
              "spatial OR peak400(naive)": lambda rel: np.maximum(SP[rel] * .99, sq(peak(np.nan_to_num(N[rel], nan=-50), 400)) * .9)}.items():
    r = score_detector(lambda rel: list(fn(rel)), corpus)
    print(k.ljust(32), {p: round(v["normalized"], 2) for p, v in r.items()}, flush=True)
