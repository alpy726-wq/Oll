"""Alternative post-processings of the saved surprise streams (no retraining, model outputs fixed)."""
import glob, os, sys, math, json
import numpy as np
from score import load_corpus, score_detector

name = sys.argv[1] if len(sys.argv) > 1 else "primeMachine_v1"
corpus = load_corpus()
S = {rel: np.load(f"results/{name}/{rel.replace('/', '__')}.npz")["surprise"] for rel in corpus}


def tail(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def numenta_ll(s, short=10, long=8640, min_hist=100, spatial=None):
    n = len(s); out = np.zeros(n); avg = []; buf = []
    for i in range(n):
        if np.isnan(s[i]):
            continue
        buf.append(s[i]); buf = buf[-short:]
        a = float(np.mean(buf))
        if len(avg) >= min_hist:
            h = np.array(avg[-long:]); mu, sd = h.mean(), max(h.std(), 1e-3)
            z = (a - mu) / sd
            p = tail(z) if z > 0 else 1 - tail(-z)
            out[i] = min(1.0, math.log(p + 1e-10) / math.log(1e-10))
        avg.append(a)
    if spatial is not None:
        out = np.maximum(out, spatial)
    return out


def spatial_flags(rel, tol=0.05):
    v = np.array(corpus[rel]["values"]); n = len(v); f = np.zeros(n)
    lo, hi = v[0], v[0]
    for i in range(1, n):
        r = hi - lo
        if r > 0 and (v[i] > hi + tol * r or v[i] < lo - tol * r):
            f[i] = 1.0
        lo, hi = min(lo, v[i]), max(hi, v[i])
    return f


variants = {
    "raw_surprise_rank": lambda rel: list(1 - 1 / (1 + np.exp(np.nan_to_num(S[rel], nan=-50) / 3))),
    "ll_short10": lambda rel: list(numenta_ll(S[rel])),
    "ll_short5": lambda rel: list(numenta_ll(S[rel], short=5)),
    "ll_short1": lambda rel: list(numenta_ll(S[rel], short=1)),
}
for k, fn in ({} if len(sys.argv) > 2 else variants).items():
    r = score_detector(fn, corpus)
    print(k.ljust(20), {p: round(v["normalized"], 2) for p, v in r.items()}, flush=True)


def robust_z(s, min_hist=100, long=8640):
    n = len(s); z = np.zeros(n); hist = []
    for i in range(n):
        if np.isnan(s[i]):
            continue
        if len(hist) >= min_hist:
            h = np.array(hist[-long:]); med = np.median(h); mad = np.median(np.abs(h - med)) * 1.4826 + 1e-3
            z[i] = (s[i] - med) / mad
        hist.append(s[i])
    return z


def peak_only(z, K):
    out = z.copy()
    for i in range(len(z)):
        if i > 0 and z[i] <= z[max(0, i - K):i].max():
            out[i] = min(z[i], 0) - 0.0  # not a new local peak -> not an alarm
    return out


def to01(z):
    return list(0.5 + 0.5 * z / (1 + np.abs(z)))


Z = {rel: robust_z(S[rel]) for rel in corpus}
SP = {rel: spatial_flags(rel) for rel in corpus}
more = {
    "robust_z": lambda rel: to01(Z[rel]),
    "robust_z_peak50": lambda rel: to01(peak_only(Z[rel], 50)),
    "robust_z_peak200": lambda rel: to01(peak_only(Z[rel], 200)),
    "robust_z+spatial": lambda rel: to01(np.maximum(Z[rel], SP[rel] * 1e3)),
    "robust_z_peak200+spatial": lambda rel: to01(peak_only(np.maximum(Z[rel], SP[rel] * 1e3), 200)),
    "spatial_only": lambda rel: list(SP[rel] * 0.9 + 0.05),
}
for k, fn in more.items():
    r = score_detector(fn, corpus)
    print(k.ljust(26), {p: round(v["normalized"], 2) for p, v in r.items()}, flush=True)
