import sys, numpy as np
nm = sys.argv[1] if len(sys.argv) > 1 else "primeMachine_v1"
sys.argv = ["x", nm, "skip"]
exec(open("explore_post.py").read().split("variants = {")[0])
SP = {rel: spatial_flags(rel) for rel in corpus}
def sm(s, k):
    s = np.nan_to_num(s, nan=-50); o = s.copy()
    for i in range(len(s)): o[i] = s[max(0, i - k + 1):i + 1].mean()
    return o
def peak(z, K):
    o = z.copy()
    for i in range(1, len(z)):
        if z[i] <= z[max(0, i - K):i].max(): o[i] = -50
    return o
sq = lambda a: 1 - 1 / (1 + np.exp(a / 3))
V = {
 "spatial OR surprise": lambda rel: np.maximum(SP[rel] * .99, sq(np.nan_to_num(S[rel], nan=-50)) * .9),
 "spatial OR mean3(surprise)": lambda rel: np.maximum(SP[rel] * .99, sq(sm(S[rel], 3)) * .9),
 "spatial OR peak100(surprise)": lambda rel: np.maximum(SP[rel] * .99, sq(peak(np.nan_to_num(S[rel], nan=-50), 100)) * .9),
 "spatial OR peak400(surprise)": lambda rel: np.maximum(SP[rel] * .99, sq(peak(np.nan_to_num(S[rel], nan=-50), 400)) * .9),
 "surprise only": lambda rel: sq(np.nan_to_num(S[rel], nan=-50)),
 "peak400(surprise) only": lambda rel: sq(peak(np.nan_to_num(S[rel], nan=-50), 400)),
}
for k, fn in V.items():
    r = score_detector(lambda rel: list(fn(rel)), corpus)
    print(k.ljust(32), {p: round(v["normalized"], 2) for p, v in r.items()}, flush=True)
