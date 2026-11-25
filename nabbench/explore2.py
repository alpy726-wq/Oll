import sys, numpy as np
sys.argv = ["x", sys.argv[1] if len(sys.argv) > 1 else "primeMachine_v1", "skip"]
exec(open("explore_post.py").read().split("variants = {")[0])
SP = {rel: spatial_flags(rel) for rel in corpus}
raw = lambda rel, k: 1 - 1 / (1 + np.exp(np.nan_to_num(S[rel], nan=-50) / k))
for k in (2, 3, 5):
    for w in (0.5, 0.8):
        fn = lambda rel: list(np.maximum(SP[rel] * 0.99, raw(rel, k) * w))
        r = score_detector(fn, corpus)
        print(f"spatial OR surprise(k={k}, cap={w})".ljust(36), {p: round(v["normalized"], 2) for p, v in r.items()}, flush=True)
