"""No-learning references on the same frozen windows: persistence Gaussian (continuous) and train unigram (text)."""
import json, math
import numpy as np
import torch
from pfm.realworld import fixed_eval_set, RealData, make_batch, DOMAINS
tr = RealData("train"); rng = np.random.default_rng(0)
sig = {}
for d in ("weather", "grid", "market", "audio"):
    b = make_batch(tr, rng, 256, 128, [d])
    same = b["s"] == b["s_out"]
    sig[d] = float((b["y"] - b["x"])[same & b["mask"]].std())
uni = np.bincount(tr.text, minlength=256) + 1.0; uni = uni / uni.sum()
out = {}
for split in ("val", "test"):
    ev = fixed_eval_set(split); r = {}
    for d, b in ev.items():
        if d == "text":
            r[d] = float(-np.log2(uni[b["y"][b["mask"]].long().numpy()]).mean())
        else:  # predict the last value of the SAME stream (persistence), Gaussian with train width
            x, s, y, so = b["x"].numpy(), b["s"].numpy(), b["y"].numpy(), b["s_out"].numpy()
            nll = []
            for i in range(x.shape[0]):
                last = {}
                for t in range(x.shape[1]):
                    last[s[i, t]] = x[i, t]
                    if b["mask"][i, t] and so[i, t] in last:
                        z = (y[i, t] - last[so[i, t]]) / sig[d]
                        nll.append(0.5 * z * z + math.log(sig[d]) + 0.5 * math.log(2 * math.pi))
            r[d] = float(np.mean(nll))
    out[split] = r
json.dump(out, open("runs/real/reference.json", "w"), indent=1)
print(json.dumps(out, indent=1))
