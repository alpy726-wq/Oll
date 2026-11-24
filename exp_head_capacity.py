"""E5a: can |s(x)|^2/Z represent a localized density at all?  Best least-squares fit of s ~ sqrt(target) for a
Gaussian target of width w on the window [-2,10]; reports excess NLL (nats) over the target's entropy.
Frequencies are divided by 'scale' (= the fixed affine input rescaling of Sec. 7)."""
import json
import numpy as np
from pfm.basis import frequencies

res = {}
xs = np.linspace(-2, 10, 6001); dx = xs[1] - xs[0]
for fam in ("sqrt_p", "log_p", "random", "geometric"):
    for R in (128, 512):
        for sc in (1, 2, 4, 8, 16):
            f = frequencies(R, fam) / sc
            M = np.exp(-2j * np.pi * np.outer(xs, f))
            row = {}
            for w in (0.3, 0.1, 0.03):
                tgt = np.exp(-0.5 * ((xs - 4) / w) ** 2) / np.sqrt(2 * np.pi * w * w)
                c, *_ = np.linalg.lstsq(M, np.sqrt(tgt).astype(complex), rcond=None)
                s2 = np.abs(M @ c) ** 2; q = s2 / (s2.sum() * dx)
                row[w] = dict(excess_nll=float(-(tgt * np.log(q + 1e-300)).sum() * dx - 0.5 * np.log(2 * np.pi * np.e * w * w)),
                              mass_outside_5w=float(q[np.abs(xs - 4) > 5 * w].sum() * dx))
            res[f"{fam},R={R},scale=1/{sc}"] = row
            print(fam, R, sc, {k: round(v["excess_nll"], 2) for k, v in row.items()}, {k: round(v["mass_outside_5w"], 3) for k, v in row.items()}, flush=True)
json.dump(res, open("runs/E5a_head_capacity.json", "w"), indent=1)
