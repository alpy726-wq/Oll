"""E9: symbolic binding with the group law (Prop. 1): superpose N key-value bindings in ONE vector, unbind each key
with its conjugate, decode the value by phase matching against the 256-byte alphabet. Capacity vs N, R, amplitude."""
import json
import numpy as np
from pfm.basis import first_primes

rng = np.random.default_rng(0)
res = {}
for R in (256, 2048):
    p = first_primes(R); f = np.sqrt(p)
    for sig in (0.5, 0.0):
        A = p ** -sig
        b = lambda x: A * np.exp(2j * np.pi * ((np.outer(x, f)) % 1.0))
        alpha = b(np.arange(256.0))
        for scheme in ("dense_int", "spaced_int", "random_real"):
          accs = {}
          for N in (1, 2, 4, 8, 16, 32, 64, 128):
            ok = tot = 0
            for trial in range(20):
                if scheme == "dense_int":     # keys 1e5 + {0..9999}: k_j + v_j can equal k_i + v exactly -> aliasing
                    keys = 1e5 + rng.choice(10_000, N, replace=False).astype(float)
                elif scheme == "spaced_int":  # typed slots 1e5 + 512*i: sums never collide inside the byte range
                    keys = 1e5 + 512.0 * rng.choice(10_000, N, replace=False)
                else:
                    keys = rng.uniform(1e5, 1e6, N)
                vals = rng.integers(0, 256, N)
                mem = (b(keys) * b(vals.astype(float))).sum(0)          # bind = elementwise product, superpose = sum
                for k, v in zip(keys, vals):
                    rec = mem * np.conj(b(np.array([k])))[0] / A ** 2   # unbind with the conjugate key (amplitude removed)
                    ok += int(np.argmax((np.conj(alpha) * rec).real.sum(1)) == v); tot += 1
            accs[N] = ok / tot
          res[f"R={R},sigma={sig},keys={scheme}"] = accs
          print(R, sig, scheme, accs, flush=True)
json.dump(res, open("runs/E9_binding.json", "w"), indent=1)
