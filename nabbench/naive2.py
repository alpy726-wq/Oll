"""Fair naive reference: the same causal standardised change d_t/sigma_t as the machine's target, scored as a Gaussian.
Saved as a surprise stream in results/naive_std/ so it goes through exactly the same post-processing."""
import os, sys, math, numpy as np, torch
sys.argv = ["x"]
from score import load_corpus
from detect import prepare
corpus = load_corpus()
os.makedirs("results/naive_std", exist_ok=True)
for rel, f in corpus.items():
    x, dt, prob, y = prepare(f)
    z = (y.numpy() - 4.0)
    s = 0.5 * z ** 2 + 0.919
    s[:prob] = np.nan
    np.savez(f"results/naive_std/{rel.replace('/', '__')}.npz", surprise=s, score=s)
print("done")
