"""E1: verify Propositions 1-4, the sigma=1/2 decade shares, typed vocabularies, and the zeta variant."""
import json, math
import numpy as np
from pfm.basis import first_primes, frequencies, amplitudes

R = 2048
p = first_primes(R)
f = np.sqrt(p)
A = p ** -0.5
rng = np.random.default_rng(0)


def b(x, f=f, A=A):
    ph = np.outer(np.atleast_1d(x).astype(np.float64), f)
    ph = 2 * np.pi * (ph - np.floor(ph))
    return A * np.exp(1j * ph)


res = {}
# Prop 1: b(a) * b(c) = A * b(a+c)
a, c = rng.uniform(-1e5, 1e5, 500), rng.uniform(-1e5, 1e5, 500)
dev = np.abs(b(a) * b(c) - A * b(a + c))
res["prop1_homomorphism"] = dict(max_abs_dev=float(dev.max()), mean_abs_dev=float(dev.mean()))
# Prop 2: constant norm
nb = (np.abs(b(np.arange(256.0))) ** 2).sum(1)
mertens = math.log(math.log(p[-1])) + 0.2614972128
res["prop2_norm"] = dict(norm2_mean=float(nb.mean()), spread=float(nb.max() - nb.min()), sum_1_over_p=float((1 / p).sum()),
                         mertens_estimate=mertens)
# Prop 3: translation covariance <b(x),b(y)> depends on x-y only
xs = rng.uniform(-1e4, 1e4, 200)
g = np.array([np.vdot(b(x + 7.0)[0], b(x)[0]) for x in xs]).real  # <b(x),b(x+7)>
closed = float((np.cos(2 * np.pi * 7 * f) / p).sum())
res["prop3_toeplitz"] = dict(mean=float(g.mean()), std=float(g.std()), closed_form=closed)
# Prop 4: continuity  1-cos ~ delta^2
x0 = 123.456
cont = {}
for d in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
    u, v = b(x0)[0], b(x0 + d)[0]
    cont[str(d)] = float(1 - np.real(np.vdot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v)))
res["prop4_continuity_1_minus_cos"] = cont
# injectivity in practice: nearest non-trivial neighbour among integers 0..1e5 (finite-R near-aliasing)
grid = np.arange(0, 20000.0)
B = b(grid[:1]) @ np.conj(b(grid)).T
cs = (B.real / nb.mean())[0]
cs[0] = -1
res["prop4_max_cos_to_other_integers_in_0_2e4"] = float(cs.max())
# exponent 1/2: decade shares
shares = {}
for s in (0.4, 0.5, 0.6):
    w = p ** (-2 * s)
    tot = w.sum()
    edges = [(10, 100), (100, 1000), (1000, 10000), (10000, 1e9)]
    shares[str(s)] = [float(w[(p >= lo) & (p < hi)].sum() / tot) for lo, hi in edges]
res["decade_shares_[10,1e2),[1e2,1e3),[1e3,1e4),[1e4,inf)"] = shares
# typed vocabulary: actions at type*1e5 + index vs byte alphabet
acts = b(1e5 + np.arange(64.0))
bytes_ = b(np.arange(256.0))
sim = np.abs(acts @ np.conj(bytes_).T).real / nb.mean()
res["typed_vocab_max_abs_cos_actions_vs_bytes"] = float(sim.max())
res["typed_vocab_mean_abs_cos"] = float(sim.mean())
res["byte_alphabet_mean_abs_cos_offdiag"] = float(((np.abs(bytes_ @ np.conj(bytes_).T) / nb.mean()) - np.eye(256)).sum() / (256 * 255))

# zeta variant: frequencies log p; Props 1-3 hold, plus multiplicative channel for integers
fl = np.log(p) / (2 * np.pi)
a, c = rng.uniform(-1e3, 1e3, 200), rng.uniform(-1e3, 1e3, 200)
res["zeta_prop1_max_dev"] = float(np.abs(b(a, fl) * b(c, fl) - A * b(a + c, fl)).max())


def mult_channel(n, s=0.5 + 0j):
    """n -> phases sum_k v_pk(n) log p_k  (completely multiplicative: n*m -> phase add)."""
    v = np.zeros(R)
    m = n
    for k, q in enumerate(p[:200].astype(int)):
        while m % q == 0:
            v[k] += 1
            m //= q
    return np.exp(-1j * v * np.log(p))


errs = []
for _ in range(200):
    m1, m2 = rng.integers(2, 3000, 2)
    errs.append(np.abs(mult_channel(m1) * mult_channel(m2) - mult_channel(m1 * m2)).max())
res["zeta_multiplicative_channel_max_dev"] = float(max(errs))

# Zeta zeros from the prime bank (dual explicit formula): the tapered prime-power bank
#   S(t) = -sum_{p^k<=X} log p * p^{-k/2} * (1 - log p^k / log X) * cos(t k log p)
# has its tallest peaks at the ordinates gamma of the zeros. This is a property of the log p basis, not of a trained model.
X = 2_000_000
P = first_primes(150_000)
P = P[P <= X]
LX = np.log(X)
t = np.linspace(12, 50, 3801)
S = np.zeros_like(t)
for k in range(1, 22):
    q = P[P ** k <= X]
    if len(q) == 0:
        break
    w = np.log(q) * q ** (-k / 2) * (1 - k * np.log(q) / LX)
    for ch in np.array_split(np.arange(len(q)), max(1, len(q) // 4000)):
        S -= w[ch].dot(np.cos(np.outer(k * np.log(q[ch]), t)))
pk = [i for i in range(1, len(t) - 1) if S[i] > S[i - 1] and S[i] > S[i + 1]]
top = sorted(float(v) for v in t[sorted(pk, key=lambda i: -S[i])[:10]])
true = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178, 40.918719, 43.327073, 48.005151, 49.773832]
res["zeta_zero_recovery_top10_peaks"] = dict(true_zeros=true, peaks=[round(v, 3) for v in top],
                                             abs_err=[round(abs(a - b), 3) for a, b in zip(top, true)])
res["effective_dimension_(sum A^2)^2/sum A^4"] = float((A ** 2).sum() ** 2 / (A ** 4).sum())
json.dump(res, open("runs/E1_basis.json", "w"), indent=1)
print(json.dumps(res, indent=1))
