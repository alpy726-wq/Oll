"""Oscillator-bank sequence model on the frozen prime basis.

No embedding table, no attention, no feed-forward block, no positional encoding.
  * time mixing: causal per-channel convolution, kernel length K, envelope (j+1)^{-1/2}, via FFT
  * depth: reversible blocks  y1 = x1 + F(x2),  y2 = x2 + G(y1)
  * inside F, G: circulant phase filter e^{i theta} (unitary), spectral operator W = U diag(e^sigma) V^T,
    U frozen orthogonal, V initialised to U (so W = I at birth), per-channel gate alpha (init 0 -> block = identity)
  * readout: phase matching against the same frozen basis
Implementation choice not fixed by the paper: F/G use a multiplicative gate (conv path x silu(input)),
which supplies the only nonlinearity. The block is invertible regardless of what F and G are.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .basis import PrimeBasis, LearnedTable


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def fft_causal_conv(u, k):
    """u: (B,T,C), k: (C,K) -> (B,T,C);  y_t = sum_j k_j u_{t-j}.  O(T log T)."""
    T, K = u.shape[1], k.shape[1]
    n = 1 << math.ceil(math.log2(T + K))
    U = torch.fft.rfft(u.transpose(1, 2), n=n)
    Kf = torch.fft.rfft(k, n=n)
    return torch.fft.irfft(U * Kf, n=n)[..., :T].transpose(1, 2)


class CausalConv(nn.Module):
    def __init__(self, C, K):
        super().__init__()
        self.K = K
        self.register_buffer("zeta", (torch.arange(K) + 1.0) ** -0.5, persistent=False)
        kap = torch.randn(C, K) * 0.02
        kap[:, 0] = 1.0
        self.kappa = nn.Parameter(kap)

    def kernel(self):
        return self.kappa * self.zeta

    def forward(self, u):
        return fft_causal_conv(u, self.kernel())

    def step(self, u_t, buf):
        """u_t: (B,C); buf: (B,K,C) holding u_{t-1..t-K}, newest first. Returns y_t, new buf."""
        buf = torch.cat([u_t.unsqueeze(1), buf[:, :-1]], 1)
        return torch.einsum("bkc,ck->bc", buf, self.kernel()), buf


class PhaseFilter(nn.Module):
    """Circulant all-pass filter: view C real channels as C/2 complex, FFT, multiply e^{i theta}, iFFT."""

    def __init__(self, C):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(C // 2))

    def forward(self, u):
        z = torch.view_as_complex(u.reshape(*u.shape[:-1], -1, 2).contiguous())
        z = torch.fft.ifft(torch.fft.fft(z, norm="ortho") * torch.polar(torch.ones_like(self.theta), self.theta), norm="ortho")
        return torch.view_as_real(z).reshape(u.shape)


class Spectral(nn.Module):
    """W = U diag(e^sigma) V^T, U frozen orthogonal."""

    def __init__(self, C, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        U, _ = torch.linalg.qr(torch.randn(C, C, generator=g))
        self.register_buffer("U", U)
        self.V = nn.Parameter(U.clone())
        self.sigma = nn.Parameter(torch.zeros(C))

    def forward(self, u):
        return ((u @ self.V) * self.sigma.exp()) @ self.U.T


class Branch(nn.Module):
    def __init__(self, C, K, seed, alpha0=0.0):
        super().__init__()
        self.conv = CausalConv(C, K)
        self.phase = PhaseFilter(C)
        self.W = Spectral(C, seed)
        self.alpha = nn.Parameter(torch.full((C,), alpha0))

    def mix(self, n, c):
        return self.alpha * self.W(self.phase(c) * Fn.silu(n))

    def forward(self, x):
        n = rms(x)
        return self.mix(n, self.conv(n))

    def step(self, x_t, buf):
        n = rms(x_t)
        c, buf = self.conv.step(n, buf)
        return self.mix(n, c), buf


class RevBlock(nn.Module):
    def __init__(self, C, K, seed, alpha0):
        super().__init__()
        self.F = Branch(C, K, 2 * seed, alpha0)
        self.G = Branch(C, K, 2 * seed + 1, alpha0)

    def forward(self, x1, x2):
        y1 = x1 + self.F(x2)
        y2 = x2 + self.G(y1)
        return y1, y2

    def inverse(self, y1, y2):
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)
        return x1, x2


class PrimeSeqModel(nn.Module):
    """d = 2R. Input x is a real tensor (B,T) (or indices for the learned-table control)."""

    def __init__(self, R=256, L=8, K=256, family="sqrt_p", n_out=256, alpha0=0.0, seed=0):
        super().__init__()
        self.R, self.L, self.K, self.family = R, L, K, family
        self.embed = LearnedTable(R, n_out) if family == "table" else (
            PrimeBasis(R, family[:-5], sigma=0.0) if family.endswith("_flat") else PrimeBasis(R, family))
        self.blocks = nn.ModuleList([RevBlock(R, K, seed * 1000 + i, alpha0) for i in range(L)])
        self.out_gain = nn.Parameter(torch.ones(2 * R))
        self.temp = nn.Parameter(torch.tensor(math.log(0.25)))
        self.n_out = n_out
        if family != "table":
            self.register_buffer("cands", torch.arange(n_out, dtype=torch.float64), persistent=False)

    # ---- trunk ------------------------------------------------------------------
    def encode(self, x):
        return self.embed(x)

    def trunk(self, e):
        x1, x2 = e[..., : self.R], e[..., self.R:]
        for b in self.blocks:
            x1, x2 = b(x1, x2)
        return torch.cat([x1, x2], -1)

    def inverse(self, h):
        y1, y2 = h[..., : self.R], h[..., self.R:]
        for b in reversed(self.blocks):
            y1, y2 = b.inverse(y1, y2)
        return torch.cat([y1, y2], -1)

    def features(self, h):
        return rms(h) * self.out_gain

    # ---- discrete readout: phase matching against the frozen basis ----------------
    def readout_matrix(self):
        return self.embed.W if self.family == "table" else self.embed(self.cands)

    def logits(self, h):
        B = self.readout_matrix()  # (n_out, 2R)
        Bn = B / B.norm(dim=-1, keepdim=True).mean()
        return self.temp.exp() * self.features(h) @ Bn.T

    def forward(self, x):
        return self.logits(self.trunk(self.encode(x)))

    # ---- constant-memory generation ---------------------------------------------
    def init_state(self, batch):
        z = lambda: torch.zeros(batch, self.K, self.R)
        return [(z(), z()) for _ in self.blocks]

    def state_bytes(self, batch=1):
        return sum(a.numel() * 4 + b.numel() * 4 for a, b in self.init_state(batch))

    @torch.no_grad()
    def step(self, x_t, state):
        e = self.encode(x_t)
        x1, x2 = e[..., : self.R], e[..., self.R:]
        new = []
        for b, (bf, bg) in zip(self.blocks, state):
            f, bf = b.F.step(x2, bf)
            x1 = x1 + f
            g, bg = b.G.step(x1, bg)
            x2 = x2 + g
            new.append((bf, bg))
        return self.logits(torch.cat([x1, x2], -1)), new


def param_breakdown(m: nn.Module):
    out = {}
    for n, p in m.named_parameters():
        key = n.split(".")[-1]
        out[key] = out.get(key, 0) + p.numel()
    return out
