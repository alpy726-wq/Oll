"""Frozen prime-frequency basis  b(x)_k = A_k * exp(2*pi*i*x*f_k),  A_k = p_k^{-1/2}.

Nothing here is a parameter. The argument x is a real number (float64), not an index.
Phases are reduced mod 1 in float64 before casting (paper, Section 7).
"""
import math
import numpy as np
import torch


def first_primes(n: int) -> np.ndarray:
    if n < 6:
        lim = 15
    else:
        lim = int(n * (math.log(n) + math.log(math.log(n)))) + 10
    sieve = np.ones(lim + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(lim ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    p = np.nonzero(sieve)[0]
    assert len(p) >= n
    return p[:n].astype(np.float64)


def frequencies(R: int, family: str = "sqrt_p", seed: int = 0) -> np.ndarray:
    """Frequency families (paper Sec. 2 'Frequency families'). All share amplitude p_k^{-1/2}."""
    p = first_primes(R)
    if family == "sqrt_p":
        return np.sqrt(p)
    if family == "log_p":  # zeta variant: e^{i t log p}  ->  f = log p / (2 pi)
        return np.log(p) / (2 * math.pi)
    if family == "random":  # random reals over the same range as sqrt_p
        rng = np.random.default_rng(seed)
        return np.sort(rng.uniform(math.sqrt(2), math.sqrt(p[-1]), R))
    if family == "geometric":  # RoPE-like: geometric ladder, wavelengths 2 .. 2*10^4
        return 0.5 * (1e-4) ** (np.arange(R) / max(R - 1, 1))
    raise ValueError(family)


def amplitudes(R: int, sigma: float = 0.5) -> np.ndarray:
    return first_primes(R) ** (-sigma)


class PrimeBasis(torch.nn.Module):
    """b: R -> C^R, stored as real [Re | Im] of width 2R. Buffers only (frozen)."""

    def __init__(self, R: int, family: str = "sqrt_p", sigma: float = 0.5, seed: int = 0, scale: float = 1.0):
        """scale: fixed affine rescaling of the input (Sec. 7); b(x) uses x/scale."""
        super().__init__()
        self.R, self.family, self.scale = R, family, scale
        self.register_buffer("f64", torch.tensor(frequencies(R, family, seed) / scale, dtype=torch.float64), persistent=False)
        self.register_buffer("A", torch.tensor(amplitudes(R, sigma), dtype=torch.float32), persistent=False)

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        """2*pi*frac(x f_k), computed in float64."""
        ph = x.to(torch.float64).unsqueeze(-1) * self.f64
        ph = ph - torch.floor(ph)
        return (2 * math.pi * ph).to(torch.float32)

    def complex(self, x: torch.Tensor) -> torch.Tensor:
        ph = self.phase(x)
        return self.A * torch.polar(torch.ones_like(ph), ph)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ph = self.phase(x)
        return torch.cat([self.A * torch.cos(ph), self.A * torch.sin(ph)], -1)


class LearnedTable(torch.nn.Module):
    """Ablation control: an ordinary learned 256 x 2R table (indices only)."""

    def __init__(self, R: int, n: int = 256):
        super().__init__()
        self.R = R
        self.W = torch.nn.Parameter(torch.randn(n, 2 * R) * (float(np.sum(amplitudes(R) ** 2)) / (2 * R)) ** 0.5)

    def forward(self, x):
        return self.W[x.long()]
