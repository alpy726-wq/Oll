"""Section 5 of the paper, implemented.

5.1  PhaseDensityHead:  q(x|h) = |s(x)|^2 / Z(h),  s(x) = <b(x), h>,  Z in closed form (pairwise sinc terms).
5.2  CTConv: kernel = sum of complex exponentials, state advanced by arbitrary dt (irregular sampling).
Baseline heads (Gaussian, mixture of Gaussians) for comparison.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .basis import PrimeBasis
from .model import RevBlock, Branch, PhaseFilter, Spectral, rms


# ----------------------------------------------------------------------------- 5.1
class PhaseDensityHead(nn.Module):
    def __init__(self, basis: PrimeBasis, a: float, c: float):
        super().__init__()
        self.basis, self.a, self.c = basis, a, c
        f, A = basis.f64, basis.A.double()
        d = f[:, None] - f[None, :]
        # I(D) = int_a^c exp(-2 pi i x D) dx
        with torch.no_grad():
            I = torch.where(d.abs() < 1e-12, torch.full_like(d, c - a).to(torch.complex128),
                            (torch.exp(-2j * math.pi * c * d) - torch.exp(-2j * math.pi * a * d)) / (-2j * math.pi * d + (d.abs() < 1e-12)))
        self.register_buffer("G", (A[:, None] * A[None, :]) * I, persistent=False)
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def coeffs(self, h):
        R = self.basis.R
        return torch.complex(h[..., :R], h[..., R:]).to(torch.complex128) * self.log_scale.exp()

    def amp(self, cf, x):
        """s(x) = sum_k c_k A_k exp(-2 pi i x f_k)."""
        return (cf * self.basis.complex(x).to(torch.complex128).conj()).sum(-1)

    def Z(self, cf):
        return torch.einsum("...k,kl,...l->...", cf, self.G, cf.conj()).real

    def nll(self, h, x):
        cf = self.coeffs(h)
        Z = self.Z(cf)
        s2 = self.amp(cf, x).abs() ** 2
        return (-(s2 + 1e-9 * Z / (self.c - self.a)).log() + Z.log()).float()

    def density_grid(self, h, n=2001):
        cf = self.coeffs(h)
        xs = torch.linspace(self.a, self.c, n, dtype=torch.float64)
        s2 = self.amp(cf.unsqueeze(-2), xs).abs() ** 2  # (..., n)
        return xs, s2 / self.Z(cf).unsqueeze(-1), s2


class GaussianHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 2)

    def nll(self, h, x):
        mu, ls = self.lin(h).unbind(-1)
        ls = ls.clamp(-7, 5)
        return 0.5 * ((x.float() - mu) / ls.exp()) ** 2 + ls + 0.5 * math.log(2 * math.pi)


class MDNHead(nn.Module):
    def __init__(self, d, M=5):
        super().__init__()
        self.M = M
        self.lin = nn.Linear(d, 3 * M)

    def nll(self, h, x):
        lw, mu, ls = self.lin(h).chunk(3, -1)
        ls = ls.clamp(-7, 5)
        lp = -0.5 * ((x.float().unsqueeze(-1) - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)
        return -torch.logsumexp(lp + torch.log_softmax(lw, -1), -1)


# ----------------------------------------------------------------------------- 5.2
class CTConv(nn.Module):
    """kappa(t) = Re sum_m c_m exp((-lambda_m + i omega_m) t) per channel.
    State s_t = exp(a dt_t) s_{t-1} + u_t (a = -lambda + i omega); read-out at query time t+q: Re sum_m c_m exp(a q) s_t.
    Memory O(C*M) at any length, and any dt / any query offset q."""

    def __init__(self, C, M=8):
        super().__init__()
        self.log_lam = nn.Parameter(torch.linspace(math.log(0.02), math.log(2.0), M).repeat(C, 1))
        self.omega = nn.Parameter(torch.linspace(0, math.pi, M).repeat(C, 1))
        cr = torch.randn(C, M) * 0.1
        cr[:, 0] += 1.0
        self.c_re = nn.Parameter(cr)
        self.c_im = nn.Parameter(torch.zeros(C, M))

    def forward(self, u, dt, q=None):
        a = torch.complex(-self.log_lam.clamp(max=math.log(4.0)).exp(), self.omega)  # (C,M), lambda <= 4 keeps the chunk scan in range
        cc = torch.complex(self.c_re, self.c_im)
        if q is not None:
            cc = cc * torch.exp(q[..., None, None].float() * a)  # (B,T,C,M)
        else:
            cc = cc.expand(u.shape[0], u.shape[1], *cc.shape)
        return self.scan(u, dt, a, cc)

    def scan(self, u, dt, a, cc, chunk=8):
        """Chunked exact scan (log-space within a chunk):
        s_t = e^{a c_t} (s_prev + sum_{j<=t} e^{-a c_j} u_j),  c = cumulative dt inside the chunk."""
        B, T, C = u.shape
        s = torch.zeros(B, C, a.shape[1], dtype=torch.complex64)
        ys = []
        for st in range(0, T, chunk):
            uc, ccc = u[:, st:st + chunk], cc[:, st:st + chunk]
            c = dt[:, st:st + chunk].float().cumsum(1)[..., None, None]  # (B,n,1,1)
            acc = torch.cumsum(torch.exp(-a * c) * uc[..., None], 1)  # (B,n,C,M)
            sc = torch.exp(a * c) * (acc + s[:, None])
            ys.append((sc * ccc).sum(-1).real)
            s = sc[:, -1]
        return torch.cat(ys, 1)

    def scan_reference(self, u, dt, a, cc):
        s = torch.zeros(u.shape[0], u.shape[2], a.shape[1], dtype=torch.complex64)
        decay = torch.exp(dt[..., None, None].float() * a)
        ys = []
        for t in range(u.shape[1]):
            s = decay[:, t] * s + u[:, t, :, None]
            ys.append((s * cc[:, t]).sum(-1).real)
        return torch.stack(ys, 1)


class CTBranch(Branch):
    def __init__(self, C, seed, M=8):
        nn.Module.__init__(self)
        self.conv = CTConv(C, M)
        self.phase = PhaseFilter(C)
        self.W = Spectral(C, seed)
        self.alpha = nn.Parameter(torch.zeros(C))

    def forward(self, x, dt, q=None):
        n = rms(x)
        return self.mix(n, self.conv(n, dt, q))


class CTSeqModel(nn.Module):
    """Continuous input (prime basis), continuous time (CTConv), continuous output (selectable head).
    Task: given (x_i, dt_i) for i<=t and the gap q_t to the next sample, predict x_{t+1}.
    mode 'ct'      : true dt drives the state, the read-out is advanced by q_t (paper 5.2)
    mode 'fixed'   : dt := 1, timestamps and gap ignored (what a fixed-step model sees)
    mode 'fixed+dt': dt := 1, but b(dt_i) and b(q_t) are added to the input (timestamps as features)"""

    def __init__(self, R=128, L=4, M=4, mode="ct", head="phase", window=(-2.0, 10.0), family="sqrt_p", seed=0, scale=8.0):
        super().__init__()
        self.R, self.mode = R, mode
        self.basis = PrimeBasis(R, family, scale=scale)
        self.F = nn.ModuleList([CTBranch(R, 2 * (seed * 100 + i), M) for i in range(L)])
        self.G = nn.ModuleList([CTBranch(R, 2 * (seed * 100 + i) + 1, M) for i in range(L)])
        self.gain = nn.Parameter(torch.ones(2 * R))
        self.head = {"phase": lambda: PhaseDensityHead(self.basis, *window),
                     "gauss": lambda: GaussianHead(2 * R), "mdn": lambda: MDNHead(2 * R)}[head]()

    def trunk(self, x, dt, q):
        e = self.basis(x)
        if self.mode == "fixed+dt":
            e = e + self.basis(dt * 3.0 + 500.0) + self.basis(q * 3.0 + 1000.0)
        if self.mode != "ct":
            dt, q = torch.ones_like(dt), None
        x1, x2 = e[..., : self.R], e[..., self.R:]
        for F, G in zip(self.F, self.G):
            x1 = x1 + F(x2, dt, q)
            x2 = x2 + G(x1, dt, q)
        return rms(torch.cat([x1, x2], -1)) * self.gain

    def nll(self, x, dt, q, y):
        return self.head.nll(self.trunk(x, dt, q), y)
