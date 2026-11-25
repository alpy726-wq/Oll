"""OneMachine: every capability of the paper in a single model.

Events, not tokens. An event is (stream s, value x, gap dt since the previous event). Streams are typed regions of R
(Sec. 2.3): an event's code is the prime-frequency code of x, bound (Prop. 1 group law) to a frozen role phase of its
stream. Several sensors, symbols and numbers therefore live in one space with no encoder and no fusion layer;
simultaneous readings are events with dt = 0.

  code(x, s)_k = A_k * exp(i (2 pi x f_k / scale_s + psi_k(s)))                       input  (Sec. 2, 2.3, 5.3)
  integer streams may add the multiplicative channel  exp(-i sum_p v_p(n) log p)        zeta variant (Sec. 2.6)
  phase memory   M <- exp(-mu dt) M + u(prev) * u(cur);  read  M * conj(u(cur))          exact bind / unbind (Prop. 1)
  trunk          reversible blocks; each branch = continuous-time oscillator mixer (Sec. 3, eq. 3)
                 + circulant phase filter e^{i theta} + Kuramoto phase coupling (Sec. 5.5) + W = U diag(e^sigma) V^T
  output         for stream s:  s(x) = <e^{i(2 pi x f/scale + psi_s)}, h>  (flat amplitude: Sec. 2.2 shows p^{-1/2} on the
                 read side limits discrimination to an effective dimension of 14);  continuous: |s|^2 / Z (closed form);  discrete: softmax of
                 log|s(c)|^2 over the stream's candidates                                   (Sec. 4.1)
  adapters       theta, sigma, alpha, Kuramoto coupling = the phase adapter (Sec. 5.6)
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .basis import frequencies, amplitudes, first_primes
from .model import rms, PhaseFilter, Spectral
from .continuous import CTConv

PHASE_PARAMS = ("theta", "sigma", "alpha", "kq", "kp", "pl")


class PhaseLinear(nn.Module):
    """Per-prime real-linear map of the phasor (Re z_k, Im z_k): a learnable, invertible 2x2 matrix per prime.
    Rotation by e^{i 2 pi c f_k} adds a constant c to the encoded value (group law); rotation by psi differences converts
    a stream role into another; reflection (conjugation) negates the value. Born as the identity."""

    def __init__(self, R):
        super().__init__()
        self.pl = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 1.0]).repeat(R, 1))  # (R, 4): [a b; c d]

    def forward(self, x1, x2):
        a, b, c, d = self.pl.unbind(-1)
        return a * x1 + b * x2, c * x1 + d * x2

    def inverse(self, y1, y2):
        a, b, c, d = self.pl.unbind(-1)
        det = a * d - b * c
        return (d * y1 - b * y2) / det, (-c * y1 + a * y2) / det


class Stream:
    def __init__(self, name, kind, scale=1.0, window=(-1.5, 1.5), n_symbols=256, integer_channel=False):
        assert kind in ("continuous", "discrete")
        self.name, self.kind, self.scale, self.window, self.n_symbols = name, kind, scale, window, n_symbols
        self.integer_channel = integer_channel


class Kuramoto(nn.Module):
    """dphi_i = eps * sum_j K_ij sin(phi_j - phi_i),  K = P Q^T (low rank), born as zero.  Norm-preserving."""

    def __init__(self, C2, rank=8):
        super().__init__()
        self.kp = nn.Parameter(torch.zeros(C2, rank))
        self.kq = nn.Parameter(torch.randn(C2, rank) * 0.1)

    def forward(self, u):
        z = torch.view_as_complex(u.reshape(*u.shape[:-1], -1, 2).contiguous())
        ph = z / (z.abs() + 1e-6)
        Ku = (ph @ self.kq.to(ph.dtype)) @ self.kp.T.to(ph.dtype)  # (..., C2) complex
        dphi = (ph.conj() * Ku).imag
        z = z * torch.polar(torch.ones_like(dphi), dphi)
        return torch.view_as_real(z).reshape(u.shape)

    def order_parameter(self, u):
        z = torch.view_as_complex(u.reshape(*u.shape[:-1], -1, 2).contiguous())
        return (z / (z.abs() + 1e-6)).mean(-1).abs()


class OneBranch(nn.Module):
    def __init__(self, C, seed, M):
        super().__init__()
        self.conv = CTConv(C, M)
        self.phase = PhaseFilter(C)
        self.sync = Kuramoto(C // 2)
        self.W = Spectral(C, seed)
        self.alpha = nn.Parameter(torch.zeros(C))

    def mix(self, n, c):
        return self.alpha * self.W(self.sync(self.phase(c)) * Fn.silu(n))

    def forward(self, x, dt, q):
        n = rms(x)
        return self.mix(n, self.conv(n, dt, q))

    def step(self, x, dt, q, s):
        n = rms(x)
        y, s = self.conv.step(n, dt, q, s)
        return self.mix(n, y), s


class OneMachine(nn.Module):
    def __init__(self, streams, R=128, L=3, M=4, family="sqrt_p", seed=0, RM=1024):
        super().__init__()
        self.streams, self.R, self.L, self.RM = list(streams), R, L, RM
        fm = torch.tensor(frequencies(RM, family), dtype=torch.float64)  # wide prime bank used only for memory addressing
        self.register_buffer("fm", fm, persistent=False)
        self.register_buffer("psim", torch.stack([2 * math.pi * torch.frac(1e5 * (i + 1) * fm) for i in range(len(streams))]), persistent=False)
        f = torch.tensor(frequencies(R, family), dtype=torch.float64)
        self.register_buffer("f", f, persistent=False)
        self.register_buffer("A", torch.tensor(amplitudes(R), dtype=torch.float64), persistent=False)
        # frozen role phases: psi(s) = 2 pi frac(1e5 * (s+1) * f)  -- the typed region type*1e5 of Sec. 2.3
        roles = torch.stack([2 * math.pi * torch.frac(1e5 * (i + 1) * f) for i in range(len(self.streams))])
        self.register_buffer("psi", roles, persistent=False)
        self.register_buffer("logp", torch.tensor(np.log(first_primes(R)), dtype=torch.float64), persistent=False)
        # fixed permutation that makes binding directional (key -> next), so unbinding retrieves the successor only
        self.register_buffer("perm", torch.randperm(R, generator=torch.Generator().manual_seed(12345)), persistent=False)
        self.F = nn.ModuleList([OneBranch(R, 2 * (seed * 100 + i), M) for i in range(L)])
        self.G = nn.ModuleList([OneBranch(R, 2 * (seed * 100 + i) + 1, M) for i in range(L)])
        self.P = nn.ModuleList([PhaseLinear(R) for _ in range(L)])
        self.gain = nn.Parameter(torch.ones(2 * R))
        self.mem_beta = nn.Parameter(torch.ones(len(self.streams)))  # per input stream: how much memory read-out enters (born on)
        self.sym_logtemp = nn.Parameter(torch.zeros(len(self.streams)))  # discrete streams: p(c) ~ |s(c)|^(2 tau)
        self.mem_logmu = nn.Parameter(torch.tensor(-6.0))     # memory decay per unit time (continuous)
        # two-band memory addressing (Sec. 5.3 bandwidth idea): the first half of the bank is "coarse" -- its frequencies
        # are divided by a learnable per-stream width, so similar (not only identical) events retrieve each other
        self.mem_logwidth = nn.Parameter(torch.full((len(self.streams),), math.log(1.0)))
        self.out_logscale = nn.Parameter(torch.tensor(0.0))
        # closed-form normaliser matrices for continuous streams (Sec. 4.1)
        for i, st in enumerate(self.streams):
            if st.kind == "continuous":
                fs = f / st.scale
                d = fs[:, None] - fs[None, :]
                a, c = st.window
                small = d.abs() < 1e-12
                I = torch.where(small, torch.full_like(d, c - a).to(torch.complex128),
                                (torch.exp(-2j * math.pi * c * d) - torch.exp(-2j * math.pi * a * d)) / (-2j * math.pi * d + small))
                self.register_buffer(f"G{i}", I, persistent=False)  # flat readout amplitude (see Sec. 2.2)

    # ------------------------------------------------------------------ codes
    def phases(self, x, s):
        """x: real values (float64), s: stream ids (long). Returns total phase (..., R) in float64."""
        out = torch.zeros(*x.shape, self.R, dtype=torch.float64)
        for i, st in enumerate(self.streams):
            m = s == i
            if not m.any():
                continue
            ph = x[m].to(torch.float64)[:, None] * (self.f / st.scale)
            ph = 2 * math.pi * (ph - torch.floor(ph)) + self.psi[i]
            if st.integer_channel:
                out[m] = ph - self.integer_phase(x[m])
            else:
                out[m] = ph
        return out

    def integer_phase(self, n):
        """multiplicative channel (Sec. 2.6): sum_p v_p(n) log p  -> n*m maps to a phase sum."""
        n = n.round().long().clamp(min=1)
        v = torch.zeros(*n.shape, self.R, dtype=torch.float64)
        rem = n.clone()
        for k, p in enumerate(first_primes(min(self.R, 60)).astype(int)):
            while True:
                hit = (rem % p == 0) & (rem > 1)
                if not hit.any():
                    break
                v[..., k] += hit.double()
                rem = torch.where(hit, rem // p, rem)
        return v * self.logp

    def code(self, x, s):
        ph = self.phases(x, s)
        return torch.cat([self.A * torch.cos(ph), self.A * torch.sin(ph)], -1).float(), ph

    # ------------------------------------------------------------------ phase memory (bind / unbind on a wide prime bank)
    def mem_phases(self, x, s):
        out = torch.zeros(*x.shape, self.RM, dtype=torch.float64)
        half = self.RM // 2
        for i, st in enumerate(self.streams):
            m = s == i
            if m.any():
                fine = x[m].to(torch.float64)[:, None] * (self.fm[half:] / st.scale)
                fine = 2 * math.pi * (fine - torch.floor(fine))
                # coarse band: low frequencies scaled down by a learnable width; kept small enough that no wrap is needed
                coarse = 2 * math.pi * x[m].to(torch.float64)[:, None] * (self.fm[:half] / self.fm[half - 1]) / self.mem_logwidth[i].exp().double() * 4.0
                out[m] = torch.cat([coarse, fine], -1) + self.psim[i]
        return out

    def memory_read(self, x, s, ph, dt):
        """Associative phase memory. Pair (event j-1 -> event j) is stored as  conj(k_{j-1}) (x) v_j,  where k is the unit
        phase code of an event on a wide RM-prime bank and v its unit code on the trunk bank. Reading at event t (before
        writing): r_t = sum_j e^{-mu (tau_t - tau_j)} <k_t, k_{j-1}>/RM  v_j  -- i.e. unbinding with the current event
        returns what followed it before. Training uses this parallel form; streaming uses the equivalent constant-size
        matrix state M (RM x R) in step()."""
        k = torch.polar(torch.ones(1, dtype=torch.float64), self.mem_phases(x, s)).to(torch.complex64)  # (B,T,RM)
        v = torch.polar(torch.ones_like(ph), ph).to(torch.complex64)  # (B,T,R)
        B, T, _ = k.shape
        sim = (k @ k.conj().transpose(1, 2)) / self.RM  # sim[t, i] = <k_t, k_i>
        tau = dt.cumsum(1)
        # pair stored at write time j uses key k_{j-1}, value v_j (1 <= j <= t-1)
        S = torch.zeros(B, T, T, dtype=torch.complex64)
        S[:, :, 1:] = sim[:, :, :-1]
        decay = torch.exp(-self.mem_logmu.exp() * (tau[:, :, None] - tau[:, None, :]).clamp(min=0))
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool), -1) & (torch.arange(T)[None, :] >= 1)
        W = S * (decay * mask).to(S.dtype)
        r = W @ v
        return torch.cat([r.real, r.imag], -1)

    # ------------------------------------------------------------------ trunk
    def embed(self, x, s, dt):
        c, ph = self.code(x, s)
        r = self.memory_read(x, s, ph, dt)
        return c + self.mem_beta[s][..., None] * r * float(self.A.mean())

    def trunk_from(self, e, dt, q):
        x1, x2 = e[..., : self.R], e[..., self.R:]
        for F, G, P in zip(self.F, self.G, self.P):
            x1 = x1 + F(x2, dt, q)
            x2 = x2 + G(x1, dt, q)
            x1, x2 = P(x1, x2)
        return torch.cat([x1, x2], -1)

    def inverse(self, h, dt, q):
        y1, y2 = h[..., : self.R], h[..., self.R:]
        for F, G, P in zip(reversed(self.F), reversed(self.G), reversed(self.P)):
            y1, y2 = P.inverse(y1, y2)
            y2 = y2 - G(y1, dt, q)
            y1 = y1 - F(y2, dt, q)
        return torch.cat([y1, y2], -1)

    def features(self, h):
        return rms(h) * self.gain

    def forward(self, x, s, dt, q):
        """x,s,dt: events 0..T-1; q: time from event t to the event being predicted. Returns features (B,T,2R)."""
        return self.features(self.trunk_from(self.embed(x, s, dt), dt, q))

    # ------------------------------------------------------------------ output
    def coeffs(self, h, s_out):
        c = torch.complex(h[..., : self.R], h[..., self.R:]).to(torch.complex128) * self.out_logscale.exp()
        return c * torch.exp(-1j * self.psi[s_out])  # unbind the role of the requested stream

    def amp(self, c, x, st):
        ph = x.to(torch.float64)[..., None] * (self.f / st.scale)
        ph = 2 * math.pi * (ph - torch.floor(ph))
        return (c * torch.exp(-1j * ph)).sum(-1)

    def nll(self, h, s_out, y):
        """negative log-likelihood of targets y for target streams s_out (per element)."""
        out = torch.zeros(y.shape, dtype=torch.float32)
        for i, st in enumerate(self.streams):
            m = s_out == i
            if not m.any():
                continue
            c = self.coeffs(h[m], torch.full((int(m.sum()),), i, dtype=torch.long))
            if st.kind == "continuous":
                Z = torch.einsum("nk,kl,nl->n", c, getattr(self, f"G{i}"), c.conj()).real
                s2 = self.amp(c, y[m], st).abs() ** 2
                out[m] = (-(s2 + 1e-9 * Z).log() + Z.log()).float()
            else:
                lg = self.discrete_logits(c, st) * self.sym_logtemp[i].exp()
                out[m] = Fn.cross_entropy(lg, y[m].long(), reduction="none")
        return out

    DISCRETE_READOUT = "power"  # "power": log|s(c)|^2 (phase-amplitude density on a discrete set); "phase": Re s(c)

    def discrete_logits(self, c, st):
        cand = torch.arange(st.n_symbols, dtype=torch.float64)
        ph = cand[:, None] * (self.f / st.scale)
        ph = 2 * math.pi * (ph - torch.floor(ph))
        s = c @ torch.exp(-1j * ph).T  # (n, n_symbols), flat readout amplitude
        if self.DISCRETE_READOUT == "phase":
            return s.real.float()  # phase matching: Re <code(c), h>
        return (s.abs() ** 2 + 1e-9).log().float()

    def density(self, h, i, xs):
        st = self.streams[i]
        c = self.coeffs(h, torch.full(h.shape[:-1], i, dtype=torch.long))
        Z = torch.einsum("...k,kl,...l->...", c, getattr(self, f"G{i}"), c.conj()).real
        return (self.amp(c.unsqueeze(-2), xs, st).abs() ** 2 / Z.unsqueeze(-1)).float()

    # ------------------------------------------------------------------ streaming (constant memory)
    def init_state(self, B):
        Mm = self.F[0].conv.c_re.shape[1]
        return dict(conv=[torch.zeros(B, self.R, Mm, dtype=torch.complex64) for _ in range(2 * self.L)],
                    M=torch.zeros(B, self.RM, self.R, dtype=torch.complex64), prev=torch.zeros(B, self.RM, dtype=torch.complex64))

    def step(self, x, s, dt, q, st):
        c, ph = self.code(x, s)
        v = torch.polar(torch.ones_like(ph), ph).to(torch.complex64)
        k = torch.polar(torch.ones(1, dtype=torch.float64), self.mem_phases(x, s)).to(torch.complex64)
        M = torch.exp(-self.mem_logmu.exp() * dt)[:, None, None] * st["M"]
        r = torch.einsum("bm,bmr->br", k, M) / self.RM
        e = c + self.mem_beta[s][..., None] * torch.cat([r.real, r.imag], -1) * float(self.A.mean())
        M = M + st["prev"].conj()[:, :, None] * v[:, None, :]
        x1, x2 = e[..., : self.R], e[..., self.R:]
        conv = []
        for i, (F, G, P) in enumerate(zip(self.F, self.G, self.P)):
            f_, s1 = F.step(x2, dt, q, st["conv"][2 * i]); x1 = x1 + f_
            g_, s2 = G.step(x1, dt, q, st["conv"][2 * i + 1]); x2 = x2 + g_
            x1, x2 = P(x1, x2)
            conv += [s1, s2]
        return self.features(torch.cat([x1, x2], -1)), dict(conv=conv, M=M, prev=k)

    def state_numel(self, st):
        return sum(t.numel() for t in st["conv"]) + st["M"].numel() + st["prev"].numel()

    # ------------------------------------------------------------------ phase adapters (Sec. 5.6)
    def adapter(self, base):
        return {k: v - base[k] for k, v in self.state_dict().items() if k.split(".")[-1] in PHASE_PARAMS}

    def apply_adapters(self, base, *adapters):
        sd = {k: v.clone() for k, v in base.items()}
        for ad in adapters:
            for k, v in ad.items():
                sd[k] = sd[k] + v
        self.load_state_dict(sd)
