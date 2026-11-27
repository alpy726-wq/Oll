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
    def __init__(self, streams, R=128, L=3, M=4, family="sqrt_p", seed=0, RM=1024, readout_amp=False, mem_orders=(1,),
                 anticipate=False, clock=False, query_head=False, seasonal=False, copy_out=False):
        super().__init__()
        # copy read-out: the memory read r_t (per context order) is already a phase code on the trunk bank, i.e. on the
        # same basis as the output coefficients, so adding g_n r_t^(n) to the output features makes the predicted
        # density peak at whatever followed the same context before (closed-form copying / induction; g_n learned,
        # born 0 so a warm start is unchanged). When no stored context matches, r_t is only crosstalk and adds little.
        self.copy_out = copy_out
        # seasonal memory ("time cells"): every event's phase code is bound to the phase of its moment on the day and
        # week circles (Prop. 1 binding, Fejer-weighted harmonics so the kernel is non-negative); reading with the phase
        # of the asked moment returns what was there at the same time of day / week before (normalised, decayed).
        # Linear in the stored pairs, so streaming keeps a constant-size state. Needs absolute times (clock data).
        self.seasonal = seasonal
        # query head (Sec. 5.2 read-out at any time): the trunk runs at q = 0, so its state never depends on what is
        # asked; one extra continuous-time branch reads the state out at the asked moment t+q (and, with the clock, at
        # that moment's calendar phase). Any future moment can be asked directly, without feeding predictions back.
        self.query_head = query_head
        # calendar clock: the phase of every event (and of the moment asked about) on the day / week / year circles,
        # the same phase encoding as the values (Sec. 2) applied to absolute time; NaN time (text, audio) = no clock
        self.clock = clock
        # anticipation (Sec. 5.2 continuous time, used for the output too): besides "what value at the asked time", the
        # machine predicts WHEN the next event comes and WHICH sense it comes from, as two more phase-amplitude read-outs
        # (a continuous density over log-gap, a discrete one over stream identities). They read a separate continuous-time
        # pathway that never sees the query time, so nothing about the next event leaks into them.
        self.anticipate, self.n_in = anticipate, len(streams)
        if anticipate:
            streams = list(streams) + [Stream("_when", "continuous", scale=2.0, window=(-1.5, 1.5)),
                                       Stream("_which", "discrete", scale=1.0, n_symbols=len(streams))]
            self.WHEN, self.WHICH = len(streams) - 2, len(streams) - 1
        # context-keyed memory: a key of order n is the binding (phase sum, Prop. 1) of the last n events, each under
        # its own fixed permutation, so "France:" and "Italy:" give different keys although both end in ':'
        self.mem_orders = tuple(mem_orders)
        gperm = torch.Generator().manual_seed(4242)
        self.register_buffer("ctx_perm", torch.stack([torch.arange(RM)] + [torch.randperm(RM, generator=gperm) for _ in range(max(self.mem_orders) - 1)]), persistent=False)
        if len(self.mem_orders) > 1:
            self.mem_order_logw = nn.Parameter(torch.zeros(len(self.mem_orders)))
        self.readout_amp = readout_amp  # ablation: read out with the p^{-1/2} amplitude instead of flat
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
        self.register_buffer("inv_scale", torch.tensor([1.0 / st.scale for st in self.streams], dtype=torch.float64), persistent=False)
        self.register_buffer("is_integer", torch.tensor([st.integer_channel for st in self.streams]), persistent=False)
        self.has_integer = any(st.integer_channel for st in self.streams)
        # fixed permutation that makes binding directional (key -> next), so unbinding retrieves the successor only
        self.register_buffer("perm", torch.randperm(R, generator=torch.Generator().manual_seed(12345)), persistent=False)
        self.F = nn.ModuleList([OneBranch(R, 2 * (seed * 100 + i), M) for i in range(L)])
        self.G = nn.ModuleList([OneBranch(R, 2 * (seed * 100 + i) + 1, M) for i in range(L)])
        self.P = nn.ModuleList([PhaseLinear(R) for _ in range(L)])
        self.gain = nn.Parameter(torch.ones(2 * R))
        if clock:
            per = [24.0 / n for n in range(1, 7)] + [168.0 / n for n in range(1, 4)] + [8766.0 / n for n in range(1, 3)]
            self.register_buffer("clock_per", torch.tensor(per, dtype=torch.float64), persistent=False)
            g = torch.Generator().manual_seed(777)
            self.clock_in = nn.Parameter(torch.randn(2 * R, 2 * len(per), generator=g) * 0.02)
            self.clock_out = nn.Parameter(torch.randn(2 * R, 2 * len(per), generator=g) * 0.02)
        if seasonal:
            H = 48
            w = torch.tensor([1 - n / (H + 1) for n in range(1, H + 1)], dtype=torch.float64)
            self.register_buffer("sea_n", torch.arange(1, H + 1, dtype=torch.float64), persistent=False)
            self.register_buffer("sea_w", torch.cat([w, w]).sqrt(), persistent=False)
            self.sea_logmu = nn.Parameter(torch.tensor(math.log(1.0 / (24 * 14))))  # forgetting per hour (two weeks)
            self.sea_gain = nn.Parameter(torch.full((2 * R,), 0.1))
        if query_head:
            self.QF = OneBranch(R, 2 * (seed * 100 + 95), M)
            self.QF.conv.qmax = 1e4
        if anticipate:
            self.AF, self.AG = OneBranch(R, 2 * (seed * 100 + 90), M), OneBranch(R, 2 * (seed * 100 + 90) + 1, M)
            self.again = nn.Parameter(torch.ones(2 * R))
        self.mem_beta = nn.Parameter(torch.ones(len(self.streams)))  # per input stream: how much memory read-out enters (born on)
        self.sym_logtemp = nn.Parameter(torch.zeros(len(self.streams)))  # discrete streams: p(c) ~ |s(c)|^(2 tau)
        self.mem_logmu = nn.Parameter(torch.tensor(-6.0))     # memory decay per unit time (continuous)
        # two-band memory addressing (Sec. 5.3 bandwidth idea): the first half of the bank is "coarse" -- its frequencies
        # are divided by a learnable per-stream width, so similar (not only identical) events retrieve each other
        self.mem_logwidth = nn.Parameter(torch.full((len(self.streams),), math.log(1.0)))
        self.out_logscale = nn.Parameter(torch.tensor(0.0))
        if copy_out:
            self.copy_gain = nn.Parameter(torch.zeros(len(self.mem_orders)))
        # closed-form normaliser matrices for continuous streams (Sec. 4.1)
        for i, st in enumerate(self.streams):
            if st.kind == "continuous":
                fs = f / st.scale
                d = fs[:, None] - fs[None, :]
                a, c = st.window
                small = d.abs() < 1e-12
                I = torch.where(small, torch.full_like(d, c - a).to(torch.complex128),
                                (torch.exp(-2j * math.pi * c * d) - torch.exp(-2j * math.pi * a * d)) / (-2j * math.pi * d + small))
                G = (self.A[:, None] * self.A[None, :]) * I if readout_amp else I
                self.register_buffer(f"G{i}", G, persistent=False)  # flat readout amplitude by default (see Sec. 2.2)

    # ------------------------------------------------------------------ codes
    def phases(self, x, s):
        """x: real values (float64), s: stream ids (long). Returns total phase (..., R) in float64. Fully vectorized."""
        ph = x.to(torch.float64)[..., None] * self.f * self.inv_scale[s][..., None]
        ph = 2 * math.pi * (ph - torch.floor(ph)) + self.psi[s]
        if self.has_integer:
            m = self.is_integer[s]
            if m.any():
                ph = ph.clone()
                ph[m] = ph[m] - self.integer_phase(x[m])
        return ph

    def integer_phase(self, n):
        """multiplicative channel (Sec. 2.6): sum_p v_p(n) log p  -> n*m maps to a phase sum."""
        n = n.round().long().clamp(min=1)
        v = torch.zeros(*n.shape, self.R, dtype=torch.float64, device=n.device)
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
        half = self.RM // 2
        xd = x.to(torch.float64)[..., None]
        fine = xd * self.fm[half:] * self.inv_scale[s][..., None]
        fine = 2 * math.pi * (fine - torch.floor(fine))
        # coarse band: low frequencies scaled down by a learnable per-stream width; small enough that no wrap is needed
        coarse = 2 * math.pi * xd * (self.fm[:half] / self.fm[half - 1]) / self.mem_logwidth.exp().double()[s][..., None] * 4.0
        return torch.cat([coarse, fine], -1) + self.psim[s]

    def order_weights(self):
        if len(self.mem_orders) == 1:
            return [1.0]
        w = self.mem_order_logw.exp()
        return [w[i] / len(self.mem_orders) for i in range(len(self.mem_orders))]

    def context_phases(self, P, n):
        """order-n context key phases: sum_{i<n} perm_i(P_{t-i}) (events before the window contribute phase 0)."""
        if n == 1:
            return P
        out = P.clone()
        for i in range(1, n):
            sh = torch.zeros_like(P)
            sh[:, i:] = P[:, :-i]
            out = out + sh[..., self.ctx_perm[i]]
        return out

    def memory_read(self, x, s, ph, dt, wg=None):
        """Associative phase memory. With context key K_t (order n: the last n events bound together), the pair
        (K_{j-1} -> event j) is stored as conj(K_{j-1}) (x) v_j, where v is the unit phase code of the event on the
        trunk bank. Reading at event t (before writing): r_t = sum_n w_n sum_j e^{-mu (tau_t - tau_j)} <K_t, K_{j-1}>/RM v_j,
        i.e. unbinding with the current context returns what followed the same context before. Training uses this
        parallel form; streaming uses the equivalent constant-size matrix state (one RM x R matrix per order)."""
        P = self.mem_phases(x, s)
        v = torch.polar(torch.ones_like(ph), ph).to(torch.complex64)  # (B,T,R)
        B, T = x.shape
        tau = dt.cumsum(1)
        decay = torch.exp(-self.mem_logmu.exp() * (tau[:, :, None] - tau[:, None, :]).clamp(min=0))
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), -1) & (torch.arange(T, device=x.device)[None, :] >= 1)
        W, Wc = 0, 0
        for o, (n, w) in enumerate(zip(self.mem_orders, self.order_weights())):
            k = torch.polar(torch.ones(1, dtype=torch.float64, device=x.device), self.context_phases(P, n)).to(torch.complex64)
            sim = (k @ k.conj().transpose(1, 2)) / self.RM  # sim[t, i] = <K_t, K_i>
            S = torch.zeros(B, T, T, dtype=torch.complex64, device=x.device)
            S[:, :, 1:] = sim[:, :, :-1]  # pair stored at write time j uses key K_{j-1}
            W = W + w * S
            if self.copy_out:
                Wc = Wc + self.copy_gain[o] * S
        if wg is not None:  # salience-gated writing: event j is stored with strength wg_j (surprise; habituation)
            v = v * wg[..., None].to(v.dtype)
        dm = (decay * mask).to(torch.complex64)
        r = (W * dm) @ v
        if self.copy_out:  # read at event t predicts event t+1: the value stored after the same context before
            rc = (Wc * dm) @ v
            self._copy_h = torch.cat([rc.real, rc.imag], -1)
        return torch.cat([r.real, r.imag], -1)

    # ------------------------------------------------------------------ trunk
    def clock_code(self, t):
        """t: absolute time in hours (NaN = none) -> (..., 2K) cos/sin of the calendar phases"""
        ok = torch.isfinite(t)
        ph = 2 * math.pi * torch.frac(torch.where(ok, t, torch.zeros_like(t)).to(torch.float64)[..., None] / self.clock_per)
        return (torch.cat([torch.cos(ph), torch.sin(ph)], -1) * ok[..., None]).float()

    def embed(self, x, s, dt, wg=None, t_in=None, t_out=None):
        c, ph = self.code(x, s)
        r = self.memory_read(x, s, ph, dt, wg)
        e = c + self.mem_beta[s][..., None] * r * float(self.A.mean())
        if self.clock and t_in is not None:
            e = e + self.clock_code(t_in) @ self.clock_in.T
            if not self.query_head:  # with a query head the asked moment enters only the query read-out
                e = e + self.clock_code(t_out) @ self.clock_out.T
        return e

    def sea_key(self, t):
        """(..., 2H) complex key of absolute time t (hours) on the day and week circles; zero where t is NaN"""
        ok = torch.isfinite(t)
        tt = torch.where(ok, t, torch.zeros_like(t)).to(torch.float64)[..., None]
        ph = torch.cat([2 * math.pi * torch.frac(tt * self.sea_n / 24.0), 2 * math.pi * torch.frac(tt * self.sea_n / 168.0)], -1)
        return torch.polar(self.sea_w.expand(ph.shape), ph) * ok[..., None]

    def seasonal_read(self, ph, t_in, t_out):
        """parallel form: r_i = sum_{j<=i} W_ij v_j / sum_j W_ij, W_ij = decay(t_i - t_j) * (1 + Re<k(t_out_i), k(t_j)>)"""
        v = torch.polar(torch.ones_like(ph), ph)  # (B,T,R) complex128
        kq, kk = self.sea_key(t_out), self.sea_key(t_in)
        K = 1.0 + (kq @ kk.conj().transpose(1, 2)).real  # (B,T,T) >= 0 (two Fejer kernels)
        ti = torch.where(torch.isfinite(t_in), t_in, torch.zeros_like(t_in)).to(torch.float64)
        age = (ti[:, :, None] - ti[:, None, :]).clamp(min=0)
        T = t_in.shape[1]
        ok = torch.isfinite(t_in)[:, None, :] & torch.isfinite(t_out)[:, :, None]
        W = K * torch.exp(-self.sea_logmu.exp().double() * age) * torch.tril(torch.ones(T, T, dtype=torch.bool, device=ph.device)) * ok
        r = (W.to(v.dtype) @ v) / (W.sum(-1, keepdim=True) + 1e-6)
        return torch.cat([r.real, r.imag], -1).float()

    def query_features(self, h0, dt, q, t_out=None):
        x1, x2 = h0[..., : self.R], h0[..., self.R:]
        hq = torch.cat([x1 + self.QF(x2, dt, q), x2], -1)
        if self.clock and t_out is not None:
            hq = hq + self.clock_code(t_out) @ self.clock_out.T
        return self.features(hq)

    def peek(self, st, q, t_out=None):
        """read the state out at several asked moments without changing it: q (B,H) -> features (B,H,2R)"""
        h0 = st["h0"]
        x1, x2 = h0[..., : self.R], h0[..., self.R:]
        n = rms(x2)
        a, cc = self.QF.conv.a_cc()
        qq = q.clamp(max=self.QF.conv.qmax).float()
        y = (st["qconv"][:, None] * cc * torch.exp(qq[..., None, None] * a)).sum(-1).real  # (B,H,R)
        f = self.QF.mix(n[:, None], y)
        hq = torch.cat([x1[:, None] + f, x2[:, None].expand_as(f)], -1)
        if self.clock and t_out is not None:
            hq = hq + self.clock_code(t_out) @ self.clock_out.T
        h = self.features(hq)
        if self.seasonal and t_out is not None and "sP" in st:
            h = h + self.sea_gain * self._seasonal_query(st, t_out)
        return h

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

    # ------------------------------------------------------------------ anticipation: when and which comes next
    def anticipation(self, e, dt):
        z = torch.zeros_like(dt)
        x1, x2 = e[..., : self.R], e[..., self.R:]
        x1 = x1 + self.AF(x2, dt, z)
        x2 = x2 + self.AG(x1, dt, z)
        return rms(torch.cat([x1, x2], -1)) * self.again

    def forward_all(self, x, s, dt, q, wg=None, t_in=None, t_out=None):
        """value features (asked at time q) and anticipation features (never see q)"""
        e = self.embed(x, s, dt, wg, t_in, t_out)
        if self.query_head:
            h = self.query_features(self.trunk_from(e, dt, torch.zeros_like(q)), dt, q, t_out)
        else:
            h = self.features(self.trunk_from(e, dt, q))
        if self.seasonal and t_in is not None:
            h = h + self.sea_gain * self.seasonal_read(self.phases(x, s), t_in, t_out)
        if self.copy_out:
            h = h + self._copy_h
        return h, self.anticipation(e, dt)

    @staticmethod
    def when_value(q):
        return torch.log1p(q.to(torch.float64).clamp(0, 1e3)) / 4.0 - 0.75  # log-gap inside the density window

    def anticipation_nll(self, ha, s_next, q):
        """(when, which) negative log-likelihoods per element"""
        sh = ha.shape[:-1]
        w = self.nll(ha, torch.full(sh, self.WHEN, dtype=torch.long, device=ha.device), self.when_value(q))
        c = self.nll(ha, torch.full(sh, self.WHICH, dtype=torch.long, device=ha.device), s_next.to(torch.float64))
        return w, c

    def forward(self, x, s, dt, q, wg=None, t_in=None, t_out=None):
        """x,s,dt: events 0..T-1; q: time from event t to the event being predicted. Returns features (B,T,2R).
        wg (optional, B,T): strength with which each event is written into the associative memory.
        t_in / t_out (optional, B,T): absolute time in hours of each event / of the event being predicted (clock)."""
        e = self.embed(x, s, dt, wg, t_in, t_out)
        if self.query_head:
            h = self.query_features(self.trunk_from(e, dt, torch.zeros_like(q)), dt, q, t_out)
        else:
            h = self.features(self.trunk_from(e, dt, q))
        if self.seasonal and t_in is not None:
            h = h + self.sea_gain * self.seasonal_read(self.phases(x, s), t_in, t_out)
        if self.copy_out:
            h = h + self._copy_h
        return h

    # ------------------------------------------------------------------ output
    def coeffs(self, h, s_out):
        c = torch.complex(h[..., : self.R], h[..., self.R:]).to(torch.complex128) * self.out_logscale.exp()
        return c * torch.exp(-1j * self.psi[s_out])  # unbind the role of the requested stream

    def amp(self, c, x, st):
        ph = x.to(torch.float64)[..., None] * (self.f / st.scale)
        ph = 2 * math.pi * (ph - torch.floor(ph))
        return (c * (self.A if self.readout_amp else 1.0) * torch.exp(-1j * ph)).sum(-1)

    def nll(self, h, s_out, y):
        """negative log-likelihood of targets y for target streams s_out (per element)."""
        out = torch.zeros(y.shape, dtype=torch.float32, device=y.device)
        for i, st in enumerate(self.streams):
            m = s_out == i
            if not m.any():
                continue
            c = self.coeffs(h[m], torch.full((int(m.sum()),), i, dtype=torch.long, device=h.device))
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
        cand = torch.arange(st.n_symbols, dtype=torch.float64, device=c.device)
        ph = cand[:, None] * (self.f / st.scale)
        ph = 2 * math.pi * (ph - torch.floor(ph))
        s = c @ ((self.A if self.readout_amp else 1.0) * torch.exp(-1j * ph)).T  # (n, n_symbols)
        if self.DISCRETE_READOUT == "phase":
            return s.real.float()  # phase matching: Re <code(c), h>
        return (s.abs() ** 2 + 1e-9).log().float()

    def density(self, h, i, xs):
        st = self.streams[i]
        c = self.coeffs(h, torch.full(h.shape[:-1], i, dtype=torch.long, device=h.device))
        Z = torch.einsum("...k,kl,...l->...", c, getattr(self, f"G{i}"), c.conj()).real
        return (self.amp(c.unsqueeze(-2), xs, st).abs() ** 2 / Z.unsqueeze(-1)).float()

    # ------------------------------------------------------------------ streaming (constant memory)
    def init_state(self, B):
        Mm = self.F[0].conv.c_re.shape[1]
        dev = self.f.device
        no = len(self.mem_orders)
        z = lambda: torch.zeros(B, self.R, Mm, dtype=torch.complex64, device=dev)
        extra = {"qconv": z()} if self.query_head else {}
        if self.seasonal:
            D = 2 * len(self.sea_n)
            c128 = lambda *sh: torch.zeros(*sh, dtype=torch.complex128, device=dev)
            extra.update(sP=c128(B, D, self.R), sQ=c128(B, D, self.R), sp=c128(B, D), sS0=c128(B, self.R),
                         sC0=torch.zeros(B, dtype=torch.float64, device=dev), st_last=torch.full((B,), float("nan"), dtype=torch.float64, device=dev))
        return dict(**extra, conv=[z() for _ in range(2 * self.L)] + ([z(), z()] if self.anticipate else []),
                    M=torch.zeros(no, B, self.RM, self.R, dtype=torch.complex64, device=dev),
                    prev=torch.zeros(no, B, self.RM, dtype=torch.complex64, device=dev),
                    hist=torch.zeros(B, max(self.mem_orders), self.RM, dtype=torch.float64, device=dev))

    def step(self, x, s, dt, q, st, wg=None, t_in=None, t_out=None):
        c, ph = self.code(x, s)
        v = torch.polar(torch.ones_like(ph), ph).to(torch.complex64)
        hist = torch.cat([self.mem_phases(x, s)[:, None], st["hist"][:, :-1]], 1)  # most recent first
        M = torch.exp(-self.mem_logmu.exp() * dt)[None, :, None, None] * st["M"]
        r, keys, rc = 0, [], 0
        for o, (n, w) in enumerate(zip(self.mem_orders, self.order_weights())):
            ph_n = sum(hist[:, i][:, self.ctx_perm[i]] for i in range(n))
            k = torch.polar(torch.ones(1, dtype=torch.float64, device=x.device), ph_n).to(torch.complex64)
            ro = torch.einsum("bm,bmr->br", k, M[o]) / self.RM
            r = r + w * ro
            if self.copy_out:
                rc = rc + self.copy_gain[o] * ro
            keys.append(k)
        e = c + self.mem_beta[s][..., None] * torch.cat([r.real, r.imag], -1) * float(self.A.mean())
        if self.clock and t_in is not None:
            e = e + self.clock_code(t_in) @ self.clock_in.T
            if not self.query_head:
                e = e + self.clock_code(t_out) @ self.clock_out.T
        qt = torch.zeros_like(q) if self.query_head else q  # with a query head the trunk never sees the asked time
        vw = v if wg is None else v * wg[:, None].to(v.dtype)
        M = M + st["prev"].conj()[:, :, :, None] * vw[None, :, None, :]
        x1, x2 = e[..., : self.R], e[..., self.R:]
        conv = []
        for i, (F, G, P) in enumerate(zip(self.F, self.G, self.P)):
            f_, s1 = F.step(x2, dt, qt, st["conv"][2 * i]); x1 = x1 + f_
            g_, s2 = G.step(x1, dt, qt, st["conv"][2 * i + 1]); x2 = x2 + g_
            x1, x2 = P(x1, x2)
            conv += [s1, s2]
        out = dict(conv=conv, M=M, prev=torch.stack(keys), hist=hist)
        if self.anticipate:
            z = torch.zeros_like(dt)
            y1, y2 = e[..., : self.R], e[..., self.R:]
            a_, s3 = self.AF.step(y2, dt, z, st["conv"][2 * self.L]); y1 = y1 + a_
            b_, s4 = self.AG.step(y1, dt, z, st["conv"][2 * self.L + 1]); y2 = y2 + b_
            out["conv"] += [s3, s4]
            out["h_ant"] = rms(torch.cat([y1, y2], -1)) * self.again
        if self.query_head:
            h0 = torch.cat([x1, x2], -1)
            n = rms(x2)
            yq, sq_ = self.QF.conv.step(n, dt, q, st["qconv"])
            hq = torch.cat([x1 + self.QF.mix(n, yq), x2], -1)
            if self.clock and t_out is not None:
                hq = hq + self.clock_code(t_out) @ self.clock_out.T
            out["qconv"], out["h0"] = sq_, h0
            h = self.features(hq)
        else:
            h = self.features(torch.cat([x1, x2], -1))
        if self.seasonal and t_in is not None:
            self._seasonal_write(ph, t_in, st, out)
            h = h + self.sea_gain * self._seasonal_query(out, t_out[:, None])[:, 0]
        if self.copy_out:
            h = h + torch.cat([rc.real, rc.imag], -1)
        return h, out

    def _seasonal_write(self, ph, t_in, st, out):
        ok = torch.isfinite(t_in)
        ti = torch.where(ok, t_in, torch.zeros_like(t_in)).double()
        last = st["st_last"]
        fac = torch.where(ok & torch.isfinite(last), torch.exp(-self.sea_logmu.exp().double() * (ti - torch.nan_to_num(last)).clamp(min=0)),
                          torch.ones_like(ti))
        v = torch.polar(torch.ones_like(ph), ph) * ok[:, None]
        kk = self.sea_key(t_in)
        f = fac.to(torch.complex128)
        out["sP"] = f[:, None, None] * st["sP"] + kk.conj()[:, :, None] * v[:, None, :]
        out["sQ"] = f[:, None, None] * st["sQ"] + kk[:, :, None] * v[:, None, :]
        out["sp"] = f[:, None] * st["sp"] + kk.conj()
        out["sS0"] = f[:, None] * st["sS0"] + v
        out["sC0"] = fac * st["sC0"] + ok.double()
        out["st_last"] = torch.where(ok, ti, last)

    def _seasonal_query(self, st, t_q):
        """t_q (B,H) -> (B,H,2R) seasonal read-out from the constant-size state"""
        kq = self.sea_key(t_q)  # (B,H,D)
        num = st["sS0"][:, None] + 0.5 * (torch.einsum("bhd,bdr->bhr", kq, st["sP"]) + torch.einsum("bhd,bdr->bhr", kq.conj(), st["sQ"]))
        den = st["sC0"][:, None] + torch.einsum("bhd,bd->bh", kq, st["sp"]).real
        r = num / (den[..., None] + 1e-6) * torch.isfinite(t_q)[..., None]
        return torch.cat([r.real, r.imag], -1).float()

    def state_numel(self, st):
        return sum(t.numel() for t in st["conv"]) + st["M"].numel() + st["prev"].numel() + st["hist"].numel()

    # ------------------------------------------------------------------ phase adapters (Sec. 5.6)
    def adapter(self, base):
        return {k: v - base[k] for k, v in self.state_dict().items() if k.split(".")[-1] in PHASE_PARAMS}

    def apply_adapters(self, base, *adapters):
        sd = {k: v.clone() for k, v in base.items()}
        for ad in adapters:
            for k, v in ad.items():
                sd[k] = sd[k] + v
        self.load_state_dict(sd)
