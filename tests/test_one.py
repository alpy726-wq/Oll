import math
"""Exactness of OneMachine (the full machine): vectorized codes, streaming == full pass (incl. memory), invertibility."""
import math
import numpy as np
import torch
from pfm.one import OneMachine
from pfm.realworld import STREAMS_REAL as STREAMS, RealData, make_batch

_DATA = RealData("train")


def batch(rng, B, T):
    """real windows from the committed frozen corpus (data/v4), mixed domains"""
    doms = ["weather", "grid", "market", "audio", "text"]
    bs = [make_batch(_DATA, rng, 1, T, [doms[i % 5]]) for i in range(B)]
    out = {k: torch.cat([b[k] for b in bs]) for k in ("x", "s", "dt", "q", "y", "s_out", "mask")}
    return out

torch.manual_seed(0)


def machine():
    m = OneMachine(STREAMS, R=64, L=2, RM=128)
    for p in m.parameters():
        p.data += 0.05 * torch.randn_like(p)
    return m.eval()


def test_vectorized_phases_match_reference():
    m = machine()
    x = torch.rand(5, 30, dtype=torch.float64) * 4 - 2
    s = torch.randint(0, len(STREAMS), (5, 30))
    x = torch.where(m.is_integer[s], x.abs().mul(20).round() + 1, x)
    ref = torch.zeros(5, 30, m.R, dtype=torch.float64)
    for i, st in enumerate(STREAMS):  # straightforward per-stream reference
        msk = s == i
        ph = x[msk][:, None] * (m.f / st.scale)
        ph = 2 * math.pi * (ph - torch.floor(ph)) + m.psi[i]
        if st.integer_channel:
            ph = ph - m.integer_phase(x[msk])
        ref[msk] = ph
    assert (torch.exp(1j * m.phases(x, s)) - torch.exp(1j * ref)).abs().max() < 1e-9


def test_streaming_equals_full_with_memory():
    m = machine()
    b = batch(np.random.default_rng(3), 3, 48)
    with torch.no_grad():
        full = m(b["x"], b["s"], b["dt"], b["q"])
        st = m.init_state(3)
        out = []
        for t in range(48):
            f, st = m.step(b["x"][:, t], b["s"][:, t], b["dt"][:, t], b["q"][:, t], st)
            out.append(f)
    assert ((torch.stack(out, 1) - full).norm() / full.norm()) < 1e-5


def test_trunk_is_invertible():
    m = machine().double()
    b = batch(np.random.default_rng(4), 2, 40)
    with torch.no_grad():
        e = m.embed(b["x"], b["s"], b["dt"]).double()
        h = m.trunk_from(e, b["dt"].double(), b["q"].double())
        assert ((m.inverse(h, b["dt"].double(), b["q"].double()) - e).norm() / e.norm()) < 1e-10


def test_densities_normalised():
    m = machine()
    xs = torch.linspace(-1.5, 1.5, 20001, dtype=torch.float64)
    with torch.no_grad():
        d = m.density(torch.randn(3, 2 * m.R), 0, xs)
    assert torch.allclose(torch.trapezoid(d, xs.float(), dim=-1), torch.ones(3), atol=1e-4)


def test_ct_mixed_batch_and_checkpoint_gradients():
    """rows with regular clocks (FFT path, grouped by dt) and irregular clocks (checkpointed scan) in one batch:
    values equal the sequential recurrence, and gradients equal those of the plain scan."""
    from pfm.continuous import CTConv
    c = CTConv(8, 3)
    for p in c.parameters():
        p.data += 0.1 * torch.randn_like(p)
    u = torch.randn(4, 30, 8, requires_grad=True)
    dt = torch.stack([torch.full((30,), 0.5), torch.full((30,), 1.0), torch.rand(30) * 3, torch.full((30,), 0.5)])
    q = torch.stack([torch.full((30,), 0.5), torch.full((30,), 1.0), torch.rand(30), torch.full((30,), 0.5)])
    out = c(u, dt, q)
    a, cc = c.a_cc()
    ref = c.scan_reference(u, dt, a, cc[None, None] * torch.exp(q[..., None, None] * a))
    assert ((out - ref).norm() / ref.norm()) < 1e-5
    g1 = torch.autograd.grad(out.square().sum(), [u, c.c_re, c.log_lam])
    g2 = torch.autograd.grad(ref.square().sum(), [u, c.c_re, c.log_lam])
    for x, y in zip(g1, g2):
        assert ((x - y).norm() / (y.norm() + 1e-12)) < 1e-4


def test_context_keyed_memory_streaming_equals_full():
    m = OneMachine(STREAMS, R=32, L=1, RM=128, mem_orders=(1, 3, 6))
    for p in m.parameters():
        p.data += 0.05 * torch.randn_like(p)
    m.eval()
    b = batch(np.random.default_rng(5), 2, 40)
    with torch.no_grad():
        full = m(b["x"], b["s"], b["dt"], b["q"])
        st = m.init_state(2); out = []
        for t in range(40):
            f, st = m.step(b["x"][:, t], b["s"][:, t], b["dt"][:, t], b["q"][:, t], st); out.append(f)
    assert ((torch.stack(out, 1) - full).norm() / full.norm()) < 1e-5


def test_copy_readout_streaming_equals_full_and_copies():
    """copy read-out: streaming == full; and with only the copy path (trunk output zeroed) a repeated byte string is
    predicted from memory: the second copy costs far fewer bits than the first"""
    from pfm.realworld import STREAMS_REAL, SID
    m = OneMachine(STREAMS, R=32, L=1, RM=128, mem_orders=(1, 3, 6), copy_out=True)
    for p in m.parameters():
        p.data += 0.05 * torch.randn_like(p)
    m.eval()
    b = batch(np.random.default_rng(6), 2, 40)
    with torch.no_grad():
        full = m(b["x"], b["s"], b["dt"], b["q"])
        st = m.init_state(2); out = []
        for t in range(40):
            f, st = m.step(b["x"][:, t], b["s"][:, t], b["dt"][:, t], b["q"][:, t], st); out.append(f)
    assert ((torch.stack(out, 1) - full).norm() / full.norm()) < 1e-5
    m = OneMachine(STREAMS_REAL, R=256, L=1, RM=2048, mem_orders=(6,), copy_out=True).eval()
    with torch.no_grad():
        m.copy_gain.fill_(30.0); m.mem_logwidth.fill_(-20.0)  # copy path only, exact (fine band) keys
        m.gain.zero_()  # trunk features off: only the memory copy path predicts
        span = list(b"the quick brown fox jumps over the lazy dog while 7 cats sleep")
        x = torch.tensor([span + span], dtype=torch.float64); s = torch.full(x.shape, SID["text"]); d = torch.ones(x.shape)
        h = m(x[:, :-1], s[:, :-1], d[:, :-1], d[:, 1:])
        bits = m.nll(h, s[:, 1:], x[:, 1:])[0] / math.log(2)
    n = len(span)
    assert bits[n + 6:].mean() < 0.5 * bits[6:n - 1].mean()


def test_anticipation_streaming_equals_full_and_no_leak():
    """when/which read-outs: streaming == full, and they do not depend on the query time q (no leak of the future)"""
    import numpy as np
    from pfm.realworld import STREAMS_REAL
    torch.manual_seed(0)
    m = OneMachine(STREAMS_REAL, R=16, L=2, M=4, RM=64, anticipate=True).eval()
    B, T = 2, 24
    s = torch.randint(0, 13, (B, T)); x = torch.rand(B, T, dtype=torch.float64) * 2 - 1
    dt = torch.rand(B, T) * 2; q = torch.rand(B, T) * 2
    with torch.no_grad():
        h, ha = m.forward_all(x, s, dt, q)
        _, ha2 = m.forward_all(x, s, dt, q * 0 + 7.0)
        assert torch.allclose(ha, ha2)  # anticipation never sees q
        st = m.init_state(B); hs, has_ = [], []
        for t in range(T):
            o, st = m.step(x[:, t], s[:, t], dt[:, t], q[:, t], st); hs.append(o); has_.append(st["h_ant"])
        assert torch.allclose(torch.stack(hs, 1), h, atol=1e-4)
        assert torch.allclose(torch.stack(has_, 1), ha, atol=1e-4)
        w, c = m.anticipation_nll(ha, s, q)
        assert torch.isfinite(w).all() and torch.isfinite(c).all()


def test_salience_gated_memory_streaming_equals_full():
    from pfm.realworld import STREAMS_REAL
    torch.manual_seed(1)
    m = OneMachine(STREAMS_REAL, R=16, L=2, M=4, RM=64, mem_orders=(1, 3)).eval()
    B, T = 2, 20
    s = torch.randint(0, 13, (B, T)); x = torch.rand(B, T, dtype=torch.float64) * 2 - 1
    dt = torch.rand(B, T) * 2; q = torch.rand(B, T) * 2; wg = torch.rand(B, T) * 2
    with torch.no_grad():
        h = m(x, s, dt, q, wg)
        assert not torch.allclose(h, m(x, s, dt, q))  # the gates matter
        st = m.init_state(B); hs = []
        for t in range(T):
            o, st = m.step(x[:, t], s[:, t], dt[:, t], q[:, t], st, wg[:, t]); hs.append(o)
        assert torch.allclose(torch.stack(hs, 1), h, atol=1e-4)


def test_clock_streaming_equals_full():
    from pfm.realworld import STREAMS_REAL
    torch.manual_seed(2)
    m = OneMachine(STREAMS_REAL, R=16, L=2, M=4, RM=64, clock=True).eval()
    B, T = 2, 20
    s = torch.randint(0, 13, (B, T)); x = torch.rand(B, T, dtype=torch.float64) * 2 - 1
    dt = torch.rand(B, T) * 2; q = torch.rand(B, T) * 2
    t_in = (4e5 + torch.cumsum(dt.double(), 1)); t_out = t_in + q.double()
    t_in[1, :5] = float("nan"); t_out[1, :5] = float("nan")  # streams without a calendar
    with torch.no_grad():
        h = m(x, s, dt, q, t_in=t_in, t_out=t_out)
        assert not torch.allclose(h, m(x, s, dt, q))
        st = m.init_state(B); hs = []
        for t in range(T):
            o, st = m.step(x[:, t], s[:, t], dt[:, t], q[:, t], st, t_in=t_in[:, t], t_out=t_out[:, t]); hs.append(o)
        assert torch.allclose(torch.stack(hs, 1), h, atol=1e-4)


def test_query_head_streaming_peek_and_state_independence():
    from pfm.realworld import STREAMS_REAL
    torch.manual_seed(3)
    m = OneMachine(STREAMS_REAL, R=16, L=2, M=4, RM=64, clock=True, query_head=True).eval()
    B, T = 2, 20
    s = torch.randint(0, 13, (B, T)); x = torch.rand(B, T, dtype=torch.float64) * 2 - 1
    dt = torch.rand(B, T) * 2; q = torch.rand(B, T) * 40  # far queries
    t_in = 4e5 + torch.cumsum(dt.double(), 1); t_out = t_in + q.double()
    with torch.no_grad():
        h = m(x, s, dt, q, t_in=t_in, t_out=t_out)
        st = m.init_state(B); hs = []
        for t in range(T):
            o, st = m.step(x[:, t], s[:, t], dt[:, t], q[:, t], st, t_in=t_in[:, t], t_out=t_out[:, t]); hs.append(o)
        assert torch.allclose(torch.stack(hs, 1), h, atol=1e-4)  # streaming == full
        # the state does not depend on the asked time: same events, other queries -> same state
        st2 = m.init_state(B)
        for t in range(T):
            _, st2 = m.step(x[:, t], s[:, t], dt[:, t], q[:, t] * 0 + 3.0, st2, t_in=t_in[:, t], t_out=t_in[:, t] + 3.0)
        assert torch.allclose(st2["h0"], st["h0"]) and torch.allclose(st2["qconv"], st["qconv"])
        # peeking several moments from the state == the step read-out at those moments
        qs = torch.tensor([[1.0, 7.0, 55.0]] * B)
        pk = m.peek(st, qs, t_out=t_in[:, -1:] + qs.double())
        assert torch.allclose(pk[:, 2], m.peek(st, qs[:, 2:], t_out=t_in[:, -1:] + 55.0)[:, 0], atol=1e-5)
        st_prev = m.init_state(B)
        for t in range(T - 1):
            _, st_prev = m.step(x[:, t], s[:, t], dt[:, t], q[:, t], st_prev, t_in=t_in[:, t], t_out=t_out[:, t])
        o7, _ = m.step(x[:, -1], s[:, -1], dt[:, -1], torch.full((B,), 7.0), st_prev, t_in=t_in[:, -1], t_out=t_in[:, -1] + 7.0)
        assert torch.allclose(o7, pk[:, 1], atol=1e-4)


def test_seasonal_memory_streaming_equals_full_and_recalls_same_hour():
    from pfm.realworld import STREAMS_REAL
    torch.manual_seed(4)
    for qh in (False, True):
        m = OneMachine(STREAMS_REAL, R=16, L=2, M=4, RM=64, clock=True, seasonal=True, query_head=qh).eval()
        B, T = 2, 30
        s = torch.randint(0, 13, (B, T)); x = torch.rand(B, T, dtype=torch.float64) * 2 - 1
        dt = torch.ones(B, T); q = torch.ones(B, T) * (5.0 if qh else 1.0)
        t_in = 4e5 + torch.cumsum(dt.double(), 1); t_out = t_in + q.double()
        t_in[1, :4] = float("nan"); t_out[1, :4] = float("nan")
        with torch.no_grad():
            h = m(x, s, dt, q, t_in=t_in, t_out=t_out)
            st = m.init_state(B); hs = []
            for t in range(T):
                o, st = m.step(x[:, t], s[:, t], dt[:, t], q[:, t], st, t_in=t_in[:, t], t_out=t_out[:, t]); hs.append(o)
            assert torch.allclose(torch.stack(hs, 1), h, atol=1e-4)
            if qh:  # peek == step read-out, seasonal part included
                pk = m.peek(st, torch.tensor([[5.0]] * B), t_out=t_out[:, -1:])
                assert torch.allclose(pk[:, 0], hs[-1], atol=1e-4)
    # recall: a value seen at 15:00 on earlier days is what the seasonal read returns at 15:00 today
    m = OneMachine(STREAMS_REAL, R=64, L=1, M=4, RM=64, clock=True, seasonal=True).eval()
    t = torch.arange(0, 24 * 5, dtype=torch.float64)[None] + 4e5 * 24  # hourly, five days, midnight aligned
    xv = torch.where((t % 24) == 15, torch.tensor(0.7, dtype=torch.float64), torch.tensor(-0.5, dtype=torch.float64))
    s = torch.full(t.shape, 3)
    with torch.no_grad():
        r = m.seasonal_read(m.phases(xv, s), t, t + 1.0)  # each position asks about the next hour
        c = torch.complex(r[..., :64], r[..., 64:]).double() * torch.exp(-1j * m.psi[3])
        grid = torch.linspace(-1.2, 1.2, 241, dtype=torch.float64)
        i = 24 * 4 + 14  # day 5, 14:00, asking about 15:00
        dens = (c[0, i][None] * torch.exp(-1j * 2 * math.pi * grid[:, None] * m.f / 2.0)).sum(-1).abs() ** 2
        assert abs(float(grid[dens.argmax()]) - 0.7) < 0.05
