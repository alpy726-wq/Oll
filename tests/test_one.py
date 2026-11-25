"""Exactness of OneMachine (the full machine): vectorized codes, streaming == full pass (incl. memory), invertibility."""
import math
import numpy as np
import torch
from pfm.one import OneMachine
from pfm.oneworld import STREAMS, batch

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
