"""Exactness properties of the machine. Run: python3 -m pytest -q tests"""
import math
import torch
from pfm.basis import PrimeBasis
from pfm.continuous import CTSeqModel, CTConv, PhaseDensityHead

torch.manual_seed(0)


def perturbed(head="phase"):
    m = CTSeqModel(R=64, L=2, head=head)
    for p in m.parameters():
        p.data += 0.05 * torch.randn_like(p)
    return m.eval()


def test_group_law():
    b = PrimeBasis(512)
    x, y = torch.rand(100).double() * 1e4, torch.rand(100).double() * 1e4
    assert (b.complex(x) * b.complex(y) - b.A * b.complex(x + y)).abs().max() < 1e-3  # fp32 storage


def test_density_normalised_closed_form():
    b = PrimeBasis(128, scale=8.0)
    h = PhaseDensityHead(b, -2.0, 10.0)
    xs = torch.linspace(-2, 10, 40001, dtype=torch.float64)
    _, d, _ = h.density_grid(torch.randn(4, 256), 40001)
    assert torch.allclose(torch.trapezoid(d, xs, dim=-1), torch.ones(4, dtype=torch.float64), atol=1e-6)
    assert (d >= 0).all()


def test_streaming_equals_full_pass():
    m = perturbed()
    x, dt, q = torch.rand(3, 50).double() * 8, torch.rand(3, 50) * 2, torch.rand(3, 50)
    with torch.no_grad():
        full = m.trunk(x, dt, q)
        st = m.init_state(3)
        out = []
        for t in range(50):
            f, st = m.step(x[:, t], dt[:, t], q[:, t], st)
            out.append(f)
    assert ((torch.stack(out, 1) - full).norm() / full.norm()) < 1e-5


def test_state_size_independent_of_length():
    m = perturbed()
    sizes = set()
    for T in (10, 1000):
        st = m.init_state(1)
        with torch.no_grad():
            for t in range(T):
                _, st = m.step(torch.rand(1).double() * 8, torch.rand(1), torch.rand(1), st)
        sizes.add(sum(s.numel() for s in st))
    assert len(sizes) == 1


def test_continuous_time_semigroup():
    """advancing the state by a then by b (no input) == advancing by a+b: time is continuous, not a step index."""
    c = CTConv(16, 4)
    a, cc = c.a_cc()
    s0 = torch.randn(2, 16, 4, dtype=torch.complex64)
    t1, t2 = 0.37, 1.91
    two = torch.exp(t2 * a) * (torch.exp(t1 * a) * s0)
    one = torch.exp((t1 + t2) * a) * s0
    assert (two - one).abs().max() < 1e-5
    # and the query offset q is the same operator: read-out at t+q equals advancing then reading at q=0
    u, z = torch.zeros(2, 16), torch.zeros(2)
    y_q, _ = c.step(u, z, torch.full((2,), t1), s0)
    y_adv, _ = c.step(u, torch.full((2,), t1), z, s0)
    assert (y_q - y_adv).abs().max() < 1e-4


def test_irregular_sampling_changes_prediction():
    """the same values at different times must give different predictions (the model actually uses time)."""
    m = perturbed()
    x = torch.rand(1, 30).double() * 8
    with torch.no_grad():
        a = m.trunk(x, torch.full((1, 30), 0.1), torch.full((1, 30), 0.1))
        b = m.trunk(x, torch.full((1, 30), 1.0), torch.full((1, 30), 1.0))
    assert (a - b).norm() / a.norm() > 1e-3
