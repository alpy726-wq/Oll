"""Shared training helpers."""
import torch


def far_targets(b, rng):
    """half of the positions ask about the next event, half about a random later event of the window (target value,
    its stream, the time until it, its absolute time); anticipation keeps the next-event targets"""
    X = torch.cat([b["x"], b["y"][:, -1:]], 1); S = torch.cat([b["s"], b["s_out"][:, -1:]], 1)
    D = torch.cat([b["dt"], b["q"][:, -1:]], 1); TA = torch.cat([b["t_in"], b["t_out"][:, -1:]], 1)
    Bn, T = b["x"].shape
    tau = D.double().cumsum(1)
    i = torch.arange(T, device=X.device)[None].expand(Bn, T)
    far = torch.from_numpy(rng.random((Bn, T)) < 0.5).to(X.device)
    jr = i + 1 + (torch.from_numpy(rng.random((Bn, T))).to(X.device) * (T - i).double()).long().clamp(max=T - 1 - i + 0)
    j = torch.where(far, jr.clamp(max=T), i + 1)
    g = lambda A: A.gather(1, j)
    out = dict(b)
    out.update(y=g(X), s_out=g(S), q=(g(tau) - tau[:, :T]).float(), t_out=g(TA), q_next=b["q"], s_next=b["s_out"])
    return out
