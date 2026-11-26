"""Oll-1: the trained continuous machine as a tool for real sensor / time-series data.

  load(path)                      -> model
  forecast(model, values, times)  -> median and 10-90% band for the next `horizon` steps (imagined futures)
  surprise(model, series)         -> per point: how unexpected each new reading was (-log density, adapted units)
  alarms(scores, ...)             -> anomaly flags from the surprise, threshold learnt on the first part of the series

Every value is seen relative to its own recent past (sensory adaptation: mean/std of the previous `adapt_window`
readings, causal), so the tool works on any unit and level without retraining. Time gaps are real: irregular
timestamps and missing readings are allowed."""
import json, math, os
import numpy as np
import torch

from pfm.one import OneMachine
from pfm.bigworld import STREAMS_BIG

GENERIC_STREAM = 18  # "ts:electricity_15min": a generic series role learnt on many real series


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    cfg = ck["config"]
    m = OneMachine(STREAMS_BIG, R=cfg["R"], L=cfg["L"], M=cfg["M"], RM=cfg["RM"]).to(device)
    m.load_state_dict({k: v.float() if v.dtype == torch.float16 else v for k, v in ck["model"].items()})
    m.eval()
    return m


def _gaps(times, n):
    """time gaps in units of the typical (median) gap; 1.0 for regular data"""
    if times is None:
        return np.ones(n)
    t = np.asarray(times, dtype=np.float64)
    d = np.diff(t, prepend=t[0] - np.median(np.diff(t)) if n > 1 else t[0] - 1)
    med = np.median(d[d > 0]) if (d > 0).any() else 1.0
    return np.clip(d / med, 0, 1e3)


def _adapt(v, w):
    """causal running level/spread: statistics of the previous w readings (excluding the current one)"""
    v = np.asarray(v, dtype=np.float64)
    mu, sd = np.empty_like(v), np.empty_like(v)
    c1 = np.concatenate([[0], np.cumsum(v)]); c2 = np.concatenate([[0], np.cumsum(v * v)])
    for i in range(len(v)):
        a, b = max(0, i - w), i
        if b - a < 2:
            mu[i], sd[i] = (v[0], 1.0) if i == 0 else (v[:i].mean(), 1.0)
            continue
        m_ = (c1[b] - c1[a]) / (b - a); var = (c2[b] - c2[a]) / (b - a) - m_ * m_
        mu[i], sd[i] = m_, math.sqrt(max(var, 0.0)) + 1e-6
    sd = np.maximum(sd, 1e-3 * (np.abs(mu) + 1e-6))
    return mu, sd


@torch.no_grad()
def surprise(model, series, adapt_window=512, stream=GENERIC_STREAM, device="cpu", normal="running", calm_frac=0.15):
    """series: list of (values, times or None). All series are streamed together (one event at a time, constant
    memory). Returns a list of arrays: surprise per point (nats; first point = nan).
    normal="running": values seen relative to the previous `adapt_window` readings (adapts to every new level).
    normal="calm":    values seen relative to the first `calm_frac` of the series (what normal operation looks like),
                      so leaving the normal range stays surprising."""
    B = len(series)
    lens = [len(v) for v, _ in series]
    T = max(lens)
    Z = np.zeros((B, T)); D = np.ones((B, T)); Mk = np.zeros((B, T), bool)
    for i, (v, t) in enumerate(series):
        if normal == "calm":
            n0 = max(10, int(calm_frac * len(v))); vv = np.asarray(v, dtype=np.float64)
            mu, sd = vv[:n0].mean(), vv[:n0].std() + 1e-6 * (abs(vv[:n0].mean()) + 1e-6)
        else:
            mu, sd = _adapt(v, adapt_window)
        Z[i, :lens[i]] = np.clip((np.asarray(v) - mu) / sd / 3.0, -1.2, 1.2)
        D[i, :lens[i]] = _gaps(t, lens[i]); Mk[i, :lens[i]] = True
    x = torch.tensor(Z, dtype=torch.float64, device=device)
    dt = torch.tensor(D, dtype=torch.float32, device=device)
    sid = torch.full((B,), stream, device=device)
    st = model.init_state(B)
    out = np.full((B, T), np.nan)
    h = None
    for k in range(T):
        if h is not None:
            out[:, k] = model.nll(h, sid, x[:, k]).cpu().numpy()
        q = dt[:, k + 1] if k + 1 < T else torch.ones(B, device=device)
        h, st = model.step(x[:, k], sid, dt[:, k], q, st)
    return [out[i, :lens[i]] for i in range(B)]


def alarms(scores, train_frac=0.15, z=4.0, smooth=3):
    """flag points whose (smoothed) surprise exceeds mean + z*std of the first `train_frac` of the series
    (the 'calm' learning period, as in NAB's probationary period). Returns a boolean array."""
    s = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0)
    if smooth > 1:
        s = np.convolve(s, np.ones(smooth) / smooth, mode="same")
    n0 = max(10, int(train_frac * len(s)))
    thr = s[1:n0].mean() + z * s[1:n0].std()
    f = s > thr
    f[:n0] = False
    return f


@torch.no_grad()
def forecast(model, values, times=None, horizon=96, paths=16, lookback=512, stream=GENERIC_STREAM, device="cpu", seed=0):
    """median and 10/90% quantiles of `paths` imagined futures, `horizon` steps of the typical gap ahead."""
    v = np.asarray(values, dtype=np.float64)[-lookback:]
    g = _gaps(times, len(values))[-lookback:]
    mu, sd = v.mean(), v.std() + 1e-6
    x = torch.tensor(np.clip((v - mu) / sd / 3.0, -1.2, 1.2), dtype=torch.float64, device=device).repeat(paths, 1)
    dt = torch.tensor(g, dtype=torch.float32, device=device).repeat(paths, 1)
    sid = torch.full((paths,), stream, device=device)
    one = torch.ones(paths, device=device)
    st = model.init_state(paths)
    for k in range(x.shape[1]):
        h, st = model.step(x[:, k], sid, dt[:, k], dt[:, k + 1] if k + 1 < x.shape[1] else one, st)
    grid = torch.linspace(-1.5, 1.5, 1501, dtype=torch.float64, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    P = np.zeros((paths, horizon))
    for k in range(horizon):
        dens = model.density(h, stream, grid).double()
        cdf = (dens / dens.sum(-1, keepdim=True)).cumsum(-1)
        u = torch.rand(paths, 1, generator=gen, dtype=cdf.dtype).to(device)
        val = grid[torch.searchsorted(cdf.contiguous(), u).clamp(max=len(grid) - 1).squeeze(1)]
        P[:, k] = val.cpu().numpy() * 3.0 * sd + mu
        h, st = model.step(val, sid, one, one, st)
    return dict(median=np.median(P, 0), q10=np.quantile(P, 0.1, 0), q90=np.quantile(P, 0.9, 0))
