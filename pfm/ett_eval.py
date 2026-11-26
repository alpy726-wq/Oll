"""Standard long-horizon forecasting protocol on ETT (Zhou et al. 2021), as used by Informer/Autoformer/DLinear/PatchTST:
borders train 12 / val 4 / test 4 months, StandardScaler fitted on train, lookback L, horizons 96/192/336/720,
MSE and MAE on standardised values averaged over all channels, steps and test windows.

OneMachine forecasts channel-independently and autoregressively: at every step it reads the median of its own
predicted density and feeds it back (constant-memory streaming). Deviation from the reference protocol, stated in every
result: test windows are taken with a stride (default 24 h for hourly data) instead of every hour, and all horizons
use the same set of window starts (those that fit the 720-step horizon). Two no-learning baselines use the identical
windows: repeat-last-value and seasonal-naive (one day)."""
import math, os
import numpy as np
import torch

HORIZONS = (96, 192, 336, 720)


def load_ett(name, data_dir):
    """rebuilds the full series from the frozen corpus (data/v4/<name>_{train,val,test}.npz = months 0-12, 12-16, 16-20)."""
    parts = [np.load(os.path.join(data_dir, f"{name}_{sp}.npz")) for sp in ("train", "val", "test")]
    t = np.concatenate([p["t"] for p in parts]) / 3600.0
    X = np.concatenate([p["x"] for p in parts])
    per_month = 30 * 24 * (4 if name.endswith("m1") or name.endswith("m2") else 1)
    tr = 12 * per_month
    mu, sd = X[:tr].mean(0), X[:tr].std(0)
    return t, (X - mu) / sd, 16 * per_month, 20 * per_month, per_month


@torch.no_grad()
def evaluate(model, sids, name, data_dir, device, L=512, stride=24, max_windows=None, adapt=False, paths=1, seed=0, chunk=256):
    """adapt=True: sensory adaptation (gain control) -- every window is seen relative to its own lookback level and
    spread (mean/std of the L past values only; RevIN-style), and the forecast is mapped back. No retraining."""
    t, Z, te_start, te_end, per_month = load_ett(name, data_dir)
    H = max(HORIZONS)
    starts = list(range(te_start, te_end - H + 1, stride))  # first forecast index of each window
    if max_windows:
        starts = starts[:max_windows]
    C = Z.shape[1]
    dtc = float(np.median(np.diff(t)))
    sq = lambda z: np.clip(z / 3.0, -1.2, 1.2)
    grid = torch.linspace(-1.5, 1.5, 1501, dtype=torch.float64, device=device)
    xs, ys, preds = [], [], []
    for c in range(C):  # channel-independent: one batch per channel, so every row shares one stream
        X = np.stack([Z[s - L:s, c] for s in starts]); Y = np.stack([Z[s:s + H, c] for s in starts])
        n0 = len(starts)
        if paths > 1:  # several imagined futures per window (sampled from the density), pointwise median at the end
            X_all, n = np.repeat(X, paths, 0), n0 * paths
        else:
            X_all, n = X, n0
        mu_w = X_all.mean(1, keepdims=True) if adapt else np.zeros((n, 1))
        sd_w = X_all.std(1, keepdims=True) + 1e-3 if adapt else np.ones((n, 1))
        gen = torch.Generator(device="cpu").manual_seed(seed + c)
        P = np.zeros((n, H))
        for r0 in range(0, n, chunk):  # rows in chunks (the memory state is rows x RM x R)
            r1 = min(n, r0 + chunk); nb = r1 - r0
            sid = torch.full((nb,), int(sids[c]), device=device)
            st = model.init_state(nb)
            x = torch.tensor(sq((X_all[r0:r1] - mu_w[r0:r1]) / sd_w[r0:r1]), dtype=torch.float64, device=device)
            one = torch.full((nb,), dtc, device=device)
            for k in range(L):
                h, st = model.step(x[:, k], sid, one, one, st)
            for k in range(H):
                dens = model.density(h, int(sids[c]), grid).double()
                cdf = (dens / dens.sum(-1, keepdim=True)).cumsum(-1)
                u = torch.rand(nb, 1, generator=gen, dtype=cdf.dtype).to(device) if paths > 1 else torch.full((nb, 1), 0.5, dtype=cdf.dtype, device=device)
                med = grid[torch.searchsorted(cdf.contiguous(), u).clamp(max=len(grid) - 1).squeeze(1)]
                P[r0:r1, k] = med.cpu().numpy() * 3.0 * sd_w[r0:r1, 0] + mu_w[r0:r1, 0]
                h, st = model.step(med, sid, one, one, st)
        if paths > 1:
            P = np.median(P.reshape(n0, paths, H), 1)
        xs.append(X); ys.append(Y); preds.append(P)
    xs, ys, preds = np.concatenate(xs), np.concatenate(ys), np.concatenate(preds)
    idx = [(s, c) for c in range(C) for s in starts]
    res = {"protocol": dict(dataset=name, lookback=L, window_stride=stride, windows=len(starts), channels=C, adaptation=adapt, sampled_paths=paths)}
    last = xs[:, -1:]
    season = 24 * (per_month // 720)  # one day: 24 hourly or 96 quarter-hourly steps
    seas = np.stack([Z[s - season:s, c] for s, c in idx])
    for Hh in HORIZONS:
        e = preds[:, :Hh] - ys[:, :Hh]
        naive = last - ys[:, :Hh]
        sn = np.tile(seas, (1, math.ceil(Hh / season)))[:, :Hh] - ys[:, :Hh]
        res[Hh] = dict(model_mse=float((e ** 2).mean()), model_mae=float(np.abs(e).mean()),
                       repeat_last_mse=float((naive ** 2).mean()), repeat_last_mae=float(np.abs(naive).mean()),
                       seasonal_naive_mse=float((sn ** 2).mean()), seasonal_naive_mae=float(np.abs(sn).mean()))
    return res


