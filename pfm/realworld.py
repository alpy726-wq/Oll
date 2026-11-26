"""Real-world event streams for OneMachine (data/v4, built by data/build_real.py). No synthetic data.

Every training example is a window of real events (stream, value, dt):
  weather : Jena climate, a random subset of 1-3 variables; missing readings are simply absent (real irregularity)
  grid    : ETT transformer load/oil temperature, a random subset of 1-3 channels
  market  : VIX daily close (log), real trading-calendar gaps; S&P 500 monthly log-returns
  audio   : raw 8 kHz waveform of real spoken digits
  text    : real books (bytes)
Values are standardised with train-split statistics only, then squashed into the density window.
"""
import os
import numpy as np
import torch
from .one import Stream

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v4")
JENA_VARS = ["p (mbar)", "T (degC)", "rh (%)", "VPdef (mbar)", "wv (m/s)", "rho (g/m**3)"]
ETT_VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

STREAMS_REAL = ([Stream(f"jena:{v}", "continuous", scale=2.0, window=(-1.5, 1.5)) for v in JENA_VARS]
                + [Stream(f"ett:{v}", "continuous", scale=2.0, window=(-1.5, 1.5)) for v in ETT_VARS]
                + [Stream("vix", "continuous", scale=2.0, window=(-1.5, 1.5)),
                   Stream("sp500_ret", "continuous", scale=2.0, window=(-1.5, 1.5)),
                   Stream("audio", "continuous", scale=2.0, window=(-1.5, 1.5)),
                   Stream("text", "discrete", scale=1.0, n_symbols=256),
                   Stream("digit", "discrete", scale=1.0, n_symbols=10)])
SID = {st.name: i for i, st in enumerate(STREAMS_REAL)}
DOMAINS = ("weather", "grid", "market", "audio", "text")


def squash(z):
    return np.clip(z / 3.0, -1.2, 1.2)  # +-3.6 std inside the window; beyond is clipped (rare, real extremes)


class RealData:
    def __init__(self, split):
        self.split = split
        L = lambda n: np.load(os.path.join(D, n + ".npz"), allow_pickle=True)
        jt = L("jena_train"); j = L(f"jena_{split}")
        names = list(jt["names"])
        cols = [names.index(v) for v in JENA_VARS]
        mu, sd = np.nanmean(jt["x"][:, cols], 0), np.nanstd(jt["x"][:, cols], 0)
        self.jena_t = j["t"] / 3600.0  # hours
        self.jena = squash((j["x"][:, cols] - mu) / sd)
        et = L("ETTh1_train"); e = L(f"ETTh1_{split}")
        mu, sd = et["x"].mean(0), et["x"].std(0)
        self.ett_t = e["t"] / 3600.0
        self.ett = squash((e["x"] - mu) / sd)
        vt = L("vix_train"); v = L(f"vix_{split}")
        lv = np.log(vt["x"][:, 3]); mu, sd = lv.mean(), lv.std()
        self.vix_t = v["t"] / 86400.0  # days
        self.vix = squash((np.log(v["x"][:, 3]) - mu) / sd)
        st_ = L("sp500_train"); sp = L(f"sp500_{split}")
        r_tr = np.diff(np.log(st_["x"][:, 0])); mu, sd = r_tr.mean(), r_tr.std()
        self.sp_t = sp["t"][1:] / 86400.0 / 30.0  # months
        self.sp = squash((np.diff(np.log(sp["x"][:, 0])) - mu) / sd)
        a = L(f"fsdd_{'train' if split != 'test' else 'test'}")
        self.audio, self.alen, self.alab = a["audio"], a["lengths"], a["label"]
        self.aoff = np.concatenate([[0], np.cumsum(self.alen)])
        self.text = L(f"text_{'train' if split != 'test' else 'test'}")["bytes"]
        if split == "val":  # validation slices of train-only sources: last 5% (never used in the train sampler below)
            n = len(self.text); self.text = self.text[int(0.95 * n):]
            k = int(0.95 * len(self.alen)); self.alen, self.alab = self.alen[k:], self.alab[k:]; self.aoff = self.aoff[k:]
        elif split == "train":
            n = len(self.text); self.text = self.text[: int(0.95 * n)]
            k = int(0.95 * len(self.alen)); self.alen, self.alab = self.alen[:k], self.alab[:k]; self.aoff = self.aoff[:k + 1]

    # ---------------------------------------------------------------------------------------------- windows
    def _multivar(self, rng, T, t, X, sid0):
        nv = int(rng.integers(1, 4))
        vars_ = np.sort(rng.choice(X.shape[1], nv, replace=False))
        steps = T // nv + 2
        i0 = int(rng.integers(0, len(t) - steps - 1))
        xs, ss, ts = [], [], []
        for i in range(i0, i0 + steps):
            for v in vars_:
                if not np.isnan(X[i, v]):
                    xs.append(X[i, v]); ss.append(sid0 + v); ts.append(t[i])
        return np.array(xs[:T + 1]), np.array(ss[:T + 1]), np.array(ts[:T + 1])

    def window(self, rng, domain, T):
        hours = 1.0  # factor from this window's time unit to absolute hours (calendar clock); None = no calendar
        if domain == "weather":
            x, s, t = self._multivar(rng, T, self.jena_t, self.jena, SID["jena:" + JENA_VARS[0]])
        elif domain == "grid":
            x, s, t = self._multivar(rng, T, self.ett_t, self.ett, SID["ett:" + ETT_VARS[0]])
        elif domain == "market":
            if rng.random() < 0.7:
                i0 = int(rng.integers(0, len(self.vix) - T - 1)); x = self.vix[i0:i0 + T + 1]; t = self.vix_t[i0:i0 + T + 1]
                s = np.full(T + 1, SID["vix"]); hours = 24.0
            else:
                i0 = int(rng.integers(0, len(self.sp) - T - 1)); x = self.sp[i0:i0 + T + 1]; t = self.sp_t[i0:i0 + T + 1]
                s = np.full(T + 1, SID["sp500_ret"]); hours = 24.0 * 30.0
        elif domain == "audio":
            k = int(rng.integers(0, len(self.alen)))
            a = self.audio[self.aoff[k]: self.aoff[k] + self.alen[k]]
            i0 = int(rng.integers(0, max(1, len(a) - T - 1)))
            x = np.clip(a[i0:i0 + T + 1] * 4.0, -1.2, 1.2)  # speech peaks ~0.3; x4 uses the window
            t = np.arange(len(x)) / 8000.0 * 1000.0  # milliseconds
            s = np.full(len(x), SID["audio"]); hours = None
        else:
            i0 = int(rng.integers(0, len(self.text) - T - 1))
            x = self.text[i0:i0 + T + 1].astype(np.float64); t = np.arange(T + 1, dtype=np.float64); s = np.full(T + 1, SID["text"])
            hours = None
        if len(x) < T + 1:  # rare: pad by repeating the last event far in the future, masked out
            pad = T + 1 - len(x)
            x = np.concatenate([x, np.repeat(x[-1:], pad)]); s = np.concatenate([s, np.repeat(s[-1:], pad)])
            t = np.concatenate([t, t[-1] + 1e3 + np.arange(pad)])
        dt = np.diff(t, prepend=t[0] - (t[1] - t[0] if len(t) > 1 else 1.0))
        mask = np.ones(T, bool)
        mask[:8] = False  # the first targets have almost no context
        tabs = t * hours if hours is not None else np.full(len(t), np.nan)
        return x, s, np.clip(dt, 0.0, 1e3), mask, tabs


def make_batch(data, rng, B, T, domains, device="cpu"):
    doms = [domains[i] for i in rng.integers(0, len(domains), B)]
    W = [data.window(rng, d, T) for d in doms]
    x = torch.tensor(np.stack([w[0] for w in W]), dtype=torch.float64, device=device)
    s = torch.tensor(np.stack([w[1] for w in W]), dtype=torch.long, device=device)
    dt = torch.tensor(np.stack([w[2] for w in W]), dtype=torch.float32, device=device)
    mask = torch.tensor(np.stack([w[3] for w in W]), device=device)
    ta = torch.tensor(np.stack([w[4] for w in W]), dtype=torch.float64, device=device)
    return dict(x=x[:, :-1], s=s[:, :-1], dt=dt[:, :-1], q=dt[:, 1:], y=x[:, 1:], s_out=s[:, 1:], mask=mask, domains=doms,
                t_in=ta[:, :-1], t_out=ta[:, 1:])


def fixed_eval_set(split, n_per_domain=64, T=128, seed=12345):
    """deterministic evaluation windows (same for every model)."""
    data = RealData(split)
    rng = np.random.default_rng(seed)
    return {d: make_batch(data, rng, n_per_domain, T, [d]) for d in DOMAINS}
