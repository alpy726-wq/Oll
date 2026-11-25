"""Main-run sampler: the frozen v4 corpus (pfm.realworld) plus the large real corpus from gpu/bigdata.py.
Every Chronos/Monash collection is its own typed stream; FineWeb-edu bytes share the text stream."""
import json, os
import numpy as np
import torch
from .one import Stream
from .realworld import STREAMS_REAL, SID, RealData

BIG_TS = ["electricity_15min", "monash_traffic", "monash_weather", "solar_1h", "taxi_30min", "exchange_rate",
          "monash_kdd_cup_2018", "uber_tlc_hourly", "wind_farms_hourly", "monash_pedestrian_counts"]
STREAMS_BIG = STREAMS_REAL + [Stream(f"ts:{n}", "continuous", scale=2.0, window=(-1.5, 1.5)) for n in BIG_TS]
SID_BIG = {st.name: i for i, st in enumerate(STREAMS_BIG)}
DOMAINS_BIG = ("weather", "grid", "market", "audio", "text", "web", "series")


class BigData:
    def __init__(self, root="/vol/big", split="train"):
        self.small = RealData(split if split != "val" else "val")
        self.split = split
        self.ts = {}
        self.names = [n for n in BIG_TS if os.path.exists(f"{root}/{n}.npz")]  # all ten on the GPU run
        for n in self.names:
            z = np.load(f"{root}/{n}.npz")
            self.ts[n] = (z["x"], z["t"], z["off"])
        self.web = np.load(f"{root}/fineweb.npy", mmap_mode="r")
        n = len(self.web)
        self.web_lo, self.web_hi = (0, int(0.98 * n)) if split == "train" else (int(0.98 * n), n)
        sizes = np.array([self.ts[k][2][-1] for k in self.names], dtype=float)
        self.ts_p = np.sqrt(sizes) / np.sqrt(sizes).sum()  # sample collections by sqrt(size)

    def _series(self, rng, T):
        k = int(rng.choice(len(self.names), p=self.ts_p)); name = self.names[k]
        x, t, off = self.ts[name]
        for _ in range(20):
            i = int(rng.integers(len(off) - 1)); a, b = off[i], off[i + 1]
            cut = a + int(0.8 * (b - a))
            lo, hi = (a, cut) if self.split == "train" else (cut, b)
            if hi - lo > T + 2:
                s = int(rng.integers(lo, hi - T - 1))
                xs, ts = x[s:s + T + 1].astype(np.float64), t[s:s + T + 1]
                dt = np.diff(ts, prepend=ts[0] - (ts[1] - ts[0]))
                return xs, np.full(T + 1, SID_BIG["ts:" + name]), np.clip(dt, 0, 1e3)
        return None

    def window(self, rng, domain, T):
        if domain == "series":
            w = None
            while w is None:
                w = self._series(rng, T)
            x, s, dt = w
            m = np.ones(T, bool); m[:8] = False
            return x, s, dt, m
        if domain == "web":
            i = int(rng.integers(self.web_lo, self.web_hi - T - 1))
            x = np.asarray(self.web[i:i + T + 1], dtype=np.float64)
            m = np.ones(T, bool); m[:8] = False
            return x, np.full(T + 1, SID["text"]), np.ones(T + 1), m
        return self.small.window(rng, domain, T)


def make_batch_big(data, rng, B, T, domain, device):
    W = [data.window(rng, domain, T) for _ in range(B)]
    x = torch.tensor(np.stack([w[0] for w in W]), dtype=torch.float64, device=device)
    s = torch.tensor(np.stack([w[1] for w in W]), dtype=torch.long, device=device)
    dt = torch.tensor(np.stack([w[2] for w in W]), dtype=torch.float32, device=device)
    mask = torch.tensor(np.stack([w[3] for w in W]), device=device)
    return dict(x=x[:, :-1], s=s[:, :-1], dt=dt[:, :-1], q=dt[:, 1:], y=x[:, 1:], s_out=s[:, 1:], mask=mask, domain=domain)
