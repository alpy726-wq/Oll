"""Mixed event world for OneMachine: continuous dynamics, multi-sensor fusion, associative recall, text.
Every sequence is a list of events (stream, value, dt). Targets: the next event, whose stream and time gap are known."""
import os
import numpy as np
import torch
from .world import gen, TASKS, _rk4
from .one import Stream

STREAMS = [
    Stream("signal", "continuous", scale=2.0, window=(-1.5, 1.5)),     # 0: six dynamical families (Sec. 5.1 world)
    Stream("sensorA", "continuous", scale=2.0, window=(-1.5, 1.5)),    # 1: prey, irregular clock
    Stream("sensorB", "continuous", scale=2.0, window=(-1.5, 1.5)),    # 2: predator, its own irregular clock
    Stream("key", "continuous", scale=0.1, window=(-1.5, 1.5)),        # 3: real-valued keys (never predicted)
    Stream("symbol", "discrete", scale=1.0, n_symbols=64),            # 4: values bound to keys
    Stream("text", "discrete", scale=1.0, n_symbols=256),             # 5: bytes
    Stream("integer", "discrete", scale=1.0, n_symbols=64, integer_channel=True),  # 6: integers (multiplicative channel)
]
TEXT = None


def _text():
    global TEXT
    if TEXT is None:
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        TEXT = {sp: np.frombuffer(open(os.path.join(root, f"prose.{sp}"), "rb").read(), dtype=np.uint8) for sp in ("train", "val")}
    return TEXT


def seq_signal(rng, T, family=None):
    fam = family or TASKS[rng.integers(len(TASKS))]
    x, dt = gen(fam, rng, T)
    return x, np.zeros(T + 1, int), dt, np.ones(T, bool), fam


def seq_fusion(rng, T):
    """Lotka-Volterra prey (A) and predator (B), each read on its own random clock, merged in time order."""
    h = 0.05
    tr = _rk4(lambda s: np.array([s[0] * (1.1 - 0.4 * s[1]), s[1] * (0.1 * s[0] - 0.4)]), rng.uniform(2, 12, 2), h, 4000)
    tA = np.cumsum(rng.exponential(0.5, T + 1)); tB = np.cumsum(rng.exponential(0.5, T + 1))
    ev = sorted([(t, 1) for t in tA] + [(t, 2) for t in tB])[: T + 1]
    times = np.array([e[0] for e in ev]); st = np.array([e[1] for e in ev])
    idx = np.minimum((times / h).astype(int), len(tr) - 1)
    val = np.where(st == 1, tr[idx, 0] / 15 - 1, tr[idx, 1] / 8 - 1)
    dt = np.diff(times, prepend=times[0] - 0.5)
    return np.clip(val, -1.2, 1.2), st, dt, np.ones(T, bool), "fusion"


def seq_recall(rng, T, n=None):
    n = n or int(rng.integers(2, T // 4 + 1))
    keys = rng.uniform(-1, 1, n); vals = rng.integers(0, 64, n)
    x, s = [], []
    for k, v in zip(keys, vals):
        x += [k, v]; s += [3, 4]
    perm = rng.permutation(n)
    for i in perm:
        x += [keys[i], vals[i]]; s += [3, 4]
    x, s = np.array(x, float), np.array(s)
    L = len(x)
    pad = T + 1 - L
    x = np.concatenate([x, np.zeros(pad)]); s = np.concatenate([s, np.full(pad, 4)])
    mask = np.zeros(T, bool)
    mask[2 * n: L - 1: 2] = True  # targets are events 2n+1, 2n+3, ... = values after each query key
    return x, s, np.ones(T + 1), mask, f"recall{n}"


def seq_text(rng, T, split="train"):
    a = _text()[split]
    i = rng.integers(0, len(a) - T - 1)
    return a[i:i + T + 1].astype(float), np.full(T + 1, 5), np.ones(T + 1), np.ones(T, bool), "text"


MIX = (("signal", 0.4), ("fusion", 0.2), ("recall", 0.2), ("text", 0.2))


def make(kind, rng, T, **kw):
    return {"signal": seq_signal, "fusion": seq_fusion, "recall": seq_recall, "text": seq_text}[kind](rng, T, **kw)


def batch(rng, B, T, kinds=None, **kw):
    kinds = kinds or [MIX[i][0] for i in rng.choice(len(MIX), B, p=[w for _, w in MIX])]
    seqs = [make(k, rng, T, **kw) for k in kinds]
    x = torch.tensor(np.stack([q[0] for q in seqs]), dtype=torch.float64)
    s = torch.tensor(np.stack([q[1] for q in seqs]), dtype=torch.long)
    dt = torch.tensor(np.stack([q[2] for q in seqs]), dtype=torch.float32)
    mask = torch.tensor(np.stack([q[3] for q in seqs]))
    return dict(x=x[:, :-1], s=s[:, :-1], dt=dt[:, :-1], q=dt[:, 1:], y=x[:, 1:], s_out=s[:, 1:], mask=mask,
                kinds=[q[4] for q in seqs])
