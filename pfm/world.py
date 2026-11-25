"""A synthetic continuous world: six signal families, five of them sampled at irregular times.
Every sample is (value x in ~[-1,1], dt = time since previous sample). Mixed sampling rates and time units by design."""
import numpy as np
import torch

TASKS = ("sines", "pendulum", "lotka", "lorenz", "oscillator", "bimodal")


def _rk4(f, y, h, n):
    out = np.empty((n, len(y)))
    out[0] = y
    for i in range(1, n):
        k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = y
    return out


def _subsample(traj, h, mean_gap, rng, n):
    """pick n samples of a fine trajectory (step h) with exponential gaps (>= one fine step)."""
    steps = np.maximum(1, np.round(rng.exponential(mean_gap / h, n))).astype(int)
    idx = np.cumsum(steps)
    idx -= idx[0]
    return traj[idx], np.concatenate([[mean_gap], steps[1:] * h])


def gen(task, rng, T):
    n = T + 1
    if task == "sines":  # 3 random sinusoids, evaluated at random times
        dt = np.clip(rng.exponential(1.0, n), 0.05, 4.0)
        t = np.cumsum(dt)
        w, ph, am = rng.uniform(0.1, 0.8, 3), rng.uniform(0, 2 * np.pi, 3), rng.uniform(0.1, 0.35, 3)
        x = (am[:, None] * np.sin(w[:, None] * t + ph[:, None])).sum(0) + 0.02 * rng.standard_normal(n)
    elif task == "pendulum":
        h, g = 0.05, 0.25
        tr = _rk4(lambda s: np.array([s[1], -np.sin(s[0])]), np.array([rng.uniform(-2.8, 2.8), 0.0]), h, int(n * g / h * 4) + 10)
        x, dt = _subsample(tr[:, 0] / np.pi, h, g, rng, n)
    elif task == "lotka":
        h, g = 0.05, 0.35
        tr = _rk4(lambda s: np.array([s[0] * (1.1 - 0.4 * s[1]), s[1] * (0.1 * s[0] - 0.4)]), rng.uniform(2, 12, 2), h, int(n * g / h * 4) + 10)
        x, dt = _subsample(tr[:, 0] / 15 - 1, h, g, rng, n)
    elif task == "lorenz":
        h, g = 0.01, 0.04
        f = lambda s: np.array([10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]])
        tr = _rk4(f, rng.uniform(-10, 10, 3) + np.array([0, 0, 25]), h, 300 + int(n * g / h * 4) + 10)[300:]
        x, dt = _subsample(tr[:, 0] / 20, h, g, rng, n)
    elif task == "oscillator":  # damped oscillator with random frequency/damping, random sample times
        dt = np.clip(rng.exponential(0.35, n), 0.02, 2.0)
        t = np.cumsum(dt)
        w, g, ph = rng.uniform(0.8, 3.0), rng.uniform(0.0, 0.15), rng.uniform(0, 2 * np.pi)
        x = 0.9 * np.exp(-g * t) * np.cos(w * t + ph) + 0.01 * rng.standard_normal(n)
    elif task == "bimodal":  # regular clock; next value = x +/- 0.25 (reflected): a genuinely two-peaked law
        dt = np.ones(n)
        x = [rng.uniform(-0.5, 0.5)]
        for _ in range(T):
            m = x[-1] + rng.choice([-0.25, 0.25]) + 0.01 * rng.standard_normal()
            x.append(m if abs(m) < 1 else x[-1] - (m - x[-1]))
        x = np.array(x)
    else:
        raise ValueError(task)
    return np.clip(x, -1.2, 1.2), dt


def to_model(x):  # value -> model coordinate x' on the density window [-2, 10]
    return x * 4 + 4


def batch(tasks, rng, B, T):
    """inputs x_0..x_{T-1} with dt_0..dt_{T-1}; query gap q_t = dt_{t+1}; target x_{t+1}."""
    xs, dts, ids = [], [], []
    for i in range(B):
        k = i % len(tasks)
        x, dt = gen(tasks[k], rng, T)
        xs.append(x); dts.append(dt); ids.append(k)
    x = torch.tensor(to_model(np.stack(xs)), dtype=torch.float64)
    dt = torch.tensor(np.stack(dts), dtype=torch.float32)
    return x[:, :-1], dt[:, :-1], dt[:, 1:], x[:, 1:], torch.tensor(ids)
