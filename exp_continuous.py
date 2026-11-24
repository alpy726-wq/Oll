"""E5: continuous output head (5.1) on a synthetic continuous world.  E6: continuous time (5.2) on irregular samples.
Values are normalised to ~[-1,1] and mapped to x' = 4(x+1) in [0,8]; densities live on the window [-2,10].
All NLLs are in nats of x' (identical units across heads, so they are directly comparable)."""
import argparse, json, math, time
import numpy as np
import torch
from pfm.continuous import CTSeqModel

ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=["heads", "time"], required=True)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--T", type=int, default=64)
ap.add_argument("--B", type=int, default=24)
ap.add_argument("--R", type=int, default=128)
ap.add_argument("--L", type=int, default=3)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--only", default="")
a = ap.parse_args()
torch.set_num_threads(a.threads)


def rk4(f, y, h, n):
    out = [y]
    for _ in range(n - 1):
        k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(y)
    return np.stack(out)


def gen(task, rng, T):
    """returns values x (T+1,), dt (T+1,)  (dt_i = time since previous sample)"""
    dt = np.ones(T + 1)
    if task == "sines":
        t = np.arange(T + 1)
        w, ph, am = rng.uniform(0.1, 0.8, 3), rng.uniform(0, 2 * np.pi, 3), rng.uniform(0.1, 0.35, 3)
        x = (am[:, None] * np.sin(w[:, None] * t + ph[:, None])).sum(0) + 0.02 * rng.standard_normal(T + 1)
    elif task == "pendulum":
        y = rk4(lambda s: np.array([s[1], -np.sin(s[0])]), np.array([rng.uniform(-2.8, 2.8), 0.0]), 0.05, 4 * (T + 1))[::4]
        x = y[:, 0] / np.pi
    elif task == "lotka":
        y = rk4(lambda s: np.array([s[0] * (1.1 - 0.4 * s[1]), s[1] * (0.1 * s[0] - 0.4)]), rng.uniform(2, 12, 2), 0.05, 6 * (T + 1))[::6]
        x = y[:, 0] / 15 - 1
    elif task == "lorenz":
        f = lambda s: np.array([10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]])
        y = rk4(f, rng.uniform(-10, 10, 3) + np.array([0, 0, 25]), 0.01, 300 + 4 * (T + 1))[300::4]
        x = y[:, 0] / 20
    elif task == "bimodal":  # x_{t+1} = x_t +/- 0.25 (reflected): a two-peaked next-value law
        x = [rng.uniform(-0.5, 0.5)]
        for _ in range(T):
            n = x[-1] + rng.choice([-0.25, 0.25]) + 0.01 * rng.standard_normal()
            x.append(n if abs(n) < 1 else x[-1] - (n - x[-1]))
        x = np.array(x)
    elif task == "irregular":  # damped oscillator sampled at random times
        dt = np.clip(rng.exponential(0.35, T + 1), 0.02, 2.0)
        t = np.cumsum(dt)
        w, g, ph = rng.uniform(0.8, 3.0), rng.uniform(0.0, 0.15), rng.uniform(0, 2 * np.pi)
        x = 0.9 * np.exp(-g * t) * np.cos(w * t + ph) + 0.01 * rng.standard_normal(T + 1)
    return np.clip(x, -1.2, 1.2), dt


def batch(tasks, rng, B, T):
    xs, dts, ids = [], [], []
    for i in range(B):
        k = i % len(tasks)
        x, dt = gen(tasks[k], rng, T)
        xs.append(x); dts.append(dt); ids.append(k)
    x = torch.tensor(np.stack(xs) * 4 + 4, dtype=torch.float64)
    dt = torch.tensor(np.stack(dts), dtype=torch.float32)
    # inputs: x_0..x_{T-1}, dt_0..dt_{T-1}; query gap q_t = dt_{t+1}; target x_{t+1}
    return x[:, :-1], dt[:, :-1], dt[:, 1:], x[:, 1:], torch.tensor(ids)


def train_eval(tasks, mode, head, seed=0, steps=a.steps):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = CTSeqModel(R=a.R, L=a.L, mode=mode, head=head, seed=seed)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    t0 = time.time()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = 3e-3 * min(1, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        x, dt, q, y, _ = batch(tasks, rng, a.B, a.T)
        loss = m.nll(x, dt, q, y)[:, 8:].mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 100 == 0:
            print(f"  {mode}/{head} step {s} nll {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    m.eval()
    erng = np.random.default_rng(10_000)
    res = {}
    with torch.no_grad():
        per = {k: [] for k in range(len(tasks))}
        for _ in range(8):
            x, dt, q, y, ids = batch(tasks, erng, 32, a.T)
            nll = m.nll(x, dt, q, y)[:, 8:].mean(1)
            for k in range(len(tasks)):
                per[k].append(nll[ids == k])
        for k, t in enumerate(tasks):
            res[t] = float(torch.cat(per[k]).mean())
    res["train_seconds"] = time.time() - t0
    return m, res


out = {}
if a.part == "heads":
    tasks = ["sines", "pendulum", "lotka", "lorenz", "bimodal"]
    heads = a.only.split(",") if a.only else ["phase", "gauss", "mdn"]
    for head in heads:
        m, r = train_eval(tasks, "ct", head)
        out[head] = r
        print(head, json.dumps(r), flush=True)
        if head == "phase":
            # shape of the predicted density on the bimodal task: mass near the two true modes vs. in between
            rng = np.random.default_rng(99)
            x, dt, q, y, ids = batch(["bimodal"], rng, 16, a.T)
            with torch.no_grad():
                h = m.trunk(x, dt, q)
                xs, dens, _ = m.head.density_grid(h[:, -1], 4001)
            cur = x[:, -1]
            m_plus = np.mean([float((dens[i] * ((xs - (cur[i] + 1.0)).abs() < 0.3)).sum() * (xs[1] - xs[0])) for i in range(16)])
            m_minus = np.mean([float((dens[i] * ((xs - (cur[i] - 1.0)).abs() < 0.3)).sum() * (xs[1] - xs[0])) for i in range(16)])
            m_mid = np.mean([float((dens[i] * ((xs - cur[i]).abs() < 0.3)).sum() * (xs[1] - xs[0])) for i in range(16)])
            Zn = float(torch.trapezoid(dens, xs, dim=-1).mean())
            out["phase_bimodal_shape"] = dict(mass_near_x_plus_step=m_plus, mass_near_x_minus_step=m_minus,
                                              mass_near_x_itself=m_mid, numeric_integral_of_density=Zn)
            np.save("runs/E5_bimodal_density.npy", np.stack([xs.numpy(), *dens[:4].numpy()]))
            np.save("runs/E5_bimodal_context.npy", np.stack([x[:4, -6:].numpy(), y[:4, -1:].repeat(1, 6).numpy()]))
            print(out["phase_bimodal_shape"], flush=True)
    json.dump(out, open("runs/E5_heads.json" if not a.only else f"runs/E5_heads_{a.only}.json", "w"), indent=1)
else:
    modes = a.only.split(",") if a.only else ["ct", "fixed+dt", "fixed"]
    for mode in modes:
        _, r = train_eval(["irregular"], mode, "phase")
        out[mode] = r
        print(mode, json.dumps(r), flush=True)
    json.dump(out, open("runs/E6_time.json" if not a.only else f"runs/E6_time_{a.only}.json", "w"), indent=1)
