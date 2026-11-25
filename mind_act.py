"""Online trial-and-error: the machine starts from scratch, acts in a continuous world, and learns only from reward.

World: a target moves (random smooth path); the agent sees the error e = target - position at irregular times, chooses
a velocity a (sampled from its own continuous output density, i.e. its policy), the world advances by the real time
gap, and a reward r = -|error| arrives. Nobody shows the right action. Learning: REINFORCE with a baseline, online,
one batch of episodes at a time. Every observation, action and reward is an event the machine itself also sees."""
import json, math, sys, time
import numpy as np
import torch
from pfm.one import OneMachine
from pfm.mind import MIND_STREAMS

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
UPDATES = int(sys.argv[2]) if len(sys.argv) > 2 else 600
import os
B, STEPS = int(os.environ.get("ACT_B", 32)), 40
torch.manual_seed(0)
rng = np.random.default_rng(0)
m = OneMachine(MIND_STREAMS, R=64, L=2, M=4, RM=256)
with torch.no_grad():
    m.mem_beta.fill_(float(os.environ.get("ACT_MEM0", 1.0)))  # memory gate at birth
opt = torch.optim.Adam(m.parameters(), lr=float(__import__("os").environ.get("ACT_LR", "1e-2")))
XS = torch.linspace(-1.5, 1.5, 601, dtype=torch.float64)
OBS, ACT, REW = 2, 3, 4


class World:
    def __init__(self, rng, B):
        self.w = rng.uniform(0.3, 1.2, (B, 2)); self.ph = rng.uniform(0, 2 * np.pi, (B, 2)); self.amp = rng.uniform(0.2, 0.5, (B, 2))
        self.t = np.zeros(B); self.p = rng.uniform(-0.5, 0.5, B)

    def target(self):
        return (self.amp * np.sin(self.w * self.t[:, None] + self.ph)).sum(1)

    def error(self):
        return np.clip(self.target() - self.p, -1.2, 1.2)

    def advance(self, a, dt):
        self.p = np.clip(self.p + 1.5 * a * dt, -1.5, 1.5)
        self.t = self.t + dt
        return -np.abs(self.target() - self.p)


def episode(policy, rng, learn=True):
    """policy in {'machine', 'random', 'proportional'}; returns events, actions, rewards."""
    w = World(rng, B)
    st = m.init_state(B)
    xs, ss, dts, rews, acts = [], [], [], [], []
    for k in range(STEPS):
        gap = float(np.clip(rng.exponential(0.3), 0.05, 1.0))
        e = w.error()
        ev_x = torch.tensor(e); ev_s = torch.full((B,), OBS)
        if policy == "machine":
            with torch.no_grad():
                h, st = m.step(ev_x, ev_s, torch.full((B,), gap), torch.zeros(B), st)
                d = m.density(h, ACT, XS).double()
                cdf = (d / d.sum(-1, keepdim=True)).cumsum(-1)
                idx = torch.searchsorted(cdf.contiguous(), torch.rand(B, 1, dtype=torch.float64)).clamp(max=len(XS) - 1).squeeze(1)
                a = XS[idx].numpy() + rng.uniform(-0.0025, 0.0025, B)
        elif policy == "random":
            a = rng.uniform(-1, 1, B)
        else:
            a = np.clip(2.0 * e, -1.2, 1.2)
        a = np.clip(a, -1.2, 1.2)
        r = w.advance(a, gap)
        if policy == "machine":
            with torch.no_grad():
                _, st = m.step(torch.tensor(a), torch.full((B,), ACT), torch.zeros(B), torch.zeros(B), st)
                _, st = m.step(torch.tensor(r), torch.full((B,), REW), torch.zeros(B), torch.zeros(B), st)
        xs += [e, a, r]; ss += [OBS, ACT, REW]; dts += [gap, 0.0, 0.0]; rews.append(r); acts.append(a)
    return np.stack(xs, 1), np.array(ss), np.array(dts), np.stack(rews, 1), np.stack(acts, 1)


def reinforce(xs, ss, dts, rews):
    x = torch.tensor(xs); s = torch.tensor(ss)[None].repeat(B, 1); dt = torch.tensor(dts, dtype=torch.float32)[None].repeat(B, 1)
    q = torch.zeros_like(dt)  # every read-out at the event's own time: identical to what the agent computed while acting
    h = m(x, s, dt, q)
    obs_pos = torch.arange(0, x.shape[1], 3)
    logp = -m.nll(h[:, obs_pos], torch.full((B, len(obs_pos)), ACT), x[:, obs_pos + 1])
    R = torch.tensor(rews, dtype=torch.float32)
    ret = torch.zeros_like(R); run = torch.zeros(B)
    for k in reversed(range(R.shape[1])):  # discounted return
        run = R[:, k] + 0.5 * run; ret[:, k] = run
    adv = ret - ret.mean(0, keepdim=True)
    adv = adv / (adv.std() + 1e-6)
    loss = -(adv * logp).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()


ref = {p: float(np.mean([episode(p, np.random.default_rng(500 + i))[3].mean() for i in range(5)])) for p in ("random", "proportional")}
curve, t0 = [], time.time()
for u in range(UPDATES):
    xs, ss, dts, rews, acts = episode("machine", rng)
    reinforce(xs, ss, dts, rews)
    if u % 25 == 0 or u == UPDATES - 1:
        m.eval()
        test = float(np.mean([episode("machine", np.random.default_rng(500 + i))[3].mean() for i in range(5)]))
        m.train()
        curve.append(dict(update=u, episodes_seen=(u + 1) * B, mean_reward=test, sec=time.time() - t0))
        print(curve[-1], "| random", round(ref["random"], 3), "| hand-made controller", round(ref["proportional"], 3), flush=True)
tag = os.environ.get("ACT_TAG", "")
torch.save(m.state_dict(), f"runs/mind_act{tag}.pt")
json.dump(dict(reference=ref, curve=curve, env={k: v for k, v in os.environ.items() if k.startswith("ACT_")}), open(f"runs/mind_act{tag}.json", "w"), indent=1)
