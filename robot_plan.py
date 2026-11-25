"""Deciding with the world model (real test episodes only). For each test moment the machine sees 512 real events of
context and a GOAL: the real joint positions 1 s later. It is offered N candidate motor-command sequences for that
second: the one the operator really executed, and N-1 taken from other test episodes. It imagines the outcome of each
(free-running, as in robot_world.imagine) and picks the candidate whose imagined final joints are closest to the goal.
Score: how often it picks the real one (chance = 1/N), and the goal error of its pick vs a random pick."""
import argparse, json, os
import numpy as np
import torch
from pfm.one import OneMachine
from robot_world import STREAMS_ROBOT, GROUP, RobotData, imagine

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--R", type=int, default=96); ap.add_argument("--L", type=int, default=3); ap.add_argument("--RM", type=int, default=512)
ap.add_argument("--N", type=int, default=4); ap.add_argument("--n", type=int, default=24)
ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", 4)))
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
m = OneMachine(STREAMS_ROBOT, R=a.R, L=a.L, M=8, RM=a.RM); m.load_state_dict(torch.load(a.ckpt)); m.eval()
te = RobotData("test")
rng = np.random.default_rng(4242)
C, H = 512, 700
W = [te.window(rng, C + H) for _ in range(a.n)]
W = [(w[0][:C + H], w[1][:C + H], w[2][:C + H], w[3]) for w in W]
cands = [[None] + [te.window(rng, C + H, ep=(w[3] + 1 + j) % len(te.eps)) for j in range(a.N - 1)] for w in W]


scores = np.zeros((a.n, a.N))
for j in range(a.N):
    foreign = None if j == 0 else [c[j] for c in cands]
    # errors per event: (group, elapsed, model_err, hold_err); split back per window by recomputing per window
    for r in range(a.n):
        E = imagine(m, [W[r][:3]], None if foreign is None else [foreign[r]])
        sel = (E[:, 0] == 0) & (E[:, 1] > 24)  # joints, last ~0.2 s of the imagined second
        scores[r, j] = float((E[sel, 2] ** 2).mean())  # distance of the imagined body to the goal body
pick = scores.argmin(1)
res = dict(N=a.N, n=a.n, picked_real_rate=float((pick == 0).mean()), chance=1.0 / a.N,
           goal_mse_of_pick=float(scores[np.arange(a.n), pick].mean()), goal_mse_random=float(scores.mean()),
           goal_mse_real_actions=float(scores[:, 0].mean()))
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
