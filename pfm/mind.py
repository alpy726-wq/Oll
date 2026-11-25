"""Mind tasks for OneMachine: (1) in-context rule induction ("logic"), (2) online trial-and-error control ("action").
All through the same machine: observations, answers and actions are events on typed streams; the continuous output
density is the answer for rules and the policy for actions."""
import math
import numpy as np
import torch
from .one import Stream

MIND_STREAMS = [
    Stream("x", "continuous", scale=2.0, window=(-1.5, 1.5)),       # 0: question / input value
    Stream("y", "continuous", scale=2.0, window=(-1.5, 1.5)),       # 1: answer value
    Stream("obs", "continuous", scale=2.0, window=(-1.5, 1.5)),     # 2: observation from the world (agent)
    Stream("act", "continuous", scale=2.0, window=(-1.5, 1.5)),     # 3: action (continuous)
    Stream("rew", "continuous", scale=2.0, window=(-1.5, 1.5)),     # 4: reward
]

# ------------------------------------------------------------------ 1. rule induction
RULES = {
    "add":    lambda x, c: x + c,
    "scale":  lambda x, c: x * c,
    "negate": lambda x, c: -x + c,
    "square": lambda x, c: x * x * np.sign(c) * 0.9,
    "abs":    lambda x, c: np.abs(x) + c - 0.5,
    "step":   lambda x, c: np.where(x > c, 0.6, -0.6),
}
TRAIN_RULES = ("add", "scale", "negate", "square", "abs", "step")


def rule_episode(rng, n_examples, rule=None, T=None):
    """x1 y1 x2 y2 ... xn yn  x* -> predict y*.  Each example is a fresh x; the query x* is new as well."""
    rule = rule or TRAIN_RULES[rng.integers(len(TRAIN_RULES))]
    c = {"add": rng.uniform(-0.5, 0.5), "scale": rng.choice([-1, 1]) * rng.uniform(0.3, 1.2), "negate": rng.uniform(-0.4, 0.4),
         "square": rng.choice([-1, 1]), "abs": rng.uniform(0.0, 0.8), "step": rng.uniform(-0.6, 0.6)}[rule]
    xs = rng.uniform(-1, 1, n_examples + 1)
    ys = np.clip(RULES[rule](xs, c), -1.2, 1.2)
    x = np.ravel(np.stack([xs, ys], 1)); s = np.tile([0, 1], n_examples + 1)
    mask = np.zeros(len(x) - 1, bool)
    mask[0::2] = True  # targets j are events j+1: answers y_i are events 1,3,5,... -> target indices 0,2,4,...; the last is the query
    return x, s, mask, rule


def rule_batch(rng, B, n_examples, rule=None):
    eps = [rule_episode(rng, n_examples, rule) for _ in range(B)]
    x = torch.tensor(np.stack([e[0] for e in eps])); s = torch.tensor(np.stack([e[1] for e in eps]))
    one = torch.ones(x.shape)
    return dict(x=x[:, :-1], s=s[:, :-1], dt=one[:, :-1], q=one[:, 1:], y=x[:, 1:], s_out=s[:, 1:],
                mask=torch.tensor(np.stack([e[2] for e in eps])), rules=[e[3] for e in eps])
