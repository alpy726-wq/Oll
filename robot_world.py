"""World model from a real body's experience (data/robot, LeRobot koch_pick_place_5_lego; no synthetic data).

One OneMachine reads the merged event stream of a real arm: 6 joint positions, 6 motor commands (actions) and 64
camera pixels (each pixel position its own stream), every event at its real time (unit: 1/30 s).
Training: next-event density on train episodes 0-39. Evaluation on test episodes 40-49 only:
  1. predictive NLL per sense (joints / pixels / actions) next to a no-learning persistence reference
  2. counterfactual actions: the same windows with the motor commands replaced by those of another episode;
     if the machine has learned what its actions do, the real body and the real image become LESS likely
  3. imagination: from 512 events of context, the machine runs 1 s ahead on its own (it generates joints and pixels
     from its own predictions, it is only told the motor commands), with the real commands vs foreign commands;
     error of imagined joints/pixels against what really happened, next to "nothing moves" (hold last value)
--no_actions trains the same machine with the motor-command events removed (does knowing the action help?)."""
import argparse, json, math, os, time
import numpy as np
import torch
from pfm.one import OneMachine, Stream

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "robot")
STREAMS_ROBOT = ([Stream(f"joint{i}", "continuous", scale=2.0, window=(-1.5, 1.5)) for i in range(6)]
                 + [Stream(f"action{i}", "continuous", scale=2.0, window=(-1.5, 1.5)) for i in range(6)]
                 + [Stream(f"pix{k}", "continuous", scale=2.0, window=(-1.5, 1.5)) for k in range(64)])
GROUP = np.array([0] * 6 + [1] * 6 + [2] * 64)  # 0 joint, 1 action, 2 pixel
GNAME = ["joints", "actions", "pixels"]
FPS = 30.0


class RobotData:
    def __init__(self, split, no_actions=False):
        z = np.load(os.path.join(D, f"{split}.npz"))
        self.t, self.s, self.x, self.off = z["t"] * FPS, z["s"], z["x"].astype(np.float64), z["off"]
        self.eps = []
        for i in range(len(self.off) - 1):
            a, b = self.off[i], self.off[i + 1]
            keep = np.arange(a, b) if not no_actions else np.arange(a, b)[GROUP[self.s[a:b]] != 1]
            self.eps.append(keep)

    def window(self, rng, T, ep=None, i0=None):
        e = int(rng.integers(len(self.eps))) if ep is None else ep
        idx = self.eps[e]
        i0 = int(rng.integers(0, len(idx) - T - 1)) if i0 is None else i0
        j = idx[i0:i0 + T + 1]
        t = self.t[j]
        dt = np.diff(t, prepend=t[0] - 1.0)
        return self.x[j], self.s[j], np.clip(dt, 0, 1e3), e, i0


def batch(W, device="cpu"):
    x = torch.tensor(np.stack([w[0] for w in W]), dtype=torch.float64, device=device)
    s = torch.tensor(np.stack([w[1] for w in W]), dtype=torch.long, device=device)
    dt = torch.tensor(np.stack([w[2] for w in W]), dtype=torch.float32, device=device)
    mask = torch.ones(x.shape[0], x.shape[1] - 1, dtype=torch.bool, device=device); mask[:, :32] = False
    return dict(x=x[:, :-1], s=s[:, :-1], dt=dt[:, :-1], q=dt[:, 1:], y=x[:, 1:], s_out=s[:, 1:], mask=mask)


def group_nll(m, b):
    with torch.no_grad():
        l = m.nll(m(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"])
    g = torch.tensor(GROUP, device=l.device)[b["s_out"]]
    return {GNAME[k]: float(l[(g == k) & b["mask"]].mean()) for k in range(3) if ((g == k) & b["mask"]).any()}


def persistence(train, test_windows):
    """no-learning reference: next value of a stream = its last value, Gaussian with the train residual width"""
    rng = np.random.default_rng(1)
    res = {k: [] for k in range(3)}
    for _ in range(64):
        x, s, *_ = train.window(rng, 2048)
        last = {}
        for v, st in zip(x, s):
            if st in last:
                res[GROUP[st]].append(v - last[st])
            last[st] = v
    sig = {k: float(np.std(v)) for k, v in res.items()}
    out = {k: [] for k in range(3)}
    for x, s, *_ in test_windows:
        last = {}
        for i, (v, st) in enumerate(zip(x, s)):
            if st in last and i > 32:
                z = (v - last[st]) / sig[GROUP[st]]
                out[GROUP[st]].append(0.5 * z * z + math.log(sig[GROUP[st]]) + 0.5 * math.log(2 * math.pi))
            last[st] = v
    return {GNAME[k]: float(np.mean(v)) for k, v in out.items()}


def median_next(m, h, s_next, grid):
    med = torch.zeros(h.shape[0], dtype=torch.float64, device=h.device)
    for i in torch.unique(s_next).tolist():
        r = s_next == i
        dens = m.density(h[r], i, grid).double()
        cdf = (dens / dens.sum(-1, keepdim=True)).cumsum(-1)
        k = torch.searchsorted(cdf.contiguous(), torch.full((int(r.sum()), 1), 0.5, dtype=cdf.dtype, device=h.device))
        med[r] = grid[k.clamp(max=len(grid) - 1).squeeze(1)]
    return med


def imagine(m, W, foreign=None, C=512, device="cpu"):
    """W: windows of length >= C+H. Context C real events, then free-running: joints/pixels from the machine's own
    median prediction, motor commands given (real ones, or from `foreign` windows). Returns per-event errors."""
    x = torch.tensor(np.stack([w[0] for w in W]), dtype=torch.float64, device=device)
    s = torch.tensor(np.stack([w[1] for w in W]), dtype=torch.long, device=device)
    dt = torch.tensor(np.stack([w[2] for w in W]), dtype=torch.float32, device=device)
    xa = x.clone()
    if foreign is not None:  # foreign motor commands, aligned by action slot (the k-th command event of the window)
        xf = np.stack([w[0] for w in foreign]); sf = np.stack([w[1] for w in foreign])
        for r in range(x.shape[0]):
            ia = np.where(GROUP[s[r].cpu().numpy()] == 1)[0]; fa = np.where(GROUP[sf[r]] == 1)[0]
            n = min(len(ia), len(fa))
            fa_by = {}
            for j in fa:  # same joint channel, in order
                fa_by.setdefault(int(sf[r][j]), []).append(j)
            used = {}
            for j in ia:
                ch = int(s[r, j]); k = used.get(ch, 0); used[ch] = k + 1
                if k < len(fa_by.get(ch, [])):
                    xa[r, j] = float(xf[r][fa_by[ch][k]])
    grid = torch.linspace(-1.5, 1.5, 601, dtype=torch.float64, device=device)
    st = m.init_state(x.shape[0])
    with torch.no_grad():
        for k in range(C):
            h, st = m.step(xa[:, k], s[:, k], dt[:, k], dt[:, k + 1], st)
        cur = median_next(m, h, s[:, C], grid)
        err, hold, tt = [], [], []
        last = {}
        for r in range(x.shape[0]):
            last[r] = {}
            for k in range(C):
                last[r][int(s[r, k])] = float(x[r, k])
        t_elapsed = torch.zeros(x.shape[0], device=device)
        for k in range(C, x.shape[1] - 1):
            g = torch.tensor(GROUP, device=device)[s[:, k]]
            inp = torch.where(g == 1, xa[:, k], cur)  # actions are given, senses are imagined
            t_elapsed = t_elapsed + dt[:, k]
            for r in range(x.shape[0]):
                if int(g[r]) != 1:
                    ch = int(s[r, k])
                    err.append((int(g[r]), float(t_elapsed[r]), float(cur[r] - x[r, k]) * 3.0, float(last[r].get(ch, 0.0) - x[r, k]) * 3.0, r))
            h, st = m.step(inp, s[:, k], dt[:, k], dt[:, k + 1], st)
            cur = median_next(m, h, s[:, k + 1], grid)
    return np.array(err)  # (n, 5): group, elapsed frames, model error, hold-last error (std units), row


def summarize_imagination(E_real, E_foreign):
    out = {}
    for g, name in ((0, "joints"), (2, "pixels")):
        for lo, hi, lab in ((0, 10, "0-0.33s"), (10, 30, "0.33-1s")):
            a = E_real[(E_real[:, 0] == g) & (E_real[:, 1] > lo) & (E_real[:, 1] <= hi)]
            f = E_foreign[(E_foreign[:, 0] == g) & (E_foreign[:, 1] > lo) & (E_foreign[:, 1] <= hi)]
            if len(a):
                out[f"{name}_{lab}"] = dict(mse_imagined_real_actions=float((a[:, 2] ** 2).mean()),
                                            mse_imagined_foreign_actions=float((f[:, 2] ** 2).mean()) if len(f) else None,
                                            mse_hold_last=float((a[:, 3] ** 2).mean()), n=int(len(a)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--minutes", type=float, default=300)
    ap.add_argument("--R", type=int, default=96)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--RM", type=int, default=512)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_actions", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", 4)))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tr = RobotData("train", a.no_actions)
    m = OneMachine(STREAMS_ROBOT, R=a.R, L=a.L, M=8, RM=a.RM, seed=a.seed).to(a.device)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    t0, log, k = time.time(), [], 0
    while k < a.steps and time.time() - t0 < a.minutes * 60:
        for g in opt.param_groups:
            prog = max(k / a.steps, (time.time() - t0) / (a.minutes * 60))  # whichever budget ends first
            g["lr"] = a.lr * min(1, (k + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))
        b = batch([tr.window(rng, a.T) for _ in range(a.B)], a.device)
        l = m.nll(m(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"])
        loss = (l * b["mask"]).sum() / b["mask"].sum()
        if torch.isfinite(loss):
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if k % 100 == 0:
            log.append(dict(step=k, loss=float(loss), sec=round(time.time() - t0))); print(log[-1], flush=True)
        k += 1
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(m.state_dict(), a.out + ".pt")  # saved before evaluation, so a failed evaluation never loses training
    m.eval()
    te, te_full = RobotData("test", a.no_actions), RobotData("test", False)
    erng = np.random.default_rng(777)
    Wt = [te.window(erng, 1024) for _ in range(40)]
    res = dict(args=vars(a), params=sum(p.numel() for p in m.parameters()), steps_done=k, train_seconds=time.time() - t0, curve=log)
    nll = {}
    for j in range(0, 40, 8):
        for kk, v in group_nll(m, batch(Wt[j:j + 8], a.device)).items():
            nll.setdefault(kk, []).append(v)
    res["test_nll"] = {kk: float(np.mean(v)) for kk, v in nll.items()}
    res["test_nll_persistence_reference"] = persistence(tr, Wt)
    if not a.no_actions:
        # 2. counterfactual: motor commands from another test episode, same channel order
        Wf = [te.window(erng, 1024, ep=(w[3] + 3) % len(te.eps)) for w in Wt]
        cf = {}
        for j in range(0, 40, 8):
            W2 = []
            for w, f in zip(Wt[j:j + 8], Wf[j:j + 8]):
                x = w[0].copy(); s = w[1]
                for ch in range(6, 12):
                    i_w = np.where(s == ch)[0]; i_f = np.where(f[1] == ch)[0]; n = min(len(i_w), len(i_f))
                    x[i_w[:n]] = f[0][i_f[:n]]
                W2.append((x, s, w[2]))
            for kk, v in group_nll(m, batch(W2, a.device)).items():
                cf.setdefault(kk, []).append(v)
        res["test_nll_foreign_actions"] = {kk: float(np.mean(v)) for kk, v in cf.items()}
        # 3. imagination, 1 s ahead (~680 events), real vs foreign commands
        Wi = Wt[:16]; Fi = Wf[:16]
        Wi = [(w[0][:512 + 700], w[1][:512 + 700], w[2][:512 + 700]) for w in Wi]
        Fi = [(w[0][:512 + 700], w[1][:512 + 700], w[2][:512 + 700]) for w in Fi]
        ti = time.time()
        E_r = imagine(m, Wi, None, device=a.device); E_f = imagine(m, Wi, Fi, device=a.device)
        res["imagination_1s"] = summarize_imagination(E_r, E_f); res["imagination_seconds"] = time.time() - ti
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    print(json.dumps({kk: v for kk, v in res.items() if kk not in ("curve", "args")}, indent=1))


if __name__ == "__main__":
    main()
