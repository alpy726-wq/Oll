"""The machine as a streaming anomaly detector on NAB.

Strictly causal, per file, no labels and no other files:
  * the first 15% (NAB probationary period) fixes a scale and warms the model up
  * then the stream is processed in chunks: each chunk is scored by the current model (predicted density of each
    value, given only the past), and only afterwards the model takes a few gradient steps on the history seen so far
  * raw score  = surprise  -log q(x_t | past)  from the machine's continuous output density
  * anomaly score = how unusual the recent surprise is relative to its own history (an anomaly-likelihood step)
"""
import argparse, csv, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pfm.continuous import CTSeqModel
from score import load_corpus, score_detector

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def prepare(f):
    v = np.array(f["values"], dtype=np.float64)
    t = np.array([x.timestamp() for x in f["times"]])
    gaps = np.diff(t, prepend=t[0] - np.median(np.diff(t)))
    dt = np.clip(gaps / np.median(np.diff(t)), 0.1, 10.0)
    n = len(v)
    prob = int(min(math.floor(0.15 * n), 750))
    lo, hi = np.quantile(v[:prob], [0.0, 1.0])
    mid = (lo + hi) / 2
    half = max((hi - lo) / 2, 1e-6 * max(1.0, abs(mid)))
    z = np.clip((v - mid) / half * 0.6, -1.2, 1.2)  # probation range -> [-0.6, 0.6]
    # target: the next change, divided by a causal robust scale of past changes (uses only rows < t)
    d = np.diff(v, prepend=v[0])
    sig = np.empty(n)
    for i in range(n):
        h = d[max(1, i - 1000):i] if i > 1 else d[1:2]
        h = h if len(h) else np.array([0.0])
        mad = np.median(np.abs(h - np.median(h))) * 1.4826
        sig[i] = max(mad, h.std() * 0.1, 1e-9 * max(1.0, abs(mid)))
    y = np.clip(d / sig, -5.8, 5.8) + 4.0  # change in units of typical change, on the density window [-2, 10]
    return torch.tensor(z * 4 + 4), torch.tensor(dt, dtype=torch.float32), prob, torch.tensor(y)


def run_file(args):
    rel, f, cfg = args
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    x, dt, prob, yt = prepare(f)
    if cfg["target"] == "value":
        yt = x
    n = len(x)
    m = CTSeqModel(R=cfg["R"], L=cfg["L"], M=4, mode="ct", head="phase", seed=cfg["seed"])
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"])
    T, C, W = cfg["T"], cfg["chunk"], cfg["context"]

    def train(upto, steps):
        if upto < T + 2:
            return
        for _ in range(steps):
            starts = rng.integers(0, upto - T - 1, cfg["B"])
            idx = torch.tensor(starts)[:, None] + torch.arange(T + 1)[None]
            xb, db, yb = x[idx], dt[idx], yt[idx]
            loss = m.nll(xb[:, :-1], db[:, :-1], db[:, 1:], yb[:, 1:])[:, 4:].mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()

    surprise = np.zeros(n)
    t0 = time.time()
    train(prob, cfg["warm_steps"])
    pos = prob
    surprise[:prob] = np.nan
    while pos < n:
        end = min(pos + C, n)
        a = max(0, pos - W)
        with torch.no_grad():  # score rows pos..end-1 using inputs up to row-1 only
            nll = m.nll(x[None, a:end - 1], dt[None, a:end - 1], dt[None, a + 1:end], yt[None, a + 1:end])[0]
        surprise[pos:end] = nll[pos - a - 1:].numpy() if pos > a else np.concatenate([[np.nan], nll.numpy()])[: end - pos]
        pos = end
        train(pos, cfg["steps_per_chunk"])
    scores = likelihood(surprise, prob, cfg)
    return rel, scores, surprise, time.time() - t0


def likelihood(s, prob, cfg):
    """anomaly score from the surprise stream: z-score of a short moving average against the history of surprise."""
    n = len(s)
    out = np.full(n, 0.5)
    hist = []
    short = cfg["short"]
    buf = []
    for i in range(n):
        if np.isnan(s[i]):
            continue
        buf.append(s[i])
        if len(buf) > short:
            buf.pop(0)
        sm = float(np.mean(buf))
        if len(hist) > 50:
            h = np.array(hist[-cfg["long"]:])
            mu, sd = h.mean(), h.std() + 1e-6
            z = (sm - mu) / sd
            out[i] = 0.5 + 0.5 * z / (1 + abs(z))
        hist.append(sm)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="primeMachine")
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--warm_steps", type=int, default=150)
    ap.add_argument("--steps_per_chunk", type=int, default=3)
    ap.add_argument("--short", type=int, default=10)
    ap.add_argument("--long", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="incr", choices=["incr", "value"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    cfg = vars(ap.parse_args())
    corpus = load_corpus()
    items = [(rel, f, cfg) for rel, f in corpus.items() if not cfg["only"] or cfg["only"] in rel]
    items.sort(key=lambda it: -len(it[1]["values"]))
    os.makedirs(os.path.join(OUT, cfg["name"]), exist_ok=True)
    results = {}
    t0 = time.time()
    with Pool(cfg["workers"]) as pool:
        for rel, sc, sur, sec in pool.imap_unordered(run_file, items):
            results[rel] = sc
            np.savez(os.path.join(OUT, cfg["name"], rel.replace("/", "__") + ".npz"), score=sc, surprise=sur)
            print(f"{len(results)}/{len(items)} {rel} {len(sc)} rows {sec:.0f}s", flush=True)
    if cfg["only"]:
        sys.exit()
    r = score_detector(lambda rel: list(results[rel]), corpus)
    r["config"] = cfg
    r["wall_seconds"] = time.time() - t0
    json.dump(r, open(os.path.join(OUT, cfg["name"] + ".json"), "w"), indent=1)
    print(json.dumps({p: round(r[p]["normalized"], 2) for p in ("standard", "reward_low_FP_rate", "reward_low_FN_rate")}))
