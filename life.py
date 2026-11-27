"""A continuous life on a computer (real data only, nothing synthetic).

The machine is born untrained (only its innate structure: frozen prime basis, continuous time, memory) and then lives
ONE uninterrupted stream of real events. Its internal state is never reset. At every event it first predicts the
next one (before seeing it), then sees it; every U events it learns from what it just lived (weights change while it
lives) and, if replay is on, also re-lives one old moment from its episodic store ("sleep"/replay against forgetting).
It acts: it chooses what to attend to next (which real source to experience), by curiosity = its own learning progress.
Sources (train splits, each walked forward in real time order, never repeated):
  weather (Jena), grid (ETT), market (VIX), audio (spoken digits), text (books), body (real robot arm: joints, motor
  commands, camera pixels).
Measured over the life:
  - prequential loss: the loss on each event BEFORE learning from it (every event is new; this is the standard measure
    of learning from experience), next to a no-learning-structure reference running on the same events
  - held-out probes (val/test data never lived) every --probe_every events: learning that generalises, and forgetting."""
import argparse, json, math, os, time
from collections import deque
import numpy as np
import torch
from pfm.one import OneMachine
from pfm.realworld import STREAMS_REAL, SID, RealData, JENA_VARS, ETT_VARS, fixed_eval_set
from robot_world import STREAMS_ROBOT, RobotData, batch as robot_batch

from pfm.bigworld import STREAMS_BIG
STREAMS_LIFE = STREAMS_REAL + STREAMS_ROBOT
ROBOT0 = len(STREAMS_REAL)


def use_big_streams():
    """born from the trained Oll-1: its 28 streams first (same role phases), the robot body after them"""
    global STREAMS_LIFE, ROBOT0
    STREAMS_LIFE = STREAMS_BIG + STREAMS_ROBOT
    ROBOT0 = len(STREAMS_BIG)
TEXT = SID["text"]
SOURCES = ["weather", "grid", "market", "audio", "text", "body"]


def build_sources():
    d = RealData("train")
    src = {}

    def multivar(t, X, sid0):
        n, k = X.shape
        tt = np.repeat(t, k); ss = np.tile(sid0 + np.arange(k), n); xx = X.reshape(-1)
        ok = ~np.isnan(xx)
        return xx[ok], ss[ok], tt[ok]
    src["weather"] = multivar(d.jena_t, d.jena, SID["jena:" + JENA_VARS[0]])
    src["grid"] = multivar(d.ett_t, d.ett, SID["ett:" + ETT_VARS[0]])
    src["market"] = (d.vix, np.full(len(d.vix), SID["vix"]), d.vix_t)
    a = np.clip(d.audio[: d.aoff[-1]] * 4.0, -1.2, 1.2)
    at = np.concatenate([np.arange(l) / 8.0 + 3000.0 * i for i, l in enumerate(d.alen)])  # ms; 3 s silence between utterances
    src["audio"] = (a, np.full(len(a), SID["audio"]), at)
    src["text"] = (d.text.astype(np.float64), np.full(len(d.text), TEXT), np.arange(len(d.text), dtype=np.float64))
    r = RobotData("train")
    src["body"] = (r.x, r.s + ROBOT0, r.t)
    # speech with its word: each real spoken digit (2 kHz, mean of 4 samples), then its real label as a "digit" event
    xs, ss, ts, t_ = [], [], [], 0.0
    for k in range(len(d.alen)):
        u = d.audio[d.aoff[k]: d.aoff[k] + d.alen[k]]; u = u[: len(u) // 4 * 4].reshape(-1, 4).mean(1)
        xs += [np.clip(u * 4.0, -1.2, 1.2), [float(d.alab[k])]]
        ss += [np.full(len(u), SID["audio"]), [SID["digit"]]]
        ts += [t_ + np.arange(len(u)) * 0.5, [t_ + len(u) * 0.5 + 50.0]]
        t_ += len(u) * 0.5 + 3000.0
    src["speech"] = (np.concatenate(xs), np.concatenate(ss), np.concatenate(ts))
    return {k: (np.asarray(x, np.float64), np.asarray(s, np.int64), np.asarray(t, np.float64)) for k, (x, s, t) in src.items()}


def probe_sets():
    ev = fixed_eval_set("val", n_per_domain=16, T=128, seed=999)
    sets = {k: ev[k] for k in ("weather", "grid", "market", "audio", "text")}
    te = RobotData("test"); rng = np.random.default_rng(999)
    b = robot_batch([te.window(rng, 256) for _ in range(16)])
    b["s"] = b["s"] + ROBOT0; b["s_out"] = b["s_out"] + ROBOT0
    sets["body"] = b
    return sets


def probe(m, sets):
    out = {}
    with torch.no_grad():
        for k, b in sets.items():
            l = m.nll(m(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"])
            out[k] = float((l * b["mask"]).sum() / b["mask"].sum())
    return out


class Reference:
    """no-learning-structure reference on the same lived events: last value of the stream with a running Gaussian width
    (continuous), running byte frequencies (text). It also adapts online, so it is a fair 'simple learner' baseline."""

    def __init__(self):
        self.last, self.var, self.n, self.cnt = {}, {}, {}, np.ones(256)

    def nll(self, x, s):
        if s == TEXT:
            p = self.cnt[int(x)] / self.cnt.sum(); self.cnt[int(x)] += 1
            return -math.log(p)
        if s not in self.last:
            self.last[s] = x; return None
        r = x - self.last[s]; self.last[s] = x
        v = self.var.get(s, 0.05); n = self.n.get(s, 0)
        out = 0.5 * r * r / v + 0.5 * math.log(2 * math.pi * v)
        self.var[s] = max((v * n + r * r) / (n + 1), 1e-6); self.n[s] = min(n + 1, 10000)
        return out


class AnticipationReference:
    """simple online learner for the same when/which questions: next-sense frequencies given the current sense
    (add-one counts), and a running Gaussian of the log-gap per current sense"""

    def __init__(self):
        self.cnt, self.mu, self.var, self.n = {}, {}, {}, {}

    def nll(self, s_prev, s, u):
        c = self.cnt.setdefault(s_prev, np.ones(len(STREAMS_LIFE)))
        rc = -math.log(c[s] / c.sum()); c[s] += 1
        mu, v, n = self.mu.get(s_prev, 0.0), self.var.get(s_prev, 0.25), self.n.get(s_prev, 0)
        rw = 0.5 * (u - mu) ** 2 / v + 0.5 * math.log(2 * math.pi * v)
        self.mu[s_prev] = mu + (u - mu) / (n + 1); self.var[s_prev] = max((v * n + (u - mu) ** 2) / (n + 1), 1e-4)
        self.n[s_prev] = min(n + 1, 10000)
        return rw, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=1_000_000)
    ap.add_argument("--minutes", type=float, default=300)
    ap.add_argument("--episode", type=int, default=1024, help="events per attention choice")
    ap.add_argument("--U", type=int, default=128, help="learn every U lived events")
    ap.add_argument("--W", type=int, default=384, help="learning window (most recent lived events)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--attention", default="curious", choices=["curious", "random"])
    ap.add_argument("--no_replay", action="store_true")
    ap.add_argument("--no_learning", action="store_true", help="control: lives the same way but never changes its weights")
    ap.add_argument("--anticipate", action="store_true", help="also predict WHEN the next event comes and WHICH sense")
    ap.add_argument("--brain", action="store_true", help="all brain functions: anticipation, surprise-gated memory writing, "
                    "surprise-modulated learning rate, speech+word grounding, confidence (metacognition) tracking")
    ap.add_argument("--init", default="", help="be born from a trained machine (e.g. models/oll-1.pt) instead of untrained")
    ap.add_argument("--rehearse", type=int, default=0, help="per update, this many windows from the original training corpus (keeps old skills)")
    ap.add_argument("--adapt_only", action="store_true", help="keep the born weights frozen; live-learning goes only into the phase "
                    "adapters (theta, sigma, alpha, kq, kp, pl), new parts and the per-stream vectors of streams the machine was not trained on")
    ap.add_argument("--probe_every", type=int, default=100_000)
    ap.add_argument("--R", type=int, default=96); ap.add_argument("--L", type=int, default=3); ap.add_argument("--RM", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", 4)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.brain:
        a.anticipate = True
    if a.init:
        use_big_streams()  # before the sources are built: the body streams move behind the trained ones
    sources = SOURCES + (["speech"] if a.brain else [])
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rehearsal_data = RealData("train") if a.rehearse else None
    src = build_sources(); cursor = {k: 0 for k in sources}
    sets = probe_sets()
    if a.init:
        ck = torch.load(a.init, map_location="cpu")
        cfg = ck["config"]; a.R, a.L, a.RM = cfg["R"], cfg["L"], cfg["RM"]
    m = OneMachine(STREAMS_LIFE, R=a.R, L=a.L, M=(cfg["M"] if a.init else 8), RM=a.RM, seed=a.seed, anticipate=a.anticipate)
    if a.init:  # every trained tensor whose shape matches; per-stream vectors: the trained streams first
        own = m.state_dict(); n_ok = 0
        for k, v in ck["model"].items():
            v = v.float() if v.dtype == torch.float16 else v
            if k in own and own[k].shape == v.shape:
                own[k] = v; n_ok += 1
            elif k in own and own[k].dim() == 1 and v.dim() == 1 and own[k].shape[0] > v.shape[0]:
                own[k][: v.shape[0]] = v; n_ok += 1
        m.load_state_dict(own)
        print("born from", a.init, ":", n_ok, "trained tensors", flush=True)
    if a.adapt_only:
        from pfm.one import PHASE_PARAMS
        n_old = len(STREAMS_BIG) if a.init else 0
        new_ok = torch.ones(len(m.streams), dtype=torch.bool); new_ok[:n_old] = False; new_ok[SID["digit"]] = True  # digit never trained
        trainable = []
        for n, p_ in m.named_parameters():
            leaf = n.split(".")[-1]
            if leaf in PHASE_PARAMS or n.startswith(("AF.", "AG.", "again")):
                trainable.append(p_)
            elif leaf in ("mem_beta", "sym_logtemp", "mem_logwidth"):
                p_.register_hook(lambda g, mk=new_ok.to(g.dtype if False else torch.float32): g * mk.to(g.dtype))
                trainable.append(p_)
            else:
                p_.requires_grad_(False)
        print("adapt_only:", sum(t.numel() for t in trainable), "of", sum(t.numel() for t in m.parameters()), "parameters learn while living", flush=True)
        opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    ref, aref = Reference(), AnticipationReference()
    st = m.init_state(1)
    buf = deque(maxlen=a.W + 1)  # recent lived events (x, s, dt)
    store, n_windows = [], 0  # episodic store of lived windows (reservoir), for replay
    lp = {k: 0.0 for k in sources}; last_loss = {k: None for k in sources}
    smu, svar = {}, {}  # running surprise statistics per stream
    meta, grounding = [], []  # (confidence entropy, actual loss, source); (lived, correct word?)
    DIGIT = SID["digit"]
    res = dict(args=vars(a), episodes=[], probes=[dict(events=0, loss=probe(m, sets))])
    print(res["probes"][-1], flush=True)
    t0, lived, updates, since_learn, next_probe = time.time(), 0, 0, 0, a.probe_every
    pending, pending_ant, prev_s = None, None, None  # predictions (features) for the next event
    prev_t = None
    while lived < a.events and time.time() - t0 < a.minutes * 60:
        # --- act: choose what to experience next
        unseen = [k for k in sources if last_loss[k] is None]
        if unseen:
            k = unseen[0]
        elif a.attention == "random" or rng.random() < 0.2:
            k = sources[int(rng.integers(len(sources)))]
        else:
            w = np.array([lp[s] for s in sources]); w = np.exp((w - w.max()) / (np.std(w) + 1e-6))
            k = sources[int(rng.choice(len(sources), p=w / w.sum()))]
        X, S, T = src[k]
        i0 = cursor[k]
        if i0 + a.episode + 1 >= len(X):
            cursor[k] = i0 = 0  # an archive lived to its end starts over (logged)
        pl, rl, n_pl = 0.0, 0.0, 0
        aw, ac, rw, rc, n_a = 0.0, 0.0, 0.0, 0.0, 0
        for i in range(i0, i0 + a.episode):
            x, s = float(X[i]), int(S[i])
            dt = 1e3 if prev_t is None or i == i0 else float(np.clip(T[i] - T[i - 1], 0, 1e3))  # attention switch = a long gap
            g = 1.0
            if pending is not None:  # score the prediction made before this event was seen
                with torch.no_grad():
                    l = float(m.nll(pending, torch.tensor([s]), torch.tensor([x], dtype=torch.float64))[0])
                    if a.brain and math.isfinite(l):
                        # surprise relative to what is usual for this sense -> writing strength (novelty / habituation)
                        mu, v = smu.get(s, l), svar.get(s, 1.0)
                        g = float(2.0 / (1.0 + math.exp(-(l - mu) / math.sqrt(v + 1e-6))))
                        smu[s] = 0.99 * mu + 0.01 * l; svar[s] = 0.99 * v + 0.01 * (l - mu) ** 2
                        if s == DIGIT:  # does it name the word it just heard?
                            c = m.coeffs(pending, torch.tensor([s]))
                            grounding.append((lived, int(m.discrete_logits(c, STREAMS_LIFE[s])[0].argmax()) == int(x)))
                        if lived % 16 == 0:  # how sure was it, and how wrong was it
                            st_ = STREAMS_LIFE[s]
                            if st_.kind == "continuous":
                                grid = torch.linspace(-1.5, 1.5, 301, dtype=torch.float64)
                                pd = m.density(pending, s, grid)[0].double(); pd = pd / pd.sum()
                                H = float(-(pd * (pd + 1e-12).log()).sum() + math.log(3.0 / 300))
                            else:
                                lg = m.discrete_logits(m.coeffs(pending, torch.tensor([s])), st_)[0] * m.sym_logtemp[s].exp()
                                pr = torch.softmax(lg.double(), -1); H = float(-(pr * (pr + 1e-12).log()).sum())
                            meta.append((H, l, k))
                r = ref.nll(x, s)
                if r is not None and math.isfinite(l):
                    pl += l; rl += r; n_pl += 1
                if a.anticipate and i > i0:  # when and which, scored on the real gap and the real next sense
                    gap = torch.tensor([dt]); w_, c_ = m.anticipation_nll(pending_ant, torch.tensor([s]), gap)
                    u = float(m.when_value(gap)[0]); rw_, rc_ = aref.nll(prev_s, s, u)
                    aw += float(w_[0].detach()); ac += float(c_[0].detach()); rw += rw_; rc += rc_; n_a += 1
            q = float(np.clip(T[i + 1] - T[i], 0, 1e3))
            with torch.no_grad():
                pending, st = m.step(torch.tensor([x], dtype=torch.float64), torch.tensor([s]), torch.tensor([dt]), torch.tensor([q]), st,
                                     torch.tensor([g]) if a.brain else None)
                pending_ant = st.get("h_ant")
            prev_s = s
            buf.append((x, s, dt, g)); prev_t = T[i]; lived += 1; since_learn += 1
            # --- learn from what was just lived (+ replay one old moment)
            if since_learn >= a.U and len(buf) == buf.maxlen and not a.no_learning:
                since_learn = 0
                win = np.array(buf)
                n_windows += 1
                if len(store) < 2000:
                    store.append(win)
                elif rng.random() < 2000 / n_windows:
                    store[int(rng.integers(2000))] = win
                wins = [win] + ([store[int(rng.integers(len(store)))]] if not a.no_replay and len(store) > 1 else [])
                for _ in range(a.rehearse):  # rehearsal: a moment of the original training data (real, train split)
                    d_ = ["weather", "grid", "market", "audio", "text"][int(rng.integers(5))]
                    xr, sr, dr, _m, _t = rehearsal_data.window(rng, d_, a.W)
                    wins.append(np.stack([xr, sr, dr, np.ones(len(xr))], 1))
                xb = torch.tensor(np.stack([w[:, 0] for w in wins]), dtype=torch.float64)
                sb = torch.tensor(np.stack([w[:, 1] for w in wins]), dtype=torch.long)
                db = torch.tensor(np.stack([w[:, 2] for w in wins]), dtype=torch.float32)
                gb = torch.tensor(np.stack([w[:, 3] for w in wins]), dtype=torch.float32)[:, :-1] if a.brain else None
                if a.brain:  # neuromodulation: learn faster from surprising moments, slower from the usual
                    for pg in opt.param_groups:
                        pg["lr"] = a.lr * float(np.clip(win[-a.U:, 3].mean(), 0.5, 2.0))
                if a.anticipate:
                    h_, ha_ = m.forward_all(xb[:, :-1], sb[:, :-1], db[:, :-1], db[:, 1:], gb)
                    w_, c_ = m.anticipation_nll(ha_, sb[:, 1:], db[:, 1:])
                    loss = (m.nll(h_, sb[:, 1:], xb[:, 1:]) + w_ + c_)[:, 32:].mean()
                else:
                    loss = m.nll(m(xb[:, :-1], sb[:, :-1], db[:, :-1], db[:, 1:]), sb[:, 1:], xb[:, 1:])[:, 32:].mean()
                if torch.isfinite(loss):
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad], 1.0); opt.step(); updates += 1
            if lived >= next_probe:
                next_probe += a.probe_every
                res["probes"].append(dict(events=lived, minutes=(time.time() - t0) / 60, loss=probe(m, sets)))
                if grounding:
                    res["probes"][-1]["words_heard"] = len(grounding)
                    res["probes"][-1]["word_accuracy_recent"] = float(np.mean([c for _, c in grounding[-100:]]))
                print(res["probes"][-1], flush=True)
                torch.save(m.state_dict(), a.out + ".pt")
                json.dump(res, open(a.out + ".json", "w"), indent=1)
        cursor[k] = i0 + a.episode
        if n_pl:
            ep_loss = pl / n_pl
            if last_loss[k] is not None:
                lp[k] = 0.7 * lp[k] + 0.3 * (last_loss[k] - ep_loss)  # learning progress on this source
            last_loss[k] = ep_loss
            res["episodes"].append(dict(events=lived, source=k, loss=ep_loss, reference=rl / n_pl, updates=updates))
            if n_a:
                res["episodes"][-1].update(when=aw / n_a, when_reference=rw / n_a, which=ac / n_a, which_reference=rc / n_a)
            if len(res["episodes"]) % 50 == 0:
                print(res["episodes"][-1], {s: round(v, 3) for s, v in lp.items()}, f"{(time.time() - t0) / 60:.1f} min", flush=True)
    res["probes"].append(dict(events=lived, minutes=(time.time() - t0) / 60, loss=probe(m, sets)))
    res["lived_events"], res["updates"], res["minutes"] = lived, updates, (time.time() - t0) / 60
    res["attention_share"] = {k: float(np.mean([e["source"] == k for e in res["episodes"]])) for k in sources}
    if a.brain:
        def spearman(u, v):
            ru, rv = np.argsort(np.argsort(u)), np.argsort(np.argsort(v))
            return float(np.corrcoef(ru, rv)[0, 1]) if len(u) > 2 else None
        res["metacognition"] = {k: dict(n=sum(1 for e in meta if e[2] == k),
                                        spearman_confidence_entropy_vs_loss=spearman([e[0] for e in meta if e[2] == k], [e[1] for e in meta if e[2] == k]))
                                for k in sources}
        gr = np.array(grounding, dtype=float).reshape(-1, 2)
        half = len(gr) // 2
        res["word_grounding"] = dict(n_words_heard=len(gr), chance=0.1, accuracy_all=float(gr[:, 1].mean()) if len(gr) else None,
                                     accuracy_first_half=float(gr[:half, 1].mean()) if half else None,
                                     accuracy_second_half=float(gr[half:, 1].mean()) if half else None)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(m.state_dict(), a.out + ".pt")
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    print("PROBES", json.dumps(res["probes"][0]), "\n", json.dumps(res["probes"][-1]), "\nATTENTION", res["attention_share"])


if __name__ == "__main__":
    main()
