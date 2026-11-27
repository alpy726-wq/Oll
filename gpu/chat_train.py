"""Oll-Chat v0: Oll-1 (warm start) trained further on real text only: general text (FineWeb-Edu, Wikipedia) and real
human dialogue / reading comprehension (OpenAssistant, Dolly, SQuAD train). Then the pre-registered evaluation:
  1. SQuAD v1.1 validation (never trained on): exact match and F1 of greedy answers from the document
  2. knowing what it does not know: does its own confidence separate right from wrong answers (AUROC, accuracy by half)
  3. held-out bits per byte (general text, dialogue)
  4. sample answers to fixed prompts
Before training the same SQuAD evaluation is run on Oll-1 as it is (reference)."""
import argparse, json, math, os, re, string, sys, time
from collections import Counter
import numpy as np
import torch
sys.path.insert(0, "/oll"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pfm.one import OneMachine
from pfm.bigworld import STREAMS_BIG
from pfm.realworld import SID

TEXT = SID["text"]
ap = argparse.ArgumentParser()
ap.add_argument("--init", default="/vol/ckpt/main.pt")
ap.add_argument("--data", default="/vol/chat")
ap.add_argument("--R", type=int, default=1024); ap.add_argument("--L", type=int, default=8)
ap.add_argument("--M", type=int, default=8); ap.add_argument("--RM", type=int, default=512)
ap.add_argument("--B", type=int, default=48); ap.add_argument("--T", type=int, default=1024)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--minutes", type=float, default=95)
ap.add_argument("--est_steps", type=int, default=6000)
ap.add_argument("--dialog_frac", type=float, default=0.45)
ap.add_argument("--n_squad", type=int, default=500)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default="/vol/results/chat.json")
ap.add_argument("--ckpt", default="/vol/ckpt/chat")
a = ap.parse_args()
dev = a.device
torch.manual_seed(0); rng = np.random.default_rng(0)
if dev == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
os.makedirs(os.path.dirname(a.out), exist_ok=True); os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)

m = OneMachine(STREAMS_BIG, R=a.R, L=a.L, M=a.M, RM=a.RM).to(dev)
if a.init and os.path.exists(a.init):
    src = torch.load(a.init, map_location=dev); src = src.get("model", src)
    own = m.state_dict()
    keep = {k: v.float() if v.dtype == torch.float16 else v for k, v in src.items() if k in own and own[k].shape == v.shape}
    m.load_state_dict(keep, strict=False)
    print("warm start", len(keep), "tensors", flush=True)
gen = np.load(f"{a.data}/general.npy", mmap_mode="r"); dia = np.load(f"{a.data}/dialog.npy", mmap_mode="r")
gcut, dcut = int(0.99 * len(gen)), int(0.99 * len(dia))  # last 1% of each: held out
squad = json.load(open(f"{a.data}/squad_val.json"))[: a.n_squad]


def window_batch(arr, lo, hi, B, T, r):
    idx = r.integers(lo, hi - T - 2, B)
    x = torch.tensor(np.stack([arr[i:i + T + 1] for i in idx]).astype(np.float64), device=dev)
    s = torch.full(x.shape, TEXT, device=dev); d = torch.ones(x.shape, device=dev)
    return x, s, d


def lm_loss(x, s, d):
    return m.nll(m(x[:, :-1], s[:, :-1], d[:, :-1], d[:, 1:]), s[:, 1:], x[:, 1:])[:, 16:].mean()


# ------------------------------------------------------------------ evaluation helpers
def norm(t):
    t = t.lower(); t = "".join(ch for ch in t if ch not in set(string.punctuation))
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return " ".join(t.split())


def f1(p, g):
    pt, gt = norm(p).split(), norm(g).split()
    common = Counter(pt) & Counter(gt); ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(pt), ns / len(gt)
    return 2 * pr * rc / (pr + rc)


@torch.no_grad()
def answer(prompts, max_new=40, chunk=128):
    """greedy answers, all prompts in lockstep (left-padded with newlines); returns texts and mean log-prob"""
    m.eval(); outs, confs = [], []
    for c0 in range(0, len(prompts), chunk):
        P = [p.encode("utf-8")[-1800:] for p in prompts[c0:c0 + chunk]]
        Lm = max(len(p) for p in P); B = len(P)
        X = torch.tensor([[10] * (Lm - len(p)) + list(p) for p in P], dtype=torch.float64, device=dev)
        s = torch.full((B,), TEXT, device=dev); one = torch.ones(B, device=dev)
        st = m.init_state(B)
        for k in range(Lm):
            h, st = m.step(X[:, k], s, one, one, st)
        done = torch.zeros(B, dtype=torch.bool, device=dev); txt = [[] for _ in range(B)]; lp = torch.zeros(B, device=dev); n = torch.zeros(B, device=dev)
        for _ in range(max_new):
            lg = (m.discrete_logits(m.coeffs(h, s), m.streams[TEXT]) * m.sym_logtemp[TEXT].exp()).float()
            logp = torch.log_softmax(lg, -1); c = logp.argmax(-1)
            stop = (c == 10) | done
            lp = lp + torch.where(stop, torch.zeros_like(lp), logp.gather(1, c[:, None])[:, 0]); n = n + (~stop).float()
            for i in range(B):
                if not stop[i]:
                    txt[i].append(int(c[i]))
            done = stop
            if bool(done.all()):
                break
            h, st = m.step(c.double(), s, one, one, st)
        outs += [bytes(t).decode("utf-8", "replace").strip() for t in txt]
        confs += (lp / n.clamp(min=1)).tolist()
    m.train()
    return outs, confs


def auroc(score, label):
    score, label = np.asarray(score), np.asarray(label, bool)
    if label.all() or (~label).all():
        return None
    r = np.argsort(np.argsort(score)) + 1.0
    return float((r[label].sum() - label.sum() * (label.sum() + 1) / 2) / (label.sum() * (~label).sum()))


def squad_eval(tag):
    prompts = [f"Document: {q['context'].strip()}\nUser: {q['question'].strip()}\nOll: " for q in squad]
    t0 = time.time(); ans, conf = answer(prompts)
    em = [max(float(norm(p) == norm(g)) for g in q["answers"]) for p, q in zip(ans, squad)]
    fs = [max(f1(p, g) for g in q["answers"]) for p, q in zip(ans, squad)]
    good = np.array(fs) >= 0.5
    order = np.argsort(conf); half = len(conf) // 2
    r = dict(n=len(squad), exact_match=float(np.mean(em)), f1=float(np.mean(fs)), seconds=time.time() - t0,
             confidence_auroc_for_f1_ge_0_5=auroc(conf, good),
             f1_most_confident_half=float(np.mean(np.array(fs)[order[half:]])), f1_least_confident_half=float(np.mean(np.array(fs)[order[:half]])),
             examples=[dict(q=q["question"], gold=q["answers"][0], answer=p, conf=round(c, 3)) for p, q, c in list(zip(ans, squad, conf))[:12]])
    print(tag, json.dumps({k: v for k, v in r.items() if k != "examples"}), flush=True)
    return r


res = dict(args=vars(a))
res["squad_before"] = squad_eval("before")
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
t0, step, log, ema = time.time(), 0, [], {}
while (time.time() - t0) / 60 < a.minutes:
    frac = min(1.0, max(step / a.est_steps, (time.time() - t0) / 60 / a.minutes))
    for g in opt.param_groups:
        g["lr"] = a.lr * min(1, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
    kind = "dialog" if rng.random() < a.dialog_frac else "general"
    x, s, d = window_batch(dia, 0, dcut, a.B, a.T, rng) if kind == "dialog" else window_batch(gen, 0, gcut, a.B, a.T, rng)
    loss = lm_loss(x, s, d)
    if not torch.isfinite(loss):
        step += 1; continue
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    ema[kind] = float(loss) / math.log(2) if kind not in ema else 0.98 * ema[kind] + 0.02 * float(loss) / math.log(2)
    step += 1
    if step % 100 == 0:
        log.append(dict(step=step, minutes=round((time.time() - t0) / 60, 2), bits_per_byte={k: round(v, 3) for k, v in ema.items()}))
        print(log[-1], flush=True)
        json.dump(dict(last=log[-1], curve=log), open(a.out.replace(".json", "_progress.json"), "w"))
    if step % 1000 == 0:
        torch.save(dict(model=m.state_dict(), step=step), a.ckpt + ".pt")
torch.save(dict(model=m.state_dict(), step=step), a.ckpt + ".pt")
res.update(steps=step, train_minutes=(time.time() - t0) / 60, bytes_read=int(step * a.B * a.T), curve=log)
with torch.no_grad():
    vr = np.random.default_rng(123)
    res["heldout_bits_per_byte"] = {k: float(np.mean([float(lm_loss(*window_batch(arr, lo, hi, 16, a.T, vr))) / math.log(2) for _ in range(8)]))
                                    for k, arr, lo, hi in (("general", gen, gcut, len(gen)), ("dialog", dia, dcut, len(dia)))}
print("heldout", res["heldout_bits_per_byte"], flush=True)
res["squad_after"] = squad_eval("after")
prompts = ["User: What is the capital of France?\nOll: ", "User: Who wrote Romeo and Juliet?\nOll: ",
           "User: How many legs does a spider have?\nOll: ", "User: What is photosynthesis?\nOll: ",
           "User: Can you give me a tip for learning a new language?\nOll: ", "User: What is 12 plus 7?\nOll: ",
           "Document: Alptekin lives in Istanbul and studies prime numbers.\nUser: Where does Alptekin live?\nOll: "]
outs, confs = answer(prompts, max_new=160)
res["samples"] = [dict(prompt=p, answer=o, confidence=round(c, 3)) for p, o, c in zip(prompts, outs, confs)]
for smp in res["samples"]:
    print(json.dumps(smp, ensure_ascii=False), flush=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
cfg = dict(R=a.R, L=a.L, M=a.M, RM=a.RM, streams="STREAMS_BIG", trained_steps=step)
torch.save(dict(config=cfg, model={k: (v.half() if v.is_floating_point() else v) for k, v in m.state_dict().items()}),
           os.path.join(os.path.dirname(a.out), "oll-chat.pt"))
print("DONE", flush=True)
