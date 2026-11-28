"""Standard zero-shot evaluation suite for byte-level Oll models (same task formats as lm-evaluation-harness):
  hellaswag, arc_easy, arc_challenge, piqa, winogrande, sciq : multiple choice by continuation log-likelihood
                                                               (acc; acc_norm = per-byte normalised)
  lambada_openai                                              : last word, greedy over its bytes (acc) and bits/byte
Datasets are the public validation/test splits from the Hugging Face hub (parquet), never trained on.
Random-guess accuracy: hellaswag/arc/sciq 25 %, piqa/winogrande 50 %, lambada 0 %."""
import argparse, ast, io, json, math, os, re, sys, time, urllib.request
import numpy as np
import torch

UA = {"User-Agent": "oll-research/0.1"}
SRC = {"hellaswag": ("Rowan/hellaswag", "default", "validation"), "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
       "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"), "piqa": ("baber/piqa", "default", "validation"),
       "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"), "sciq": ("allenai/sciq", "default", "test"),
       "lambada_openai": ("EleutherAI/lambada_openai", "en", "test")}
RANDOM = dict(hellaswag=0.25, arc_easy=0.25, arc_challenge=0.25, piqa=0.5, winogrande=0.5, sciq=0.25, lambada_openai=0.0)


def rows(task, cache):
    import pyarrow.parquet as pq
    d, c, sp = SRC[task]
    path = os.path.join(cache, f"{task}.parquet")
    if not os.path.exists(path):
        os.makedirs(cache, exist_ok=True)
        b = urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/datasets/{d}/parquet/{c}/{sp}/0.parquet", headers=UA), timeout=120).read()
        open(path, "wb").write(b)
    return pq.read_table(path).to_pylist()


def _hs(t):
    t = t.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", t).replace("  ", " ")


def items(task, cache):
    """-> list of (list of (context, continuation) per choice, gold index)"""
    out = []
    for r in rows(task, cache):
        if task == "hellaswag":
            ctx = _hs(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize())
            ends = r["endings"] if isinstance(r["endings"], list) else ast.literal_eval(r["endings"])
            out.append(([(ctx, " " + _hs(e)) for e in ends], int(r["label"])))
        elif task.startswith("arc"):
            ch = r["choices"] if isinstance(r["choices"], dict) else ast.literal_eval(r["choices"])
            labels = list(ch["label"])
            if r["answerKey"] not in labels:
                continue
            ctx = f"Question: {r['question']}\nAnswer:"
            out.append(([(ctx, " " + t) for t in ch["text"]], labels.index(r["answerKey"])))
        elif task == "piqa":
            ctx = f"Question: {r['goal']}\nAnswer:"
            out.append(([(ctx, " " + r["sol1"]), (ctx, " " + r["sol2"])], int(r["label"])))
        elif task == "winogrande":
            i = r["sentence"].index("_")
            cont = " " + r["sentence"][i + 1:].strip()
            out.append(([(r["sentence"][:i] + r["option1"], cont), (r["sentence"][:i] + r["option2"], cont)], int(r["answer"]) - 1))
        elif task == "sciq":
            ctx = f"{r['support'].strip()}\nQuestion: {r['question']}\nAnswer:".lstrip()
            ch = [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]]
            out.append(([(ctx, " " + c) for c in ch], 3))
        elif task == "lambada_openai":
            t = r["text"]; i = t.rindex(" ")
            out.append(([(t[:i], t[i:])], 0))
    return out


class Scorer:
    """continuation log-likelihood (nats) and greedy-match of a byte-level Oll model"""

    def __init__(self, m, device="cpu", max_len=1024, batch=16):
        from pfm.realworld import SID
        self.m, self.dev, self.max_len, self.batch, self.TEXT = m.eval(), device, max_len, batch, SID["text"]

    @torch.no_grad()
    def score(self, pairs):
        res = []
        for b0 in range(0, len(pairs), self.batch):
            P = pairs[b0:b0 + self.batch]
            seqs, nc = [], []
            for ctx, cont in P:
                c, k = ctx.encode("utf-8"), cont.encode("utf-8")
                full = (b"\n" + c + k)[-(self.max_len + 1):]
                seqs.append(list(full)); nc.append(min(len(k), len(full) - 1))
            Lm = max(len(s) for s in seqs)
            X = torch.tensor([s + [10] * (Lm - len(s)) for s in seqs], dtype=torch.float64, device=self.dev)
            s_ = torch.full(X.shape, self.TEXT, device=self.dev); d_ = torch.ones(X.shape, device=self.dev)
            h = self.m(X[:, :-1], s_[:, :-1], d_[:, :-1], d_[:, 1:])
            lp = self.m.symbol_logp(h, self.TEXT) if hasattr(self.m, "symbol_logp") else None
            for i, (s, n) in enumerate(zip(seqs, nc)):
                L = len(s); pos = torch.arange(L - 1 - n, L - 1, device=self.dev)
                tgt = X[i, L - n:L].long()
                li = lp[i, pos]
                res.append((float(li.gather(1, tgt[:, None]).sum()), bool((li.argmax(-1) == tgt).all()), n))
        return res


def evaluate(m, tasks, cache, device="cpu", limit=0, max_len=1024, batch=16):
    sc = Scorer(m, device, max_len, batch); out = {}
    for task in tasks:
        t0 = time.time(); its = items(task, cache)
        if limit:
            its = [its[i] for i in np.random.default_rng(0).permutation(len(its))[:limit]]
        pairs = [p for ch, _ in its for p in ch]
        scored = sc.score(pairs); k = 0; acc = accn = 0; bits = nb = 0
        for ch, gold in its:
            r = scored[k:k + len(ch)]; k += len(ch)
            if task == "lambada_openai":
                acc += r[0][1]; bits -= r[0][0] / math.log(2); nb += r[0][2]
            else:
                ll = [x[0] for x in r]; lln = [x[0] / max(x[2], 1) for x in r]
                acc += int(np.argmax(ll) == gold); accn += int(np.argmax(lln) == gold)
        n = len(its)
        out[task] = dict(n=n, acc=acc / n, random=RANDOM[task], seconds=round(time.time() - t0, 1))
        if task == "lambada_openai":
            out[task]["bits_per_byte_last_word"] = bits / max(nb, 1)
        else:
            out[task]["acc_norm"] = accn / n
        print(task, json.dumps(out[task]), flush=True)
    return out


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/oll-1.pt")
    ap.add_argument("--tasks", default=",".join(SRC))
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/oll_evals"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from oll.core import load
    m = load(a.model, a.device)
    res = evaluate(m, a.tasks.split(","), a.cache, a.device, a.limit)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(dict(model=a.model, limit=a.limit, res=res), open(a.out, "w"), indent=1)
