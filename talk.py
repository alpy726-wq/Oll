"""Can the machine learn to talk on a CPU?  OneMachine (all capabilities kept, text stream) vs a same-size transformer,
same data, same wall-clock budget. Ends with samples written by each model, one byte at a time."""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
from pfm.one import OneMachine
from pfm.oneworld import STREAMS

TEXT_STREAM = 5
DATA = "data"


class GPT(nn.Module):
    def __init__(self, d=128, L=4, H=4, T=256):
        super().__init__()
        self.emb = nn.Embedding(256, d); self.pos = nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, H, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L)
        self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, 256)

    def forward(self, x):
        T = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(T))
        return self.out(self.ln(self.enc(h, mask=nn.Transformer.generate_square_subsequent_mask(T), is_causal=True)))


def loss_and_logits(model, arch, x, y):
    if arch == "gpt":
        lg = model(x.long())
        return nn.functional.cross_entropy(lg.reshape(-1, 256), y.reshape(-1).long()), None
    s = torch.full(x.shape, TEXT_STREAM, dtype=torch.long)
    d = torch.ones(x.shape)
    h = model(x, s, d, d)
    return model.nll(h, s, y).mean(), None


@torch.no_grad()
def sample(model, arch, prompt, n=400, temp=0.8, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = list(prompt.encode())
    if arch == "gpt":
        for _ in range(n):
            x = torch.tensor(out[-256:])[None]
            p = torch.softmax(model(x)[0, -1] / temp, -1)
            out.append(int(torch.multinomial(p, 1, generator=g)))
    else:  # streaming: constant memory, one byte at a time
        st = model.init_state(1)
        one = torch.ones(1); s = torch.tensor([TEXT_STREAM])
        for b in out[:-1]:
            _, st = model.step(torch.tensor([float(b)], dtype=torch.float64), s, one, one, st)
        cur = out[-1]
        for _ in range(n):
            h, st = model.step(torch.tensor([float(cur)], dtype=torch.float64), s, one, one, st)
            c = model.coeffs(h, s)
            lg = model.discrete_logits(c, STREAMS[TEXT_STREAM])[0] * model.sym_logtemp[TEXT_STREAM].exp()
            cur = int(torch.multinomial(torch.softmax(lg / temp, -1), 1, generator=g))
            out.append(cur)
    return bytes(out).decode("utf-8", errors="replace")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["machine", "gpt"], required=True)
    ap.add_argument("--minutes", type=float, default=150)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--R", type=int, default=256)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--readout", default="power", choices=["power", "phase"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--M", type=int, default=4, help="oscillator modes per channel")
    ap.add_argument("--mem0", type=float, default=1.0, help="initial memory gate")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    tr = np.frombuffer(open(f"{DATA}/talk.train", "rb").read(), dtype=np.uint8)
    va = np.frombuffer(open(f"{DATA}/talk.val", "rb").read(), dtype=np.uint8)
    OneMachine.DISCRETE_READOUT = a.readout
    model = GPT() if a.arch == "gpt" else OneMachine(STREAMS, R=a.R, L=a.L, M=a.M, RM=512)
    lr = 1e-3 if a.arch == "gpt" else a.lr
    if a.arch == "machine":
        with torch.no_grad():
            model.mem_beta.fill_(a.mem0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    vrng = np.random.default_rng(99)
    vs = vrng.integers(0, len(va) - a.T - 1, 32)
    VX = torch.tensor(np.stack([va[i:i + a.T + 1] for i in vs]).astype(np.float64))
    os.makedirs("runs/talk", exist_ok=True)
    log, t0, step = [], time.time(), 0
    budget = a.minutes * 60
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        for g_ in opt.param_groups:
            g_["lr"] = lr * min(1, (step + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        idx = rng.integers(0, len(tr) - a.T - 1, a.B)
        X = torch.tensor(np.stack([tr[i:i + a.T + 1] for i in idx]).astype(np.float64))
        loss, _ = loss_and_logits(model, a.arch, X[:, :-1], X[:, 1:])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1
        if step % a.eval_every == 0:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([loss_and_logits(model, a.arch, VX[i:i + 8, :-1], VX[i:i + 8, 1:])[0].item() for i in range(0, 32, 8)]))
            model.train()
            log.append(dict(step=step, minutes=(time.time() - t0) / 60, bytes_seen=step * a.B * a.T, val_bits_per_byte=vl / math.log(2)))
            print(log[-1], flush=True)
    model.eval()
    prompts = ["Alice was beginning to ", "Once upon a time there was a little ", "\"I do not know,\" said the "]
    samples = {p: sample(model, a.arch, p) for p in prompts}
    torch.save(model.state_dict(), f"runs/talk/{a.arch}{a.tag}.pt")
    json.dump(dict(args=vars(a), params=sum(p.numel() for p in model.parameters()), curve=log, samples=samples),
              open(f"runs/talk/{a.arch}{a.tag}.json", "w"), indent=1)
    for p, s in samples.items():
        print("=" * 60); print(s)
