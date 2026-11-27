"""Oll-1 alive on live internet streams for a few minutes (read-only: it only listens, predicts and learns).
  - Wikipedia recent changes (stream.wikimedia.org): titles and edit comments typed right now by people worldwide (bytes)
  - Bitcoin spot price (api.coinbase.com), polled every 2 s (a real number with its real time)
Two copies live the same moments: one keeps learning (phase adapters only, the trained weights stay frozen), one is
frozen. Every event is predicted before it is seen (prequential), so the difference is learning from live experience."""
import argparse, json, math, queue, threading, time, urllib.request
from collections import deque
import numpy as np
import torch
from oll.core import load
from pfm.one import PHASE_PARAMS
from pfm.realworld import SID

TEXT, PRICE = SID["text"], 18  # bytes; a generic series stream for the price


def wiki_reader(q, stop):
    req = urllib.request.Request("https://stream.wikimedia.org/v2/stream/recentchange", headers={"User-Agent": "oll-research/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        for line in r:
            if stop.is_set():
                return
            line = line.decode("utf-8", "replace")
            if line.startswith("data: "):
                try:
                    e = json.loads(line[6:])
                except Exception:
                    continue
                txt = f"{e.get('title', '')}: {e.get('comment', '')}"[:160] + "\n"
                if q.qsize() < 50:  # keep only fresh moments
                    q.put(("text", txt, time.time()))


def price_reader(q, stop):
    while not stop.is_set():
        try:
            d = json.loads(urllib.request.urlopen("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=5).read())
            q.put(("price", float(d["data"]["amount"]), time.time()))
        except Exception:
            pass
        stop.wait(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--U", type=int, default=192)
    ap.add_argument("--W", type=int, default=192)
    ap.add_argument("--out", default="runs/live/live.json")
    a = ap.parse_args()
    torch.set_num_threads(4); torch.manual_seed(0)
    living, frozen = load("models/oll-1.pt"), load("models/oll-1.pt")
    train = [p for n, p in living.named_parameters() if n.split(".")[-1] in PHASE_PARAMS]
    for p in living.parameters():
        p.requires_grad_(any(p is t for t in train))
    opt = torch.optim.AdamW(train, lr=a.lr, weight_decay=0.0)
    st = {"living": living.init_state(1), "frozen": frozen.init_state(1)}
    pend = {"living": None, "frozen": None}
    q, stop = queue.Queue(), threading.Event()
    threading.Thread(target=wiki_reader, args=(q, stop), daemon=True).start()
    threading.Thread(target=price_reader, args=(q, stop), daemon=True).start()
    buf = deque(maxlen=a.W + 1)
    prices = deque(maxlen=60)
    log, seen, t0, since, updates, last_price_t = [], [], time.time(), 0, 0, None
    minute = {"text_bits": {"living": [], "frozen": []}, "price_nll": {"living": [], "frozen": []}, "events": 0}
    one = torch.ones(1)

    def live_event(x, s, dt):
        nonlocal since, updates
        for k, m in (("living", living), ("frozen", frozen)):
            with torch.no_grad():
                if pend[k] is not None:
                    l = float(m.nll(pend[k], torch.tensor([s]), torch.tensor([x], dtype=torch.float64))[0])
                    if s == TEXT:
                        minute["text_bits"][k].append(l / math.log(2))
                    else:
                        minute["price_nll"][k].append(l)
                pend[k], st[k] = m.step(torch.tensor([x], dtype=torch.float64), torch.tensor([s]), torch.tensor([dt]), one, st[k])
        minute["events"] += 1
        buf.append((x, s, dt)); since += 1
        if since >= a.U and len(buf) == buf.maxlen:  # learn from the last moments (adapters only)
            since = 0
            w = np.array(buf)
            xb = torch.tensor(w[None, :, 0], dtype=torch.float64); sb = torch.tensor(w[None, :, 1], dtype=torch.long)
            db = torch.tensor(w[None, :, 2], dtype=torch.float32)
            loss = living.nll(living(xb[:, :-1], sb[:, :-1], db[:, :-1], db[:, 1:]), sb[:, 1:], xb[:, 1:])[:, 16:].mean()
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(train, 1.0); opt.step(); updates += 1

    while time.time() - t0 < a.minutes * 60:
        try:
            kind, v, tt = q.get(timeout=1.0)
        except queue.Empty:
            continue
        if kind == "text":
            if len(seen) < 12:
                seen.append(v.strip())
            for i, b in enumerate(v.encode("utf-8")):
                live_event(float(b), TEXT, 1.0 if i else 50.0)
        else:
            prices.append(v)
            if len(prices) >= 5:  # the price relative to its own last minutes (sensory adaptation, causal)
                mu, sd = np.mean(list(prices)[:-1]), np.std(list(prices)[:-1]) + 1e-6
                z = float(np.clip((v - mu) / sd / 3.0, -1.2, 1.2))
                live_event(z, PRICE, 1.0 if last_price_t is None else (tt - last_price_t) / 2.0)
            last_price_t = tt
        if time.time() - t0 >= (len(log) + 1) * 60:
            m_ = lambda k, key: float(np.mean(minute[key][k])) if minute[key][k] else None
            log.append(dict(minute=len(log) + 1, events=minute["events"], updates=updates,
                            text_bits_living=m_("living", "text_bits"), text_bits_frozen=m_("frozen", "text_bits"),
                            price_nll_living=m_("living", "price_nll"), price_nll_frozen=m_("frozen", "price_nll")))
            print(json.dumps(log[-1]), flush=True)
            minute = {"text_bits": {"living": [], "frozen": []}, "price_nll": {"living": [], "frozen": []}, "events": 0}
    stop.set()
    # what it now expects after a typical edit opening
    prompt = "Category:"
    out = {}
    for k, m in (("living", living), ("frozen", frozen)):
        s_ = m.init_state(1); g = torch.Generator().manual_seed(0); txt = []
        with torch.no_grad():
            for b in prompt.encode():
                h, s_ = m.step(torch.tensor([float(b)], dtype=torch.float64), torch.tensor([TEXT]), one, one, s_)
            for _ in range(60):
                lg = m.discrete_logits(m.coeffs(h, torch.tensor([TEXT])), m.streams[TEXT])[0] * m.sym_logtemp[TEXT].exp()
                c = int(torch.multinomial(torch.softmax(lg.double() / 0.7, -1), 1, generator=g))
                if c == 10:
                    break
                txt.append(c); h, s_ = m.step(torch.tensor([float(c)], dtype=torch.float64), torch.tensor([TEXT]), one, one, s_)
        out[k] = prompt + bytes(txt).decode("utf-8", "replace")
    res = dict(minutes=a.minutes, per_minute=log, updates=updates, examples_seen=seen, continuation_after_Category=out)
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
