"""Oll-1 wanders Wikipedia on its own (read-only, polite: one request per 2 s).
At each page it reads the opening text (predicting every byte before seeing it, and learning: phase adapters only),
then looks at the page's links and decides where to go by its own measure: it follows the link whose title
surprises it most (where it expects to know least). Nothing about the route is given by us except the start.
Logged: the path, how surprised it was by each page, the candidates it weighed, and what it chose."""
import argparse, json, math, os, random, time, urllib.parse, urllib.request
from collections import deque
import numpy as np
import torch
from oll.core import load
from pfm.one import PHASE_PARAMS
from pfm.realworld import SID

TEXT = SID["text"]
UA = {"User-Agent": "OllResearchBot/0.1 (read-only research prototype)"}
_last = [0.0]


def get(url):
    """polite GET: at most one request per 2 s; on 429 wait (Retry-After, at most 60 s) and retry"""
    import urllib.error
    for attempt in range(6):
        wait = 2.0 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20).read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(min(60, int(e.headers.get("Retry-After", "10") or 10)))
    raise RuntimeError("rate limited")


REST = "https://en.wikipedia.org/api/rest_v1/page/"


def page(title):
    """opening text (REST summary) and the article links of the page (REST html, internal wiki links only)"""
    import re
    q = urllib.parse.quote(title.replace(" ", "_"), safe="")
    summ = json.loads(get(REST + "summary/" + q))
    html = get(REST + "html/" + q).decode("utf-8", "replace")
    links = re.findall(r'rel="mw:WikiLink" href="\./([^"#?]+)"', html)
    links = [urllib.parse.unquote(l).replace("_", " ") for l in links if ":" not in urllib.parse.unquote(l)]
    return summ.get("title", title), (summ.get("extract") or "")[:900], list(dict.fromkeys(links))


def random_title():
    return json.loads(get(REST.replace("page/", "page/random/summary")))["title"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--start", default="")
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/live/wander.json")
    a = ap.parse_args()
    torch.set_num_threads(4); torch.manual_seed(0); random.seed(0)
    m = load("models/oll-1.pt")
    train = [p for n, p in m.named_parameters() if n.split(".")[-1] in PHASE_PARAMS]
    for p in m.parameters():
        p.requires_grad_(any(p is t for t in train))
    opt = torch.optim.AdamW(train, lr=a.lr, weight_decay=0.0)
    one = torch.ones(1)
    st = m.init_state(1); h = None
    buf = deque(maxlen=193); since = [0]

    def feed(text, learn=True):
        """read bytes, return mean surprise (bits/byte) measured before each byte is seen"""
        nonlocal st, h
        bits = []
        for i, b in enumerate(text.encode("utf-8")):
            with torch.no_grad():
                if h is not None:
                    bits.append(float(m.nll(h, torch.tensor([TEXT]), torch.tensor([float(b)], dtype=torch.float64))[0]) / math.log(2))
                h, st = m.step(torch.tensor([float(b)], dtype=torch.float64), torch.tensor([TEXT]), one, one, st)
            buf.append(float(b)); since[0] += 1
            if learn and since[0] >= 192 and len(buf) == buf.maxlen:
                since[0] = 0
                x = torch.tensor([list(buf)], dtype=torch.float64); s = torch.full(x.shape, TEXT); d = torch.ones(x.shape)
                loss = m.nll(m(x[:, :-1], s[:, :-1], d[:, :-1], d[:, 1:]), s[:, 1:], x[:, 1:])[:, 16:].mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(train, 1.0); opt.step()
        return float(np.mean(bits)) if bits else None

    def surprise_of(title):
        """how unexpected this title is to it right now (bits/byte), without changing its state"""
        s_, h_ = {k: (v.clone() if torch.is_tensor(v) else [c.clone() for c in v]) for k, v in st.items()}, h
        bits = []
        with torch.no_grad():
            for b in (title + "\n").encode("utf-8"):
                bits.append(float(m.nll(h_, torch.tensor([TEXT]), torch.tensor([float(b)], dtype=torch.float64))[0]) / math.log(2))
                h_, s_ = m.step(torch.tensor([float(b)], dtype=torch.float64), torch.tensor([TEXT]), one, one, s_)
        return float(np.mean(bits))

    if a.start:
        title = a.start
    else:
        title = random_title()
    path, visited, t0 = [], set(), time.time()
    while time.time() - t0 < a.minutes * 60:
        title, text, links = page(title)
        visited.add(title)
        s_page = feed("\n" + title + "\n" + text + "\n")
        links = [l for l in links if l not in visited and not l.startswith(("List of", "Index of"))]
        step = dict(page=title, read_bytes=len(text.encode()), surprise_bits_per_byte=s_page, n_links=len(links))
        if not links:
            path.append(step); print(json.dumps(step, ensure_ascii=False), flush=True); break
        cand = random.sample(links, min(a.candidates, len(links)))
        scored = sorted(((surprise_of(c), c) for c in cand), reverse=True)
        step.update(chose=scored[0][1], chose_surprise=scored[0][0], least_surprising=scored[-1][1],
                    candidates={c: round(v, 2) for v, c in scored})
        path.append(step)
        print(json.dumps({k: v for k, v in step.items() if k != "candidates"}, ensure_ascii=False), flush=True)
        title = scored[0][1]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(minutes=a.minutes, rule="follow the link whose title surprises it most", path=path), open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
