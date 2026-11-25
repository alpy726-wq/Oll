"""Real-facts test of the phase memory mechanism (no training): 243 real "Country:Capital" facts from the ISO country
table are streamed ONCE into the memory as text. Then, for every country, the prompt "\\nCountry:" is given and the
capital is read back character by character from the memory alone (no trunk, no learned weights involved).
Measured: exact full-string recall, per-character accuracy, and exact erase (one fact removed, all others unchanged).
Compares memory keys of order 1 (previous event only, the old design) with context keys (orders 1+8+16)."""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pfm.one import OneMachine
from pfm.realworld import STREAMS_REAL, SID

facts = json.load(open("data/v4/facts.json"))
text = "".join(f"\n{f['country']}:{f['capital']}" for f in facts) + "\n"
data = torch.tensor(list(text.encode("utf-8")), dtype=torch.float64)
TS = SID["text"]
res = {"n_facts": len(facts), "bytes_written": int(len(data))}


def run(orders, RM=4096, erase_idx=None, width=1.0, order_weight_power=0.0):
    m = OneMachine(STREAMS_REAL, R=64, L=1, M=2, RM=RM, mem_orders=orders).eval()
    with torch.no_grad():
        m.mem_logwidth.fill_(float(np.log(width)))  # width of the "similar values" band; small = exact symbols
        m.mem_logmu.fill_(-20.0)  # no forgetting over the 4670 written bytes (default e^-6 per step forgets in ~400)
        if len(orders) > 1:  # longer contexts weigh more (back-off: the longest matching context dominates)
            m.mem_order_logw.copy_(order_weight_power * torch.log(torch.tensor(orders, dtype=torch.float32)))
    w = m.order_weights()
    one = torch.ones(1)

    def write_all():
        st = m.init_state(1)
        with torch.no_grad():
            for b in data:
                _, st = m.step(b[None], torch.tensor([TS]), one, one, st)
        return st

    st = write_all()

    def ask(st0, prompt, maxlen=40):
        st = {k: (v if k != "conv" else [c.clone() for c in v]) for k, v in st0.items()}
        out = []
        with torch.no_grad():
            for b in prompt.encode("utf-8"):
                _, st = m.step(torch.tensor([float(b)], dtype=torch.float64), torch.tensor([TS]), one, one, st)
            for _ in range(maxlen):
                # memory-only read-out: the retrieval for the current context, decoded against the byte alphabet
                hist = st["hist"]
                r = 0
                for o, (n, wn) in enumerate(zip(m.mem_orders, w)):
                    ph_n = sum(hist[:, i][:, m.ctx_perm[i]] for i in range(n))
                    k = torch.polar(torch.ones(1, dtype=torch.float64), ph_n).to(torch.complex64)
                    r = r + wn * torch.einsum("bm,bmr->br", k, st["M"][o]) / m.RM
                cf = r.to(torch.complex128) * torch.exp(-1j * m.psi[TS])
                ch = int(m.discrete_logits(cf, STREAMS_REAL[TS])[0].argmax())
                if ch == ord("\n"):
                    break
                out.append(ch)
                _, st = m.step(torch.tensor([float(ch)], dtype=torch.float64), torch.tensor([TS]), one, one, st)
        return bytes(out).decode("utf-8", errors="replace")

    answers = [ask(st, f"\n{f['country']}:") for f in facts]
    exact = np.mean([a == f["capital"] for a, f in zip(answers, facts)])
    chars = np.mean([np.mean([i < len(a) and a[i] == c for i, c in enumerate(f["capital"])]) for a, f in zip(answers, facts)])
    out = dict(orders=list(orders), RM=RM, coarse_width=width, exact_capital_recall=float(exact), char_accuracy=float(chars),
               examples=[(f["country"], f["capital"], a) for a, f in list(zip(answers, facts))[:8]])
    if erase_idx is not None:  # erase one whole fact: subtract every binding written while streaming its line
        f = facts[erase_idx]
        line = f"\n{f['country']}:{f['capital']}"
        start = text.index(line)
        # re-stream the prefix to recover the exact keys used for this line, then subtract its pairs
        st_pre = m.init_state(1)
        with torch.no_grad():
            for b in data[:start]:
                _, st_pre = m.step(b[None], torch.tensor([TS]), one, one, st_pre)
            st_e = {k: v for k, v in st.items()}
            M = st["M"].clone(); cur = st_pre
            for b in data[start:start + len(line.encode("utf-8")) + 1]:
                _, nxt = m.step(b[None], torch.tensor([TS]), one, one, cur)
                M = M - (nxt["M"] - cur["M"])  # the pair this event added (no decay: mu ~ e^-6 per step, negligible here)
                cur = nxt
            st_e["M"] = M
        after = [ask(st_e, f"\n{g['country']}:") for g in facts]
        out["erase"] = dict(erased_country=f["country"], erased_recalled_before=answers[erase_idx] == f["capital"],
                            erased_recalled_after=after[erase_idx] == f["capital"],
                            other_correct_before=int(sum(answers[i] == facts[i]["capital"] for i in range(len(facts)) if i != erase_idx)),
                            other_correct_after=int(sum(after[i] == facts[i]["capital"] for i in range(len(facts)) if i != erase_idx)),
                            other_answers_changed=int(sum(after[i] != answers[i] for i in range(len(facts)) if i != erase_idx)))
    return out


res["order_1_only"] = run((1,), width=0.01)
res["context_key_8"] = run((8,), width=0.01)
res["context_keys_4_8_12"] = run((4, 8, 12), width=0.01)
res["context_keys_4_8_12_RM16384"] = run((4, 8, 12), RM=16384, width=0.01, erase_idx=0)
json.dump(res, open("audit/facts_memory.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "examples"} if isinstance(v, dict) else v) for k, v in res.items()}, indent=1, ensure_ascii=False))
print("examples:", res["context_keys_4_8_12_RM16384"]["examples"][:8])
