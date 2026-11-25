"""GPU selection by measurement: training-step time of the actual workloads on several GPU types, then cost per
1000 steps from Modal's list prices. Writes runs/gpu/bench.json."""
import json, os
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICE_PER_HOUR = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "L40S": 1.95, "A100-80GB": 2.50, "H100": 3.95, "H200": 4.54, "B200": 6.25, "B300": 7.10}
GPUS = ["L4", "A10", "L40S", "H100", "B200"]
app = modal.App("oll-bench")
image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.9.0", "numpy==2.3.4")
         .add_local_dir(ROOT, "/oll", ignore=["data/real", "data/text_dl", "runs", ".git", "**/__pycache__", "*.pt"]))

WORKLOADS = {  # name: (arch, size args, batch, seq)
    "machine_small": ("machine", dict(R=96, L=3), 64, 128),
    "gpt_small": ("gpt", dict(d=40, L=3), 64, 128),
    "machine_large": ("machine", dict(R=1024, L=8), 32, 512),
    "gpt_large": ("gpt", dict(d=512, L=8), 32, 512),
}


@app.function(image=image, timeout=1800)
def bench():
    import sys, time, subprocess
    import numpy as np, torch
    sys.path.insert(0, "/oll"); os.chdir("/oll")
    from pfm.one import OneMachine
    from pfm.eventgpt import EventGPT
    from pfm.realworld import STREAMS_REAL, RealData, make_batch, DOMAINS
    data = RealData("train"); rng = np.random.default_rng(0)
    out = {"gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()}
    for name, (arch, kw, B, T) in WORKLOADS.items():
        torch.manual_seed(0)
        m = (OneMachine(STREAMS_REAL, M=8, RM=512, **kw) if arch == "machine" else EventGPT(STREAMS_REAL, T=T, **kw)).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        batches = [make_batch(data, rng, B, T, [d], "cuda") for d in DOMAINS for _ in range(2)]
        times = []
        for i in range(40):
            b = batches[i % len(batches)]
            torch.cuda.synchronize(); t = time.time()
            l = m.nll(m(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"])
            loss = (l * b["mask"]).sum() / b["mask"].sum()
            opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize(); times.append(time.time() - t)
        out[name] = dict(params=sum(p.numel() for p in m.parameters()), sec_per_step=float(np.median(times[10:])),
                         events_per_sec=float(B * T / np.median(times[10:])), peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
        del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return out


@app.local_entrypoint()
def main():
    calls = {g: bench.with_options(gpu=g).spawn() for g in GPUS}
    res = {}
    for g, c in calls.items():
        try:
            r = c.get()
        except Exception as e:  # a GPU type may be unavailable
            r = {"error": str(e)[:300]}
        for k, v in r.items():
            if isinstance(v, dict):
                v["usd_per_1000_steps"] = round(v["sec_per_step"] * 1000 / 3600 * PRICE_PER_HOUR[g], 4)
        res[g] = r
        print(g, json.dumps(r), flush=True)
    os.makedirs("runs/gpu", exist_ok=True)
    json.dump(dict(prices_usd_per_hour=PRICE_PER_HOUR, results=res), open("runs/gpu/bench.json", "w"), indent=1)
