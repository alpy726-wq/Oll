"""Where does training time go on the GPU? Main-model size, real batches from every domain, fwd+bwd step time with the
old complex128 scan and the float32 scan, plus the top CUDA ops for one irregular and one regular batch.
Writes /vol/results/profile_speed.json"""
import json, sys, time
import numpy as np
import torch
sys.path.insert(0, "/oll")
from pfm.one import OneMachine
from pfm.bigworld import STREAMS_BIG, BigData, make_batch_big
from pfm import continuous

dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True
m = OneMachine(STREAMS_BIG, R=1024, L=8, M=8, RM=512).to(dev)
data = BigData("/vol/big", "train"); rng = np.random.default_rng(0)
batches = {d: make_batch_big(data, rng, 32, 512, d, dev) for d in ("series", "web", "weather", "grid", "market", "audio", "text")}


def step(b):
    l = m.nll(m(b["x"], b["s"], b["dt"], b["q"]), b["s_out"], b["y"]).mean()
    l.backward(); m.zero_grad(set_to_none=True)


out = {}
for fast in (False,):
    continuous.CTConv.FAST_SCAN = fast
    times = {}
    for d, b in batches.items():
        step(b); torch.cuda.synchronize()
        t = time.time()
        for _ in range(3):
            step(b)
        torch.cuda.synchronize(); times[d] = (time.time() - t) / 3
    out["fast_scan" if fast else "old_scan"] = times
    print("fast" if fast else "old", json.dumps({k: round(v, 3) for k, v in times.items()}), flush=True)
from torch.profiler import profile, ProfilerActivity
continuous.CTConv.FAST_SCAN = False
for d in ("weather", "web"):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        step(batches[d]); torch.cuda.synchronize()
    out["top_ops_" + d] = p.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    print(out["top_ops_" + d], flush=True)
out["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
json.dump(out, open("/vol/results/profile_speed.json", "w"), indent=1)
