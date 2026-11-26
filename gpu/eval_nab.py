"""Oll-1 on the real NAB anomaly benchmark + export of the deliverable weights (fp16, with config).
Writes /vol/results/nab.json and /vol/results/oll-1.pt"""
import json, sys, time
import numpy as np
import torch
sys.path.insert(0, "/oll")
from pfm.one import OneMachine
from pfm.bigworld import STREAMS_BIG
from oll.core import surprise, alarms
from oll.nab import load_nab, evaluate

dev = "cuda"
ck = torch.load("/vol/ckpt/main.pt", map_location=dev)
cfg = json.load(open("/vol/results/main.json"))["args"]
config = dict(R=cfg["R"], L=cfg["L"], M=cfg["M"], RM=cfg["RM"], streams="STREAMS_BIG", trained_steps=int(ck["step"]))
import os
if not os.path.exists("/vol/results/oll-1.pt"):
  torch.save(dict(config=config, model={k: (v.half() if v.is_floating_point() else v) for k, v in ck["model"].items()}),
             "/vol/results/oll-1.pt")
m = OneMachine(STREAMS_BIG, R=cfg["R"], L=cfg["L"], M=cfg["M"], RM=cfg["RM"]).to(dev)
m.load_state_dict(ck["model"]); m.eval()
S = load_nab("/oll/data/nab")
NORMAL = sys.argv[1] if len(sys.argv) > 1 else "running"
suffix = "" if NORMAL == "running" else "_" + NORMAL
t0 = time.time()
sc = surprise(m, [(s["v"], s["t"]) for s in S], device=dev, normal=NORMAL)
res = evaluate(S, sc, alarms)
res["seconds"] = time.time() - t0
res["normal"] = NORMAL
res["rule"] = "alarm = 3-point smoothed surprise > mean + 4 std of the first 15% of the series; no label tuning"
json.dump(res, open(f"/vol/results/nab{suffix}.json", "w"), indent=1)
np.savez_compressed(f"/vol/results/nab_scores{suffix}.npz", **{s["name"].replace("/", "__"): x for s, x in zip(S, sc)})
print(json.dumps({k: v for k, v in res.items() if k != "per_series"}, indent=1))
