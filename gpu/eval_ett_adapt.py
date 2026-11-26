"""ETT standard protocol for the trained main-run machine, with and without sensory adaptation (per-window gain control
from the lookback only). Same windows and baselines as main.json. Writes /vol/results/ett_adapt.json"""
import json, sys
import torch
sys.path.insert(0, "/oll")
from pfm.one import OneMachine
from pfm.bigworld import STREAMS_BIG
from pfm.realworld import SID, ETT_VARS
from pfm.ett_eval import evaluate

dev = "cuda"
ck = torch.load("/vol/ckpt/main.pt", map_location=dev)
cfg = json.load(open("/vol/results/main.json"))["args"]
m = OneMachine(STREAMS_BIG, R=cfg["R"], L=cfg["L"], M=cfg["M"], RM=cfg["RM"]).to(dev)
m.load_state_dict(ck["model"]); m.eval()
sids = [SID["ett:" + v] for v in ETT_VARS]
out = {}
for name, stride in (("ETTh2", 24), ("ETTh1", 24), ("ETTm1", 96)):
    out[name + "_adapt"] = evaluate(m, sids, name, "/oll/data/v4", dev, L=512, stride=stride, adapt=True)
    print(name, json.dumps({h: out[name + "_adapt"][h] for h in (96, 720)}), flush=True)
    json.dump(out, open("/vol/results/ett_adapt.json", "w"), indent=1)
