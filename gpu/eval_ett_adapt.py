"""ETT standard protocol for the trained main-run machine: sensory adaptation (per-window gain control from the
lookback only), with the median rollout and with 16 imagined futures (sampled paths, pointwise median).
Same windows and baselines as main.json. No retraining. Writes /vol/results/ett_paths.json"""
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
with torch.no_grad():
    for name, stride in (("ETTh1", 24), ("ETTm1", 96), ("ETTh2", 24)):
        out[name + "_adapt_paths16"] = evaluate(m, sids, name, "/oll/data/v4", dev, L=512, stride=stride, adapt=True, paths=16)
        print(name, json.dumps({h: out[name + "_adapt_paths16"][h] for h in (96, 720)}), flush=True)
        json.dump(out, open("/vol/results/ett_paths.json", "w"), indent=1)
