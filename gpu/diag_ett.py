"""Why does the main-run machine fail on ETTh2 (MSE ~2.1 vs 0.36 for repeat-last) but not on ETTh1/ETTm1?
Per channel, H=96, 40 test windows: model MSE, repeat-last MSE, mean/std of the model's median path vs the truth.
Two stream assignments: (a) the ETTh1-trained "ett:*" streams (as in the main evaluation); (b) one generic hourly
series stream (ts:electricity_15min) for every channel. Writes /vol/results/diag_ett.json"""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, "/oll")
from pfm.one import OneMachine
from pfm.bigworld import STREAMS_BIG
from pfm.realworld import SID, ETT_VARS
from pfm.ett_eval import load_ett

dev = "cuda"
ck = torch.load("/vol/ckpt/main.pt", map_location=dev)
cfg = json.load(open("/vol/results/main.json"))["args"]
m = OneMachine(STREAMS_BIG, R=cfg["R"], L=cfg["L"], M=cfg["M"], RM=cfg["RM"]).to(dev)
m.load_state_dict(ck["model"]); m.eval()
grid = torch.linspace(-1.5, 1.5, 1501, dtype=torch.float64, device=dev)
out = {}
for name in ("ETTh2", "ETTh1"):
    t, Z, a0, b0, pm = load_ett(name, "/oll/data/v4")
    L, H = 512, 96
    starts = list(range(a0, b0 - 720 + 1, 24))[:40]
    dtc = float(np.median(np.diff(t)))
    for mode in ("ett_streams", "generic_stream"):
        rows = {}
        for c in range(7):
            sid_c = SID["ett:" + ETT_VARS[c]] if mode == "ett_streams" else 18
            X = np.stack([Z[s - L:s, c] for s in starts]); Y = np.stack([Z[s:s + H, c] for s in starts]); n = len(starts)
            sid = torch.full((n,), sid_c, device=dev); st = m.init_state(n)
            x = torch.tensor(np.clip(X / 3.0, -1.2, 1.2), dtype=torch.float64, device=dev); one = torch.full((n,), dtc, device=dev)
            with torch.no_grad():
                for k in range(L):
                    h, st = m.step(x[:, k], sid, one, one, st)
                P = np.zeros((n, H))
                for k in range(H):
                    dens = m.density(h, sid_c, grid).double(); cdf = (dens / dens.sum(-1, keepdim=True)).cumsum(-1)
                    med = grid[torch.searchsorted(cdf.contiguous(), torch.full((n, 1), 0.5, dtype=cdf.dtype, device=dev)).clamp(max=len(grid) - 1).squeeze(1)]
                    P[:, k] = med.cpu().numpy() * 3.0
                    h, st = m.step(med, sid, one, one, st)
            rows[ETT_VARS[c]] = dict(model_mse=float(((P - Y) ** 2).mean()), repeat_last_mse=float(((X[:, -1:] - Y) ** 2).mean()),
                                     pred_mean=float(P.mean()), pred_std=float(P.std()), true_mean=float(Y.mean()), true_std=float(Y.std()),
                                     first_step_err=float(np.abs(P[:, 0] - Y[:, 0]).mean()))
        rows["avg_model_mse"] = float(np.mean([r["model_mse"] for k, r in rows.items() if isinstance(r, dict)]))
        out[f"{name}_{mode}"] = rows
        print(name, mode, json.dumps(rows["avg_model_mse"]), flush=True)
json.dump(out, open("/vol/results/diag_ett.json", "w"), indent=1)
print(json.dumps(out, indent=1))
