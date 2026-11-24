"""E7: phase adapters, adapter composition (5.6) and alignment-free merging, from runs/main.pt.
Needs: runs/adapt_phase_vim, runs/adapt_dense_vim, runs/adapt_phase_ch, runs/branch_prose, runs/branch_code."""
import json, os, sys
import torch
from pfm.model import PrimeSeqModel
from pfm.data import eval_all

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
args = json.load(open("runs/main.json"))["args"]
D = ["prose", "code", "json", "vimdoc", "cheader"]


def model(sd):
    m = PrimeSeqModel(R=args["R"], L=args["L"], K=args["K"], family=args["family"])
    m.load_state_dict(sd)
    return m


def ev(sd):
    return eval_all(model(sd), 512, 32, D)


base = torch.load("runs/main.pt")
sds = {k: torch.load(f"runs/{k}.pt") for k in ("adapt_phase_vim", "adapt_dense_vim", "adapt_phase_ch", "branch_prose", "branch_code")}
res = {"base": ev(base)}
PH = ("theta", "sigma", "alpha")


def delta(sd, keys=None):
    return {k: sd[k] - base[k] for k in base if (keys is None or k.split(".")[-1] in keys) and not torch.equal(sd[k], base[k])}


for k in ("adapt_phase_vim", "adapt_dense_vim", "adapt_phase_ch"):
    d = delta(sds[k])
    n = sum(v.numel() for v in d.values())
    res[k] = dict(eval=ev(sds[k]), params_moved=n, params_moved_frac=n / sum(v.numel() for v in base.values()),
                  adapter_bytes_fp32=4 * n, base_bytes_fp32=4 * sum(v.numel() for v in base.values()))

# composition: base + d_vim + d_ch (phase adapters only)
dv, dc = delta(sds["adapt_phase_vim"]), delta(sds["adapt_phase_ch"])
comp = {k: base[k] + dv.get(k, 0) + dc.get(k, 0) for k in base}
res["compose_phase_vim_plus_ch"] = ev(comp)
# subtraction: (base + d_vim + d_ch) - d_vim should return to base + d_ch exactly
back = {k: comp[k] - dv.get(k, 0) for k in base}
res["compose_then_subtract_max_abs_dev_vs_ch_adapter"] = max(float((back[k] - sds["adapt_phase_ch"][k]).abs().max()) for k in base)
# merging: plain average of two full fine-tuned branches sharing one coordinate system
avg = {k: 0.5 * (sds["branch_prose"][k] + sds["branch_code"][k]) for k in base}
res["branch_prose"] = ev(sds["branch_prose"])
res["branch_code"] = ev(sds["branch_code"])
res["merge_average"] = ev(avg)
json.dump(res, open("runs/E7_adapt.json", "w"), indent=1)
print(json.dumps(res, indent=1))
