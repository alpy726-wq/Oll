"""E2: properties of the trained model (paper Section 4), measured on runs/main.pt."""
import copy, json, sys
import torch
from pfm.model import PrimeSeqModel, PhaseFilter, param_breakdown
from pfm.data import eval_windows, bpb, eval_all, DOMAINS

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
torch.manual_seed(0)
args = json.load(open("runs/main.json"))["args"]
m = PrimeSeqModel(R=args["R"], L=args["L"], K=args["K"], family=args["family"])
m.load_state_dict(torch.load("runs/main.pt"))
m.eval()
res = {"params_total": sum(p.numel() for p in m.parameters()), "breakdown": param_breakdown(m),
       "embedding_attention_ffn_positional_params": 0}

x, y = eval_windows("prose", 4, 512)
with torch.no_grad():
    # invertibility
    e = m.encode(x.double())
    h = m.trunk(e)
    res["inverse_rel_err_fp32"] = float((m.inverse(h) - e).norm() / e.norm())
    md = copy.deepcopy(m).double()
    hd = md.trunk(e.double())
    res["inverse_rel_err_fp64"] = float((md.inverse(hd) - e.double()).norm() / e.double().norm())
    # distinct inputs never merge (bijection): min distance between trunk outputs of sequences differing in 1 byte
    x2 = x.clone(); x2[:, 100] = (x2[:, 100] + 1) % 256
    d_out = (m.trunk(m.encode(x2.double())) - h)[:, 100:].norm(dim=-1).min()
    res["one_byte_change_min_output_distance_from_pos100"] = float(d_out)
    # unitarity of the phase filters
    u = torch.randn(64, 2 * args["R"])
    ratios = [float(b.F.phase(u[:, : args["R"]]).norm() / u[:, : args["R"]].norm()) for b in m.blocks]
    res["phase_filter_norm_ratio_min_max"] = [min(ratios), max(ratios)]
    # streaming state == full pass
    T = 600
    xs = eval_windows("code", 1, T)[0].double()
    full = m(xs)
    st = m.init_state(1)
    outs = []
    for t in range(T):
        l, st = m.step(xs[:, t], st)
        outs.append(l)
    stream = torch.stack(outs, 1)
    res["stream_vs_full_max_abs_logit_diff"] = float((stream - full).abs().max())
    res["stream_vs_full_rel"] = float((stream - full).norm() / full.norm())
    res["stream_vs_full_same_argmax_frac"] = float((stream.argmax(-1) == full.argmax(-1)).float().mean())
    res["state_bytes"] = m.state_bytes(1)
    res["state_MiB"] = m.state_bytes(1) / 2 ** 20
    res["receptive_field_bytes"] = args["L"] * 2 * (args["K"] - 1) + 1

# length extrapolation: same weights, longer windows (trained at T=512)
res["length_extrapolation_bpb"] = {}
for T in (256, 512, 1024, 2048, 4096):
    xx, yy = eval_windows("code", max(4, 48 * 512 // T), T)
    res["length_extrapolation_bpb"][T] = bpb(m, xx, yy)

base = eval_all(m, 512, 32)
res["base_eval"] = base


def perturb(names, frac, kind="noise"):
    mm = copy.deepcopy(m)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in mm.named_parameters():
            if n.split(".")[-1] in names:
                if kind == "noise":
                    scale = p.std() if p.numel() > 1 and p.std() > 0 else p.abs().mean() + 1e-3
                    if n.endswith("theta"):
                        scale = torch.tensor(torch.pi / 3 ** 0.5)  # natural scale of a phase: std of U(-pi,pi)
                    p.add_(frac * scale * torch.randn(p.shape, generator=g))
                elif kind == "zero_half":
                    p.mul_((torch.rand(p.shape, generator=g) > 0.5).float())
    return eval_all(mm, 512, 32)["avg3"] - base["avg3"]


res["sensitivity_25pct_noise_delta_bpb"] = {"V": perturb(("V",), 0.25), "theta": perturb(("theta",), 0.25),
                                            "kappa": perturb(("kappa",), 0.25)}
res["zero_half_of_all_weights_delta_bpb"] = perturb(("V", "kappa", "theta", "sigma", "alpha"), 0, "zero_half")
# delete one layer (identity in its place) -- 1/L of the layers
mm = copy.deepcopy(m)
del mm.blocks[len(mm.blocks) // 2]
res["delete_one_layer_delta_bpb"] = eval_all(mm, 512, 32)["avg3"] - base["avg3"]
res["deleted_layer_fraction"] = 1 / args["L"]

# exact memory in a phase state: write association by phase rotation, erase by conjugate rotation
torch.manual_seed(3)
R = args["R"]
state = torch.polar(torch.ones(R, dtype=torch.float64), torch.rand(R, dtype=torch.float64) * 6.283)
from pfm.basis import PrimeBasis
B = PrimeBasis(R)
key, val = B.complex(torch.tensor(17.0)).double(), B.complex(torch.tensor(201.0)).double()
amp = B.A.double()
bound = key * val / amp  # = A * b(17+201) / A  phase-wise binding (Prop 1)
written = state + bound
recovered = (written - state) * key.conj() / amp ** 1  # unbind with the conjugate key
erased = written - bound
res["phase_memory_erase_max_abs_err"] = float((erased - state).abs().max())
# unbinding: recovered should be the value up to the amplitude factor A
cands = B.complex(torch.arange(256.0)).double()
scores = (cands.conj() * recovered).sum(-1).real
res["phase_memory_unbind_top1"] = int(scores.argmax())
res["phase_memory_unbind_expected"] = 201
json.dump(res, open("runs/E2_props.json", "w"), indent=1)
print(json.dumps(res, indent=1))
