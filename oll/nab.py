"""Real anomaly detection test on NAB (all real* series, official labelled windows). No tuning on the labels:
one fixed rule (alarm = smoothed surprise above mean + 4 std of the first 15% of the series)."""
import csv, glob, json, os
import numpy as np


def load_nab(root):
    W = json.load(open(os.path.join(root, "windows.json")))
    out = []
    for k, wins in sorted(W.items()):
        rows = list(csv.DictReader(open(os.path.join(root, k.replace("/", "__")))))
        t = np.array([np.datetime64(r["timestamp"].replace(" ", "T")) for r in rows]).astype("datetime64[s]").astype(np.float64)
        v = np.array([float(r["value"]) for r in rows])
        lab = np.zeros(len(v), bool)
        wi = []
        for a, b in wins:
            ta = np.datetime64(a.replace(" ", "T")[:19]).astype("datetime64[s]").astype(np.float64)
            tb = np.datetime64(b.replace(" ", "T")[:19]).astype("datetime64[s]").astype(np.float64)
            lab |= (t >= ta) & (t <= tb); wi.append((ta, tb))
        out.append(dict(name=k, t=t, v=v, label=lab, windows=wi))
    return out


def auc(score, label):
    s = np.nan_to_num(score, nan=np.nanmin(score)); pos, neg = s[label], s[~label]
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def runs(flags):
    """contiguous True runs as (start, end) index pairs"""
    d = np.diff(np.concatenate([[0], flags.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def evaluate(series, scores, alarm_fn):
    per = []
    for S, sc in zip(series, scores):
        sm = np.convolve(np.nan_to_num(sc, nan=0.0), np.ones(3) / 3, mode="same")
        fl = alarm_fn(sc)
        R = runs(fl)
        hit_runs = sum(1 for a, b in R if S["label"][a:b].any())
        det = sum(1 for ta, tb in S["windows"] if fl[(S["t"] >= ta) & (S["t"] <= tb)].any())
        per.append(dict(name=S["name"], n=int(len(sc)), auc=auc(sm, S["label"]), windows=len(S["windows"]), windows_detected=det,
                        alarm_runs=len(R), alarm_runs_in_windows=hit_runs))
    A = [p["auc"] for p in per if p["auc"] is not None]
    nw = sum(p["windows"] for p in per); nd = sum(p["windows_detected"] for p in per)
    na = sum(p["alarm_runs"] for p in per); nh = sum(p["alarm_runs_in_windows"] for p in per)
    return dict(series=len(per), mean_point_auc=float(np.mean(A)), median_point_auc=float(np.median(A)),
                labelled_windows=nw, windows_detected=nd, window_recall=nd / max(nw, 1),
                alarm_runs=na, alarm_runs_inside_windows=nh, alarm_precision=nh / max(na, 1), per_series=per)
