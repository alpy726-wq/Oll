"""Oll-1 command line.
  python -m oll forecast data.csv [--horizon 96]      -> forecast_<name>.csv (median, q10, q90)
  python -m oll alarms data.csv                       -> alarms_<name>.csv (surprise and alarm per reading)
CSV: "timestamp,value" or just "value" (one per line). Weights: models/oll-1.pt"""
import argparse, csv, io, os
import numpy as np
import torch
from oll.core import load, forecast, surprise, alarms
from oll.alarm_select import relative_alarms

DEFAULT_RULE = dict(W=288, w=6, k=4.0)


def read(path):
    rows = [r for r in csv.reader(open(path)) if r and any(c.strip() for c in r)]
    try:
        float(rows[0][-1]); body = rows
    except ValueError:
        body = rows[1:]
    v = np.array([float(r[-1]) for r in body]); t = None
    if len(body[0]) >= 2:
        try:
            t = np.array([np.datetime64(r[0].strip().replace(" ", "T")[:19]) for r in body]).astype("datetime64[s]").astype(np.float64)
        except Exception:
            t = None
    return v, t


def main():
    ap = argparse.ArgumentParser(prog="python -m oll")
    ap.add_argument("cmd", choices=["forecast", "alarms"])
    ap.add_argument("csv")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "oll-1.pt"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load(a.weights, dev)
    v, t = read(a.csv)
    name = os.path.splitext(os.path.basename(a.csv))[0]
    if a.cmd == "forecast":
        f = forecast(m, v, t, horizon=a.horizon, paths=a.paths, device=dev)
        with open(f"forecast_{name}.csv", "w") as fh:
            fh.write("step,median,q10,q90\n")
            for k in range(a.horizon):
                fh.write(f"{k + 1},{f['median'][k]:.6g},{f['q10'][k]:.6g},{f['q90'][k]:.6g}\n")
        print(f"forecast_{name}.csv")
    else:
        s = surprise(m, [(v, t)], device=dev, normal="calm")[0]
        fl = alarms(s)
        with open(f"alarms_{name}.csv", "w") as fh:
            fh.write("index,value,surprise,alarm\n")
            for i in range(len(v)):
                fh.write(f"{i},{v[i]:.6g},{s[i]:.4f},{int(fl[i])}\n")
        print(f"alarms_{name}.csv: {int(fl.sum())} alarm points")


if __name__ == "__main__":
    main()
