"""Oll command line.
  python -m oll ask notes.txt [more.txt ...] [--q "question"] -> answers from your documents (Oll-Reader, CPU)
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


def ask(argv):
    """answer questions from documents; says when it is not sure (confidence below --sure)"""
    from oll.reader import Reader
    p = argparse.ArgumentParser(prog="python -m oll ask")
    p.add_argument("docs", nargs="+")
    p.add_argument("--q", action="append", default=[])
    p.add_argument("--sure", type=float, default=0.3)
    p.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "oll-reader.pt"))
    a = p.parse_args(argv)
    text = "\n".join(open(d, encoding="utf-8", errors="replace").read() for d in a.docs)
    r = Reader(a.weights)

    def one(q):
        o = r.ask(text, q)
        if o["confidence"] >= a.sure:
            print(f"Oll: {o['answer']}   (confidence {o['confidence']:.2f}; from: \"{o.get('sentence', '')[:160]}\")")
        else:
            print(f"Oll: I am not sure. Best guess: {o['answer']} (confidence {o['confidence']:.2f})")

    if a.q:
        for q in a.q:
            print("You:", q); one(q)
        return
    print("Ask about your documents (empty line to quit).")
    while True:
        try:
            q = input("You: ").strip()
        except EOFError:
            break
        if not q:
            break
        one(q)


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ask":
        return ask(sys.argv[2:])
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
