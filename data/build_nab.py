"""Numenta Anomaly Benchmark (NAB): real-world series with labelled anomaly windows (machine temperature before a
failure, AWS server metrics, ad exchange, traffic, Twitter volume). Only the real* categories. Downloads the CSVs and
the official window labels into data/nab/."""
import json, os, urllib.request
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nab")
BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
os.makedirs(OUT, exist_ok=True)
lab = json.loads(urllib.request.urlopen(f"{BASE}/labels/combined_windows.json").read())
keep = {k: v for k, v in lab.items() if k.split("/")[0].startswith("real")}
for k in keep:
    p = os.path.join(OUT, k.replace("/", "__"))
    if not os.path.exists(p):
        urllib.request.urlretrieve(f"{BASE}/data/{k}", p)
json.dump(keep, open(os.path.join(OUT, "windows.json"), "w"), indent=1)
print(len(keep), "series,", sum(len(v) for v in keep.values()), "labelled anomaly windows")
