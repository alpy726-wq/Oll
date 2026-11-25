"""Large real-world corpus for the main run (built on the GPU provider, cached on a Modal volume).

Time series: ten public real-world collections from the Chronos/Monash archive (autogluon/chronos_datasets on the
Hugging Face hub): electricity (15 min), traffic occupancy, weather, solar power, taxi demand, exchange rates, air
quality (KDD Cup 2018), Uber pickups, wind farm output, pedestrian counts. ETT and Jena are NOT in this list: they
stay as held-out benchmarks.
Text: the first FineWeb-edu "sample-10BT" shard (educational web text), plus our book corpus.

Per series, the last 20% of time steps are held out (validation); standardisation uses the first 80% only."""
import io, json, os, urllib.request
import numpy as np

HF = "https://huggingface.co/datasets"
TS = ["electricity_15min", "monash_traffic", "monash_weather", "solar_1h", "taxi_30min", "exchange_rate",
      "monash_kdd_cup_2018", "uber_tlc_hourly", "wind_farms_hourly", "monash_pedestrian_counts"]
TEXT_BYTES = 400_000_000


def fetch(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.rename(path + ".part", path)
    return path


def build(out="/vol/big"):
    import pyarrow.parquet as pq
    os.makedirs(out, exist_ok=True)
    meta = {}
    for name in TS:
        dst = f"{out}/{name}.npz"
        if os.path.exists(dst):
            meta[name] = json.load(open(dst + ".json")); continue
        p = fetch(f"{HF}/autogluon/chronos_datasets/resolve/main/{name}/train-00000-of-00001.parquet", f"{out}/raw/{name}.parquet")
        tab = pq.read_table(p)
        cols = tab.column_names
        tcol = "timestamp"
        vcol = "target" if "target" in cols else [c for c in cols if c not in ("id", "timestamp")][0]
        vals, times, offs, n_series = [], [], [0], 0
        for ts, v in zip(tab.column(tcol).to_pylist(), tab.column(vcol).to_pylist()):
            v = np.asarray(v, dtype=np.float64)
            t = np.asarray([x.timestamp() for x in ts], dtype=np.float64) / 3600.0
            ok = np.isfinite(v)
            if ok.sum() < 200:
                continue
            v, t = v[ok], t[ok]
            cut = int(0.8 * len(v))
            mu, sd = v[:cut].mean(), v[:cut].std() + 1e-8
            z = np.clip((v - mu) / sd / 3.0, -1.2, 1.2).astype(np.float32)
            vals.append(z); times.append(t.astype(np.float64)); offs.append(offs[-1] + len(z)); n_series += 1
        np.savez(dst, x=np.concatenate(vals), t=np.concatenate(times), off=np.array(offs))
        meta[name] = dict(series=n_series, points=int(offs[-1]))
        json.dump(meta[name], open(dst + ".json", "w"))
    if not os.path.exists(f"{out}/fineweb.npy"):
        p = fetch(f"{HF}/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet", f"{out}/raw/fineweb.parquet")
        pf = pq.ParquetFile(p)
        buf, n = [], 0
        for rg in range(pf.num_row_groups):
            for txt in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                b = txt.encode("utf-8") + b"\n"
                buf.append(b); n += len(b)
            if n >= TEXT_BYTES:
                break
        arr = np.frombuffer(b"".join(buf)[:TEXT_BYTES], dtype=np.uint8)
        np.save(f"{out}/fineweb.npy", arr)
    meta["fineweb_bytes"] = int(np.load(f"{out}/fineweb.npy", mmap_mode="r").shape[0])
    json.dump(meta, open(f"{out}/META.json", "w"), indent=1)
    return meta
