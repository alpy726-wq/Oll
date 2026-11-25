"""Real driving data for an action-conditioned world model: comma2k19 (comma.ai, 2018), the public "demo" shards on the
Hugging Face hub (66 one-minute segments of real highway driving, Toyota RAV4 / Honda Civic, California).

Each sensor keeps its own clock (no resampling):
  action  : steering_angle (driver)                        ~100 Hz CAN
  vehicle : speed, wheel speeds (front-left, rear-left)    ~50-100 Hz CAN
  inertia : accelerometer x/y/z, gyro z (yaw rate)          ~100 Hz IMU
Split by segment (no time overlap): the last 16 segments are test. Standardisation uses train segments only.
Output: data/drive/{train,test}.npz with per-event arrays (t [s], stream id, value) per segment, and META.json."""
import json, os, urllib.request
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drive")
URL = "https://huggingface.co/datasets/commaai/comma2k19/resolve/main/data/demo-0000{}-of-00003.parquet"
CHANNELS = [  # (name, time column, value column, index or None)
    ("steering_angle", "processed_log__CAN__steering_angle__t", "processed_log__CAN__steering_angle__value", None),
    ("speed", "processed_log__CAN__speed__t", "processed_log__CAN__speed__value", 0),
    ("wheel_fl", "processed_log__CAN__wheel_speed__t", "processed_log__CAN__wheel_speed__value", 0),
    ("wheel_rl", "processed_log__CAN__wheel_speed__t", "processed_log__CAN__wheel_speed__value", 2),
    ("acc_x", "processed_log__IMU__accelerometer__t", "processed_log__IMU__accelerometer__value", 0),
    ("acc_y", "processed_log__IMU__accelerometer__t", "processed_log__IMU__accelerometer__value", 1),
    ("acc_z", "processed_log__IMU__accelerometer__t", "processed_log__IMU__accelerometer__value", 2),
    ("gyro_z", "processed_log__IMU__gyro__t", "processed_log__IMU__gyro__value", 2),
]


def main():
    import pyarrow.parquet as pq
    os.makedirs(OUT + "/raw", exist_ok=True)
    segs = []
    for i in range(3):
        p = f"{OUT}/raw/demo{i}.parquet"
        if not os.path.exists(p):
            urllib.request.urlretrieve(URL.format(i), p)
        tab = pq.read_table(p, columns=["segment_id", "log"])
        for sid, log in zip(tab.column("segment_id").to_pylist(), tab.column("log").to_pylist()):
            chans = []
            for name, tc, vc, idx in CHANNELS:
                t = np.asarray(log[tc], dtype=np.float64)
                v = np.asarray(log[vc], dtype=np.float64)
                v = v[:, idx] if idx is not None and v.ndim == 2 else v.reshape(len(t), -1)[:, 0]
                ok = np.isfinite(t) & np.isfinite(v)
                chans.append((t[ok], v[ok]))
            segs.append((sid, chans))
    segs.sort(key=lambda s: s[0])
    train, test = segs[:-16], segs[-16:]
    mu = [np.mean(np.concatenate([s[1][c][1] for s in train])) for c in range(len(CHANNELS))]
    sd = [np.std(np.concatenate([s[1][c][1] for s in train])) + 1e-9 for c in range(len(CHANNELS))]
    for split, S in (("train", train), ("test", test)):
        ts, ss, xs, off = [], [], [], [0]
        for sid, chans in S:
            t = np.concatenate([c[0] for c in chans])
            s = np.concatenate([np.full(len(c[0]), k) for k, c in enumerate(chans)])
            x = np.concatenate([(c[1] - mu[k]) / sd[k] for k, c in enumerate(chans)])
            o = np.argsort(t, kind="stable")  # one merged event stream in real time order
            ts.append(t[o] - t[o][0]); ss.append(s[o]); xs.append(np.clip(x[o] / 3.0, -1.2, 1.2)); off.append(off[-1] + len(o))
        np.savez_compressed(f"{OUT}/{split}.npz", t=np.concatenate(ts), s=np.concatenate(ss), x=np.concatenate(xs).astype(np.float32),
                            off=np.array(off), segments=np.array([s[0] for s in S]))
    rates = {CHANNELS[k][0]: float(np.median([len(s[1][k][0]) / max(s[1][k][0][-1] - s[1][k][0][0], 1e-9) for s in train])) for k in range(len(CHANNELS))}
    meta = dict(source="commaai/comma2k19 demo shards", segments_train=len(train), segments_test=len(test),
                channels=[c[0] for c in CHANNELS], mean=mu, std=sd, median_rate_hz=rates,
                events_train=int(sum(sum(len(c[0]) for c in s[1]) for s in train)))
    json.dump(meta, open(f"{OUT}/META.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
