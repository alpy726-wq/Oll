"""Real robot experience for the world-model phase: LeRobot "koch_pick_place_5_lego" (a real Koch v1.1 arm picking
and placing Lego bricks; 50 episodes, 30 fps). Senses and actions of a body, all real recordings:
  proprioception : 6 joint positions (observation.state)        30 Hz
  motor commands : 6 joint targets (action)                     30 Hz
  vision         : camera "laptop", 8x8 grey pixels             5 Hz   (one stream per pixel position, so the
                   position is the stream's role phase: a pixel event is its value bound to its place, Prop. 1)
Split by episode: episodes 40-49 are test. Joint and action standardisation from train episodes only.
Output: data/robot/{train,test}.npz (merged event stream per episode, real time order) and META.json."""
import json, os, subprocess, urllib.request
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot")
REPO = "https://huggingface.co/datasets/lerobot/koch_pick_place_5_lego/resolve/main"
CAM, PIX, FPS_IMG = "observation.images.laptop", 8, 5


def fetch(rel):
    p = os.path.join(OUT, "raw", rel.replace("/", "__"))
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        urllib.request.urlretrieve(f"{REPO}/{rel}", p + ".part"); os.rename(p + ".part", p)
    return p


def decode(path):
    """all frames of a video file at FPS_IMG, as PIXxPIX grey uint8, with their timestamps"""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={FPS_IMG},scale={PIX}:{PIX}:flags=area,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, PIX, PIX)
    return fr, np.arange(len(fr)) / FPS_IMG


def main():
    import pyarrow.parquet as pq
    data = pq.read_table(fetch("data/chunk-000/file-000.parquet")).to_pydict()
    eps = pq.read_table(fetch("meta/episodes/chunk-000/file-000.parquet")).to_pydict()
    state = np.array(data["observation.state"], dtype=np.float64); action = np.array(data["action"], dtype=np.float64)
    ep_of = np.array(data["episode_index"]); ts = np.array(data["timestamp"], dtype=np.float64)
    videos = {}
    for k in range(len(eps["episode_index"])):
        f = eps[f"videos/{CAM}/file_index"][k]
        if f not in videos:
            videos[f] = decode(fetch(f"videos/{CAM}/chunk-000/file-{f:03d}.mp4"))
    train_eps = [e for e in eps["episode_index"] if e < 40]
    trm = np.isin(ep_of, train_eps)
    smu, ssd = state[trm].mean(0), state[trm].std(0) + 1e-6
    amu, asd = action[trm].mean(0), action[trm].std(0) + 1e-6
    for split, keep in (("train", lambda e: e < 40), ("test", lambda e: e >= 40)):
        T, S, X, off, names = [], [], [], [0], []
        for k, e in enumerate(eps["episode_index"]):
            if not keep(e):
                continue
            m = ep_of == e
            t_j = ts[m]
            ev_t = [np.repeat(t_j, 6), np.repeat(t_j, 6)]
            ev_s = [np.tile(np.arange(6), len(t_j)), np.tile(6 + np.arange(6), len(t_j))]
            ev_x = [((state[m] - smu) / ssd).ravel() / 3.0, ((action[m] - amu) / asd).ravel() / 3.0]
            fr, ft = videos[eps[f"videos/{CAM}/file_index"][k]]
            a, b = eps[f"videos/{CAM}/from_timestamp"][k], eps[f"videos/{CAM}/to_timestamp"][k]
            sel = (ft >= a) & (ft < b)
            img_t = ft[sel] - a
            ev_t.append(np.repeat(img_t, PIX * PIX)); ev_s.append(np.tile(12 + np.arange(PIX * PIX), sel.sum()))
            ev_x.append(fr[sel].reshape(-1).astype(np.float64) / 255.0 * 2.0 - 1.0)
            t = np.concatenate(ev_t); s = np.concatenate(ev_s); x = np.concatenate(ev_x)
            o = np.lexsort((s, t))  # time order; at equal time: joints, actions, then pixels in raster order
            T.append(t[o]); S.append(s[o]); X.append(np.clip(x[o], -1.2, 1.2)); off.append(off[-1] + len(o)); names.append(int(e))
        np.savez_compressed(f"{OUT}/{split}.npz", t=np.concatenate(T), s=np.concatenate(S), x=np.concatenate(X).astype(np.float32),
                            off=np.array(off), episodes=np.array(names))
    meta = dict(source="lerobot/koch_pick_place_5_lego", camera=CAM, pixels=f"{PIX}x{PIX} grey @ {FPS_IMG} Hz",
                streams=[f"joint{i}" for i in range(6)] + [f"action{i}" for i in range(6)] + [f"pix{r}_{c}" for r in range(PIX) for c in range(PIX)],
                episodes_train=len(train_eps), episodes_test=50 - len(train_eps))
    json.dump(meta, open(f"{OUT}/META.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "streams"}))


if __name__ == "__main__":
    main()
