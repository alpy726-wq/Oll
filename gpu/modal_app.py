"""Modal app: runs training jobs on GPUs with the frozen real-data corpus committed in data/v4 (every file is checked
against its SHA-256 in data/v4/MANIFEST.json before training). Launched from GitHub Actions (gpu/launch.py), because the coding sandbox cannot speak gRPC to Modal."""
import json, os, subprocess, sys
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = modal.App("oll")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "curl")
         .pip_install("torch==2.9.0", "numpy==2.3.4")
         .add_local_dir(ROOT, "/oll", ignore=["data/real", "data/text_dl", "runs", ".git", "**/__pycache__", "*.pt"]))
vol = modal.Volume.from_name("oll-data", create_if_missing=True)  # kept for checkpoints


def _ensure_data():
    import hashlib
    os.chdir("/oll")
    man = json.load(open("data/v4/MANIFEST.json"))
    for k, v in man.items():
        f = f"data/v4/{k}.npz" if os.path.exists(f"data/v4/{k}.npz") else f"data/v4/{k}.json"
        if hashlib.sha256(open(f, "rb").read()).hexdigest() != v["sha256"]:
            raise RuntimeError(f"corpus file {f} does not match its frozen hash")


@app.function(image=image, volumes={"/vol": vol}, timeout=3600, cpu=4)
def build_data():
    _ensure_data()
    return json.load(open("/oll/data/v4/MANIFEST.json"))


def _train(args: list, script: str):
    _ensure_data()
    out = "/tmp/run"
    cmd = [sys.executable, script, *args, "--device", "cuda", "--threads", "8", "--out", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return dict(error=p.stderr[-4000:], cmd=cmd)
    res = json.load(open(out + ".json"))
    res["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return res


@app.function(image=image, volumes={"/vol": vol}, gpu="B200", timeout=6 * 3600)
def train_b200(args: list, script: str = "real_train.py"):
    return _train(args, script)


@app.function(image=image, volumes={"/vol": vol}, gpu="H100", timeout=6 * 3600)
def train_h100(args: list, script: str = "real_train.py"):
    return _train(args, script)


def _multi(cmds: dict):
    """several small runs at once on ONE GPU (small models leave a GPU mostly idle); each value is a shell command
    that writes JSON files under runs/; returns {relative path: parsed JSON} plus logs"""
    import glob, time
    os.chdir("/oll")
    t0 = time.time()
    procs = {k: subprocess.Popen(c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for k, c in cmds.items()}
    logs = {k: p.communicate()[0][-3000:] for k, p in procs.items()}
    out = {f: json.load(open(f)) for f in glob.glob("runs/**/*.json", recursive=True)}
    pts = {f: open(f, "rb").read() for f in glob.glob("runs/**/*.pt", recursive=True)}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return dict(files=out, pts=pts, logs=logs, gpu=gpu, wall_seconds=time.time() - t0)


@app.function(image=image, gpu="L4", cpu=8, timeout=4 * 3600)
def multi_l4(cmds: dict):
    return _multi(cmds)
