"""Modal app for the main run. Detached: training continues on Modal even if the GitHub runner that started it stops.
  python gpu/main_app.py start <gpu> <minutes> [extra main_train.py args...]   -> spawns, prints the call id
  python gpu/main_app.py fetch                                                 -> copies /vol/results/*.json to runs/gpu/main/"""
import json, os, subprocess, sys
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = modal.App("oll-main")
image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.9.0", "numpy==2.3.4", "pyarrow==21.0.0")
         .add_local_dir(ROOT, "/oll", ignore=["data/real", "data/text_dl", "runs", ".git", "**/__pycache__", "*.pt"]))
vol = modal.Volume.from_name("oll-main", create_if_missing=True)
GPU = os.environ.get("OLL_GPU", "L40S")


@app.function(image=image, volumes={"/vol": vol}, gpu=GPU, timeout=24 * 3600, cpu=8, memory=32768)
def main(args: list):
    sys.path.insert(0, "/oll"); os.chdir("/oll")
    sys.path.insert(0, "/oll/gpu")
    from bigdata import build
    print("corpus", json.dumps(build("/vol/big")), flush=True); vol.commit()
    os.makedirs("/vol/results", exist_ok=True)
    p = subprocess.Popen([sys.executable, "gpu/main_train.py", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail, n = [], 0
    for line in p.stdout:
        print(line, end="", flush=True); tail = (tail + [line])[-200:]; n += 1
        if n % 5 == 0:  # progress.json is rewritten every 200 steps; make it visible to `fetch`
            vol.commit()
    p.wait(); vol.commit()
    return dict(returncode=p.returncode, tail="".join(tail[-60:]))


@app.function(image=image, volumes={"/vol": vol}, gpu=GPU, timeout=4 * 3600, cpu=8, memory=32768)
def caps(args: list):
    os.chdir("/oll")
    vol.reload()
    p = subprocess.run([sys.executable, "gpu/eval_caps.py", *args], capture_output=True, text=True)
    vol.commit()
    return dict(returncode=p.returncode, tail=(p.stdout + p.stderr)[-6000:])


@app.function(image=image, volumes={"/vol": vol}, timeout=600)
def read_results():
    vol.reload()
    out = {}
    for f in os.listdir("/vol/results") if os.path.exists("/vol/results") else []:
        out[f] = open(f"/vol/results/{f}").read()
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "caps":
        with modal.enable_output(), app.run(detach=True):
            c = caps.spawn(sys.argv[2:])
            print("SPAWNED", c.object_id)
    elif cmd == "start":
        with modal.enable_output(), app.run(detach=True):
            c = main.spawn(sys.argv[2:])
            print("SPAWNED", c.object_id)
    else:
        with app.run():
            res = read_results.remote()
        os.makedirs("runs/gpu/main", exist_ok=True)
        for f, txt in res.items():
            open(f"runs/gpu/main/{f}", "w").write(txt)
        print("fetched", list(res))
