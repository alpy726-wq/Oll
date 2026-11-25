"""Run gpu/jobs/<name>.json of the form {"gpu": "L4", "cmds": {name: shell command}} on one Modal GPU in parallel;
writes every returned JSON file to runs/gpu/<job>/ (keeping its path below runs/)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modal
from modal_app import app, multi_l4

job_path = sys.argv[1]
job = json.load(open(job_path))
name = os.path.splitext(os.path.basename(job_path))[0]
with modal.enable_output(), app.run():
    r = {"L4": multi_l4}[job.get("gpu", "L4")].remote(job["cmds"])
for f, v in r["files"].items():
    p = os.path.join("runs", "gpu", name, os.path.relpath(f, "runs"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(v, open(p, "w"), indent=1)
for f, v in r.get("pts", {}).items():
    p = os.path.join("runs", "gpu", name, os.path.relpath(f, "runs"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(v)
os.makedirs(os.path.join("runs", "gpu", name), exist_ok=True)
json.dump(dict(gpu=r["gpu"], wall_seconds=r["wall_seconds"], logs=r["logs"]), open(os.path.join("runs", "gpu", name, "_logs.json"), "w"), indent=1)
print(r["gpu"], r["wall_seconds"], json.dumps({k: v[-600:] for k, v in r["logs"].items()}, indent=1))
