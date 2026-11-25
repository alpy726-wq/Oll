"""Run a job file (gpu/jobs/*.json) on Modal and write every result to runs/gpu/<job>/<run>.json.
Job file: {"gpu": "B200"|"H100", "script": "real_train.py", "runs": {"name": ["--arg", "value", ...], ...}}"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modal
from modal_app import app, build_data, train_b200, train_h100

job_path = sys.argv[1]
job = json.load(open(job_path))
name = os.path.splitext(os.path.basename(job_path))[0]
out_dir = os.path.join("runs", "gpu", name)
os.makedirs(out_dir, exist_ok=True)
fn = {"B200": train_b200, "H100": train_h100}[job.get("gpu", "B200")]
with modal.enable_output(), app.run():
    print("corpus:", json.dumps(build_data.remote())[:300])
    calls = {k: fn.spawn(v, job.get("script", "real_train.py")) for k, v in job["runs"].items()}
    for k, c in calls.items():
        r = c.get()
        json.dump(r, open(os.path.join(out_dir, k + ".json"), "w"), indent=1)
        print(k, json.dumps({kk: r.get(kk) for kk in ("test", "val", "error", "gpu", "train_seconds")})[:800], flush=True)
