"""Oll-1 web app (Modal, scales to zero; a small GPU only while a request runs).
Upload a CSV (timestamp,value  or just value) -> forecast with 10-90% band + per-point surprise and alarms.
Deploy: modal deploy gpu/web_app.py"""
import base64, csv, io, os, sys
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.9.0", "numpy==2.3.4", "fastapi[standard]==0.115.6", "matplotlib==3.9.2", "python-multipart==0.0.20")
         .add_local_dir(ROOT, "/oll", ignore=["data/real", "data/text_dl", "data/robot", "runs", ".git", "**/__pycache__", "*.pt", "*.npz"]))
vol = modal.Volume.from_name("oll-main", create_if_missing=True)
app = modal.App("oll-web")

PAGE = """<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Oll-1</title><style>
:root{--bg:#fafaf8;--fg:#1c1c1c;--mut:#666;--card:#fff;--line:#e4e2dc;--acc:#2f5d8a}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--mut:#9a9a9a;--card:#1d1d1d;--line:#333;--acc:#7fb0e0}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:14px 0}
h1{font-size:26px;margin:0 0 4px}p.m{color:var(--mut);margin:0 0 10px}
button,input[type=submit]{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:10px 16px;font-size:15px;cursor:pointer}
a.b{display:inline-block;margin:4px 6px 4px 0;padding:8px 12px;border:1px solid var(--line);border-radius:8px;color:var(--fg);text-decoration:none;font-size:14px}
img{width:100%;height:auto;border-radius:6px;background:#fff}table{border-collapse:collapse;font-size:14px}td{padding:3px 10px 3px 0}
</style></head><body><main>
<h1>Oll-1</h1><p class="m">Sürekli zamanlı makine: sensör / zaman serisi tahmini (güven aralığıyla) ve anomali (arıza) alarmı.
Düzensiz zaman ve eksik ölçüm olabilir. Birim ve seviye fark etmez.</p>
<div class="card"><form action="/run" method="post" enctype="multipart/form-data">
<p><b>CSV yükle</b> — sütunlar: <code>zaman,değer</code> ya da sadece <code>değer</code></p>
<input type="file" name="file" accept=".csv,text/csv" required> <br><br>
Tahmin ufku (adım): <input type="number" name="horizon" value="96" min="1" max="500" style="width:80px">
&nbsp;<input type="submit" value="Çalıştır"></form></div>
<div class="card"><p><b>Gerçek örneklerle dene</b> (NAB gerçek verileri, etiketli arızalar):</p>
__DEMOS__</div>
__RESULT__
</main></body></html>"""

DEMOS = {"makine_sicakligi": "realKnownCause__machine_temperature_system_failure.csv",
         "ambient_sicaklik": "realKnownCause__ambient_temperature_system_failure.csv",
         "sunucu_cpu": "realAWSCloudwatch__ec2_cpu_utilization_825cc2.csv",
         "taksi_talebi": "realKnownCause__nyc_taxi.csv"}

_model = {}


def get_model():
    if "m" not in _model:
        sys.path.insert(0, "/oll")
        import torch
        from oll.core import load
        _model["m"] = load("/vol/results/oll-1.pt", "cuda" if torch.cuda.is_available() else "cpu")
    return _model["m"]


def parse_csv(text):
    import numpy as np
    rows = [r for r in csv.reader(io.StringIO(text)) if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError("boş dosya")
    try:
        float(rows[0][-1]); body = rows
    except ValueError:
        body = rows[1:]
    vals = np.array([float(r[-1]) for r in body])
    times = None
    if len(body[0]) >= 2:
        try:
            times = np.array([np.datetime64(r[0].strip().replace(" ", "T")[:19]) for r in body]).astype("datetime64[s]").astype(np.float64)
        except Exception:
            try:
                times = np.array([float(r[0]) for r in body])
            except Exception:
                times = None
    return vals, times


def run(vals, times, horizon, windows=None):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from oll.core import forecast, surprise, alarms
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = get_model()
    n_an = min(len(vals), 4000)
    v_an = vals[-n_an:]; t_an = None if times is None else times[-n_an:]
    sc = surprise(m, [(v_an, t_an)], device=dev, normal="calm")[0]
    fl = alarms(sc)
    fc = forecast(m, vals, times, horizon=horizon, paths=16, lookback=512, device=dev)
    out = []
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=110)
    hist = vals[-min(len(vals), 4 * horizon):]
    xh = np.arange(-len(hist), 0); xf = np.arange(horizon)
    ax.plot(xh, hist, color="#333", lw=1, label="geçmiş")
    ax.fill_between(xf, fc["q10"], fc["q90"], color="#2f5d8a", alpha=0.25, label="%10–90 aralık")
    ax.plot(xf, fc["median"], color="#2f5d8a", lw=1.6, label="tahmin (ortanca)")
    ax.axvline(0, color="#999", lw=0.8); ax.legend(loc="upper left", fontsize=8); ax.set_title("Tahmin", fontsize=11)
    fig.tight_layout(); b = io.BytesIO(); fig.savefig(b, format="png"); plt.close(fig); out.append(base64.b64encode(b.getvalue()).decode())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.6), dpi=110, sharex=True)
    x = np.arange(n_an)
    a1.plot(x, v_an, color="#333", lw=0.8)
    if windows is not None:
        for (i0, i1) in windows:
            a1.axvspan(i0, i1, color="#e0a030", alpha=0.25)
    a1.scatter(x[fl], v_an[fl], color="#c0392b", s=10, zorder=3, label="alarm")
    a1.set_title("Veri ve alarmlar" + (" (turuncu: etiketli gerçek arıza pencereleri)" if windows else ""), fontsize=11)
    a2.plot(x, np.nan_to_num(sc), color="#2f5d8a", lw=0.7); a2.set_title("Şaşırma (beklenmedik olma) puanı", fontsize=11)
    fig.tight_layout(); b = io.BytesIO(); fig.savefig(b, format="png"); plt.close(fig); out.append(base64.b64encode(b.getvalue()).decode())
    nalarm = int(sum(1 for i in range(1, len(fl)) if fl[i] and not fl[i - 1]) + int(fl[0]))
    table = "".join(f"<tr><td>+{k + 1}</td><td>{fc['median'][k]:.4g}</td><td>{fc['q10'][k]:.4g} – {fc['q90'][k]:.4g}</td></tr>" for k in range(min(horizon, 12)))
    return (f"<div class='card'><img src='data:image/png;base64,{out[0]}'><table><tr><td><b>adım</b></td><td><b>tahmin</b></td>"
            f"<td><b>%10–90</b></td></tr>{table}</table></div><div class='card'><p>{nalarm} alarm bölgesi "
            f"(son {n_an} ölçüm). Eşik: serinin ilk %15'indeki şaşırma ortalaması + 4 standart sapma.</p>"
            f"<p class='m'>DENEYSEL: alarm özelliği gerçek NAB arızalarında henüz zayıf (47 seri, 110 etiketli arıza: "
            f"arızaların %33'ü yakalandı, alarmların %16'sı gerçek arızaya denk geldi). Tahmin ve güven aralığı ölçülmüş ve güvenilirdir.</p>"
            f"<img src='data:image/png;base64,{out[1]}'></div>")


def page(result=""):
    demos = "".join(f"<a class='b' href='/demo/{k}'>{k.replace('_', ' ')}</a>" for k in DEMOS)
    return PAGE.replace("__DEMOS__", demos).replace("__RESULT__", result)


@app.function(image=image, volumes={"/vol": vol}, gpu="T4", scaledown_window=120, timeout=600)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse
    api = FastAPI()

    @api.get("/", response_class=HTMLResponse)
    def index():
        return page()

    @api.post("/run", response_class=HTMLResponse)
    async def run_csv(file: UploadFile = File(...), horizon: int = Form(96)):
        try:
            vals, times = parse_csv((await file.read()).decode("utf-8", errors="replace"))
            if len(vals) < 50:
                raise ValueError("en az 50 ölçüm gerekli")
            return page(run(vals, times, max(1, min(horizon, 500))))
        except Exception as e:
            return page(f"<div class='card'>Hata: {e}</div>")

    @api.get("/demo/{name}", response_class=HTMLResponse)
    def demo(name: str):
        sys.path.insert(0, "/oll")
        import numpy as np
        from oll.nab import load_nab
        S = {s["name"].replace("/", "__"): s for s in load_nab("/oll/data/nab")}
        s = S[DEMOS[name]]
        n = min(len(s["v"]), 4000); off = len(s["v"]) - n
        tt = s["t"][-n:]
        wins = [(int(np.searchsorted(tt, a)), int(np.searchsorted(tt, b))) for a, b in s["windows"]]
        return page(f"<div class='card'><b>{name.replace('_', ' ')}</b> — {s['name']}</div>" + run(s["v"], s["t"], 96, wins))

    @api.post("/api/forecast")
    async def api_forecast(file: UploadFile = File(...), horizon: int = Form(96)):
        from oll.core import forecast
        vals, times = parse_csv((await file.read()).decode())
        fc = forecast(get_model(), vals, times, horizon=horizon, device="cuda")
        return JSONResponse({k: v.tolist() for k, v in fc.items()})

    return api
