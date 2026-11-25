"""Real-world corpus for OneMachine (no synthetic data).

Every source is a real recording or real text. Splits are fixed by time, file or speaker, and frozen: the manifest
stores a SHA-256 per split file. Test splits are never used for training, tuning or model selection.

  sensors  : Jena climate 2009-2016 (Max Planck, 10-min, 14 variables) and ETT transformer data (hourly / 15-min)
  finance  : VIX daily OHLC 1990-, S&P 500 monthly 1871-
  audio    : Free Spoken Digit Dataset (3000 real recordings, 6 speakers, 8 kHz) with the spoken digit as a label
  text     : Project Gutenberg books (NLTK) and the Brown corpus
  facts    : ISO country table (country, capital, currency, dial code, continent)   -- evaluation of memory only
  reasoning: ARC (Chollet) training 400 / evaluation 400 tasks
  anomaly  : NAB 58 real series -- evaluation only, never trained on
"""
import csv, glob, hashlib, io, json, os, wave, zipfile
from datetime import datetime
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "real")
OUT = os.path.join(ROOT, "v4")
os.makedirs(OUT, exist_ok=True)
manifest = {}


def save(name, **arrays):
    path = os.path.join(OUT, name + ".npz")
    np.savez_compressed(path, **arrays)
    manifest[name] = dict(sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
                          shapes={k: list(np.asarray(v).shape) for k, v in arrays.items()})


def epoch(dt):
    return dt.timestamp()


# ---------------------------------------------------------------- sensors: Jena climate (split by year)
rows = list(csv.reader(open(os.path.join(RAW, "jena_climate_2009_2016.csv"))))
head, rows = rows[0], rows[1:]
t = np.array([epoch(datetime.strptime(r[0], "%d.%m.%Y %H:%M:%S")) for r in rows])
X = np.array([[float(v) for v in r[1:]] for r in rows])
X[X == -9999.0] = np.nan  # documented sensor faults in wv / max. wv
years = np.array([datetime.fromtimestamp(v).year for v in t])
for split, mask in (("train", years <= 2014), ("val", years == 2015), ("test", years == 2016)):
    save(f"jena_{split}", t=t[mask], x=X[mask], names=np.array(head[1:]))

# ---------------------------------------------------------------- sensors: ETT (standard 12/4/4 month split)
for f in ("ETTh1", "ETTh2", "ETTm1"):
    rows = list(csv.reader(open(os.path.join(RAW, f + ".csv"))))
    head, rows = rows[0], rows[1:]
    t = np.array([epoch(datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S")) for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows])
    per_month = 30 * 24 * (4 if f.endswith("m1") else 1)
    a, b = 12 * per_month, 16 * per_month
    for split, sl in (("train", slice(0, a)), ("val", slice(a, b)), ("test", slice(b, b + 4 * per_month))):
        save(f"{f}_{split}", t=t[sl], x=X[sl], names=np.array(head[1:]))

# ---------------------------------------------------------------- finance
rows = list(csv.reader(open(os.path.join(RAW, "vix.csv"))))[1:]
t = np.array([epoch(datetime.strptime(r[0], "%Y-%m-%d")) for r in rows]); X = np.array([[float(v) for v in r[1:]] for r in rows])
yr = np.array([datetime.fromtimestamp(v).year for v in t])
for split, mask in (("train", yr <= 2015), ("val", (yr >= 2016) & (yr <= 2018)), ("test", yr >= 2019)):
    save(f"vix_{split}", t=t[mask], x=X[mask], names=np.array(["open", "high", "low", "close"]))
rows = list(csv.reader(open(os.path.join(RAW, "sp500.csv"))))[1:]
rows = [r for r in rows if float(r[1]) > 0]
t = np.array([epoch(datetime.strptime(r[0], "%Y-%m-%d")) for r in rows]); X = np.array([[float(r[1])] for r in rows])
yr = np.array([datetime.fromtimestamp(v).year for v in t])
for split, mask in (("train", yr <= 1990), ("val", (yr > 1990) & (yr <= 2005)), ("test", yr > 2005)):
    save(f"sp500_{split}", t=t[mask], x=X[mask], names=np.array(["sp500"]))

# ---------------------------------------------------------------- audio: FSDD (standard split: take 0-4 of every digit/speaker = test)
wav_train, lab_train, spk_train, wav_test, lab_test, spk_test = [], [], [], [], [], []
for f in sorted(glob.glob(os.path.join(RAW, "fsdd/recordings/*.wav"))):
    digit, speaker, idx = os.path.basename(f)[:-4].split("_")
    w = wave.open(f)
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16 if w.getsampwidth() == 2 else np.uint8).astype(np.float32)
    a = a / 32768.0 if w.getsampwidth() == 2 else (a - 128) / 128.0
    (wav_test if int(idx) < 5 else wav_train).append(a)
    (lab_test if int(idx) < 5 else lab_train).append(int(digit))
    (spk_test if int(idx) < 5 else spk_train).append(speaker)
for split, W, Lb, S in (("train", wav_train, lab_train, spk_train), ("test", wav_test, lab_test, spk_test)):
    lens = np.array([len(w) for w in W])
    save(f"fsdd_{split}", audio=np.concatenate(W), lengths=lens, label=np.array(Lb), speaker=np.array(S), rate=np.array(8000))

# ---------------------------------------------------------------- text: Gutenberg (held-out books) + Brown (held-out files)
zg = zipfile.ZipFile(os.path.join(ROOT, "text_dl/gutenberg.zip"))
test_books = {"gutenberg/carroll-alice.txt", "gutenberg/austen-persuasion.txt", "gutenberg/chesterton-thursday.txt"}
tr, te = [], []
for n in sorted(zg.namelist()):
    if n.endswith(".txt") and "README" not in n:
        (te if n in test_books else tr).append(zg.read(n))
zb = zipfile.ZipFile(os.path.join(ROOT, "text_dl/brown.zip"))
bn = sorted(n for n in zb.namelist() if n.startswith("brown/c") and len(os.path.basename(n)) == 4)
for i, n in enumerate(bn):  # every 10th Brown file is test
    raw = zb.read(n).decode("latin-1")
    words = [w.rsplit("/", 1)[0] for w in raw.split()]
    (te if i % 10 == 0 else tr).append(" ".join(words).encode("utf-8", errors="ignore"))
save("text_train", bytes=np.frombuffer(b"\n".join(tr), dtype=np.uint8))
save("text_test", bytes=np.frombuffer(b"\n".join(te), dtype=np.uint8))

# ---------------------------------------------------------------- facts (evaluation of the editable memory)
facts = []
for r in csv.DictReader(open(os.path.join(RAW, "country-codes.csv"), encoding="utf-8")):
    name = r.get("CLDR display name") or r.get("UNTERM English Short") or ""
    cap = r.get("Capital", "")
    if name and cap:
        facts.append(dict(country=name, capital=cap, currency=r.get("ISO4217-currency_alphabetic_code", ""),
                          dial=r.get("Dial", ""), continent=r.get("Continent", "")))
json.dump(facts, open(os.path.join(OUT, "facts.json"), "w"), ensure_ascii=False, indent=0)
manifest["facts"] = dict(sha256=hashlib.sha256(open(os.path.join(OUT, "facts.json"), "rb").read()).hexdigest(), n=len(facts))

# ---------------------------------------------------------------- reasoning: ARC
for split, d in (("train", "training"), ("test", "evaluation")):
    tasks = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(RAW, f"arc/data/{d}/*.json")))}
    json.dump(tasks, open(os.path.join(OUT, f"arc_{split}.json"), "w"))
    manifest[f"arc_{split}"] = dict(sha256=hashlib.sha256(open(os.path.join(OUT, f"arc_{split}.json"), "rb").read()).hexdigest(), n=len(tasks))

json.dump(manifest, open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=1)
print(json.dumps({k: v.get("shapes", v.get("n")) for k, v in manifest.items()}, indent=0))
