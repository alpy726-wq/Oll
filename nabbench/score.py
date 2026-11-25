"""Independent re-implementation of the Numenta Anomaly Benchmark (NAB v1.1) scoring:
windowed scaled-sigmoid reward, probationary period, one corpus-wide threshold optimised per profile,
normalised so that the null detector scores 0 and a perfect detector 100.
Validated by reproducing the published scores of existing detectors from their result files (see validate()).
"""
import csv, glob, json, math, os, sys
from datetime import datetime

NAB = os.environ.get("NAB_DIR", "/home/user/nab_upstream")
PROFILES = {"standard": (1.0, 1.0, 0.11), "reward_low_FP_rate": (1.0, 1.0, 0.22), "reward_low_FN_rate": (1.0, 2.0, 0.11)}  # tp, fn, fp


def ts(s):
    s = s.strip()
    return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")


def sig_scaled(pos):
    return -1.0 if pos > 3.0 else 2 / (1 + math.exp(5 * pos)) - 1.0


def sweep(times, scores, windows, name, tpw, fnw, fpw):
    """per-row (anomaly score, weighted sweep score, window name) exactly as nab.sweeper.calcSweepScore."""
    windows = list(windows)
    maxTP = sig_scaled(-1.0)
    prob = min(math.floor(0.15 * len(times)), 0.15 * 5000)
    index = {t: i for i, t in enumerate(times)}
    out = []
    cur = curname = curW = curR = prevW = prevR = None
    for i, (t, a) in enumerate(zip(times, scores)):
        if windows and t == windows[0][0]:
            cur = windows.pop(0)
            curname = f"{name}|{cur[0]}"
            curR = index[cur[1]]
            curW = float(curR - index[cur[0]] + 1)
        if cur is not None:
            w = sig_scaled(-(curR - i + 1) / curW) * tpw / maxTP
        else:
            u = -1.0 if prevR is None else sig_scaled(abs(prevR - i) / float(prevW - 1))
            w = u * fpw
        out.append((a, w, curname if i >= prob else "probationary"))
        if cur is not None and t == cur[1]:
            prevR, prevW = i, curW
            cur = curname = curW = curR = None
    return out


def best_threshold_score(rows, fnw):
    rows = sorted([r for r in rows if r[2] != "probationary"], key=lambda r: r[0], reverse=True)
    parts = {"fp": 0.0}
    for r in rows:
        if r[2] is not None:
            parts[r[2]] = -fnw
    total = sum(parts.values())
    best, best_th, th = total, 1.1, 1.1
    for a, w, n in rows:
        if a != th:
            if total > best:
                best, best_th = total, th
            th = a
        if n is None:
            parts["fp"] += w
            total += w
        else:
            if w > parts[n]:
                total += w - parts[n]
                parts[n] = w
    if total > best:
        best, best_th = total, th
    return best, best_th


def load_corpus():
    labels = json.load(open(os.path.join(NAB, "labels/combined_windows.json")))
    files = {}
    for rel, wins in labels.items():
        with open(os.path.join(NAB, "data", rel)) as f:
            r = csv.reader(f); next(r)
            rows = [(ts(a), float(b)) for a, b in r]
        files[rel] = dict(times=[t for t, _ in rows], values=[v for _, v in rows],
                          windows=[(ts(a), ts(b)) for a, b in wins])
    return files


def score_detector(get_scores, corpus=None):
    """get_scores(rel) -> list of anomaly scores in [0,1] for every row. Returns normalised scores per profile."""
    corpus = corpus or load_corpus()
    nwin = sum(len(f["windows"]) for f in corpus.values())
    res = {}
    allscores = {rel: get_scores(rel) for rel in corpus}
    for prof, (tpw, fnw, fpw) in PROFILES.items():
        rows = []
        for rel, f in corpus.items():
            rows += sweep(f["times"], allscores[rel], f["windows"], rel, tpw, fnw, fpw)
        raw, th = best_threshold_score(rows, fnw)
        null = -fnw * nwin
        res[prof] = dict(normalized=100 * (raw - null) / (nwin * tpw - null), raw=raw, threshold=th)
    return res


def result_file_scores(detector):
    def get(rel):
        d, fn = os.path.split(rel)
        with open(os.path.join(NAB, "results", detector, d, f"{detector}_{fn}")) as f:
            return [float(r["anomaly_score"]) for r in csv.DictReader(f)]
    return get


def validate(detectors=("numenta", "ARTime", "randomCutForest", "windowedGaussian", "random", "null")):
    corpus = load_corpus()
    pub = json.load(open(os.path.join(NAB, "results/final_results.json")))
    for d in detectors:
        r = score_detector(result_file_scores(d), corpus)
        print(d.ljust(18), " ".join(f"{p}: ours {r[p]['normalized']:6.2f} / published {pub[d][p]:6.2f}" for p in PROFILES), flush=True)


if __name__ == "__main__":
    validate()
