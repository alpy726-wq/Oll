"""Does the document talk about the question at all? Support = share of the question's (idf-weighted) meaning found in
the best sentence. SQuAD v1.1 validation questions with their own document (answerable) vs with a document of
another article (the answer is not there). AUROC, and the support threshold that keeps 95 % of answerable questions."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oll.reader import Reader
val = json.load(open(sys.argv[1])); r = Reader(); rng = np.random.default_rng(0)
idx = rng.choice(len(val), 2000, replace=False); own, other = [], []
for k in idx:
    q = val[k]; w = val[int(rng.integers(len(val)))]
    while w["title"] == q["title"] if "title" in w else w["context"] == q["context"]:
        w = val[int(rng.integers(len(val)))]
    own.append(r.ask(q["context"], q["question"])["support"]); other.append(r.ask(w["context"], q["question"])["support"])
own, other = np.array(own), np.array(other)
sc = np.concatenate([own, other]); lab = np.r_[np.ones(len(own)), np.zeros(len(other))].astype(bool)
rk = np.argsort(np.argsort(sc)) + 1.0
auroc = float((rk[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum()))
thr = float(np.quantile(own, 0.05))
res = dict(n=len(own), auroc=auroc, threshold_keep_95pct_answerable=thr, unanswerable_flagged=float((other < thr).mean()),
           support_own_median=float(np.median(own)), support_other_median=float(np.median(other)))
print(json.dumps(res)); json.dump(res, open("runs/chat/reader_support.json", "w"), indent=1)
