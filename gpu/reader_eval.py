"""Evaluate the delivered Oll-Reader (oll/reader.py, models/oll-reader.pt) on all of SQuAD v1.1 validation."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from oll.reader import Reader, norm, f1
val = json.load(open(sys.argv[1])); out = {}
for unk in (False, True):
    r = Reader(unk_rare=unk); t0 = time.time(); em, fs, cf = [], [], []
    for q in val:
        o = r.ask(q["context"], q["question"])
        em.append(max(float(norm(o["answer"]) == norm(g)) for g in q["answers"])); fs.append(max(f1(o["answer"], g) for g in q["answers"])); cf.append(o["confidence"])
    em, fs, cf = map(np.array, (em, fs, cf))
    out["unseen_word_weight_" + ("rarest" if unk else "zero")] = dict(n=len(em), exact_match=float(em.mean()), f1=float(fs.mean()),
        by_confidence={f">={t}": dict(answered=float((cf >= t).mean()), exact_match=float(em[cf >= t].mean())) for t in (0.2, 0.4, 0.6, 0.8)}, seconds=time.time() - t0)
    print(json.dumps(out), flush=True)
json.dump(out, open("runs/chat/reader_eval.json", "w"), indent=1)
