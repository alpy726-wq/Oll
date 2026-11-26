"""Choose the alarm rule WITHOUT tuning on the test labels: 2-fold over NAB series (even / odd index).
The rule is chosen on one half (best F1 of window recall and alarm precision) and scored on the other half.
Rule family ("surprise relative to usual surprise"): z = (s - mean of the previous W surprises) / their std,
smoothed over w points, alarm if z > k, after a warm-up of W points. Plus the fixed first-15% rule."""
import itertools, json, sys
import numpy as np
from oll.nab import load_nab, evaluate
from oll.core import alarms


def relative_alarms(scores, W=288, w=6, k=4.0):
    s = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0)
    c1 = np.concatenate([[0], np.cumsum(s)]); c2 = np.concatenate([[0], np.cumsum(s * s)])
    i = np.arange(len(s)); a = np.maximum(0, i - W)
    n = np.maximum(i - a, 1)
    mu = (c1[i] - c1[a]) / n; sd = np.sqrt(np.maximum((c2[i] - c2[a]) / n - mu * mu, 1e-9))
    z = (s - mu) / sd
    if w > 1:
        z = np.convolve(z, np.ones(w) / w, mode="full")[: len(z)]  # causal smoothing
    f = z > k
    f[:W] = False
    return f


def f1(r):
    p, q = r["alarm_precision"], r["window_recall"]
    return 0.0 if p + q == 0 else 2 * p * q / (p + q)


def main(scores_npz, root="data/nab", out="runs/nab_rule_selection.json"):
    S = load_nab(root)
    Z = np.load(scores_npz)
    sc = [Z[s["name"].replace("/", "__")] for s in S]
    grid = list(itertools.product((144, 288, 576), (1, 6, 24), (3.0, 4.0, 5.0, 6.0)))
    res = {"fixed_first15pct_rule_all": {k: v for k, v in evaluate(S, sc, alarms).items() if k != "per_series"}}
    folds = []
    for fold in (0, 1):
        tr = [i for i in range(len(S)) if i % 2 == fold]; te = [i for i in range(len(S)) if i % 2 != fold]
        best = max(grid, key=lambda g: f1(evaluate([S[i] for i in tr], [sc[i] for i in tr], lambda x: relative_alarms(x, *g))))
        r = evaluate([S[i] for i in te], [sc[i] for i in te], lambda x: relative_alarms(x, *best))
        folds.append(dict(chosen_on=f"series index %2=={fold}", rule=dict(W=best[0], w=best[1], k=best[2]),
                          test={k: v for k, v in r.items() if k != "per_series"}, test_f1=f1(r)))
    res["two_fold"] = folds
    tot = lambda key: sum(f["test"][key] for f in folds)
    res["two_fold_pooled_test"] = dict(windows_detected=tot("windows_detected"), labelled_windows=tot("labelled_windows"),
                                       window_recall=tot("windows_detected") / tot("labelled_windows"),
                                       alarm_runs=tot("alarm_runs"), alarm_runs_inside_windows=tot("alarm_runs_inside_windows"),
                                       alarm_precision=tot("alarm_runs_inside_windows") / max(tot("alarm_runs"), 1))
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
