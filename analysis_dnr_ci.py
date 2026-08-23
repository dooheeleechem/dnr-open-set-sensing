"""Full-pipeline bootstrap interval for the DNR of each split.

Every bootstrap iteration resamples the measurements within each class and then
re-fits the entire estimation chain on the resampled source known classes:
per-feature standardization, Ledoit-Wolf shrunk pooled within-class covariance,
class centroids, and finally the ratio. The interval therefore carries the
uncertainty of the metric as well as of the centroids, which the earlier
centroid-only version held fixed.

Usage:  python analysis_dnr_ci.py N     (process up to N splits not yet in the
output file; run repeatedly until every split is present)
"""
from __future__ import annotations
import sys, csv, os
import numpy as np
from pathlib import Path
import analysis_main as A

B = 400
OUT = "results/dnr_uncertainty_full.csv"
FIELDS = ["src", "tgt", "unknown", "unknown_name", "DNR", "lo", "hi", "n_boot"]


def boot_split(batches, src, tgt, unknown, rs):
    Xs, ys = batches[src]["X"], batches[src]["y"]
    Xt, yt = batches[tgt]["X"], batches[tgt]["y"]
    ok_s = {c for c in set(ys) if (ys == c).sum() >= A.MIN_PER_CLASS}
    ok_t = {c for c in set(yt) if (yt == c).sum() >= A.MIN_PER_CLASS}
    known = sorted((ok_s & ok_t) - {unknown})
    if unknown not in ok_t or len(known) < 3:
        return None
    src_pool = {c: Xs[ys == c] for c in known}
    tgt_pool = {c: Xt[yt == c] for c in known}
    unk_pool = Xt[yt == unknown]

    def one(sample):
        if sample:
            s = {c: src_pool[c][rs.integers(0, len(src_pool[c]), len(src_pool[c]))] for c in known}
            t = {c: tgt_pool[c][rs.integers(0, len(tgt_pool[c]), len(tgt_pool[c]))] for c in known}
            u = unk_pool[rs.integers(0, len(unk_pool), len(unk_pool))]
        else:
            s, t, u = src_pool, tgt_pool, unk_pool
        Xk = np.vstack([s[c] for c in known])
        yk = np.concatenate([np.full(len(s[c]), c) for c in known])
        scaler, P = A.fit_space(Xk, yk)                      # metric re-fitted here
        cs = {c: scaler.transform(s[c]).mean(0) for c in known}
        ct = {c: scaler.transform(t[c]).mean(0) for c in known}
        cu = scaler.transform(u).mean(0)
        dn = min(A.maha(cu, cs[c], P) for c in known)
        dd = float(np.mean([A.maha(cs[c], ct[c], P) for c in known]))
        return dd / dn if dn > 0 else np.nan

    point = one(False)
    vals = np.array([one(True) for _ in range(B)])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(src=src, tgt=tgt, unknown=unknown, unknown_name=A.GAS_NAMES[unknown],
                DNR=float(point), lo=float(lo), hi=float(hi), n_boot=int(len(vals)))


def already_done():
    if not os.path.exists(OUT):
        return set()
    import pandas as pd
    d = pd.read_csv(OUT)
    return {(int(a), int(b), int(c)) for a, b, c in zip(d.src, d.tgt, d.unknown)}


def main(limit):
    batches = A.load_all(Path("./data/Dataset"))
    settings = {"Setting 1": [(1, k) for k in range(2, 11)],
                "Setting 2": [(k, k + 1) for k in range(1, 10)]}
    combos, seen = [], set()
    for pairs in settings.values():
        for s, t in pairs:
            for u in sorted(A.GAS_NAMES):
                if (s, t, u) not in seen:
                    seen.add((s, t, u)); combos.append((s, t, u))
    done_set = already_done()
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        done = 0
        for i, (s, t, u) in enumerate(combos):
            if (s, t, u) in done_set or done >= limit:
                continue
            rs = np.random.default_rng(A.SEED + i)     # per-split seed, order independent
            r = boot_split(batches, s, t, u, rs)
            if r:
                w.writerow(r); fh.flush(); done += 1
                print(f"  [{i:3d}] {s}->{t} u={u}  DNR {r['DNR']:.3f} "
                      f"[{r['lo']:.3f}, {r['hi']:.3f}]", flush=True)
    print(f"{done} new splits written; {len(done_set)+done} total")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10**9)
