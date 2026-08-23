"""
Stage 4  per-split sampling uncertainty of DNR
Stage 5  drift compensation with and without the unknown class in the alignment

Reads the raw dataset; writes results/extra_statistics2.json.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from pathlib import Path
import analysis_main as A

SEED = A.SEED
B_SPLIT = 400
rs = np.random.default_rng(SEED)


def dnr_bootstrap(batches, src, tgt, unknown, n_boot=B_SPLIT):
    """Resample measurements within each class to propagate centroid uncertainty.

    The metric (standardizer and precision matrix) is held fixed at its
    protocol-defined value, estimated once from the source known classes; only
    the centroids are resampled. The interval therefore expresses how precisely
    the displacement of each class is known, which is the quantity that decides
    whether a split sits above or below the crossover.
    """
    Xs, ys = batches[src]["X"], batches[src]["y"]
    Xt, yt = batches[tgt]["X"], batches[tgt]["y"]
    ok_s = {c for c in set(ys) if (ys == c).sum() >= A.MIN_PER_CLASS}
    ok_t = {c for c in set(yt) if (yt == c).sum() >= A.MIN_PER_CLASS}
    known = sorted((ok_s & ok_t) - {unknown})
    if unknown not in ok_t or len(known) < 3:
        return None
    m = np.isin(ys, known)
    scaler, P = A.fit_space(Xs[m], ys[m])
    Zs, Zt = scaler.transform(Xs), scaler.transform(Xt)
    src_pools = {c: Zs[ys == c] for c in known}
    tgt_pools = {c: Zt[yt == c] for c in known}
    unk_pool = Zt[yt == unknown]

    def one(sample):
        cs = {c: (src_pools[c][rs.integers(0, len(src_pools[c]), len(src_pools[c]))]
                  if sample else src_pools[c]).mean(0) for c in known}
        ct = {c: (tgt_pools[c][rs.integers(0, len(tgt_pools[c]), len(tgt_pools[c]))]
                  if sample else tgt_pools[c]).mean(0) for c in known}
        cu = (unk_pool[rs.integers(0, len(unk_pool), len(unk_pool))]
              if sample else unk_pool).mean(0)
        dn = min(A.maha(cu, cs[c], P) for c in known)
        dd = float(np.mean([A.maha(cs[c], ct[c], P) for c in known]))
        return dd / dn if dn > 0 else np.nan

    point = one(False)
    boot = np.array([one(True) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return dict(DNR=float(point), lo=float(lo), hi=float(hi))


def compensate_known_only(batches, src, tgt, unknown, mode):
    """Alignment statistics estimated from KNOWN target samples only.

    This is an oracle variant: it requires knowing which target samples are
    known, which a deployed detector does not. It is included to separate the
    effect of compensation itself from the effect of letting unknown samples
    influence the transformation.
    """
    Xs, Xt = batches[src]["X"], batches[tgt]["X"]
    yt = batches[tgt]["y"]
    ref = Xt[yt != unknown]
    if mode == "mean":
        Xt2 = Xt - ref.mean(0) + Xs.mean(0)
    else:
        Cs = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1])
        Ct = np.cov(ref, rowvar=False) + np.eye(Xs.shape[1])
        def isqrt(M):
            w, V = np.linalg.eigh(M); return V @ np.diag(1/np.sqrt(np.maximum(w,1e-10))) @ V.T
        def sqrt(M):
            w, V = np.linalg.eigh(M); return V @ np.diag(np.sqrt(np.maximum(w,0))) @ V.T
        Xt2 = (Xt - ref.mean(0)) @ isqrt(Ct) @ sqrt(Cs) + Xs.mean(0)
    out = dict(batches); out[tgt] = {**batches[tgt], "X": Xt2}
    return out


def main():
    batches = A.load_all(Path("./data/Dataset"))
    settings = {"Setting 1": [(1, k) for k in range(2, 11)],
                "Setting 2": [(k, k + 1) for k in range(1, 10)]}
    seen, rows = set(), []
    for setting, pairs in settings.items():
        for src, tgt in pairs:
            for unk in sorted(A.GAS_NAMES):
                if (src, tgt, unk) in seen:
                    continue
                ci = dnr_bootstrap(batches, src, tgt, unk)
                if ci is None:
                    continue
                seen.add((src, tgt, unk))
                r = dict(setting=setting, src=src, tgt=tgt, unknown=unk,
                         unknown_name=A.GAS_NAMES[unk], **ci)
                for mode in ("mean", "coral"):
                    ko = compensate_known_only(batches, src, tgt, unk, mode)
                    dk = A.compute_dnr(ko, src, tgt, unk, "maha")
                    ak = A.openset_auroc(ko, src, tgt, unk)
                    r[f"DNR_{mode}_knownonly"] = dk["DNR"] if dk else np.nan
                    r[f"dnovel_{mode}_knownonly"] = dk["d_novel"] if dk else np.nan
                    r[f"ddrift_{mode}_knownonly"] = dk["d_drift"] if dk else np.nan
                    r[f"AUROC_{mode}_knownonly"] = ak.get("AUROC_maha", np.nan)
                rows.append(r)
        print(f"  {setting} done")
    df = pd.DataFrame(rows)
    df.to_csv("results/dnr_uncertainty.csv", index=False)

    base = pd.read_csv("results/raw_results.csv")
    base["pair"] = base.src.astype(str) + "->" + base.tgt.astype(str)
    base = base.drop_duplicates(subset=["pair", "unknown"])
    mg = df.merge(base[["src","tgt","unknown","d_novel_maha","d_drift_maha",
                        "d_novel_maha_mean","d_drift_maha_mean",
                        "d_novel_maha_coral","d_drift_maha_coral",
                        "DNR_maha_mean","DNR_maha_coral","AUROC_maha_mean","AUROC_maha_coral"]],
                  on=["src","tgt","unknown"])
    out = {
        "n_splits": int(len(df)),
        "dnr_uncertainty": {
            "definitely_below_1": int((df.hi < 1).sum()),
            "uncertain_spans_1": int(((df.lo <= 1) & (df.hi >= 1)).sum()),
            "definitely_above_1": int((df.lo > 1).sum()),
            "median_relative_width": float(((df.hi - df.lo) / df.DNR).median()),
        },
        "compensation_unknown_contamination": {
            mode: {
                "median_DNR_all_target": float(mg[f"DNR_maha_{mode}"].median()),
                "median_DNR_known_only": float(mg[f"DNR_{mode}_knownonly"].median()),
                "median_dnovel_ratio_all_target": float((mg[f"d_novel_maha_{mode}"]/mg.d_novel_maha).median()),
                "median_dnovel_ratio_known_only": float((mg[f"dnovel_{mode}_knownonly"]/mg.d_novel_maha).median()),
                "median_ddrift_ratio_all_target": float((mg[f"d_drift_maha_{mode}"]/mg.d_drift_maha).median()),
                "median_ddrift_ratio_known_only": float((mg[f"ddrift_{mode}_knownonly"]/mg.d_drift_maha).median()),
                "median_AUROC_all_target": float(mg[f"AUROC_maha_{mode}"].median()),
                "median_AUROC_known_only": float(mg[f"AUROC_{mode}_knownonly"].median()),
            } for mode in ("mean", "coral")
        },
    }
    with open("results/extra_statistics2.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
