"""
Extended statistics for the DNR manuscript, addressing dependence structure,
detector specificity and the location of the breakpoint.

Stage 1  cluster-aware bootstrap (batch-pair clusters, analyte clusters)
Stage 2  detector-specific correlations
Stage 3  segmented regression with a free breakpoint

Run after analysis_main.py. Reads results/raw_results.csv.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

SEED = 20260822
B = 2000

d = pd.read_csv("results/raw_results.csv")
d["pair"] = d.src.astype(str) + "->" + d.tgt.astype(str)
u = d.drop_duplicates(subset=["pair", "unknown"]).reset_index(drop=True)

DETECTORS = {"Mahalanobis": "AUROC_maha", "kNN": "AUROC_knn", "MSP": "AUROC_msp"}
PREDICTORS = ["DNR_maha", "DNR_energy", "DNR_euclid", "MMD_ratio",
              "d_drift_maha", "d_novel_maha", "PAD_drift", "MMD_drift", "interval"]


# ---------------------------------------------------------------- bootstraps
def _rank_avg(a):
    """Average ranks, fully vectorized (ties handled as scipy.rankdata does)."""
    a = np.asarray(a, float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), int); inv[sorter] = np.arange(len(a))
    arr = a[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _rho(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    rx, ry = _rank_avg(x), _rank_avg(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return np.nan if den == 0 else float((rx * ry).sum() / den)


def cluster_boot(df, xcol, ycol, cluster_col, n_boot=B, seed=SEED):
    """Cluster bootstrap: resample whole clusters with replacement.

    The 95 rows are not independent. Each batch pair contributes up to six rows
    (one per withheld analyte) built from the same measurements, and each analyte
    recurs across pairs. Resampling clusters rather than rows keeps that
    dependence inside the resampling unit.
    """
    rs = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    idx_by_cluster = {c: df.index[df[cluster_col] == c].to_numpy() for c in clusters}
    obs = _rho(df[xcol], df[ycol])
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rs.choice(clusters, len(clusters), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in pick])
        boot[b] = _rho(df.loc[idx, xcol].to_numpy(), df.loc[idx, ycol].to_numpy())
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def iid_boot(df, xcol, ycol, n_boot=B, seed=SEED):
    rs = np.random.default_rng(seed)
    x, y = df[xcol].to_numpy(), df[ycol].to_numpy()
    obs = _rho(x, y)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        i = rs.integers(0, len(x), len(x))
        boot[b] = _rho(x[i], y[i])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


# ------------------------------------------------- Stage 3: segmented model
def segmented_fit(x, y, grid=None):
    """Least-squares two-segment continuous fit; breakpoint chosen by grid search."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if grid is None:
        grid = np.linspace(np.quantile(x, 0.10), np.quantile(x, 0.90), 81)
    best = (np.inf, None, None)
    for c in grid:
        X = np.column_stack([np.ones_like(x), x, np.maximum(x - c, 0.0)])
        XtX = X.T @ X + 1e-10 * np.eye(3)
        beta = np.linalg.solve(XtX, X.T @ y)
        rss = float(((y - X @ beta) ** 2).sum())
        if rss < best[0]:
            best = (rss, c, beta)
    rss, c, beta = best
    # linear (no breakpoint) comparison
    Xl = np.column_stack([np.ones_like(x), x])
    bl, *_ = np.linalg.lstsq(Xl, y, rcond=None)
    rss_lin = float(((y - Xl @ bl) ** 2).sum())
    return dict(breakpoint_log10=float(c), breakpoint_DNR=float(10 ** c),
                slope_left=float(beta[1]), slope_right=float(beta[1] + beta[2]),
                rss_segmented=rss, rss_linear=rss_lin,
                r2_gain=float(1 - rss / rss_lin))


def segmented_ci(df, xcol, ycol, cluster_col, n_boot=600, seed=SEED):
    rs = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    idx_by_cluster = {c: df.index[df[cluster_col] == c].to_numpy() for c in clusters}
    bps = []
    for b in range(n_boot):
        pick = rs.choice(clusters, len(clusters), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in pick])
        sub = df.loc[idx]
        if sub[xcol].nunique() < 10:
            continue
        try:
            bps.append(segmented_fit(np.log10(sub[xcol]), sub[ycol])["breakpoint_DNR"])
        except Exception:
            continue
    lo, hi = np.nanpercentile(bps, [2.5, 97.5])
    return float(lo), float(hi), len(bps)


def main():
    out = {"n_unique_splits": int(len(u)),
           "n_batch_pairs": int(u.pair.nunique()),
           "n_analytes": int(u.unknown.nunique())}

    # ---- Stage 1 + 2 -----------------------------------------------------
    tbl = {}
    for det_name, det in DETECTORS.items():
        tbl[det_name] = {}
        for p in PREDICTORS:
            r_i, lo_i, hi_i = iid_boot(u, p, det)
            r_p, lo_p, hi_p = cluster_boot(u, p, det, "pair")
            r_a, lo_a, hi_a = cluster_boot(u, p, det, "unknown")
            # report the widest (most conservative) of the two cluster schemes
            lo_c, hi_c = min(lo_p, lo_a), max(hi_p, hi_a)
            tbl[det_name][p] = dict(rho=round(r_i, 3),
                                    ci_iid=[round(lo_i, 3), round(hi_i, 3)],
                                    ci_pair=[round(lo_p, 3), round(hi_p, 3)],
                                    ci_analyte=[round(lo_a, 3), round(hi_a, 3)],
                                    ci_cluster=[round(lo_c, 3), round(hi_c, 3)])
        print(f"  {det_name} done")
    out["detector_specific"] = tbl

    # ---- Stage 3 ---------------------------------------------------------
    seg = segmented_fit(np.log10(u.DNR_maha), u.AUROC_maha)
    lo, hi, nb = segmented_ci(u, "DNR_maha", "AUROC_maha", "pair")
    seg["breakpoint_DNR_ci"] = [round(lo, 3), round(hi, 3)]
    seg["n_bootstrap_fits"] = nb
    out["segmented_regression"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in seg.items()}

    with open("results/extra_statistics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["segmented_regression"], indent=2))
    return out


if __name__ == "__main__":
    main()
