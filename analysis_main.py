"""
DNR (Drift-to-Novelty Ratio) analysis on the UCI Gas Sensor Array Drift Dataset.

Companion analysis to:
  Kim et al., Orthogonal Multimodal Sensing and AI Fusion for the Recognition of
  Unknown Chemical Threats: A Critical Review. Chemosensors 2026.

Research questions
  RQ1  How do d_novel and d_drift compare in a common metric space, and where do
       they cross (DNR = 1)?
  RQ2  Does DNR predict open-set performance?
  RQ3  How widely does DNR vary across the split families used in the literature
       (Setting 1: batch 1 -> K; Setting 2: batch K -> K+1)?
  RQ4  Does drift compensation lower DNR, and does it do so by shrinking d_drift
       alone or by also shrinking d_novel?

Data
  UCI Gas Sensor Array Drift Dataset, DOI 10.24432/C5RP6W (CC BY 4.0).
  Vergara et al., Sens. Actuators B 2012, 166-167, 320-329.
  Rodriguez-Lujan et al., Chemom. Intell. Lab. Syst. 2014, 130, 123-134.
  Expected layout:  <DATA_DIR>/batch1.dat ... batch10.dat
  Line format:      <class>;<concentration> 1:<v1> 2:<v2> ... 128:<v128>

Usage
  python analysis_main.py --data-dir ./data --out-dir ./results
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 20260822
rng = np.random.default_rng(SEED)

GAS_NAMES = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia",
             4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}
N_BATCHES = 10
MIN_PER_CLASS = 10   # a class with fewer samples gives an unusable centroid


# --------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------
def load_batch(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, conc) for one batch file."""
    X, y, conc = [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            head, *feats = line.split()
            # Two variants circulate: "<class>;<concentration>" and bare "<class>".
            if ";" in head:
                label, concentration = head.split(";")
            else:
                label, concentration = head, "nan"
            y.append(int(label))
            conc.append(float(concentration))
            X.append([float(tok.split(":")[1]) for tok in feats])
    return np.asarray(X, float), np.asarray(y, int), np.asarray(conc, float)


def load_all(data_dir: Path) -> dict[int, dict]:
    batches = {}
    for b in range(1, N_BATCHES + 1):
        f = data_dir / f"batch{b}.dat"
        if not f.exists():
            raise FileNotFoundError(f"missing {f}")
        X, y, c = load_batch(f)
        batches[b] = {"X": X, "y": y, "conc": c}
        print(f"  batch{b:<2d}  n={len(y):5d}  classes={sorted(set(y))}")
    return batches


# --------------------------------------------------------------------------
# 2. Distances
#    All distances are computed in ONE feature space fitted on the source
#    batch, so that d_novel and d_drift are directly comparable.
# --------------------------------------------------------------------------
def fit_space(X_src: np.ndarray, y_src: np.ndarray):
    """Standardizer + pooled within-class covariance (Ledoit-Wolf shrinkage)."""
    scaler = StandardScaler().fit(X_src)
    Z = scaler.transform(X_src)
    centred = np.vstack([Z[y_src == c] - Z[y_src == c].mean(0)
                         for c in np.unique(y_src)])
    cov = LedoitWolf().fit(centred)
    P = np.linalg.inv(cov.covariance_)          # precision matrix
    return scaler, P


def maha(a: np.ndarray, b: np.ndarray, P: np.ndarray) -> float:
    d = np.atleast_1d(a - b)
    return float(np.sqrt(max(d @ P @ d.T, 0.0)))


def energy_distance(A: np.ndarray, B: np.ndarray, max_n: int = 400) -> float:
    """Two-sample energy distance; subsampled for tractability."""
    if len(A) > max_n:
        A = A[rng.choice(len(A), max_n, replace=False)]
    if len(B) > max_n:
        B = B[rng.choice(len(B), max_n, replace=False)]
    ab = cdist(A, B).mean()
    aa = cdist(A, A).mean()
    bb = cdist(B, B).mean()
    return float(max(2 * ab - aa - bb, 0.0))


def centroid(Z: np.ndarray) -> np.ndarray:
    return Z.mean(0)


# --------------------------------------------------------------------------
# 3. DNR
# --------------------------------------------------------------------------
def compute_dnr(batches, src: int, tgt: int, unknown: int, metric: str = "maha"):
    """
    Both displacements are measured FROM the source known-class manifold, in the
    space fitted on the source known classes, because that manifold is what a
    detector trained on the source actually holds.

      d_drift : mean over known classes of || centroid_target(c) - centroid_source(c) ||
                i.e. how far a KNOWN class has moved by test time.
      d_novel : min over known classes of || centroid_target(unknown) - centroid_source(c) ||
                i.e. how far the UNKNOWN class sits from the nearest known class.

    DNR = d_drift / d_novel. DNR >= 1 means a known analyte has been displaced as
    far as an unknown one, so no score defined on that manifold can separate them.

    Measuring the unknown class in the TARGET batch (not the source) matters: it
    is the only definition under which drift compensation, which acts on the
    target, can move d_novel as well as d_drift.
    """
    Xs, ys = batches[src]["X"], batches[src]["y"]
    Xt, yt = batches[tgt]["X"], batches[tgt]["y"]
    ok_s = {c for c in set(ys) if (ys == c).sum() >= MIN_PER_CLASS}
    ok_t = {c for c in set(yt) if (yt == c).sum() >= MIN_PER_CLASS}
    known = sorted((ok_s & ok_t) - {unknown})
    if unknown not in ok_t or len(known) < 3:
        return None

    scaler, P = fit_space(Xs[np.isin(ys, known)], ys[np.isin(ys, known)])
    Zs, Zt = scaler.transform(Xs), scaler.transform(Xt)

    cen_s = {c: centroid(Zs[ys == c]) for c in known}
    cen_t = {c: centroid(Zt[yt == c]) for c in known}
    cen_u = centroid(Zt[yt == unknown])          # unknown observed at test time

    if metric == "maha":
        d_novel = min(maha(cen_u, cen_s[c], P) for c in known)
        drifts = [maha(cen_s[c], cen_t[c], P) for c in known]
    elif metric == "euclid":
        d_novel = min(float(np.linalg.norm(cen_u - cen_s[c])) for c in known)
        drifts = [float(np.linalg.norm(cen_s[c] - cen_t[c])) for c in known]
    elif metric == "energy":
        d_novel = min(energy_distance(Zt[yt == unknown], Zs[ys == c]) for c in known)
        drifts = [energy_distance(Zs[ys == c], Zt[yt == c]) for c in known]
    else:
        raise ValueError(metric)

    d_drift = float(np.mean(drifts))
    return {"d_novel": d_novel, "d_drift": d_drift,
            "d_drift_max": float(np.max(drifts)),
            "DNR": d_drift / d_novel if d_novel > 0 else np.nan,
            "metric": metric}


# --------------------------------------------------------------------------
# 4. Open-set scoring baselines (deliberately simple and reproducible)
# --------------------------------------------------------------------------
def openset_auroc(batches, src: int, tgt: int, unknown: int) -> dict[str, float]:
    """Train on KNOWN classes of source; score target samples; AUROC for unknown."""
    Xs, ys = batches[src]["X"], batches[src]["y"]
    Xt, yt = batches[tgt]["X"], batches[tgt]["y"]
    ok_s = {c for c in set(ys) if (ys == c).sum() >= MIN_PER_CLASS}
    ok_t = {c for c in set(yt) if (yt == c).sum() >= MIN_PER_CLASS}
    known = sorted((ok_s & ok_t) - {unknown})
    if unknown not in ok_t or len(known) < 3:
        return {}

    m = np.isin(ys, known)
    scaler, P = fit_space(Xs[m], ys[m])
    Zs, Zt = scaler.transform(Xs[m]), scaler.transform(Xt)
    ys_k = ys[m]
    is_unknown = (yt == unknown).astype(int)

    cen = {c: centroid(Zs[ys_k == c]) for c in known}

    # (a) minimum Mahalanobis distance to a known centroid
    s_maha = np.array([min(maha(z, cen[c], P) for c in known) for z in Zt])
    # (b) k-nearest-neighbour distance to the known training set
    D = cdist(Zt, Zs)
    s_knn = np.sort(D, axis=1)[:, :5].mean(1)
    # (c) softmax-confidence complement of a linear discriminant
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=3000, random_state=SEED).fit(Zs, ys_k)
    s_msp = 1.0 - clf.predict_proba(Zt).max(1)

    return {"AUROC_maha": roc_auc_score(is_unknown, s_maha),
            "AUROC_knn": roc_auc_score(is_unknown, s_knn),
            "AUROC_msp": roc_auc_score(is_unknown, s_msp)}



# --------------------------------------------------------------------------
# 4b. Competing domain-shift measures (reviewer defence)
#     MMD and proxy A-distance are the standard descriptors of domain shift.
#     They are magnitudes, not ratios; the comparison tests whether the
#     predictive power of DNR comes from the distance or from the ratio.
# --------------------------------------------------------------------------
def _sub(A, n=250):
    return A[rng.choice(len(A), n, replace=False)] if len(A) > n else A


def mmd2_rbf(A: np.ndarray, B: np.ndarray) -> float:
    """Unbiased MMD^2 with an RBF kernel, median-heuristic bandwidth."""
    A, B = _sub(A), _sub(B)
    Z = np.vstack([A, B])
    d = cdist(Z, Z)
    sigma = np.median(d[d > 0]) if (d > 0).any() else 1.0
    g = 1.0 / (2 * sigma ** 2)
    Kaa, Kbb, Kab = (np.exp(-g * cdist(A, A) ** 2),
                     np.exp(-g * cdist(B, B) ** 2),
                     np.exp(-g * cdist(A, B) ** 2))
    na, nb = len(A), len(B)
    np.fill_diagonal(Kaa, 0.0)
    np.fill_diagonal(Kbb, 0.0)
    return float(Kaa.sum() / (na * (na - 1)) + Kbb.sum() / (nb * (nb - 1))
                 - 2 * Kab.mean())


def proxy_a_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    PAD = 2(1 - 2 eps), with eps the held-out error of a source/target
    discriminator (Ben-David et al.). A single stratified 50/50 split is used and
    the liblinear solver is chosen because the two domains are close to linearly
    separable in 128 dimensions, which makes lbfgs run to its iteration cap.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    A, B = _sub(A), _sub(B)
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y,
                                          random_state=SEED)
    clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=200).fit(Xtr, ytr)
    return float(2 * (2 * clf.score(Xte, yte) - 1))


def shift_measures(batches, src: int, tgt: int, unknown: int) -> dict[str, float]:
    Xs, ys = batches[src]["X"], batches[src]["y"]
    Xt, yt = batches[tgt]["X"], batches[tgt]["y"]
    ok_s = {c for c in set(ys) if (ys == c).sum() >= MIN_PER_CLASS}
    ok_t = {c for c in set(yt) if (yt == c).sum() >= MIN_PER_CLASS}
    known = sorted((ok_s & ok_t) - {unknown})
    if unknown not in ok_t or len(known) < 3:
        return {}
    scaler, _ = fit_space(Xs[np.isin(ys, known)], ys[np.isin(ys, known)])
    Zs_k = scaler.transform(Xs[np.isin(ys, known)])
    Zt_k = scaler.transform(Xt[np.isin(yt, known)])
    Zt_u = scaler.transform(Xt[yt == unknown])

    mmd_drift = mmd2_rbf(Zs_k, Zt_k)      # domain shift, knowns only
    mmd_novel = mmd2_rbf(Zs_k, Zt_u)      # separation of the unknown
    return {"MMD_drift": mmd_drift,
            "MMD_novel": mmd_novel,
            "MMD_ratio": mmd_drift / mmd_novel if mmd_novel > 0 else np.nan,
            "PAD_drift": proxy_a_distance(Zs_k, Zt_k)}


# --------------------------------------------------------------------------
# 5. Drift compensation (RQ4) — mean-shift / CORAL-style alignment
# --------------------------------------------------------------------------
def compensate(batches, src: int, tgt: int, mode: str = "mean"):
    """Return a copy of the target batch aligned to the source batch."""
    Xs, Xt = batches[src]["X"], batches[tgt]["X"]
    if mode == "mean":
        Xt2 = Xt - Xt.mean(0) + Xs.mean(0)
    elif mode == "coral":
        Cs = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1])
        Ct = np.cov(Xt, rowvar=False) + np.eye(Xt.shape[1])
        def isqrt(M):
            w, V = np.linalg.eigh(M)
            return V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-10))) @ V.T
        def sqrt(M):
            w, V = np.linalg.eigh(M)
            return V @ np.diag(np.sqrt(np.maximum(w, 0))) @ V.T
        Xt2 = (Xt - Xt.mean(0)) @ isqrt(Ct) @ sqrt(Cs) + Xs.mean(0)
    else:
        raise ValueError(mode)
    out = dict(batches)
    out[tgt] = {**batches[tgt], "X": Xt2}
    return out


# --------------------------------------------------------------------------
# 6. Main sweep
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--out-dir", type=Path, default=Path("./results"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading UCI Gas Sensor Array Drift Dataset")
    batches = load_all(args.data_dir)

    settings = {
        "Setting 1": [(1, k) for k in range(2, N_BATCHES + 1)],
        "Setting 2": [(k, k + 1) for k in range(1, N_BATCHES)],
    }

    rows = []
    for setting, pairs in settings.items():
        for src, tgt in pairs:
            for unknown in sorted(GAS_NAMES):
                base = {"setting": setting, "src": src, "tgt": tgt,
                        "interval": tgt - src, "unknown": unknown,
                        "unknown_name": GAS_NAMES[unknown],
                        "n_src": int(len(batches[src]["y"])),
                        "n_tgt": int(len(batches[tgt]["y"])),
                        "n_unknown_tgt": int((batches[tgt]["y"] == unknown).sum())}
                for metric in ("maha", "euclid", "energy"):
                    d = compute_dnr(batches, src, tgt, unknown, metric)
                    if d is None:
                        break
                    base[f"d_novel_{metric}"] = d["d_novel"]
                    base[f"d_drift_{metric}"] = d["d_drift"]
                    base[f"DNR_{metric}"] = d["DNR"]
                else:
                    base.update(openset_auroc(batches, src, tgt, unknown))
                    base.update(shift_measures(batches, src, tgt, unknown))
                    for mode in ("mean", "coral"):
                        comp = compensate(batches, src, tgt, mode)
                        dc = compute_dnr(comp, src, tgt, unknown, "maha")
                        ac = openset_auroc(comp, src, tgt, unknown)
                        if dc:
                            base[f"DNR_maha_{mode}"] = dc["DNR"]
                            base[f"d_novel_maha_{mode}"] = dc["d_novel"]
                            base[f"d_drift_maha_{mode}"] = dc["d_drift"]
                        base[f"AUROC_maha_{mode}"] = ac.get("AUROC_maha", np.nan)
                    rows.append(base)
                    print(f"  {setting}  {src}->{tgt}  unknown={GAS_NAMES[unknown]:<12s}"
                          f"  DNR={base.get('DNR_maha', float('nan')):.3f}"
                          f"  AUROC={base.get('AUROC_maha', float('nan')):.3f}")

    df = pd.DataFrame(rows)
    df["pair"] = df.src.astype(str) + "->" + df.tgt.astype(str)
    df.to_csv(args.out_dir / "raw_results.csv", index=False)

    # The pair 1->2 is simultaneously Setting 1 (K=2) and Setting 2 (K=1); the
    # literature reports it twice as well. Deduplicate before pooled statistics.
    n_dup = int(df.duplicated(subset=["pair", "unknown"]).sum())
    df = df.drop_duplicates(subset=["pair", "unknown"]).reset_index(drop=True)

    # ---- summary metrics -------------------------------------------------
    summary = {
        "seed": SEED,
        "n_splits_unique": int(len(df)),
        "n_duplicate_rows_removed": n_dup,
        "frac_AUROC_below_chance": float((df.AUROC_maha < 0.5).mean()),
        "DNR_maha": {
            "min": float(df.DNR_maha.min()), "max": float(df.DNR_maha.max()),
            "median": float(df.DNR_maha.median()),
            "frac_above_1": float((df.DNR_maha > 1).mean()),
        },
        "DNR_by_setting": df.groupby("setting").DNR_maha
                            .agg(["min", "median", "max"]).to_dict(),
        "spearman_DNR_vs_AUROC": {
            k: float(df[["DNR_maha", k]].corr(method="spearman").iloc[0, 1])
            for k in ("AUROC_maha", "AUROC_knn", "AUROC_msp")
        },
        "spearman_competing_predictors_vs_AUROC_maha": {
            k: float(df[[k, "AUROC_maha"]].corr(method="spearman").iloc[0, 1])
            for k in ("DNR_maha", "DNR_euclid", "DNR_energy", "MMD_ratio",
                      "MMD_drift", "MMD_novel", "PAD_drift", "d_drift_maha",
                      "d_novel_maha", "interval")
        },
        "PAD_saturation_frac_above_1p95": float((df.PAD_drift > 1.95).mean()),
        "spearman_interval_vs_AUROC": float(
            df[["interval", "AUROC_maha"]].corr(method="spearman").iloc[0, 1]),
        "compensation_effect": {
            mode: {
                "median_DNR_before": float(df.DNR_maha.median()),
                "median_DNR_after": float(df[f"DNR_maha_{mode}"].median()),
                "median_d_drift_ratio": float(
                    (df[f"d_drift_maha_{mode}"] / df.d_drift_maha).median()),
                "median_d_novel_ratio": float(
                    (df[f"d_novel_maha_{mode}"] / df.d_novel_maha).median()),
                "median_AUROC_before": float(df.AUROC_maha.median()),
                "median_AUROC_after": float(df[f"AUROC_maha_{mode}"].median()),
            } for mode in ("mean", "coral")
        },
    }
    with open(args.out_dir / "experiment_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n--- summary ---")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out_dir/'raw_results.csv'} and experiment_summary.json")


if __name__ == "__main__":
    main()
