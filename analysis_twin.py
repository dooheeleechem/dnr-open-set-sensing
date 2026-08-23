"""External replication on the Twin Gas Sensor Arrays dataset (UCI 361).

The shift mechanism differs from the one in the main analysis. There the
domains are batches separated in time and the shift is temporal drift. Here the
domains are five nominally identical eight-sensor boards recorded in the same
campaign, so the shift is unit-to-unit variation. If the drift-to-novelty ratio
describes a geometric property rather than an artifact of temporal drift, the
association with open-set performance should reappear.
"""
from __future__ import annotations
import json, itertools
import numpy as np, pandas as pd
import analysis_main as A

GAS_ID = {"CO": 1, "Ethanol": 2, "Ethylene": 3, "Methane": 4}
B = 2000
rs = np.random.default_rng(A.SEED)


def rank(x):
    x = np.asarray(x, float); n = len(x); o = np.argsort(x, kind="mergesort")
    r = np.empty(n); r[o] = np.arange(1, n + 1)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, r)
    return (s / cnt)[inv]


def rho(x, y):
    a, b = rank(x) , rank(y)
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def cluster_ci(x, y, clusters, n_boot=B):
    uc = np.unique(clusters)
    idx = {c: np.where(clusters == c)[0] for c in uc}
    out = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[c] for c in rs.choice(uc, len(uc), replace=True)])
        if len(np.unique(x[pick])) < 3:
            continue
        out.append(rho(x[pick], y[pick]))
    return [float(v) for v in np.percentile(out, [2.5, 97.5])]


def main():
    df = pd.read_csv("results/twin_features.csv")
    fc = [c for c in df.columns if c.startswith("f")]
    boards = {}
    for b, g in df.groupby("board"):
        boards[int(b)] = {"X": g[fc].values.astype(float),
                          "y": np.array([GAS_ID[n] for n in g.gas_name])}
    print("boards:", {k: v["X"].shape for k, v in boards.items()})

    rows = []
    for src, tgt in itertools.permutations(sorted(boards), 2):
        for name, u in GAS_ID.items():
            r = {"src": src, "tgt": tgt, "unknown": u, "unknown_name": name,
                 "pair": f"{src}->{tgt}"}
            ok = True
            for met in ("maha", "euclid", "energy"):
                d = A.compute_dnr(boards, src, tgt, u, met)
                if d is None:
                    ok = False; break
                r[f"DNR_{met}"] = d["DNR"]
                if met == "maha":
                    r["d_drift"], r["d_novel"] = d["d_drift"], d["d_novel"]
            if not ok:
                continue
            a = A.openset_auroc(boards, src, tgt, u)
            r.update({k: v for k, v in a.items() if k.startswith("AUROC")})
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv("results/twin_results.csv", index=False)
    print(f"\nsplits: {len(t)}")

    # unordered board pair is the resampling unit; a board appears in many splits
    t["cluster"] = t.apply(lambda r: "-".join(sorted([str(r.src), str(r.tgt)])), axis=1)
    cl = t.cluster.values
    print(f"clusters: {t.cluster.nunique()} board pairs, {t.unknown.nunique()} analytes")

    out = {"n_splits": int(len(t)), "n_clusters": int(t.cluster.nunique())}
    print(f"\n{'predictor':16s}{'detector':12s}{'rho':>8s}   cluster 95% CI")
    corr = {}
    for det, col in (("maha", "AUROC_maha"), ("knn", "AUROC_knn"), ("msp", "AUROC_msp")):
        if col not in t:
            continue
        for pred in ("DNR_maha", "DNR_euclid", "DNR_energy", "d_drift", "d_novel"):
            v = rho(t[pred].values, t[col].values)
            ci = cluster_ci(t[pred].values, t[col].values, cl)
            corr[f"{pred}|{det}"] = {"rho": v, "cluster_ci": ci}
            print(f"{pred:16s}{det:12s}{v:+8.3f}   [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    out["correlations"] = corr
    out["DNR_maha"] = {"min": float(t.DNR_maha.min()), "median": float(t.DNR_maha.median()),
                       "max": float(t.DNR_maha.max()),
                       "frac_above_1": float((t.DNR_maha > 1).mean())}
    out["AUROC_maha"] = {"median": float(t.AUROC_maha.median()),
                         "frac_below_half": float((t.AUROC_maha < 0.5).mean())}
    print(f"\nDNR  min {t.DNR_maha.min():.3f}  median {t.DNR_maha.median():.3f}  "
          f"max {t.DNR_maha.max():.3f}  above 1: {(t.DNR_maha>1).mean()*100:.1f}%")
    print(f"AUROC median {t.AUROC_maha.median():.3f}  below 0.5: {(t.AUROC_maha<0.5).mean()*100:.1f}%")
    lo = t[t.DNR_maha <= 1]; hi = t[t.DNR_maha > 1]
    print(f"  DNR<=1  n={len(lo):3d}  median AUROC {lo.AUROC_maha.median():.3f}")
    print(f"  DNR>1   n={len(hi):3d}  median AUROC {hi.AUROC_maha.median():.3f}")
    out["stratified"] = {"below_1": {"n": int(len(lo)), "median_AUROC": float(lo.AUROC_maha.median())},
                         "above_1": {"n": int(len(hi)), "median_AUROC": float(hi.AUROC_maha.median())}}
    with open("results/twin_analysis.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
