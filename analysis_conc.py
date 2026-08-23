"""Concentration confounding analysis.

The drift dataset at different concentrations (UCI 270) carries the same
features as the drift dataset (UCI 224) with the analyte concentration
appended to the label field. Row alignment between the two releases was
verified exhaustively before this script was run, so concentrations can be
attached to every sample of the main analysis without re-deriving features.

Three questions are addressed.
  Q1  Do the concentration composition of a split and the DNR covary?
  Q2  Does the DNR-AUROC association survive control for concentration?
  Q3  Does it survive restriction to a common concentration window?
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from pathlib import Path
import analysis_main as A

CONC_LO, CONC_HI = 50.0, 250.0


def load_conc():
    out = {}
    for b in range(1, 11):
        c = [float(l.split()[0].split(";")[1])
             for l in Path(f"data_conc/batch{b}.dat").read_text().strip().split("\n")]
        out[b] = np.asarray(c)
    return out


def rank(x):
    x = np.asarray(x, float); n = len(x); o = np.argsort(x, kind="mergesort")
    r = np.empty(n); r[o] = np.arange(1, n + 1)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, r)
    return (s / cnt)[inv]


def partial_rho(x, y, controls):
    """Spearman partial correlation: correlate rank residuals after removing
    the linear projection of the ranked control variables."""
    X = np.column_stack([rank(c) for c in controls])
    X = np.column_stack([np.ones(len(X)), X])
    rx, ry = rank(x), rank(y)
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    batches = A.load_all(Path("./data/Dataset"))
    conc = load_conc()
    for b in batches:
        assert len(conc[b]) == len(batches[b]["y"]), b
        batches[b]["c"] = conc[b]

    base = pd.read_csv("results/raw_results.csv")
    base["pair"] = base.src.astype(str) + "->" + base.tgt.astype(str)
    base = base.drop_duplicates(subset=["pair", "unknown"]).reset_index(drop=True)

    rows = []
    for _, r in base.iterrows():
        s, t, u = int(r.src), int(r.tgt), int(r.unknown)
        ys, yt = batches[s]["y"], batches[t]["y"]
        cs, ct = batches[s]["c"], batches[t]["c"]
        known = sorted({c for c in set(ys) if (ys == c).sum() >= A.MIN_PER_CLASS}
                       & {c for c in set(yt) if (yt == c).sum() >= A.MIN_PER_CLASS} - {u})
        if not known or (yt == u).sum() < A.MIN_PER_CLASS:
            continue
        lg = lambda v: float(np.median(np.log10(v)))
        src_known = np.mean([lg(cs[ys == c]) for c in known])
        tgt_known = np.mean([lg(ct[yt == c]) for c in known])
        unk_c = lg(ct[yt == u])
        rows.append(dict(
            src=s, tgt=t, unknown=u,
            # how far the unknown analyte sits from the knowns in concentration
            conc_gap=abs(unk_c - src_known),
            # how much the concentration composition itself shifts between batches
            conc_shift=abs(tgt_known - src_known),
            # spread of concentrations among the known classes in the source
            conc_spread=float(np.std([lg(cs[ys == c]) for c in known])),
        ))
    cdf = pd.DataFrame(rows)
    m = base.merge(cdf, on=["src", "tgt", "unknown"])
    print(f"splits with concentration annotation: {len(m)}")

    out = {"n_splits": int(len(m))}
    print("\n=== Q1  concentration descriptors vs DNR and AUROC ===")
    q1 = {}
    for v in ("conc_gap", "conc_shift", "conc_spread"):
        a = float(np.corrcoef(rank(m[v]), rank(m.DNR_maha))[0, 1])
        b = float(np.corrcoef(rank(m[v]), rank(m.AUROC_maha))[0, 1])
        q1[v] = {"rho_vs_DNR": a, "rho_vs_AUROC": b}
        print(f"  {v:12s}  vs DNR {a:+.3f}   vs AUROC {b:+.3f}")
    out["Q1"] = q1

    print("\n=== Q2  partial correlation, DNR vs AUROC ===")
    raw = float(np.corrcoef(rank(m.DNR_maha), rank(m.AUROC_maha))[0, 1])
    p1 = partial_rho(m.DNR_maha, m.AUROC_maha, [m.conc_gap])
    p2 = partial_rho(m.DNR_maha, m.AUROC_maha, [m.conc_gap, m.conc_shift])
    p3 = partial_rho(m.DNR_maha, m.AUROC_maha, [m.conc_gap, m.conc_shift, m.conc_spread])
    print(f"  unconditional                      {raw:+.3f}")
    print(f"  | conc_gap                         {p1:+.3f}")
    print(f"  | conc_gap, conc_shift             {p2:+.3f}")
    print(f"  | conc_gap, conc_shift, conc_spread {p3:+.3f}")
    out["Q2"] = {"unconditional": raw, "ctrl_gap": p1,
                 "ctrl_gap_shift": p2, "ctrl_all": p3}

    print(f"\n=== Q3  restriction to {CONC_LO:.0f}-{CONC_HI:.0f} ppmv ===")
    sub = {}
    for b in batches:
        k = (batches[b]["c"] >= CONC_LO) & (batches[b]["c"] <= CONC_HI)
        sub[b] = {"X": batches[b]["X"][k], "y": batches[b]["y"][k]}
        print(f"  batch{b:<3d} {k.sum():5d} / {len(k):5d} retained")
    res = []
    for _, r in m.iterrows():
        s, t, u = int(r.src), int(r.tgt), int(r.unknown)
        d = A.compute_dnr(sub, s, t, u, "maha")
        a = A.openset_auroc(sub, s, t, u)
        if d and "AUROC_maha" in a:
            res.append((d["DNR"], a["AUROC_maha"], r.DNR_maha, r.AUROC_maha))
    res = np.array(res)
    rho_sub = float(np.corrcoef(rank(res[:, 0]), rank(res[:, 1]))[0, 1])
    rho_full_same = float(np.corrcoef(rank(res[:, 2]), rank(res[:, 3]))[0, 1])
    print(f"\n  splits surviving the window: {len(res)} of {len(m)}")
    print(f"  rho(DNR, AUROC) within window      {rho_sub:+.3f}")
    print(f"  rho(DNR, AUROC) same splits, full  {rho_full_same:+.3f}")
    print(f"  median DNR within window {np.median(res[:,0]):.3f}  full {np.median(res[:,2]):.3f}")
    out["Q3"] = {"window_ppmv": [CONC_LO, CONC_HI], "n_splits": int(len(res)),
                 "rho_within_window": rho_sub, "rho_same_splits_full": rho_full_same,
                 "median_DNR_window": float(np.median(res[:, 0])),
                 "median_DNR_full": float(np.median(res[:, 2])),
                 "frac_above_1_window": float((res[:, 0] > 1).mean()),
                 "frac_above_1_full": float((res[:, 2] > 1).mean())}
    with open("results/concentration_analysis.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
