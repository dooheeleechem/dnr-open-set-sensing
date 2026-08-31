"""Breakpoint and sanity checks for the twin gas sensor arrays replication.

Produces results/twin_extra.json, which make_figure4.py consumes. Two things are
computed here.

Stage 1  a continuous two-segment regression of open-set performance on
         log10(DNR) with the breakpoint chosen by grid search, and a cluster
         bootstrap interval on the breakpoint over unordered board pairs. The
         segmented machinery is imported from analysis_extra so that the primary
         dataset and the replication are fit by identical code.

Stage 2  checks that the features derived from the raw traces behave as
         intended: a multinomial logistic model must separate the four analytes,
         and must also be able to say which board produced a recording, which is
         what makes the board a meaningful domain.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_extra import segmented_fit, segmented_ci, SEED

GAS_ID = {"CO": 1, "Ethanol": 2, "Ethylene": 3, "Methane": 4}


def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000))


def main():
    t = pd.read_csv("results/twin_results.csv")
    t["cluster"] = t.apply(lambda r: "-".join(sorted([str(r.src), str(r.tgt)])), axis=1)

    seg = segmented_fit(np.log10(t.DNR_maha.values), t.AUROC_maha.values)
    lo, hi, n_ok = segmented_ci(t, "DNR_maha", "AUROC_maha", "cluster", n_boot=600, seed=SEED)
    out = {"breakpoint_DNR": seg["breakpoint_DNR"],
           "breakpoint_DNR_ci": [round(lo, 3), round(hi, 3)],
           "slope_left": seg["slope_left"],
           "slope_right": seg["slope_right"],
           "r2_gain": seg["r2_gain"]}
    print(f"breakpoint DNR {seg['breakpoint_DNR']:.3f}  CI [{lo:.3f}, {hi:.3f}]"
          f"  RSS gain {seg['r2_gain']*100:.1f}%  ({n_ok} resamples)")

    df = pd.read_csv("results/twin_features.csv")
    fc = [c for c in df.columns if c.startswith("f")]
    X = df[fc].values.astype(float)
    y = np.array([GAS_ID[n] for n in df.gas_name])
    b = df.board.values.astype(int)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    acc = float(cross_val_score(clf(), X, y, cv=cv).mean())
    out["closed_set_cv_acc"] = round(acc, 4)
    print(f"closed-set 5-fold accuracy {acc:.4f}  (chance {1/len(GAS_ID):.3f})")

    lobo = {}
    for held in sorted(np.unique(b)):
        m = b != held
        lobo[f"B{held}"] = round(float(clf().fit(X[m], y[m]).score(X[~m], y[~m])), 4)
    out["leave_one_board_out_acc"] = lobo
    out["leave_one_board_out_mean"] = round(float(np.mean(list(lobo.values()))), 4)
    print("leave-one-board-out:", lobo, "mean", out["leave_one_board_out_mean"])

    bacc = float(cross_val_score(clf(), X, b, cv=cv).mean())
    out["board_predictable_acc"] = round(bacc, 4)
    out["n_recordings"] = int(len(df))
    out["n_features"] = int(len(fc))
    print(f"board identification accuracy {bacc:.4f}  (chance {1/len(np.unique(b)):.3f})")

    with open("results/twin_extra.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote results/twin_extra.json")


if __name__ == "__main__":
    main()
