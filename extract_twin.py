"""Feature extraction for the Twin Gas Sensor Arrays dataset (UCI 361).

Each recording is a 600 s, 100 Hz acquisition from an eight-sensor metal-oxide
array. The gas is admitted at 60 s and withdrawn at 180 s, which the onset
detection reported in the accompanying log confirms.

The eight features per sensor follow Vergara et al. 2012, the definition used
to build the drift dataset analyzed in the main text, so that the two datasets
carry comparable descriptors: the steady-state response, its normalized form,
and the extrema of exponential moving average transients at three smoothing
constants over the rising and the decaying phase.
"""
from __future__ import annotations
import zipfile, io, re
import numpy as np, pandas as pd
from scipy.signal import lfilter

FS = 100.0
BASE_END, GAS_ON, GAS_OFF, DECAY_END = 55.0, 60.0, 180.0, 300.0
ALPHAS = (0.1, 0.01, 0.001)
GASES = {"GEy": "Ethylene", "GEa": "Ethanol", "GCO": "CO", "GMe": "Methane"}
PAT = re.compile(r"B(\d)_G(\w\w)_F(\d+)_R(\d+)\.txt$")


def ema(x, a):
    return lfilter([a], [1.0, -(1.0 - a)], np.diff(x, prepend=x[0]))


def features(t, X):
    base = np.median(X[t < BASE_END], 0)
    rise = (t >= GAS_ON) & (t <= GAS_OFF)
    dec = (t > GAS_OFF) & (t <= DECAY_END)
    out = []
    for j in range(X.shape[1]):
        x = X[:, j].astype(np.float64)
        d = x[rise] - base[j]
        dR = d[np.argmax(np.abs(d))]
        out += [dR, dR / (abs(base[j]) + 1e-9)]
        for a in ALPHAS:
            out.append(ema(x[rise], a).max())
        for a in ALPHAS:
            out.append(ema(x[dec], a).min())
    return out


def main():
    z = zipfile.ZipFile("data_twin/twin+gas+sensor+arrays.zip")
    names = sorted(n for n in z.namelist() if n.endswith(".txt"))
    rows = []
    for i, n in enumerate(names, 1):
        m = PAT.search(n)
        board, gas, conc, rep = int(m[1]), "G" + m[2], int(m[3]), int(m[4])
        d = pd.read_csv(io.BytesIO(z.read(n)), sep=r"\s+", header=None,
                        dtype=np.float32).values
        rows.append(dict(board=board, gas=gas, gas_name=GASES[gas], conc=conc,
                         rep=rep, **{f"f{k:02d}": v for k, v in
                                     enumerate(features(d[:, 0], d[:, 1:]))}))
        if i % 80 == 0:
            print(f"  {i}/{len(names)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/twin_features.csv", index=False)
    print("\nwrote results/twin_features.csv", df.shape)
    print(pd.crosstab(df.board, df.gas_name))
    fc = [c for c in df.columns if c.startswith("f")]
    print(f"\nfeatures {len(fc)}  non-finite {int((~np.isfinite(df[fc].values)).sum())}")


if __name__ == "__main__":
    main()
