# DNR: a difficulty axis for open-set evaluation in chemical sensing

Analysis code for

> S. Kang and D.-H. Lee, *When Drift Outweighs Novelty: A Difficulty Axis for
> Open-Set Evaluation in Chemical Sensing* (submitted).

Companion to Kim et al., *Chemosensors* **2026**, 14, 189,
https://doi.org/10.3390/chemosensors14090189

## Quick start

```bash
pip install -r requirements.txt
# download the dataset first (see "Data" below) into ./data/
python analysis_main.py --data-dir ./data --out-dir ./results
python make_figure2.py
python make_figure3.py
```

`raw_results.csv` and `experiment_summary.json` in this repository are the
outputs of that command, so the results can be inspected without re-running it.

## Files

| File | Purpose |
|---|---|
| `analysis_main.py` | full analysis, single seed, no manual steps |
| `make_figure1.py` | Figure 1, built from values published in ref. 4 of the paper |
| `make_figure2.py` | Figure 2, four-panel core result |
| `make_figure3.py` | Figure 3, DNR landscape |
| `raw_results.csv` | one row per split-by-unknown combination |
| `experiment_summary.json` | headline metrics reported in the paper |
| `requirements.txt` | pinned minimum versions |

## What it computes

Two displacements, both measured from the source known-class manifold, in one
feature space fitted on the source known classes.

    d_drift = mean over known classes of  || centroid_target(c) - centroid_source(c) ||
    d_novel = min  over known classes of  || centroid_target(unknown) - centroid_source(c) ||
    DNR     = d_drift / d_novel

`DNR >= 1` means a known analyte has been displaced by drift as far as an unknown
analyte sits from the library. No novelty score defined on that manifold can then
separate the two.

The unknown class is measured in the **target** batch, not the source. That choice
matters: it is the only definition under which drift compensation, which acts on
the target, can move `d_novel` as well as `d_drift`. Under the source-side
definition `d_novel` is invariant to compensation by construction, and RQ4 becomes
unanswerable.

## Distances

Three metrics are computed in parallel so that no conclusion rests on one choice.

| key | metric | notes |
|---|---|---|
| `maha` | Mahalanobis | pooled within-class covariance, Ledoit-Wolf shrinkage |
| `euclid` | Euclidean | on standardized features |
| `energy` | energy distance | two-sample, subsampled to 400 per class |

## Splits

Both split families used in the literature, following Yao et al. 2023.

- **Setting 1** — source = batch 1, target = batch K (K = 2..10). Interval grows.
- **Setting 2** — source = batch K, target = batch K+1. Interval fixed.

Each split is repeated with each of the six gases held out as the unknown class,
giving 108 split-by-unknown combinations.

## Open-set baselines

Three deliberately simple, fully reproducible scorers. The point is not to win a
benchmark; it is to test whether DNR predicts performance regardless of scorer.

- `AUROC_maha` — minimum Mahalanobis distance to a known centroid
- `AUROC_knn` — mean distance to the 5 nearest known training samples
- `AUROC_msp` — one minus the maximum softmax probability of a multinomial logistic model

## Drift compensation (RQ4)

- `mean` — mean-shift alignment of the target batch onto the source
- `coral` — second-order (covariance) alignment

For each, the code reports the ratio by which `d_drift` and `d_novel` change. If
compensation shrinks both by similar factors, DNR barely improves and the
apparent gain in known-class accuracy is bought at the cost of novelty
separability. That is the RQ4 result.

## Running it

```bash
pip install -r requirements.txt
python analysis_main.py --data-dir ./data --out-dir ./results
```

Expected input: `data/batch1.dat` ... `data/batch10.dat`, in the original UCI
format, one sample per line:

```
<class>;<concentration> 1:<v1> 2:<v2> ... 128:<v128>
```

Outputs:

- `results/raw_results.csv` — one row per split-by-unknown combination, 26 columns
- `results/experiment_summary.json` — headline metrics, including the Spearman
  correlation between DNR and each AUROC, and between batch interval and AUROC

## Reproducibility

`SEED = 20260822` fixes the energy-distance subsampling and the logistic solver.
No other stochastic component. Verified end to end on synthetic data with the
same file format before the real data was obtained.

## Data

UCI Gas Sensor Array Drift Dataset, DOI `10.24432/C5RP6W`, CC BY 4.0.
Cite Vergara et al., *Sens. Actuators B* **2012**, 166-167, 320-329 and
Rodriguez-Lujan et al., *Chemom. Intell. Lab. Syst.* **2014**, 130, 123-134,
as the dataset page requires.

Not redistributed here. Download separately into `data/`.
