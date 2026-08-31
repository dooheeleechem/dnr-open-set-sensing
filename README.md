# Drift-to-Novelty Ratio (DNR) for open-set evaluation in chemical sensing

Analysis code for the manuscript

> **When Drift Outweighs Novelty: A Difficulty Axis for Open-Set Evaluation in Chemical Sensing**
> Seungmin Kang and Doo-Hee Lee

The drift-to-novelty ratio places the displacement of a known class and the
separation of an unknown class on one scale, so that the difficulty of an
open-set evaluation can be reported alongside openness. This repository
reproduces every number, table and figure in the paper.

## Datasets

None of the raw data is redistributed here. Download from the UCI Machine
Learning Repository and place as shown.

| Dataset | DOI | Place under |
|---|---|---|
| Gas Sensor Array Drift | [10.24432/C5RP6W](https://doi.org/10.24432/C5RP6W) | `data/Dataset/batch{1..10}.dat` |
| Gas Sensor Array Drift at Different Concentrations | [10.24432/C5MK6M](https://doi.org/10.24432/C5MK6M) | `data_conc/batch{1..10}.dat` |
| Twin gas sensor arrays | [10.24432/C5MW3K](https://doi.org/10.24432/C5MW3K) | `data_twin/twin+gas+sensor+arrays.zip` |

All three are distributed under CC BY 4.0. The twin archive is 195 MB
compressed and 2.77 GB expanded; `extract_twin.py` streams it from the zip
without expanding it to disk.

For convenience, `twin_features.csv` contains the 640 x 64 feature matrix that
`extract_twin.py` produces, so the replication analysis can be run without
downloading the raw traces.

## Reproducing the paper

```bash
pip install -r requirements.txt

python analysis_main.py      # 95 splits, DNR, AUROC, MMD, PAD  -> raw_results.csv
python analysis_extra.py     # cluster bootstrap, detector-specific rho, segmented fit
python analysis_extra2.py    # oracle compensation, centroid-only DNR intervals
python analysis_dnr_ci.py    # full-pipeline per-split DNR intervals (run repeatedly)
python analysis_conc.py      # concentration confounding control
python extract_twin.py       # feature extraction from raw twin traces
python analysis_twin.py      # replication on 80 board-pair splits
python analysis_twin_extra.py  # twin breakpoint and feature sanity checks

python make_figure1.py       # openness is blind        (Figure 1)
python make_figure2.py       # DNR core results         (Figure 2)
python make_figure4.py       # external replication     (Figure 3)
python make_figure3.py       # DNR landscape            (Figure S1)
python make_toc.py           # table-of-contents graphic
```

The script names follow the order in which the analyses were written, not the
figure order in the submitted manuscript. `make_figure4.py` produces Figure 3 and
`make_figure3.py` produces Figure S1 of the Supporting Information.

Every script fixes the same seed (`SEED = 20260822`), reads from `results/` and
writes to `results/`. The output files listed below ship at the repository root,
so copy them into `results/` before running any script that consumes them, or
regenerate them with the analysis scripts.

## Outputs included here

| File | Contents |
|---|---|
| `raw_results.csv` | per-split DNR, AUROC and competing descriptors, 100 rows |
| `experiment_summary.json` | headline statistics of the primary analysis |
| `extra_statistics.json` | cluster bootstrap intervals, descriptor x detector table, segmented regression |
| `extra_statistics2.json` | per-split DNR uncertainty summary, compensation with and without unknown contamination |
| `dnr_uncertainty.csv` | per-split DNR interval, metric held fixed |
| `dnr_uncertainty_full.csv` | per-split DNR interval, whole estimation chain re-fitted per resample |
| `dnr_uncertainty_full.json` | summary of the full-pipeline intervals |
| `concentration_analysis.json` | concentration descriptors, partial correlations, restricted-window reanalysis |
| `twin_features.csv` | 640 x 64 features derived from the twin arrays raw traces |
| `twin_results.csv` | per-split DNR and AUROC for the 80 board-pair splits |
| `twin_analysis.json` | replication correlations with cluster bootstrap intervals |
| `twin_extra.json` | twin breakpoint, closed-set and board-identification checks |

## Notes on method

- Distances are Mahalanobis under a Ledoit-Wolf shrunk pooled within-class
  covariance, estimated from the source known classes only. Euclidean and
  energy-distance variants are computed in parallel.
- `d_novel` uses the unknown class **in the target batch**, so that drift
  compensation is able to affect it.
- Twin-array features are cut at the valve events of the published acquisition
  protocol: 50 s of carrier air, 100 s of gas, 450 s of purging, so the baseline
  window is 0-45 s, the rising phase 50-150 s and the decaying phase 150-300 s.
  Averaging the normalized response of 120 recordings on a common time grid puts
  the onset of the rise at 54.5 s and the peak at 136.5 s, which is the protocol
  plus sensor lag.
- Confidence intervals for correlations are cluster bootstrap intervals over
  batch pairs and over analytes, reported as the union of the two. This is a
  deliberately conservative sensitivity interval, not a formal multiway cluster
  interval. Resampling splits independently understates the uncertainty because
  a batch pair contributes up to six splits.
- Per-split DNR intervals re-fit the standardization, the shrunk covariance and
  the centroids on every bootstrap resample. Holding the metric fixed narrows
  the interval by a factor of 1.5 in median width.
- Compensation is computed in a deployable variant, with alignment statistics
  from the entire target batch, and an oracle variant, with statistics from
  known target samples only. The contrast isolates the effect of unknown
  samples entering the transformation.

## Citation

Cite the manuscript, and the datasets under their own DOIs.

## License

MIT for the code. The datasets keep their own CC BY 4.0 licenses.
