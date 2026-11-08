# FitNMF

**Inferring viral variants and fitness advantages from longitudinal wastewater sequencing data**

FitNMF - an integrated viral dynamics and non-negative matrix factorization model that leverages the temporal structure of wastewater sequencing data and the underlying variant competition to jointly infer mutational signatures, relative abundances, and fitness advantages of circulating variants directly from wastewater next-generation sequencing data.

> Rimaite A., Beerenwinkel N. *Inferring viral variants and fitness advantages from longitudinal
> wastewater sequencing data.* medRxiv, 2026. [https://doi.org/10.64898/2026.11.06.26367358](https://doi.org/10.64898/2026.11.06.26367358).

This repository contains the FitNMF code and the notebooks that produce the results of the manuscript.

## Installation

```bash
git clone https://github.com/cbg-ethz/FitNMF.git
cd FitNMF
```

## Data

- **Input matrices:** one CSV per location, `data/joint_input/<location>/datamatrix_snvs.csv`. Rows are mutations
  (the same rows in the same order in every location), columns are sampling dates (`YYYY_MM_DD`), values are
  mutation frequencies in [0, 1], and empty cells are missing values.
- **Raw sequencing data:** ENA project PRJEB44932, processed with
  V-pipe.
- **Reference signatures:** CoV-Spectrum (`resources/variants_definitions_sars/variant_signatures_covspectrum_2026_08_27.csv`).
- **Cross-validation results:** `results/cross_validation_results/`

## Choosing the hyperparameters

The number of variants $k$ and the regularization strengths $\lambda_W$, $\lambda_H$, $\lambda_f$ have to be chosen by 10-fold cross-validation on randomly held-out
entries (Methods, Sec. 2.5).

1. Run one cross-validation per candidate $k$. The $\lambda$ grid is set inside `held_out_cv.py`: uncomment the
   block for real or simulated data (the values of Supplementary Table S3).
2. `summarize_cross_validation_results.py` (a Snakemake script) combines the per-rank results into
   `cross_validation_summary_snv.csv`, the mean held-out error for every $k$ and $(\lambda_W, \lambda_H, \lambda_f)$.
3. Choose $k$ as the smallest value for which $k+1$ lowers the error by less than 2%. Cross-validation-optimal parameters for Zurich 2021–2024: $k = 10$, $\lambda_W = 0.01$, $\lambda_H = 0.1$, $\lambda_f = 5$.

## Reproducing the results


**Real wastewater data (Zurich 2021–2024 and six WWTPs)**

1. `parameter_selection.ipynb` (real-data cells): cross-validation error against $k$, selected $\lambda$ per $k$.
2. `SC2_3y_factorization.ipynb`: the Zurich fit and the joint six-WWTP fit with the selected values (Fig. 2), the comparison with Covvfit (Fig. 3, `evaluation/fig3_growth_advantages.py`) and the early
   detection experiment (Fig. 4). Fig. 4 also needs a cross-validation per time window: the window datasets are written
   by the commented cell "Generate the window-datasets", and the best hyperparameters per window are read from
   `results/cross_validation_results/windows_ZH_real_time_detection/cross_validation_best_by_window_snv.csv`.

**Simulations**

1. `simulating_data.ipynb` (sequential take-over and co-circulation) and `genetic_drift_simulations.ipynb` (neutral
   drift).
2. Cross-validation of the 36 datasets for $k = 1, \dots, 6$.
The summary goes to `results/cross_validation_results/simulations/cross_validation_summary_snv.csv`.
3. FitNMF with the cross-validation-selected $\lambda$'s
   produces the figures. `parameter_selection.ipynb` (simulated-data cells) shows the cross-validation
   diagnostics.



