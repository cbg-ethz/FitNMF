import argparse
import numpy as np
import pandas as pd
from nmf_cross_validation import joint_nmf_cross_validation, best_lambda
from typing import Optional

"""
This script performs cross-validation to select the best hyperparameters (lambda_w, lambda_h, lambda_f) for joint multi-location NMF. 
The parameters need to be specified. The parameters used to generate results in the manuscript, please find in Supplementary and fill in the following lists. You can also change them to explore a different range of values.
Please note that the CV procedure may take some time to run, especially for larger datasets and a wide range of hyperparameter values. 
"""
def main(input_paths: dict,
        rank: int,
        output_csv: str,
        n_cores: int = 1,
        bandwidth: float = 10.0,
        seed: Optional[int] = None):

    # ── Load one data matrix per location ─────────────────────────────
    data = {}
    for loc, path in input_paths.items():
        df = pd.read_csv(path, header=0, index_col=0)
        df.sort_index(axis=1, inplace=True)  # Sort columns by sample name (the date)

        data[loc] = df

    # ── Cross-validation ──────────────────────────────────────────────

# Enter the hyperparameter values to test here. These are the values used in the paper (Supplementary Table S3), but you can change them if you want to explore a different range of values.
# Real data (Zurich 2021-24): 10 x 14 x 14 = 1,960 combinations
 #   lambda_w_values = [0, 0.001, 0.01, 0.1, 0.5, 1, 5, 7, 10, 20]
 #   lambda_h_values = [0, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 20, 30, 40, 50, 100, 1000]
 #   lambda_f_values = [0, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 20, 30, 40, 50, 100, 1000]
# Simulated data: 23 x 28 x 28 = 18,032 combinations
 #   lambda_w_values = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
 #   lambda_h_values = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]
 #   lambda_f_values = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]
    results_df, cv_summary = joint_nmf_cross_validation(
        
        data=data,
        rank=rank,
        lambda_w_values=lambda_w_values,
        lambda_h_values=lambda_h_values,
        lambda_f_values=lambda_f_values,
        K=10,   # number of folds for cross-validation
        n_run=10, # number of random restarts for NMF fitting 
        max_iter=5000,# Maximum number of iterations for NMF fitting
        rel_tol=1e-4, # Relative tolerance for convergence in NMF fitting
        bandwidth=bandwidth,
        n_jobs=n_cores,
        random_state=seed,     # joint_nmf_cross_validation seeds its own fold
                                # shuffle via np.random.default_rng, which does NOT
                                # read the legacy global np.random.seed() -- must be
                                # passed explicitly or --seed silently does nothing
        output_csv=output_csv   # save directly inside CV
    )

    best = best_lambda(cv_summary)
    best_lw, best_lh, best_lf = best


    print(f"\nBest hyperparameters for rank={rank}:")
    print(f"  lambda_w = {best_lw}")
    print(f"  lambda_h = {best_lh}")
    if best_lf is not None:
        print(f"  lambda_f = {best_lf}")
    print(f"\nCV summary:")
    print(cv_summary.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cross-validation for lambda selection (joint multi-location NMF)')

    parser.add_argument('-i', '--input', required=True, nargs='+',
                        help="One or more location=path.csv pairs, e.g. "
                             "-i ZH=zh.csv GR=geneva.csv -- each CSV is a "
                             "mutation frequency data matrix for that location")
    parser.add_argument('-r', '--rank', required=True, type=int,
                        help='Expected number of lineages (shared across all locations)')
    parser.add_argument('-o', '--output', required=True,
                        help='CSV file name for CV results')
    parser.add_argument('--cores', type=int, default=1,
                        help='Number of cores to use for parallelisation')
    parser.add_argument('--bandwidth', type=float, default=10,
                        help='Bandwidth parameter for Gaussian kernel')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    input_paths = {}
    for pair in args.input:
        if '=' not in pair:
            parser.error(f"--input entries must be location=path.csv, got '{pair}'")
        loc, path = pair.split('=', 1)
        input_paths[loc] = path

    main(
        input_paths=input_paths,
        rank=args.rank,
        output_csv=args.output,
        n_cores=args.cores,
        bandwidth=args.bandwidth,
        seed=args.seed
    )
