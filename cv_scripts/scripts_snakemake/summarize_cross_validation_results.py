"""
Summary of cross validation results for all locations and all expected variants numbers
"""

import pandas as pd
import re


input_files_paths = snakemake.input
output_csv        = snakemake.output.output_csv_combined

all_results = []

for filepath in input_files_paths:
    results_df_all_folds = pd.read_csv(filepath)

    # ── Average over folds ────────────────────────────────────────────────────
    group_cols = ['lambda_w', 'lambda_h', 'lambda_f']

    results_df = (results_df_all_folds
                  .groupby(group_cols)['squared_error_loss']
                  .agg(['mean', 'std'])
                  .reset_index()
                  .rename(columns={'mean': 'mean_error', 'std': 'std_error'})
                  .sort_values('mean_error')
                  .reset_index(drop=True))

    # ── Extract location from path ────────────────────────────────────────────
    parts    = filepath.split('/')
    location = parts[-2]

    # ── Extract rank from filename ────────────────────────────────────────────
    match      = re.search(r'cross_validation_(\d+)_snvs', filepath)
    n_variants = int(match.group(1)) if match else None


    results_df['location']     = location
    results_df['n_variants']   = n_variants


    all_results.append(results_df)


combined_results_df = pd.concat(all_results, axis=0, ignore_index=True)

# ── Reorder columns ───────────────────────────────────────────────────────────
col_order = ['location', 'n_variants', 
             'lambda_w', 'lambda_h', 'lambda_f',
             'mean_error', 'std_error']
combined_results_df = combined_results_df[col_order]

combined_results_df.to_csv(output_csv, index=False)