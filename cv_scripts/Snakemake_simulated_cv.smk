# Snakefile_simulated_cv
# Snakefile for cross-validation of simulated data (12 x 3 = 36 CSV files)
import os
configfile: "config/config_simulated.yaml"

SIM_DIR = config.get("simulated_data_dir")
SIMS    = sorted(glob_wildcards(os.path.join(SIM_DIR, "{sim}.csv")).sim)
assert len(SIMS) == 36, f"expected 36 simulated CSV files in {SIM_DIR}, found {len(SIMS)}: {SIMS}"

SIM_RESULTS_DIR = config.get("sim_results_dir")
N_VARIANTS_LIST = config["expected_variants_number"]
BANDWIDTH       = config["bandwidth_variants"]

wildcard_constraints:
    sim="|".join(SIMS),
    n_variants=r"\d+"


rule all:
    input:
        expand(f"{SIM_RESULTS_DIR}/{{sim}}/cross_validation_{{n_variants}}_snvs.csv",
               sim=SIMS, n_variants=N_VARIANTS_LIST),
        f"{SIM_RESULTS_DIR}/cross_validation_summary_snv.csv",


rule cv_simulated:
    input:
        datamatrix=f"{SIM_DIR}/{{sim}}.csv"
    output:
        f"{SIM_RESULTS_DIR}/{{sim}}/cross_validation_{{n_variants}}_snvs.csv"
    threads: 20
    resources:
        mem_per_cpu_mb=5000
    params:
        bandwidth=BANDWIDTH
    shell:
        "python scripts_snakemake/held_out_cv.py "
        "-i {wildcards.sim}={input.datamatrix} "
        "-r {wildcards.n_variants} "
        "-o {output} "
        "--cores {threads} "
        "--bandwidth {params.bandwidth}"


rule summarize_cv_simulated:
    input:
        expand(f"{SIM_RESULTS_DIR}/{{sim}}/cross_validation_{{n_variants}}_snvs.csv",
               sim=SIMS, n_variants=N_VARIANTS_LIST)
    output:
        output_csv_combined=f"{SIM_RESULTS_DIR}/cross_validation_summary_snv.csv"
    threads: 1
    resources:
        mem_per_cpu_mb=4000
    script:
        "scripts_snakemake/summarize_cross_validation_results.py"