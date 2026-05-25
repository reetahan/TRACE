import argparse
import os
from datetime import datetime
from em import EM_algorithm
from welfare import evaluate_simulation_output
import json
import pandas as pd
import numpy as np
import pickle
from data_ingestion import *
from util import log_and_print
from file_config import *
from list_length import return_nyc_list_params, return_nyc_list_params_per_district
from constants import DISTRICT_TO_BOROUGH_MAPPING, LEARNING_RATE


def run_real(max_iter=20, M=15, K=12,
             sampling_n_jobs=32, max_iter_opt=5, seed=40, eta=LEARNING_RATE,
             profile_timing=False, outfile=None, imputation_file=None,
             save_best_params=True, save_best_sample=False, is_gaussian_per_district=False):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_filepath = imputation_file
    outfile_name_entry = "default"
    if(not imputation_file):
        df_filepath = f"{POLISHED_DATA_DIR}/{MAIN_AGG_APP_STATS_FILEPATH}"
    else:
        outfile_name_entry = df_filepath.split('/')[-1].replace('.csv', '').replace('xlsx', '')
    outfile = f'{EXP_OUT_FOLDER}nyc_res_logs/{timestamp}/real_experiment_K={K}_M={M}_LR={eta}_iter={max_iter}_opt={max_iter_opt}_{outfile_name_entry}_{timestamp}.txt'

    df = read_data(df_filepath)
    match_stats_df = read_data(
        f"{RAW_DATA_DIR}/{MAIN_AGG_MATCH_STATS_FILEPATH}",
        sheet=MAIN_AGG_MATCH_STATS_FILEPATH_SHEET,
        is_first_row_header=True
    )
    school_info_df = read_data(
        f"{RAW_DATA_DIR}/{SCHOOL_INFO_STATS_FILEPATH}",
        sheet=SCHOOL_INFO_STATS_FILEPATH_SHEET
    )
    addtl_school_info_df = read_data(
        f"{RAW_DATA_DIR}/{ADDTL_SCHOOL_INFO_STATS_FILEPATH}",
        sheet=ADDTL_SCHOOL_INFO_STATS_FILEPATH_SHEET
    )

    nyc_config_path = f"{POLISHED_DATA_DIR}/{NYC_CONFIG_FILEPATH}"
    priority_config = None
    if os.path.exists(nyc_config_path):
        with open(nyc_config_path) as f:
            priority_config = json.load(f)
        log_and_print(f"Loaded NYC priority config from {nyc_config_path}", outfile)
    else:
        log_and_print(f"No config file at {nyc_config_path}, proceeding with default info!", outfile)
    

    df, match_stats_df, school_info_df = nyc_preprocess_data(
        df, match_stats_df, school_info_df, addtl_school_info_df
    )
    list_length_params = return_nyc_list_params()

    if is_gaussian_per_district:
        list_length_params = return_nyc_list_params_per_district(df, match_stats_df)

    log_and_print(f"======== Data Loading and Preprocessing Complete =========", outfile)
    log_and_print(f"Parameters:\nMax_iter: {max_iter}\nM: {M}\nK: {K}\nLR: {eta}\nSeed: {seed}\n \
                  Profile Timing: {profile_timing}\nNum Sampling Jobs: {sampling_n_jobs}\n \
                List Length Params: {list_length_params}\nSave Parameters: {save_best_params}\n \
                Save Sample of Best Rankings: {save_best_sample}\nLength of main DF: {len(df)} \
                Length of school info DF: {len(school_info_df)}\n Length of Match Stats DF: \
                {len(match_stats_df)}", outfile)
    log_and_print(f"Entering EM Algorithm...", outfile)

    experiment_results = EM_algorithm(
        df,
        match_stats_df,
        school_info_df,
        max_iter=max_iter,
        M_simulations=M,
        K=K,
        outfile=outfile,
        sampling_n_jobs=sampling_n_jobs,
        max_iter_opt=max_iter_opt,
        seed=seed,
        profile_timing=profile_timing,
        eta=eta,
        priority_config=priority_config,
        district_to_region=DISTRICT_TO_BOROUGH_MAPPING,
        list_length_params=list_length_params,
        save_best_sample = save_best_sample
    )

    params = experiment_results.params
    syn_rankings = experiment_results.syn_rankings
    syn_districts = experiment_results.syn_districts
    syn_attrs = experiment_results.syn_attrs
    syn_rankings_idx = experiment_results.syn_rankings_idx
    matches_idx = experiment_results.matches_idx
    log_likelihoods = experiment_results.log_likelihoods

    params_path = outfile.replace('.txt', '_params.pkl')
    with open(params_path, 'wb') as f:
        pickle.dump(params, f)
    log_and_print(f"Saved params to {params_path}", log_file=outfile)

    rows = []
    for i, (ranking, district) in enumerate(zip(syn_rankings, syn_districts)):
        row = {'student_id': i, 'district': district}
        for j, school in enumerate(ranking[:10]):
            row[f'choice_{j+1}'] = school
        rows.append(row)

    syn_df = pd.DataFrame(rows)
    syn_path = outfile.replace('.txt', '_synthetic_rankings.csv')
    syn_df.to_csv(syn_path, index=False)
    log_and_print(f"Saved synthetic rankings ({len(syn_df)} students) to {syn_path}", log_file=outfile)

    attr_df = pd.DataFrame(syn_attrs) if syn_attrs is not None else pd.DataFrame()
    attr_df['district'] = list(syn_districts)

    welfare_results = evaluate_simulation_output(
        sim_output={
            'rankings_as_indices': syn_rankings_idx,
            'matches_idx': matches_idx,
            'student_attributes': attr_df,
        },
        categories=['district'],
        output_dir=outfile.replace('.txt', '_welfare'),
    )
    log_and_print(
        f"Welfare: avg rank={welfare_results.rank_stats['avg_rank']:.3f}, "
        f"pct_matched={welfare_results.rank_stats['pct_matched']:.1f}%",
        log_file=outfile,
    )

    log_and_print(f"===== RUN COMPLETE =====", log_file=outfile)
    log_and_print(f"Log-likelihood trajectory: {log_likelihoods}", log_file=outfile)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=5, help='Number of mixture components for real data')
    parser.add_argument('--M', type=int, default=10, help='Number of simulations per evaluation')
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum EM iterations')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate for sigma nudges')
    parser.add_argument('--max_iter_opt', type=int, default=10, help='Maximum Optimizer iterations')
    parser.add_argument('--seed', type=int, default=DATA_GENERATION_SEED, help='Random seed for synthetic experiments')
    parser.add_argument('--n_jobs', type=int, default=64, help='Number of parallel workers')
    parser.add_argument('--profile_timing', action='store_true', help='Enable detailed timing logs')
    parser.add_argument('--imputation_file', type=str, default=None, help='Path to imputation file')
    parser.add_argument('--outfile', type=str, default=None, help='Output file for logs')
    parser.add_argument('--save_params', action='store_true', help='Enable saving of parameters to a pickle file')
    parser.add_argument('--run_gaussian_per_district', action='store_true', help='Make list length mean district dependent')
    parser.add_argument('--save_best_sample', action='store_true', help='Enable saving sample of preference profile from best parameters to CSV')
    args = parser.parse_args()
    
    run_real(
        outfile=args.outfile,
        max_iter=args.max_iter,
        M=args.M,
        K=args.K,
        sampling_n_jobs=args.n_jobs,
        max_iter_opt=args.max_iter_opt,
        seed=args.seed,
        eta=args.lr,
        profile_timing=args.profile_timing,
        save_best_params=args.save_params,
        save_best_sample=args.save_best_sample,
        imputation_file=args.imputation_file,
        is_gaussian_per_district=args.run_gaussian_per_district
    )
