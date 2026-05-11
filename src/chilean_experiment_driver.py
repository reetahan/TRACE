import argparse
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from em import EM_algorithm
from data_ingestion import read_data, preprocess_chilean_data
from constants import LEARNING_RATE
from util import log_and_print
from file_config import *
from welfare import evaluate_simulation_output
from list_length import return_chilean_list_params
import json

def run_chilean_data_experiment(
    max_iter=20,
    M=10,
    K=6,
    sampling_n_jobs=64,
    max_iter_opt=10,
    seed=40,
    eta=LEARNING_RATE,
    profile_timing=False,
    outfile=None,
    imputation_file=None,
    save_best_params=True, 
    save_best_sample=False,
    max_p=None
):
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if outfile is None:
        outfile = f'{EXP_OUT_FOLDER}/chile_res_logs/{timestamp}/chilean_experiment_K={K}_M={M}_iter={max_iter}_opt={max_iter_opt}_lr={eta}_seed={seed}_{timestamp}.txt'

    indv_df = read_data(f"{CHILEAN_DATA_DIR}/{CHILEAN_INDV_PREF_FILEPATH}")
    match_df = read_data(f"{CHILEAN_DATA_DIR}/{CHILEAN_MATCH_OUTCOME_FILEPATH}")
    school_cap_df = read_data(f"{CHILEAN_DATA_DIR}/{CHILEAN_SCHOOL_CAPACITY_FILEPATH}")
    school_cap_reg_df = read_data(f"{CHILEAN_DATA_DIR}/{CHILEAN_SCHOOL_CAPACITY_BY_REGION_FILEPATH}")

    chile_config_path = f"{CHILEAN_DATA_DIR}/{CHILE_CONFIG_FILEPATH}"
    priority_config = None
    if os.path.exists(chile_config_path):
        with open(chile_config_path) as f:
            priority_config = json.load(f)
        log_and_print(f"Loaded Chilean priority config: {chile_config_path}", outfile)

    df, match_stats_df, school_info_df = preprocess_chilean_data(
        indv_df, match_df, school_cap_reg_df, school_cap_df
    )

    district_to_region = {str(r): str(r) for r in df['Residential District'].unique()}
    list_length_params = return_chilean_list_params(indv_df=indv_df)

    log_and_print(f'Match stats sample:\n{match_stats_df.head()}', outfile)

    log_and_print(f"======== Data Loading and Preprocessing Complete =========", outfile)
    log_and_print(f"Parameters:\nMax_iter: {max_iter}\nM: {M}\nK: {K}\nSeed: {seed}\n \
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
        district_to_region=district_to_region,
        list_length_params=list_length_params,
        save_best_sample = save_best_sample,
        max_p=max_p
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
        categories=['district', 'female'],
        output_dir=outfile.replace('.txt', '_welfare'),
    )
    log_and_print(
        f"Welfare: avg rank={welfare_results.rank_stats['avg_rank']:.3f}, "
        f"pct_matched={welfare_results.rank_stats['pct_matched']:.1f}%",
        log_file=outfile,
    )

    log_and_print(f"!!!!!!!! RUN COMPLETE !!!!!!!!", log_file=outfile)
    log_and_print(f"Log-likelihood trajectory: {log_likelihoods}", log_file=outfile)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=5, help='Number of mixture components for real data')
    parser.add_argument('--M', type=int, default=10, help='Number of simulations per evaluation')
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum EM iterations')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate for sigma nudges')
    parser.add_argument('--max_iter_opt', type=int, default=10, help='Maximum Optimizer iterations')
    parser.add_argument('--max_p', type=int, default=10, help='Default number of top choices to run loss function against for Chile')
    parser.add_argument('--seed', type=int, default=DATA_GENERATION_SEED, help='Random seed for synthetic experiments')
    parser.add_argument('--n_jobs', type=int, default=64, help='Number of parallel workers')
    parser.add_argument('--profile_timing', action='store_true', help='Enable detailed timing logs')
    parser.add_argument('--imputation_file', type=str, default=None, help='Path to imputation file')
    parser.add_argument('--outfile', type=str, default=None, help='Output file for logs')
    parser.add_argument('--save_params', action='store_true', help='Enable saving of parameters to a pickle file')
    parser.add_argument('--save_best_sample', action='store_true', help='Enable saving sample of preference profile from best parameters to CSV')
    args = parser.parse_args()
    
    
    run_chilean_data_experiment(
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
        max_p=args.max_p,
        imputation_file=args.imputation_file,
    )