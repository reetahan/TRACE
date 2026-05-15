import pickle
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

from data_ingestion import read_data, nyc_preprocess_data
from em import run_single_simulation
from file_config import RAW_DATA_DIR, POLISHED_DATA_DIR, EXP_OUT_FOLDER, DATA_GENERATION_SEED
from file_config import (
    MAIN_AGG_APP_STATS_FILEPATH,
    MAIN_AGG_MATCH_STATS_FILEPATH,
    MAIN_AGG_MATCH_STATS_FILEPATH_SHEET,
    SCHOOL_INFO_STATS_FILEPATH,
    SCHOOL_INFO_STATS_FILEPATH_SHEET,
    ADDTL_SCHOOL_INFO_STATS_FILEPATH,
    ADDTL_SCHOOL_INFO_STATS_FILEPATH_SHEET,
    NYC_CONFIG_FILEPATH,
)
from constants import DISTRICT_TO_BOROUGH_MAPPING
from util import log_and_print

OBS_COLOR  = "#0a17d1"
SIM_COLOR  = "#c47806"
BAR_WIDTH = 0.38
FONT_SIZE = 13
LEGEND_FONT_SIZE = 11

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params',      required=True, help='Path to saved params pickle')
    parser.add_argument('--output',      default='utilization_by_district.png')
    parser.add_argument('--imputation',  default=None, help='Optional imputation file path')
    parser.add_argument('--seed',        type=int, default=DATA_GENERATION_SEED)
    parser.add_argument('--list_length_mean', type=float, default=7)
    parser.add_argument('--list_length_std',  type=float, default=2)
    args = parser.parse_args()

    # Data loading — mirrors experiment driver exactly
    df_filepath = args.imputation or f"{POLISHED_DATA_DIR}/{MAIN_AGG_APP_STATS_FILEPATH}"

    print("Loading data...")
    df_raw = read_data(df_filepath)
    match_stats_df = read_data(
        f"{RAW_DATA_DIR}/{MAIN_AGG_MATCH_STATS_FILEPATH}",
        sheet=MAIN_AGG_MATCH_STATS_FILEPATH_SHEET,
        is_first_row_header=True,
    )
    school_info_df = read_data(
        f"{RAW_DATA_DIR}/{SCHOOL_INFO_STATS_FILEPATH}",
        sheet=SCHOOL_INFO_STATS_FILEPATH_SHEET,
    )
    addtl_school_info_df = read_data(
        f"{RAW_DATA_DIR}/{ADDTL_SCHOOL_INFO_STATS_FILEPATH}",
        sheet=ADDTL_SCHOOL_INFO_STATS_FILEPATH_SHEET,
    )
    df, match_stats_df, school_info_df = nyc_preprocess_data(
        df_raw, match_stats_df, school_info_df, addtl_school_info_df
    )
    print(f"  Schools: {df['School DBN'].nunique()}, Students: {int(match_stats_df['Total Applicants'].sum())}")

    # Priority config
    nyc_config_path = f"{POLISHED_DATA_DIR}/{NYC_CONFIG_FILEPATH}"
    priority_config = None
    if os.path.exists(nyc_config_path):
        with open(nyc_config_path) as f:
            priority_config = json.load(f)
        print(f"  Loaded priority config from {nyc_config_path}")
    else:
        print(f"  No config at {nyc_config_path}, proceeding without priority config")

    # Load params
    print(f"Loading params from {args.params}...")
    with open(args.params, 'rb') as f:
        params = pickle.load(f)
    print(f"  K={len(params['global_phis'])} components, phis={params['global_phis']}")

    # Single simulation
    n_students = int(match_stats_df['Total Applicants'].sum())
    lottery = np.random.default_rng(args.seed).permutation(n_students)

    list_length_params = {
        'list_length_mode': 'gaussian',
        'list_length_mean': args.list_length_mean,
        'list_length_std':  args.list_length_std,
        'list_length_min':  1,
    }

    print("Running simulation...")
    agg = run_single_simulation(
        params=params,
        df=df,
        match_stats_df=match_stats_df,
        school_info_df=school_info_df,
        lottery_fixed=lottery,
        mallows_seed=args.seed,
        priority_config=priority_config,
        district_to_region=DISTRICT_TO_BOROUGH_MAPPING,
        list_length_params=list_length_params,
        save_best_sample=False,
    )

    # District-level utilization
    all_schools = df['School DBN'].unique()
    sim_filled  = pd.Series(agg['filled'], index=all_schools)
    capacities  = school_info_df.set_index('School DBN')['Capacity']
    obs_util    = school_info_df.set_index('School DBN')['Utilization']
    sim_util    = (sim_filled / capacities * 100).clip(upper=100)

    def district_from_dbn(dbn):
        return int(str(dbn)[:2])

    district_obs, district_sim = {}, {}
    for dbn in all_schools:
        d = district_from_dbn(dbn)
        if dbn in obs_util.index and np.isfinite(obs_util[dbn]):
            district_obs.setdefault(d, []).append(float(obs_util[dbn]))
            district_sim.setdefault(d, []).append(float(sim_util.get(dbn, np.nan)))

    districts  = sorted(district_obs.keys())
    obs_means  = [np.nanmean(district_obs[d]) for d in districts]
    sim_means  = [np.nanmean(district_sim[d]) for d in districts]
    mae_by_d   = [abs(o - s) for o, s in zip(obs_means, sim_means)]
    overall_mae = np.mean(mae_by_d)
    print(f"Overall district-level utilization MAE: {overall_mae:.2f}%")

    diffs = [o - s for o, s in zip(obs_means, sim_means)]
    print(f"\n── Utilization Summary ────────────────────────")
    print(f"  Overall obs utilization:     {np.mean(obs_means):.2f}%")
    print(f"  Overall sim utilization:     {np.mean(sim_means):.2f}%")
    print(f"  Max district diff (abs):     {max(abs(d) for d in diffs):+.2f}pp  (District {districts[max(range(len(diffs)), key=lambda i: abs(diffs[i]))]})")
    print(f"  Min district diff (abs):     {min(abs(d) for d in diffs):+.2f}pp  (District {districts[min(range(len(diffs)), key=lambda i: abs(diffs[i]))]})")
    print(f"  Mean district diff:          {np.mean(diffs):+.2f}pp")
    print(f"  Overall MAE:                 {overall_mae:.2f}%")
    print(f"──────────────────────────────────────────────\n")

    util_df = pd.DataFrame({
        'district': districts,
        'obs_util': obs_means,
        'sim_util': sim_means,
        'diff_util': diffs,
        'abs_diff_util': [abs(d) for d in diffs],
    })
    util_csv = args.output.replace('.png', '_district_util.csv')
    util_df.to_csv(util_csv, index=False)
    print(f"Saved: {util_csv}")

    x = np.arange(len(districts))
    fig1, ax1 = plt.subplots(figsize=(11, 4))
    ax1.bar(x - BAR_WIDTH/2, obs_means, BAR_WIDTH, label='Observed', color=OBS_COLOR, alpha=0.85)
    ax1.bar(x + BAR_WIDTH/2, sim_means, BAR_WIDTH, label='Simulated', color=SIM_COLOR, alpha=0.85)
    ax1.set_xticks(x)
    labels = [str(d) for d in districts]
    ax1.set_xticklabels(labels)
    plt.setp(ax1.get_xticklabels(), rotation=0, ha='center', fontsize=FONT_SIZE)
    ax1.set_ylabel('Average School Utilization (%)', fontsize=FONT_SIZE)
    ax1.set_xlabel("Residential District", fontsize=FONT_SIZE)
    ax1.legend(fontsize=LEGEND_FONT_SIZE, loc='upper center', bbox_to_anchor=(0.5, 1.12),
          ncol=2, borderaxespad=0)
    ax1.set_ylim(0, 105)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='y', labelsize=FONT_SIZE)
    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    output_util = args.output.replace('.png', '_utilization.png')
    fig1.savefig(output_util, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {output_util}")



if __name__ == '__main__':
    main()