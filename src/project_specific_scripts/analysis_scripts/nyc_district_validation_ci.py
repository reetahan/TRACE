"""
Figs 11-15 style NYC observed-vs-simulated bars by residential district, with
95% CIs on the simulated bars from N independent replicate simulations at a
single fitted params.pkl -- not a single log's "best iteration" point
estimate. NYC analogue of chile_region_validation_ci.py.

Uses the post-nyc_priority_attributes.py bugfix code (student_attrs no
longer silently discarded/redrawn), so these replicates reflect the current
(fixed) matching pipeline.

Usage:
    python3 src/project_specific_scripts/nyc_district_validation_ci.py \
        --params_pkl <path to params.pkl> \
        --n_runs 10 --seed 42 --out nyc_validation_ci
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_ingestion import read_data, nyc_preprocess_data
from em import run_single_simulation
from list_length import return_nyc_list_params
from constants import DISTRICT_TO_BOROUGH_MAPPING
from file_config import (
    POLISHED_DATA_DIR, RAW_DATA_DIR,
    MAIN_AGG_APP_STATS_FILEPATH, MAIN_AGG_MATCH_STATS_FILEPATH,
    MAIN_AGG_MATCH_STATS_FILEPATH_SHEET, SCHOOL_INFO_STATS_FILEPATH,
    SCHOOL_INFO_STATS_FILEPATH_SHEET, ADDTL_SCHOOL_INFO_STATS_FILEPATH,
    ADDTL_SCHOOL_INFO_STATS_FILEPATH_SHEET, NYC_CONFIG_FILEPATH,
)

OBS_COLOR = "#0a17d1"
SIM_COLOR = "#c47806"
FONT_SIZE = 13
BAR_WIDTH = 0.38


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_pkl', required=True)
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_jobs', type=int, default=32)
    parser.add_argument('--out', default='nyc_validation_ci')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pickle
    with open(args.params_pkl, 'rb') as f:
        params = pickle.load(f)
    print(f"K={len(params['global_phis'])}, global_phis={params['global_phis']}")

    df_raw = read_data(f"{POLISHED_DATA_DIR}/{MAIN_AGG_APP_STATS_FILEPATH}")
    match_stats_raw = read_data(
        f"{RAW_DATA_DIR}/{MAIN_AGG_MATCH_STATS_FILEPATH}",
        sheet=MAIN_AGG_MATCH_STATS_FILEPATH_SHEET, is_first_row_header=True,
    )
    school_info_raw = read_data(
        f"{RAW_DATA_DIR}/{SCHOOL_INFO_STATS_FILEPATH}", sheet=SCHOOL_INFO_STATS_FILEPATH_SHEET,
    )
    addtl_school_info_raw = read_data(
        f"{RAW_DATA_DIR}/{ADDTL_SCHOOL_INFO_STATS_FILEPATH}", sheet=ADDTL_SCHOOL_INFO_STATS_FILEPATH_SHEET,
    )
    df, match_stats_df, school_info_df = nyc_preprocess_data(
        df_raw, match_stats_raw, school_info_raw, addtl_school_info_raw
    )
    list_length_params = return_nyc_list_params()

    nyc_config_path = f"{POLISHED_DATA_DIR}/{NYC_CONFIG_FILEPATH}"
    priority_config = None
    if os.path.exists(nyc_config_path):
        with open(nyc_config_path) as f:
            priority_config = json.load(f)
        print(f"Loaded priority config: {nyc_config_path}")

    districts = sorted(match_stats_df['Residential District'].unique(), key=lambda d: int(d))
    n_students_total = int(match_stats_df['Total Applicants'].sum())
    print(f"n districts: {len(districts)}, n students: {n_students_total}")

    # Observed, straight from the real match_stats_df -- no simulation needed.
    metric_names = ['top3', 'top5', 'top10', 'unmatched']
    obs_by_district = {}
    for d in districts:
        row = match_stats_df[match_stats_df['Residential District'] == d].iloc[0]
        obs_by_district[d] = np.array([
            row['% Matches to Choice 1-3'], row['% Matches to Choice 1-5'],
            row['% Matches to Choice 1-10'], row['Unmatched'],
        ], dtype=float)

    # school_info_df['District'] is zero-padded ('01'..'32', from the DBN prefix
    # dbn[:2]), while match_stats_df/our `districts` use unpadded '1'..'32' --
    # verified empirically, not assumed. Normalize here or every utilization
    # lookup below silently misses.
    school_info_df = school_info_df.copy()
    school_info_df['District'] = school_info_df['District'].astype(int).astype(str)
    obs_util_by_district = (
        school_info_df.groupby('District')['Utilization'].mean().to_dict()
    )

    # N independent replicate simulations at the fixed fitted params.
    region_runs = {m: [] for m in metric_names}
    util_region_runs = []
    for run in range(args.n_runs):
        print(f"Run {run + 1}/{args.n_runs}...")
        rng_lottery = np.random.default_rng(seed=args.seed + run)
        lottery_fixed = rng_lottery.permutation(n_students_total)
        agg = run_single_simulation(
            params=params, df=df, match_stats_df=match_stats_df, school_info_df=school_info_df,
            lottery_fixed=lottery_fixed, mallows_seed=args.seed + run, sampling_n_jobs=args.n_jobs,
            priority_config=priority_config, district_to_region=DISTRICT_TO_BOROUGH_MAPPING,
            list_length_params=list_length_params,
        )
        # compute_aggregates builds its row order via np.unique(district_assignments),
        # and district_assignments only ever contains values from
        # params['districts'].keys() (verified directly -- string labels like
        # '1','10','11',...,'2','20',... -- lexicographic, NOT our numeric display
        # order). Recomputing that exact order here rather than assuming alignment
        # with our display-sorted `districts` list is exactly the kind of silent
        # index mismatch that broke Chile's priority pipeline earlier in this project.
        agg_districts = sorted(params['districts'].keys())
        sim_by_district = {d: agg['match_stats'][i, :] for i, d in enumerate(agg_districts)}
        for i, m in enumerate(metric_names):
            region_runs[m].append({d: v[i] for d, v in sim_by_district.items()})

        # agg['filled'] is indexed by df['School DBN'].unique() (see em.py:247),
        # NOT school_info_df's order/set -- same alignment-risk class as the
        # district fix above, so pull the order from the exact object passed
        # into run_single_simulation rather than a differently-ordered table.
        all_schools_order = df['School DBN'].unique()
        filled_by_school = dict(zip(all_schools_order, agg['filled']))
        cap_map = school_info_df.set_index('School DBN')['Capacity'].to_dict()
        school_to_district = school_info_df.set_index('School DBN')['District'].to_dict()
        sim_util_num, sim_util_den = {}, {}
        for s, filled in filled_by_school.items():
            d = school_to_district.get(s)
            cap = cap_map.get(s, 0)
            if d is None or cap <= 0:
                continue
            u = min(100.0, 100.0 * filled / cap)
            sim_util_num[d] = sim_util_num.get(d, 0) + u
            sim_util_den[d] = sim_util_den.get(d, 0) + 1
        util_region_runs.append({d: sim_util_num[d] / sim_util_den[d] for d in sim_util_num})

    def mean_ci(records, district):
        vals = np.array([rec.get(district, np.nan) for rec in records], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan, np.nan
        return float(np.mean(vals)), float(1.96 * np.std(vals) / np.sqrt(len(vals)))

    dump_rows = []
    for d in districts:
        row = {'district': d}
        for m in metric_names:
            mean, ci = mean_ci(region_runs[m], d)
            row[f'obs_{m}'] = obs_by_district[d][metric_names.index(m)]
            row[f'sim_{m}_mean'] = mean
            row[f'sim_{m}_ci95'] = ci
        u_mean, u_ci = mean_ci(util_region_runs, d)
        row['obs_util'] = obs_util_by_district.get(d, np.nan)
        row['sim_util_mean'] = u_mean
        row['sim_util_ci95'] = u_ci
        dump_rows.append(row)
    dump_df = pd.DataFrame(dump_rows)
    dump_path = out_dir / 'nyc_district_validation_ci.csv'
    dump_df.to_csv(dump_path, index=False)
    print(f"Saved: {dump_path}")

    def plot_bar(metric_label, obs_col, sim_mean_col, sim_ci_col, fname, ylim=105):
        x = np.arange(len(districts))
        fig, ax = plt.subplots(figsize=(11, 4))
        obs_vals = dump_df[obs_col].to_numpy()
        sim_vals = dump_df[sim_mean_col].to_numpy()
        sim_ci = dump_df[sim_ci_col].to_numpy()
        ax.bar(x - BAR_WIDTH / 2, obs_vals, BAR_WIDTH, label='Observed', color=OBS_COLOR, alpha=0.85)
        ax.bar(x + BAR_WIDTH / 2, sim_vals, BAR_WIDTH, label='Simulated', color=SIM_COLOR, alpha=0.85,
               yerr=sim_ci, error_kw={'ecolor': '#000000', 'elinewidth': 1.4, 'capsize': 3})
        ax.set_xticks(x)
        ax.set_xticklabels(districts)
        ax.set_xlabel('Residential District', fontsize=FONT_SIZE)
        ax.set_ylabel(metric_label, fontsize=FONT_SIZE)
        ax.set_ylim(0, ylim)
        ax.legend(fontsize=11, loc='upper right')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=FONT_SIZE - 2)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_dir / fname}")

    plot_bar('Unmatched Rate (%)', 'obs_unmatched', 'sim_unmatched_mean', 'sim_unmatched_ci95', 'fig11_nyc_unmatched_district_ci.png')
    plot_bar('Top-3 Match Rate (%)', 'obs_top3', 'sim_top3_mean', 'sim_top3_ci95', 'fig12_nyc_top3_district_ci.png')
    plot_bar('Top-5 Match Rate (%)', 'obs_top5', 'sim_top5_mean', 'sim_top5_ci95', 'fig13_nyc_top5_district_ci.png')
    plot_bar('Top-10 Match Rate (%)', 'obs_top10', 'sim_top10_mean', 'sim_top10_ci95', 'fig14_nyc_top10_district_ci.png')
    plot_bar('Average School Utilization (%)', 'obs_util', 'sim_util_mean', 'sim_util_ci95', 'fig15_nyc_utilization_district_ci.png')


if __name__ == '__main__':
    main()
