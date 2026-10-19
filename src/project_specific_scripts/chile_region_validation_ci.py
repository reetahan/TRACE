"""
Region-level observed vs. simulated validation bars (Figs 16-20 style) with
95% CIs on the simulated bars, built from N independent replicate
simulations at a single fitted params.pkl -- not a single log's "best
iteration" point estimate.

Usage:
    python3 src/project_specific_scripts/chile_region_validation_ci.py \
        --params_pkl <path to params.pkl> \
        --n_runs 10 --seed 42 --out validation_ci
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_ingestion import read_data, preprocess_chilean_data
from em import run_single_simulation
from list_length import return_chilean_list_params
from constants import CHILE_PROVINCE_TO_REGION_MAPPING
from file_config import (
    CHILEAN_DATA_DIR,
    CHILEAN_INDV_PREF_PROVINCE_SANTIAGOZONES_FILEPATH,
    CHILEAN_MATCH_OUTCOME_PROVINCE_SANTIAGOZONES_FILEPATH,
    CHILEAN_SCHOOL_CAPACITY_BY_REGION_PROVINCE_SANTIAGOZONES_FILEPATH,
    CHILE_CONFIG_FILEPATH,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chile_region_cumulative_stb_mtb import ROMAN_BY_REGION, ROMAN_ORDER

OBS_COLOR = "#0a17d1"
SIM_COLOR = "#c47806"
FONT_SIZE = 13
BAR_WIDTH = 0.38


def aggregate_district_vec_to_region(values_by_district, weights, mapping):
    """values_by_district: {district: np.array of stats}. Returns {region: weighted-mean array}."""
    region_sum, region_w = {}, {}
    for d, v in values_by_district.items():
        r = mapping.get(d, d)
        w = weights.get(d, 1)
        region_sum[r] = region_sum.get(r, 0) + w * np.asarray(v, dtype=float)
        region_w[r] = region_w.get(r, 0) + w
    return {r: region_sum[r] / region_w[r] for r in region_sum}


def aggregate_district_scalar_to_region(values_by_district, weights, mapping):
    region_sum, region_w = {}, {}
    for d, v in values_by_district.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        r = mapping.get(d, d)
        w = weights.get(d, 1)
        region_sum[r] = region_sum.get(r, 0) + w * v
        region_w[r] = region_w.get(r, 0) + w
    return {r: region_sum[r] / region_w[r] for r in region_sum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_pkl', required=True)
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_p', type=int, default=10)
    parser.add_argument('--n_jobs', type=int, default=32)
    parser.add_argument('--out', default='validation_ci')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pickle
    with open(args.params_pkl, 'rb') as f:
        params = pickle.load(f)
    print(f"K={len(params['global_phis'])}, global_phis={params['global_phis']}")

    indv_df = read_data(f"{CHILEAN_DATA_DIR}/{CHILEAN_INDV_PREF_PROVINCE_SANTIAGOZONES_FILEPATH}")
    match_df = read_data(f"{CHILEAN_DATA_DIR}/{CHILEAN_MATCH_OUTCOME_PROVINCE_SANTIAGOZONES_FILEPATH}")
    school_cap_df = read_data(f"{CHILEAN_DATA_DIR}/{CHILEAN_SCHOOL_CAPACITY_BY_REGION_PROVINCE_SANTIAGOZONES_FILEPATH}")

    df, match_stats_df, school_info_df = preprocess_chilean_data(
        indv_df, match_df, school_cap_df, is_province_level=True
    )
    list_length_params = return_chilean_list_params(indv_df=indv_df)

    chile_config_path = f"{CHILEAN_DATA_DIR}/{CHILE_CONFIG_FILEPATH}"
    priority_config = None
    import os
    if os.path.exists(chile_config_path):
        with open(chile_config_path) as f:
            priority_config = json.load(f)
        print(f"Loaded priority config: {chile_config_path}")

    district_to_region = {str(r): str(r) for r in df['Residential District'].unique()}

    districts = sorted(df['Residential District'].unique())
    weights = {d: int(match_stats_df[match_stats_df['Residential District'] == d]['Total Applicants'].iloc[0])
               for d in districts}
    n_students_total = sum(weights.values())
    print(f"n districts: {len(districts)}, n students: {n_students_total}")

    # Observed, region-level (same weighting scheme as the simulated side).
    n_stats = max_p_metrics = args.max_p + 1
    metric_names = [f'top{p}' for p in range(1, args.max_p + 1)] + ['unmatched']
    obs_by_district = {}
    for d in districts:
        row = match_stats_df[match_stats_df['Residential District'] == d].iloc[0]
        cols = [f'% Matches to Choice 1-{p}' for p in range(1, args.max_p + 1)]
        vec = [row[c] if c in row else np.nan for c in cols] + [row['Unmatched']]
        obs_by_district[d] = np.array(vec, dtype=float)
    obs_by_region = aggregate_district_vec_to_region(obs_by_district, weights, CHILE_PROVINCE_TO_REGION_MAPPING)

    # Utilization is per-school; aggregate to district via school->district map, then district->region.
    school_to_district = {}
    for d in districts:
        for s in params['districts'][d]['schools']:
            school_to_district[s] = d
    obs_util_map = school_info_df.set_index('School DBN')['Utilization'].to_dict()
    cap_map = school_info_df.set_index('School DBN')['Capacity'].to_dict()
    obs_util_by_district_num, obs_util_by_district_den = {}, {}
    for s, u in obs_util_map.items():
        d = school_to_district.get(s)
        if d is None or not np.isfinite(u):
            continue
        obs_util_by_district_num[d] = obs_util_by_district_num.get(d, 0) + u
        obs_util_by_district_den[d] = obs_util_by_district_den.get(d, 0) + 1
    obs_util_by_district = {d: obs_util_by_district_num[d] / obs_util_by_district_den[d] for d in obs_util_by_district_num}
    obs_util_by_region = aggregate_district_scalar_to_region(obs_util_by_district, weights, CHILE_PROVINCE_TO_REGION_MAPPING)

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
            priority_config=priority_config, district_to_region=district_to_region,
            list_length_params=list_length_params, raw_capacity_df=school_cap_df,
            max_p=args.max_p,
        )
        sim_by_district = {d: agg['match_stats'][i, :] for i, d in enumerate(districts)}
        sim_by_region = aggregate_district_vec_to_region(sim_by_district, weights, CHILE_PROVINCE_TO_REGION_MAPPING)
        for i, m in enumerate(metric_names):
            region_runs[m].append({r: v[i] for r, v in sim_by_region.items()})

        filled_by_school = dict(zip(df['School DBN'].unique(), agg['filled']))
        sim_util_by_district_num, sim_util_by_district_den = {}, {}
        for s, filled in filled_by_school.items():
            d = school_to_district.get(s)
            cap = cap_map.get(s, 0)
            if d is None or cap <= 0:
                continue
            u = min(100.0, 100.0 * filled / cap)
            sim_util_by_district_num[d] = sim_util_by_district_num.get(d, 0) + u
            sim_util_by_district_den[d] = sim_util_by_district_den.get(d, 0) + 1
        sim_util_by_district = {d: sim_util_by_district_num[d] / sim_util_by_district_den[d] for d in sim_util_by_district_num}
        util_region_runs.append(aggregate_district_scalar_to_region(sim_util_by_district, weights, CHILE_PROVINCE_TO_REGION_MAPPING))

    region_name_by_roman = {v: k for k, v in ROMAN_BY_REGION.items()}
    regions_present = [r for r in ROMAN_ORDER if region_name_by_roman.get(r) in obs_by_region]

    def mean_ci(records, region):
        vals = np.array([rec.get(region, np.nan) for rec in records], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan, np.nan
        return float(np.mean(vals)), float(1.96 * np.std(vals) / np.sqrt(len(vals)))

    dump_rows = []
    for roman in regions_present:
        full_name = region_name_by_roman[roman]
        row = {'region_roman': roman, 'region_name': full_name}
        for m in metric_names:
            mean, ci = mean_ci(region_runs[m], full_name)
            row[f'obs_{m}'] = obs_by_region.get(full_name, [np.nan] * n_stats)[metric_names.index(m)]
            row[f'sim_{m}_mean'] = mean
            row[f'sim_{m}_ci95'] = ci
        u_mean, u_ci = mean_ci(util_region_runs, full_name)
        row['obs_util'] = obs_util_by_region.get(full_name, np.nan)
        row['sim_util_mean'] = u_mean
        row['sim_util_ci95'] = u_ci
        dump_rows.append(row)
    dump_df = pd.DataFrame(dump_rows)
    dump_path = out_dir / 'chile_region_validation_ci.csv'
    dump_df.to_csv(dump_path, index=False)
    print(f"Saved: {dump_path}")

    def plot_bar(metric_label, obs_col, sim_mean_col, sim_ci_col, fname, ylim=105):
        x = np.arange(len(regions_present))
        fig, ax = plt.subplots(figsize=(11, 4))
        obs_vals = dump_df[obs_col].to_numpy()
        sim_vals = dump_df[sim_mean_col].to_numpy()
        sim_ci = dump_df[sim_ci_col].to_numpy()
        ax.bar(x - BAR_WIDTH / 2, obs_vals, BAR_WIDTH, label='Observed', color=OBS_COLOR, alpha=0.85)
        ax.bar(x + BAR_WIDTH / 2, sim_vals, BAR_WIDTH, label='Simulated', color=SIM_COLOR, alpha=0.85,
               yerr=sim_ci, error_kw={'ecolor': '#000000', 'elinewidth': 1.4, 'capsize': 3})
        ax.set_xticks(x)
        ax.set_xticklabels(regions_present)
        ax.set_xlabel('Region', fontsize=FONT_SIZE)
        ax.set_ylabel(metric_label, fontsize=FONT_SIZE)
        ax.set_ylim(0, ylim)
        ax.legend(fontsize=11, loc='upper right')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=FONT_SIZE - 2)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_dir / fname}")

    plot_bar('Unmatched Rate (%)', 'obs_unmatched', 'sim_unmatched_mean', 'sim_unmatched_ci95', 'fig16_chile_unmatched_region_ci.png')
    plot_bar('Top-3 Match Rate (%)', 'obs_top3', 'sim_top3_mean', 'sim_top3_ci95', 'fig17_chile_top3_region_ci.png')
    plot_bar('Top-5 Match Rate (%)', 'obs_top5', 'sim_top5_mean', 'sim_top5_ci95', 'fig18_chile_top5_region_ci.png')
    plot_bar('Top-10 Match Rate (%)', 'obs_top10', 'sim_top10_mean', 'sim_top10_ci95', 'fig19_chile_top10_region_ci.png')
    plot_bar('Average School Utilization (%)', 'obs_util', 'sim_util_mean', 'sim_util_ci95', 'fig20_chile_utilization_region_ci.png')


if __name__ == '__main__':
    main()
