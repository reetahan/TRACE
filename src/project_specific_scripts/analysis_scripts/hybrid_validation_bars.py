
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chile_region_cumulative_stb_mtb import ROMAN_BY_REGION, ROMAN_ORDER

OBS_COLOR = "#0a17d1"
SIM_COLOR = "#c47806"
FONT_SIZE = 13
BAR_WIDTH = 0.38

NYC_METRICS = [
    ('Unmatched Rate (%)', 'unmatched', 'unmatched', 'fig11_nyc_unmatched_district_hybrid.png'),
    ('Top-3 Match Rate (%)', 'top3', 'top3', 'fig12_nyc_top3_district_hybrid.png'),
    ('Top-5 Match Rate (%)', 'top5', 'top5', 'fig13_nyc_top5_district_hybrid.png'),
    ('Top-10 Match Rate (%)', 'top10', 'top10', 'fig14_nyc_top10_district_hybrid.png'),
]
CHILE_METRICS = [
    ('Unmatched Rate (%)', 'unmatched', 'unmatched', 'fig16_chile_unmatched_region_hybrid.png'),
    ('Top-3 Match Rate (%)', 'top3', 'top3', 'fig17_chile_top3_region_hybrid.png'),
    ('Top-5 Match Rate (%)', 'top5', 'top5', 'fig18_chile_top5_region_hybrid.png'),
    ('Top-10 Match Rate (%)', 'top10', 'top10', 'fig19_chile_top10_region_hybrid.png'),
]


def plot_bar(x_labels, obs_vals, sim_vals, sim_ci, metric_label, xlabel, fname, out_dir, ylim=105,
             figsize=(11, 4), rotation=0):
    import numpy as np
    x = np.arange(len(x_labels))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - BAR_WIDTH / 2, obs_vals, BAR_WIDTH, label='Observed', color=OBS_COLOR, alpha=0.85)
    ax.bar(x + BAR_WIDTH / 2, sim_vals, BAR_WIDTH, label='Simulated', color=SIM_COLOR, alpha=0.85,
           yerr=sim_ci, error_kw={'ecolor': '#000000', 'elinewidth': 1.4, 'capsize': 3})
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=rotation)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(metric_label, fontsize=FONT_SIZE)
    ax.set_ylim(0, ylim)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=FONT_SIZE - 2)
    fig.tight_layout()
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def run_nyc(args, out_dir):
    log_fit = pd.read_csv(args.log_fit_csv).set_index('district')
    log_util = pd.read_csv(args.log_util_csv).set_index('district')
    ci = pd.read_csv(args.ci_csv).set_index('district')

    districts = sorted(set(log_fit.index) & set(ci.index), key=lambda d: int(d))
    print(f"n districts in both sources: {len(districts)}")

    for metric_label, log_col, ci_col, fname in NYC_METRICS:
        obs_vals = [log_fit.loc[d, f'obs_{log_col}'] for d in districts]
        sim_vals = [log_fit.loc[d, f'sim_{log_col}'] for d in districts]  # mean from log snapshot
        sim_ci = [ci.loc[d, f'sim_{ci_col}_ci95'] for d in districts]    # CI from N-replicate
        plot_bar(districts, obs_vals, sim_vals, sim_ci, metric_label, 'Residential District', fname, out_dir)

    # Utilization
    districts_u = sorted(set(log_util.index) & set(ci.index), key=lambda d: int(d))
    obs_vals = [log_util.loc[d, 'obs_util'] for d in districts_u]
    sim_vals = [log_util.loc[d, 'sim_util'] for d in districts_u]
    sim_ci = [ci.loc[d, 'sim_util_ci95'] for d in districts_u]
    plot_bar(districts_u, obs_vals, sim_vals, sim_ci, 'Average School Utilization (%)', 'Residential District',
              'fig15_nyc_utilization_district_hybrid.png', out_dir)


def run_chile(args, out_dir):
    log_fit = pd.read_csv(args.log_fit_csv).set_index('district')
    log_util = pd.read_csv(args.log_util_csv).set_index('district')
    ci = pd.read_csv(args.ci_csv).set_index('region_name')

    region_name_by_roman = {v: k for k, v in ROMAN_BY_REGION.items()}
    regions_present = [r for r in ROMAN_ORDER
                        if region_name_by_roman.get(r) in log_fit.index
                        and region_name_by_roman.get(r) in ci.index]
    print(f"n regions in both sources: {len(regions_present)}")
    region_names = [region_name_by_roman[r] for r in regions_present]

    for metric_label, log_col, ci_col, fname in CHILE_METRICS:
        obs_vals = [log_fit.loc[n, f'obs_{log_col}'] for n in region_names]
        sim_vals = [log_fit.loc[n, f'sim_{log_col}'] for n in region_names]
        sim_ci = [ci.loc[n, f'sim_{ci_col}_ci95'] for n in region_names]
        plot_bar(regions_present, obs_vals, sim_vals, sim_ci, metric_label, 'Region', fname, out_dir)

    region_names_u = [n for n in region_names if n in log_util.index and n in ci.index]
    regions_present_u = [ROMAN_BY_REGION[n] for n in region_names_u]
    obs_vals = [log_util.loc[n, 'obs_util'] for n in region_names_u]
    sim_vals = [log_util.loc[n, 'sim_util'] for n in region_names_u]
    sim_ci = [ci.loc[n, 'sim_util_ci95'] for n in region_names_u]
    plot_bar(regions_present_u, obs_vals, sim_vals, sim_ci, 'Average School Utilization (%)', 'Region',
              'fig20_chile_utilization_region_hybrid.png', out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['nyc', 'chile'], required=True)
    parser.add_argument('--log_fit_csv', required=True, help='<mode>_val_district_fit_1.csv (log-snapshot means)')
    parser.add_argument('--log_util_csv', required=True, help='<mode>_val_district_util_1.csv (log-snapshot means)')
    parser.add_argument('--ci_csv', required=True, help='N-replicate CI dump CSV (CIs only)')
    parser.add_argument('--out', default='validation_hybrid')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'nyc':
        run_nyc(args, out_dir)
    else:
        run_chile(args, out_dir)


if __name__ == '__main__':
    main()
