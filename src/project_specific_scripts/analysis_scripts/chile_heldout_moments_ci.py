"""
Held-out-moments comparison (Fig 21 / regional panels style), with 95% CIs on
each config's simulated curve from N independent replicate simulations at
that config's fixed, already-fitted params.pkl -- not the single-log point
estimate trace_experiment_comparison_plot.py uses. Observed has no CI (it's
the real population statistic, not a sample estimate).

The 6 configs/params.pkl paths below were identified by matching each
heldout script's --loss_top_p args to chilean_experiment_driver.py's
_loss_label() naming convention, then picking the most recent post-indexing-
-fix run (Oct 14-15 batch) that actually has a saved params.pkl -- verified
against experimental_output/10_14_Heldouts_Progress.txt's point estimates
and checked for zero "not in ranking" warnings (the pre-fix bug's signature).

Usage:
    python3 src/project_specific_scripts/chile_heldout_moments_ci.py \
        --n_runs 10 --seed 42 --out chile_heldout_ci

    # Replot only (e.g. after a styling change), reusing a previous run's
    # mean/CI dump instead of resimulating:
    python3 src/project_specific_scripts/chile_heldout_moments_ci.py \
        --from_csv chile_heldout_ci/chile_heldout_moments_ci.csv --out chile_heldout_ci
"""
import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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

# Configs with a params.pkl that's actually compatible (61-district Santiago
# split) with this script's data loading -- these get N-replicate-simulation
# CIs.
CONFIGS = {
    'ALL MOMENTS (FULL)':
        'experimental_output/chile_res_logs/20261014_102833/chilean_experiment_K=6_M=15_iter=13_opt=15_lr=0.05_maxp=10_loss=allTopP+unm_seed=40_20261014_102833_params.pkl',
    'TOP-1 ONLY':
        'experimental_output/chile_res_logs/20261014_104428/chilean_experiment_K=6_M=15_iter=13_opt=15_lr=0.05_maxp=10_loss=top1_seed=40_20261014_104428_params.pkl',
    'TOP-1 AND UNMATCHED ONLY':
        'experimental_output/chile_res_logs/20261014_105426/chilean_experiment_K=6_M=15_iter=13_opt=15_lr=0.05_maxp=10_loss=top1+unm_seed=40_20261014_105426_params.pkl',
    'TOP-1,2,3,4,5 AND UNMATCHED ONLY':
        'experimental_output/chile_res_logs/20261015_023409/chilean_experiment_K=6_M=15_iter=13_opt=15_lr=0.05_maxp=10_loss=top1-2-3-4-5+unm_seed=40_20261015_023409_params.pkl',
    'TOP-3,5,10 AND UNMATCHED ONLY':
        'experimental_output/chile_res_logs/20261015_025413/chilean_experiment_K=6_M=15_iter=13_opt=15_lr=0.05_maxp=10_loss=top3-5-10+unm_seed=40_20261015_025413_params.pkl',
}

# Configs with NO usable params.pkl (the only one on disk is fit on a
# different, incompatible districting scheme -- see the district-set
# validation check below). Plotted as a single point-estimate line with a
# zero-width (i.e. invisible) CI band, parsed directly out of the
# already-region-aggregated Match Rate Analysis block for that experiment,
# rather than re-simulating.
NO_CI_CONFIGS = {
    'TOP-1,2,3 AND UNMATCHED ONLY': ('experimental_output/10_14_Heldouts_Progress.txt',
                                      'TOP-1,2,3 AND UNMATCHED ONLY'),
}

PALETTE = ('#19d308', '#e67e22', '#8e44ad', '#16a085', '#d62728', '#9467bd', '#7f7f7f')
OBSERVED_COLOR = '#0a17d1'

METRIC_LINE_RE = re.compile(r'^\s*(top\d+|unmatched):\s*obs=([\d.]+)%\s+sim=([\d.]+)%')
EXPERIMENT_RE = re.compile(r'^EXPERIMENT:\s*(.+?)\s*$', re.MULTILINE)


def parse_no_ci_config(log_path, experiment_name, max_p):
    """Pull one EXPERIMENT block's Overall + per-region sim values out of a
    trace_experiment_comparison_plot.py-style concatenated log (already
    region-aggregated by validation_generation.py upstream). Returns
    (overall_sim: dict[metric->value], regions_sim: dict[region->dict])."""
    text = Path(log_path).read_text()
    matches = list(EXPERIMENT_RE.finditer(text))
    block = None
    for i, m in enumerate(matches):
        if m.group(1).strip() == experiment_name:
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[m.end():end]
            break
    if block is None:
        raise ValueError(f"No EXPERIMENT block named {experiment_name!r} found in {log_path}")

    lines = block.splitlines()
    overall_start = next((i for i, l in enumerate(lines) if l.strip() == 'Overall:'), None)
    district_start = next((i for i, l in enumerate(lines) if l.strip() == 'Per district:'), None)
    if overall_start is None or district_start is None:
        raise ValueError(f"Block {experiment_name!r} in {log_path} missing Overall:/Per district: sections")

    def parse_metric_lines(line_slice):
        out = {}
        for l in line_slice:
            m = METRIC_LINE_RE.match(l)
            if m:
                out[m.group(1)] = float(m.group(3))  # sim value
        return out

    overall_sim = parse_metric_lines(lines[overall_start + 1:district_start])
    regions_sim = {}
    current = None
    for l in lines[district_start + 1:]:
        dm = re.match(r'^\s{2}(.+):\s*$', l)
        if dm:
            current = dm.group(1).strip()
            regions_sim[current] = {}
            continue
        mm = METRIC_LINE_RE.match(l)
        if mm and current is not None:
            regions_sim[current][mm.group(1)] = float(mm.group(3))
    return overall_sim, regions_sim


def mean_ci(values):
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(1.96 * np.std(arr) / np.sqrt(len(arr)))


def _plot_comparison_standalone(obs_curve, sims, plot_metrics, title, output_path):
    """sims: dict[config_name] -> (mean_list, ci_list), both in plot_metrics order."""
    x = np.arange(len(plot_metrics))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, obs_curve, marker='o', linewidth=2, color=OBSERVED_COLOR, label='Observed')
    for i, (name, (mean, ci)) in enumerate(sims.items()):
        color = PALETTE[i % len(PALETTE)]
        mean_arr, ci_arr = np.array(mean), np.array(ci)
        ax.errorbar(x, mean_arr, yerr=ci_arr, marker='o', linewidth=2, color=color, label=name,
                    ecolor=color, elinewidth=1.4, capsize=3, capthick=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_metrics)
    ax.set_xlabel('Rank cutoff (p)')
    ax.set_ylabel('Match rate (%)')
    ax.set_title(title)
    all_vals = list(obs_curve) + [v for mean, _ in sims.values() for v in mean]
    lower = max(0, np.floor(min(all_vals) - 5))
    upper = min(100, np.ceil(max(all_vals) + 5))
    ax.set_ylim(lower, upper)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_jobs', type=int, default=32)
    parser.add_argument('--max_p', type=int, default=10)
    parser.add_argument('--out', default='chile_heldout_ci')
    parser.add_argument('--from_csv', default=None,
                         help='Path to a previous chile_heldout_moments_ci.csv dump -- skips all '
                              'simulation/parsing and replots from those saved mean/CI values.')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    indv_df = read_data(f"{CHILEAN_DATA_DIR}/{CHILEAN_INDV_PREF_PROVINCE_SANTIAGOZONES_FILEPATH}")
    match_df = read_data(f"{CHILEAN_DATA_DIR}/{CHILEAN_MATCH_OUTCOME_PROVINCE_SANTIAGOZONES_FILEPATH}")
    school_cap_df = read_data(f"{CHILEAN_DATA_DIR}/{CHILEAN_SCHOOL_CAPACITY_BY_REGION_PROVINCE_SANTIAGOZONES_FILEPATH}")

    df, match_stats_df, school_info_df = preprocess_chilean_data(
        indv_df, match_df, school_cap_df, is_province_level=True
    )
    list_length_params = return_chilean_list_params(indv_df=indv_df)

    chile_config_path = f"{CHILEAN_DATA_DIR}/{CHILE_CONFIG_FILEPATH}"
    priority_config = None
    if os.path.exists(chile_config_path):
        with open(chile_config_path) as f:
            priority_config = json.load(f)
        print(f"Loaded priority config: {chile_config_path}")

    district_to_region_sim_arg = {str(r): str(r) for r in df['Residential District'].unique()}
    districts = sorted(df['Residential District'].unique())
    weights = {d: int(match_stats_df[match_stats_df['Residential District'] == d]['Total Applicants'].iloc[0])
               for d in districts}
    n_students_total = sum(weights.values())
    print(f"n districts: {len(districts)}, n students: {n_students_total}")

    metric_names = [f'top{p}' for p in range(1, args.max_p + 1)] + ['unmatched']

    obs_by_district = {}
    for d in districts:
        row = match_stats_df[match_stats_df['Residential District'] == d].iloc[0]
        cols = [f'% Matches to Choice 1-{p}' for p in range(1, args.max_p + 1)]
        vec = [row[c] if c in row else np.nan for c in cols] + [row['Unmatched']]
        obs_by_district[d] = np.array(vec, dtype=float)

    def aggregate_to_region(values_by_district):
        region_sum, region_w = {}, {}
        for d, v in values_by_district.items():
            r = CHILE_PROVINCE_TO_REGION_MAPPING.get(d, d)
            w = weights.get(d, 1)
            region_sum[r] = region_sum.get(r, 0) + w * np.asarray(v, dtype=float)
            region_w[r] = region_w.get(r, 0) + w
        return {r: region_sum[r] / region_w[r] for r in region_sum}

    obs_by_region = aggregate_to_region(obs_by_district)
    obs_overall = sum(weights[d] * obs_by_district[d] for d in districts) / n_students_total

    plot_metrics = [f'top{p}' for p in range(1, args.max_p + 1)] + ['matched']

    if args.from_csv:
        print(f"Replot-only mode: reading {args.from_csv}, skipping all simulation/parsing.")
        csv_df = pd.read_csv(args.from_csv)
        all_config_names = list(csv_df['config'].unique())

        def curve_from_csv(scope, name):
            sub = csv_df[(csv_df['config'] == name) & (csv_df['scope'] == scope)].set_index('metric')
            mean, ci = [], []
            for m in metric_names[:-1]:  # top1..top_max_p
                mean.append(sub.loc[m, 'sim_mean'])
                ci.append(sub.loc[m, 'sim_ci95'])
            mean.append(100 - sub.loc['unmatched', 'sim_mean'])
            ci.append(sub.loc['unmatched', 'sim_ci95'])
            return mean, ci

        sims_overall = {name: curve_from_csv('overall', name) for name in all_config_names}
        obs_curve_overall = list(obs_overall[:args.max_p]) + [100 - obs_overall[-1]]
        _plot_comparison_standalone(
            obs_curve_overall, sims_overall, plot_metrics,
            'Overall top-p comparison (95% CI; TOP-1,2,3 AND UNMATCHED ONLY has no CI -- see script docstring)',
            out_dir / 'overall_top_p_comparison_ci.png'
        )
        for region, obs_vec in obs_by_region.items():
            if not all(((csv_df['config'] == name) & (csv_df['scope'] == region)).any() for name in all_config_names):
                print(f"Skipping {region}: missing from at least one config's saved results")
                continue
            sims_region = {name: curve_from_csv(region, name) for name in all_config_names}
            obs_curve_region = list(obs_vec[:args.max_p]) + [100 - obs_vec[-1]]
            fname = re.sub(r'[^A-Za-z0-9_.-]+', '_', region).strip('_')
            _plot_comparison_standalone(
                obs_curve_region, sims_region, plot_metrics, f'{region} top-p comparison (with 95% CI)',
                out_dir / f'{fname}_top_p_comparison_ci.png'
            )
        print("Done (replot-only).")
        return

    # Fail fast, before burning any compute: every config's params.pkl must use
    # the SAME district split as `districts` (the Santiago-macrozone, 61-way
    # split this script's data loading uses). A params.pkl fit on the plain
    # (54-district, unsplit 'Santiago') granularity will crash deep inside
    # run_single_simulation's match_stats_df lookup -- verified the hard way,
    # this check exists so that failure happens up front instead.
    expected_districts = set(districts)
    loaded_params = {}
    for name, pkl_path in CONFIGS.items():
        with open(pkl_path, 'rb') as f:
            params = pickle.load(f)
        loaded_params[name] = params
        got = set(params['districts'].keys())
        if got != expected_districts:
            missing = expected_districts - got
            extra = got - expected_districts
            raise SystemExit(
                f"District-set mismatch for config {name!r} ({pkl_path}):\n"
                f"  missing from params: {sorted(missing)}\n"
                f"  extra in params:     {sorted(extra)}\n"
                f"This params.pkl was fit on a different districting scheme "
                f"(likely the plain, unsplit-Santiago province level) than the "
                f"other configs -- it is not comparable, fix CONFIGS before rerunning."
            )
    print(f"All {len(CONFIGS)} simulated configs use the same district split -- proceeding "
          f"({len(NO_CI_CONFIGS)} additional config(s) will be plotted from a parsed point estimate, no CI).")

    results = {}
    for name in CONFIGS:
        print(f"\n=== {name} ===")
        params = loaded_params[name]
        overall_runs = {m: [] for m in metric_names}
        region_runs = {}
        for run in range(args.n_runs):
            print(f"  Run {run + 1}/{args.n_runs}...")
            rng_lottery = np.random.default_rng(seed=args.seed + run)
            lottery_fixed = rng_lottery.permutation(n_students_total)
            agg = run_single_simulation(
                params=params, df=df, match_stats_df=match_stats_df, school_info_df=school_info_df,
                lottery_fixed=lottery_fixed, mallows_seed=args.seed + run, sampling_n_jobs=args.n_jobs,
                priority_config=priority_config, district_to_region=district_to_region_sim_arg,
                list_length_params=list_length_params, raw_capacity_df=school_cap_df,
                max_p=args.max_p,
            )
            sim_by_district = {d: agg['match_stats'][i, :] for i, d in enumerate(districts)}
            sim_by_region = aggregate_to_region(sim_by_district)
            sim_overall = sum(weights[d] * sim_by_district[d] for d in districts) / n_students_total
            for i, m in enumerate(metric_names):
                overall_runs[m].append(sim_overall[i])
                for r, v in sim_by_region.items():
                    region_runs.setdefault(r, {mm: [] for mm in metric_names})
                    region_runs[r][m].append(v[i])
        results[name] = {'overall': overall_runs, 'regions': region_runs}

    # No-CI configs: parse the single point estimate directly, wrapped as a
    # 1-element list so mean_ci() naturally gives ci=0 (std of one value) and
    # every downstream function (curve_for, plot_comparison) needs no special
    # casing -- a zero-width fill_between band is visually just a line.
    for name, (log_path, experiment_name) in NO_CI_CONFIGS.items():
        print(f"\n=== {name} (no CI -- parsed point estimate from {log_path}) ===")
        overall_sim, regions_sim = parse_no_ci_config(log_path, experiment_name, args.max_p)
        overall_runs = {m: [overall_sim[m]] for m in metric_names if m in overall_sim}
        region_runs = {r: {m: [v[m]] for m in metric_names if m in v} for r, v in regions_sim.items()}
        results[name] = {'overall': overall_runs, 'regions': region_runs}

    all_config_names = list(CONFIGS) + list(NO_CI_CONFIGS)

    def curve_for(obs_vec, sim_runs_dict):
        sim_mean, sim_ci = [], []
        for m in metric_names[:-1]:
            mean, ci = mean_ci(sim_runs_dict[m])
            sim_mean.append(mean)
            sim_ci.append(ci)
        mean_u, ci_u = mean_ci(sim_runs_dict['unmatched'])
        sim_mean.append(100 - mean_u)
        sim_ci.append(ci_u)
        return sim_mean, sim_ci

    # Overall
    sims_overall = {name: curve_for(obs_overall, results[name]['overall']) for name in all_config_names}
    obs_curve_overall = list(obs_overall[:args.max_p]) + [100 - obs_overall[-1]]
    _plot_comparison_standalone(
        obs_curve_overall, sims_overall, plot_metrics,
        'Overall top-p comparison (95% CI; TOP-1,2,3 AND UNMATCHED ONLY has no CI -- see script docstring)',
        out_dir / 'overall_top_p_comparison_ci.png'
    )

    # Per region
    for region, obs_vec in obs_by_region.items():
        if not all(region in results[name]['regions'] for name in all_config_names):
            print(f"Skipping {region}: missing from at least one config's results")
            continue
        sims_region = {name: curve_for(obs_vec, results[name]['regions'][region]) for name in all_config_names}
        obs_curve_region = list(obs_vec[:args.max_p]) + [100 - obs_vec[-1]]
        fname = re.sub(r'[^A-Za-z0-9_.-]+', '_', region).strip('_')
        _plot_comparison_standalone(
            obs_curve_region, sims_region, plot_metrics, f'{region} top-p comparison (with 95% CI)',
            out_dir / f'{fname}_top_p_comparison_ci.png'
        )

    # Raw dump for reference/reuse.
    dump_rows = []
    for name in all_config_names:
        for m in metric_names:
            if m not in results[name]['overall']:
                continue
            mean, ci = mean_ci(results[name]['overall'][m])
            dump_rows.append({'config': name, 'scope': 'overall', 'metric': m,
                               'sim_mean': mean, 'sim_ci95': ci})
        for region in results[name]['regions']:
            for m in metric_names:
                if m not in results[name]['regions'][region]:
                    continue
                mean, ci = mean_ci(results[name]['regions'][region][m])
                dump_rows.append({'config': name, 'scope': region, 'metric': m,
                                   'sim_mean': mean, 'sim_ci95': ci})
    pd.DataFrame(dump_rows).to_csv(out_dir / 'chile_heldout_moments_ci.csv', index=False)
    print(f"Saved: {out_dir / 'chile_heldout_moments_ci.csv'}")
    print("Done.")


if __name__ == '__main__':
    main()
