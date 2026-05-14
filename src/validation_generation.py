"""
parse_validation.py
Generates individual validation plots from an EM log file.
Usage: python parse_validation.py --log <path> --out <dir>
"""

import re
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from constants import CHILE_PROVINCE_TO_REGION_MAPPING

METRICS = ['top3', 'top5', 'top10', 'unmatched']
METRIC_LABELS = {
    'top3':      'Top-3 Match Rate (%)',
    'top5':      'Top-5 Match Rate (%)',
    'top10':     'Top-10 Match Rate (%)',
    'unmatched': 'Unmatched Rate (%)',
}

OBS_COLOR  = "#0a17d1"
SIM_COLOR  = "#19d308"
FONT_SIZE  = 10
LEGEND_FONT_SIZE = 8
CHILE_REGION_LABEL_SIZE = 8
CHILE_ROTATION = 45
BAR_WIDTH  = 0.38

def aggregate_scalar_to_region(block, province_to_region, province_students):
    region_obs = {}
    region_sim = {}
    region_weights = {}

    for province, vals in block.items():
        region = province_to_region.get(str(province), str(province))
        n = province_students.get(str(province), 1)
        region_obs[region] = region_obs.get(region, 0) + n * vals['obs']
        region_sim[region] = region_sim.get(region, 0) + n * vals['sim']
        region_weights[region] = region_weights.get(region, 0) + n

    return {
        r: {'obs': region_obs[r] / region_weights[r],
            'sim': region_sim[r] / region_weights[r]}
        for r in region_obs
    }

def aggregate_to_region(block, province_to_region, province_students):

    region_obs_weighted = {}
    region_sim_weighted = {}
    region_weights = {}

    for province, vals in block.items():
        region = province_to_region.get(str(province), str(province))
        n = province_students.get(str(province), 1)  # fallback to 1 if missing
        if region not in region_obs_weighted:
            region_obs_weighted[region] = np.zeros(len(vals['obs']))
            region_sim_weighted[region] = np.zeros(len(vals['sim']))
            region_weights[region] = 0
        region_obs_weighted[region] += n * np.array(vals['obs'])
        region_sim_weighted[region] += n * np.array(vals['sim'])
        region_weights[region] += n

    return {
        r: {
            'obs': (region_obs_weighted[r] / region_weights[r]).tolist(),
            'sim': (region_sim_weighted[r] / region_weights[r]).tolist(),
        }
        for r in region_obs_weighted
    }

def get_metric_label(key):
    if key == 'unmatched':
        return 'Unmatched Rate (%)'
    m = re.match(r'top(\d+)', key)
    if m:
        return f'Top-{m.group(1)} Match Rate (%)'
    return key

def parse_log(path):
    """
    Returns best-iteration district fits, overall log-likelihood,
    per-school utilization at best iteration, and MAE utilization.
    """
    with open(path) as f:
        lines = f.readlines()

    best_ll      = -np.inf
    best_block   = {}   # district -> {'obs': [...], 'sim': [...]}
    best_util    = {}   # school_dbn -> {'obs': float, 'sim': float}
    best_mae_util = None

    current_district_obs = {}
    current_district_sim = {}
    current_util = {}
    current_dist_util = {}

    i = 0
    min_mae_util = None
    
    while i < len(lines):
        line = lines[i].rstrip()

        # District fit lines
        m = re.match(r'^\s*District (.+):\s*$', line.strip())
        if m:
            d = m.group(1).strip()
            obs_line = lines[i+1].strip() if i+1 < len(lines) else ''
            sim_line = lines[i+2].strip() if i+2 < len(lines) else ''
            obs_vals = re.findall(r'([\d.]+)%', obs_line)
            sim_vals = re.findall(r'([\d.]+)%', sim_line)
            if obs_vals and sim_vals and 'Observed' in obs_line and 'Simulated' in sim_line:
                current_district_obs[d] = [float(x) for x in obs_vals]
                current_district_sim[d] = [float(x) for x in sim_vals]
            i += 1
            continue

        # Global School Utilization header — reset current util block
        if 'Global School Utilization' in line:
            current_util = {}
            i += 1
            continue

        # School utilization lines  e.g. "  02M416_prog1: Obs=75.0%, Sim=60.2%, Diff=+14.8%"
        m = re.match(r'^\s+(\S+):\s+Obs=\s*([\d.]+)%,\s+Sim=\s*([\d.]+)%', line)
        if m:
            dbn  = m.group(1)
            obs  = float(m.group(2))
            sim  = float(m.group(3))
            current_util[dbn] = {'obs': obs, 'sim': sim}
            i += 1
            continue

        m = re.match(r'^\s+DIST_UTIL\s+(.+):\s+Obs=\s*([\d.]+)%,\s+Sim=\s*([\d.]+)%', line)
        if m:
            d = m.group(1).strip()
            current_dist_util[d] = {'obs': float(m.group(2)), 'sim': float(m.group(3))}
            i += 1
            continue

        # MAE utilization
        m = re.search(r'Mean Absolute Utilization Error:\s+([\d.]+)%', line)
        if m:
            current_mae_util = float(m.group(1))
            if min_mae_util is None or current_mae_util < min_mae_util:
                min_mae_util = current_mae_util
            i += 1
            continue

        # Log-likelihood — track the combined score reported per eval
        m = re.search(r'Match stats log-likelihood:\s*([-\d.]+),\s*Util penalty:\s*([-\d.]+)', line)
        if m:
            ll = float(m.group(1))
            if ll > best_ll:
                best_ll       = ll
                best_block    = {d: {'obs': current_district_obs[d],
                                     'sim': current_district_sim[d]}
                                 for d in current_district_obs
                                 if d in current_district_sim}
                best_util     = dict(current_util)
                best_dist_util = dict(current_dist_util)
                best_mae_util = current_mae_util if 'current_mae_util' in dir() else None
            i += 1
            continue

        i += 1

    return best_ll, best_block, best_util, best_mae_util, min_mae_util, best_dist_util


def district_from_dbn(dbn):
    """Extract district number from school DBN like '02M416_prog1' -> 2"""
    m = re.match(r'^(\d{2})', dbn)
    return int(m.group(1)) if m else None


# ── plotting ────────────────────────────────────────────────────────────────

def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {Path(path).name}")


def plot_metric(best_block, metric_idx, metric_key, out_path, mode='nyc'):
    districts = sorted(best_block.keys())
    if(mode == 'nyc'):
        districts = sorted(best_block.keys(), key=lambda d: int(d) if d.isdigit() else d)
    obs_vals  = [best_block[d]['obs'][metric_idx] for d in districts]
    sim_vals  = [best_block[d]['sim'][metric_idx] for d in districts]

    x   = np.arange(len(districts))
    fig, ax = plt.subplots(figsize=(11, 4))

    bars_obs = ax.bar(x - BAR_WIDTH/2, obs_vals, BAR_WIDTH,
                      label='Observed', color=OBS_COLOR, alpha=0.85)
    bars_sim = ax.bar(x + BAR_WIDTH/2, sim_vals, BAR_WIDTH,
                      label='Simulated', color=SIM_COLOR, alpha=0.85)


    ax.set_xticks(x)
    if mode == 'chile':
        labels = [re.sub(r'^de[l]?\s+', '', str(d).replace("Región", "").strip()) for d in districts]
        plt.setp(ax.get_xticklabels(), rotation=CHILE_ROTATION, ha='right', fontsize=CHILE_REGION_LABEL_SIZE)
    else:
        labels = [str(d) for d in districts]
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=FONT_SIZE)
    ax.set_xticklabels(labels)
    ax.set_ylabel(get_metric_label(metric_key), fontsize=FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
    ax.set_ylim(0, 105)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)

    fig.tight_layout()
    savefig(fig, out_path)


def plot_utilization_by_district(best_util, out_path, mode='nyc'):
    """Aggregate per-school utilization to district level."""
    district_obs = {}
    district_sim = {}
    for dbn, vals in best_util.items():
        d = district_from_dbn(dbn)
        if d is None:
            continue
        district_obs.setdefault(d, []).append(vals['obs'])
        district_sim.setdefault(d, []).append(vals['sim'])

    districts = sorted(set(district_obs) | set(district_sim))
    if(mode == 'nyc'):
        districts = sorted(best_block.keys(), key=lambda d: int(d) if d.isdigit() else d)
    obs_means = [np.mean(district_obs.get(d, [np.nan])) for d in districts]
    sim_means = [np.mean(district_sim.get(d, [np.nan])) for d in districts]

    x   = np.arange(len(districts))
    fig, ax = plt.subplots(figsize=(11, 4))

    ax.bar(x - BAR_WIDTH/2, obs_means, BAR_WIDTH,
           label='Observed', color=OBS_COLOR, alpha=0.85)
    ax.bar(x + BAR_WIDTH/2, sim_means, BAR_WIDTH,
           label='Simulated', color=SIM_COLOR, alpha=0.85)

    ax.set_xticks(x)
    if mode == 'chile':
        labels = [re.sub(r'^de[l]?\s+', '', str(d).replace("Región", "").strip()) for d in districts]
        plt.setp(ax.get_xticklabels(), rotation=CHILE_ROTATION, ha='right', fontsize=CHILE_REGION_LABEL_SIZE)
    else:
        labels = [str(d) for d in districts]
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=FONT_SIZE)
    print(districts)
    ax.set_xlabel('District (inferred from school DBN)', fontsize=FONT_SIZE)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Average School Utilization (%)', fontsize=FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
    ax.set_ylim(0, 105)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)

    fig.tight_layout()
    savefig(fig, out_path)


def plot_dist_util(dist_util, out_path, mode='nyc'):
    districts = sorted(dist_util.keys(), key=str)
    if mode == 'nyc':
        districts = sorted(dist_util.keys(), key=lambda d: int(d) if str(d).isdigit() else d)
    else:
        districts = sorted(dist_util.keys(), key=str)
    obs_means = [dist_util[d]['obs'] for d in districts]
    sim_means = [dist_util[d]['sim'] for d in districts]
    x = np.arange(len(districts))
    fig, ax = plt.subplots(figsize=(max(11, len(districts) * 0.7), 4))
    ax.bar(x - BAR_WIDTH/2, obs_means, BAR_WIDTH, label='Observed', color=OBS_COLOR, alpha=0.85)
    ax.bar(x + BAR_WIDTH/2, sim_means, BAR_WIDTH, label='Simulated', color=SIM_COLOR, alpha=0.85)

    ax.set_xticks(x)
    if mode == 'chile':
        labels = [re.sub(r'^de[l]?\s+', '', str(d).replace("Región", "").strip()) for d in districts]
        plt.setp(ax.get_xticklabels(), rotation=CHILE_ROTATION, ha='right', fontsize=CHILE_REGION_LABEL_SIZE)
    else:
        labels = [str(d) for d in districts]
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=FONT_SIZE)

    ax.set_xticklabels(labels)
    ax.set_ylabel('Average School Utilization (%)', fontsize=FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
    ax.set_ylim(0, 105)
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig(fig, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--out', default='validation_plots')
    parser.add_argument('--match_stats', required=False, default=None)
    parser.add_argument('--mode', choices=['nyc', 'chile'], default='nyc')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)


    print(f"Parsing {args.log} ...")
    best_ll, best_block, best_util, best_mae_util, min_mae_util, best_dist_util = parse_log(args.log)

    if best_block:
        sample_vals = next(iter(best_block.values()))
        n_stats = len(sample_vals['obs'])
        metrics_here = [f'top{p}' for p in range(1, n_stats)] + ['unmatched']
        idx_map = {k: i for i, k in enumerate(metrics_here)}

        if 'top3' in idx_map:
            top3_idx = idx_map['top3']
            obs_top3 = [best_block[d]['obs'][top3_idx] for d in best_block]
            sim_top3 = [best_block[d]['sim'][top3_idx] for d in best_block]
            diffs    = [o - s for o, s in zip(obs_top3, sim_top3)]

            districts_list = list(best_block.keys())
            abs_diffs = [abs(d) for d in diffs]
            print(f"\n── Top-3 Match Rate Summary ──────────────────")
            print(f"  Overall obs top-3:           {np.mean(obs_top3):.2f}%")
            print(f"  Overall sim top-3:           {np.mean(sim_top3):.2f}%")
            print(f"  Max district diff (abs):     {max(abs_diffs):.2f}pp  ({districts_list[abs_diffs.index(max(abs_diffs))]})")
            print(f"  Min district diff (abs):     {min(abs_diffs):.2f}pp  ({districts_list[abs_diffs.index(min(abs_diffs))]})")
            print(f"  Mean abs district diff:      {np.mean(abs_diffs):.2f}pp")
            print(f"──────────────────────────────────────────────\n")

    if args.match_stats:
        match_stats_df = pd.read_excel(args.match_stats)
        province_students = match_stats_df.set_index('Provincia')['n_students'].to_dict()
        province_students = {str(k): int(v) for k, v in province_students.items()}
        best_block = aggregate_to_region(best_block, CHILE_PROVINCE_TO_REGION_MAPPING, province_students)
        best_dist_util = aggregate_scalar_to_region(best_dist_util, CHILE_PROVINCE_TO_REGION_MAPPING, province_students)

        if best_dist_util:
            dist_list = list(best_dist_util.keys())
            obs_util_vals = [best_dist_util[d]['obs'] for d in dist_list]
            sim_util_vals = [best_dist_util[d]['sim'] for d in dist_list]
            util_diffs = [o - s for o, s in zip(obs_util_vals, sim_util_vals)]
            abs_util_diffs = [abs(d) for d in util_diffs]
            print(f"\n── Utilization Summary ────────────────────────")
            print(f"  Overall obs utilization:     {np.mean(obs_util_vals):.1f}%")
            print(f"  Overall sim utilization:     {np.mean(sim_util_vals):.1f}%")
            print(f"  Max district diff (abs):     {max(abs_util_diffs):.1f}pp  ({dist_list[abs_util_diffs.index(max(abs_util_diffs))]})")
            print(f"  Min district diff (abs):     {min(abs_util_diffs):.1f}pp  ({dist_list[abs_util_diffs.index(min(abs_util_diffs))]})")
            print(f"  Mean abs district diff:      {np.mean(abs_util_diffs):.1f}pp")
            print(f"──────────────────────────────────────────────\n")


    for region in sorted(best_block.keys()):
        vals = best_block[region]
        n_stats = len(vals['obs'])
        # find top3, top10, unmatched indices
        # metrics list matches what parse_log produces: [top1, top2, ..., unmatched]
        metrics_here = [f'top{p}' for p in range(1, n_stats)] + ['unmatched']
        idx = {k: i for i, k in enumerate(metrics_here)}
        top3    = vals['obs'][idx['top3']]   if 'top3'    in idx else None
        top10   = vals['obs'][idx['top10']]  if 'top10'   in idx else None
        unmatched = vals['obs'][idx['unmatched']]
        top3_sim    = vals['sim'][idx['top3']]   if 'top3'    in idx else None
        top10_sim   = vals['sim'][idx['top10']]  if 'top10'   in idx else None
        unmatched_sim = vals['sim'][idx['unmatched']]
        print(f"  {region}:")
        if top3 is not None:
            print(f"    top3:      obs={top3:.2f}%  sim={top3_sim:.2f}%  diff={top3-top3_sim:+.2f}")
        if top10 is not None:
            print(f"    top10:     obs={top10:.2f}%  sim={top10_sim:.2f}%  diff={top10-top10_sim:+.2f}")
        print(f"    unmatched: obs={unmatched:.2f}%  sim={unmatched_sim:.2f}%  diff={unmatched-unmatched_sim:+.2f}")


    print(f"\n── Validation Stats ──────────────────────────")
    print(f"  Overall log-likelihood (best iter): {best_ll:.2f}")
    print(f"  MAE utilization (at best LL eval):  {best_mae_util:.2f}%" if best_mae_util else "  MAE utilization: not found")
    print(f"  MAE utilization (best ever):        {min_mae_util:.2f}%" if min_mae_util else "")
    print(f"  Districts with fit data:            {len(best_block)}")
    print(f"  Schools with utilization data:      {len(best_util)}")
    print(f"──────────────────────────────────────────────\n")

    if not best_block:
        print("No district fit data found — check log format.")
        raise SystemExit(1)

    # Individual metric plots
    n_stats = len(next(iter(best_block.values()))['obs'])
    if n_stats == 4:
        metrics = ['top3', 'top5', 'top10', 'unmatched']
    else:
        metrics = [f'top{p}' for p in range(1, n_stats)] + ['unmatched']
    for mi, key in enumerate(metrics):
        plot_metric(best_block, mi, key,
                    out_dir / f'{args.mode}_val_{key}_by_district.png', mode=args.mode)

    
    if best_dist_util:
        plot_dist_util(best_dist_util, out_dir / f'{args.mode}_val_utilization_by_district.png', mode=args.mode)
    elif len(best_util) >= 20 and args.mode == 'nyc':
        plot_utilization_by_district(best_util, out_dir / f'{args.mode}_val_utilization_by_district.png', mode=args.mode)
    else:
        print(f"\nSkipping district utilization plot — only {len(best_util)} schools in best eval.")