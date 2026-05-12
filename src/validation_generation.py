"""
parse_validation.py
Generates individual validation plots from an EM log file.
Usage: python parse_validation.py --log <path> --out <dir>
"""

import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

METRICS = ['top3', 'top5', 'top10', 'unmatched']
METRIC_LABELS = {
    'top3':      'Top-3 Match Rate (%)',
    'top5':      'Top-5 Match Rate (%)',
    'top10':     'Top-10 Match Rate (%)',
    'unmatched': 'Unmatched Rate (%)',
}

OBS_COLOR  = "#0a17d1"
SIM_COLOR  = "#19d308"
FONT_SIZE  = 9
BAR_WIDTH  = 0.38


# ── parsing ────────────────────────────────────────────────────────────────

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


def plot_metric(best_block, metric_idx, metric_key, out_path):
    districts = sorted(best_block.keys())
    obs_vals  = [best_block[d]['obs'][metric_idx] for d in districts]
    sim_vals  = [best_block[d]['sim'][metric_idx] for d in districts]

    x   = np.arange(len(districts))
    fig, ax = plt.subplots(figsize=(11, 4))

    bars_obs = ax.bar(x - BAR_WIDTH/2, obs_vals, BAR_WIDTH,
                      label='Observed', color=OBS_COLOR, alpha=0.85)
    bars_sim = ax.bar(x + BAR_WIDTH/2, sim_vals, BAR_WIDTH,
                      label='Simulated', color=SIM_COLOR, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in districts], fontsize=FONT_SIZE*0.5, rotation=80)
    ax.set_xlabel('Residential District', fontsize=FONT_SIZE)
    ax.set_ylabel(METRIC_LABELS[metric_key], fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)

    fig.tight_layout()
    savefig(fig, out_path)


def plot_utilization_by_district(best_util, out_path):
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
    obs_means = [np.mean(district_obs.get(d, [np.nan])) for d in districts]
    sim_means = [np.mean(district_sim.get(d, [np.nan])) for d in districts]

    x   = np.arange(len(districts))
    fig, ax = plt.subplots(figsize=(11, 4))

    ax.bar(x - BAR_WIDTH/2, obs_means, BAR_WIDTH,
           label='Observed', color=OBS_COLOR, alpha=0.85)
    ax.bar(x + BAR_WIDTH/2, sim_means, BAR_WIDTH,
           label='Simulated', color=SIM_COLOR, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in districts], fontsize=FONT_SIZE*0.5, rotation=80)
    ax.set_xlabel('District (inferred from school DBN)', fontsize=FONT_SIZE)
    ax.set_ylabel('Average School Utilization (%)', fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)

    fig.tight_layout()
    savefig(fig, out_path)

def plot_dist_util(dist_util, out_path):
    districts = sorted(dist_util.keys(), key=str)
    obs_means = [dist_util[d]['obs'] for d in districts]
    sim_means = [dist_util[d]['sim'] for d in districts]
    x = np.arange(len(districts))
    fig, ax = plt.subplots(figsize=(max(11, len(districts) * 0.7), 4))
    ax.bar(x - BAR_WIDTH/2, obs_means, BAR_WIDTH, label='Observed', color=OBS_COLOR, alpha=0.85)
    ax.bar(x + BAR_WIDTH/2, sim_means, BAR_WIDTH, label='Simulated', color=SIM_COLOR, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in districts], fontsize=FONT_SIZE*0.5, rotation=80)
    ax.set_xlabel('District', fontsize=FONT_SIZE)
    ax.set_ylabel('Average School Utilization (%)', fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig(fig, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--out', default='validation_plots')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {args.log} ...")
    best_ll, best_block, best_util, best_mae_util, min_mae_util, best_dist_util = parse_log(args.log)


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
                    out_dir / f'val_{key}_by_district.png')

    # Utilization by district
    if best_dist_util:
        plot_dist_util(best_dist_util, out_dir / 'val_utilization_by_district.png')
    elif len(best_util) >= 20:
        plot_utilization_by_district(best_util, out_dir / 'val_utilization_by_district.png')
    else:
        print(f"\nSkipping district utilization plot — only {len(best_util)} schools in best eval.")