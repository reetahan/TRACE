

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from file_config import DATA_GENERATION_SEED
from chile_priority_attributes import _prepare_school_capacity_table
from chilean_real_welfare_comparison import (
    load_df, build_applications_long, run_matching,
)

N_DECILES = 10

AGG_TO_SOURCE = {
    'min': 'min_topk',
    'median': 'median_topk',
    'mean': 'mean_topk',
    'rank_weighted_avg': 'rank_weighted_topk',
    'first_choice': 'first_choice',
    'matched_school': 'matched_school',
}
CONDENSED_AGGS = ['median', 'mean', 'rank_weighted_avg', 'first_choice', 'matched_school']
AGG_TITLES = {
    'min': 'Min (top-5)',
    'median': 'Median (top-5)',
    'mean': 'Mean (top-5)',
    'rank_weighted_avg': 'Rank-weighted avg (top-5)',
    'first_choice': 'First choice',
    'matched_school': 'Matched school',
}


def compute_student_records(student_ids, student_rankings, matches, lottery_df, decile_source, topk=5):
    """
    Per-student matched/rank_pos, plus the scalar used for decile bucketing.
    """
    if decile_source == 'single':
        decile_val = lottery_df.drop_duplicates('student_idx').set_index('student_idx')['lottery']
    elif decile_source == 'min_topk':
        decile_val = (
            lottery_df[lottery_df['preference_number'] <= topk]
            .groupby('student_idx')['lottery'].min()
        )
    elif decile_source == 'median_topk':
        decile_val = (
            lottery_df[lottery_df['preference_number'] <= topk]
            .groupby('student_idx')['lottery'].median()
        )
    elif decile_source == 'mean_topk':
        decile_val = (
            lottery_df[lottery_df['preference_number'] <= topk]
            .groupby('student_idx')['lottery'].mean()
        )
    elif decile_source == 'rank_weighted_topk':
        topk_df = lottery_df[lottery_df['preference_number'] <= topk].copy()
        topk_df['weight'] = 1.0 / topk_df['preference_number']
        topk_df['weighted_lottery'] = topk_df['lottery'] * topk_df['weight']
        grouped = topk_df.groupby('student_idx')
        decile_val = grouped['weighted_lottery'].sum() / grouped['weight'].sum()
    elif decile_source == 'first_choice':
        decile_val = (
            lottery_df.sort_values(['student_idx', 'preference_number'])
            .groupby('student_idx').first()['lottery']
        )
    elif decile_source == 'matched_school':
        matched_df = pd.DataFrame({
            'student_idx': np.arange(len(student_ids)),
            'school_idx': np.asarray(matches).astype(int),
        })
        lookup = lottery_df.drop_duplicates(subset=['student_idx', 'school_idx'], keep='first')
        merged = matched_df.merge(
            lookup[['student_idx', 'school_idx', 'lottery']],
            on=['student_idx', 'school_idx'], how='left',
        )
        decile_val = merged.set_index('student_idx')['lottery']
        worst_fallback = lottery_df.groupby('student_idx')['lottery'].min()
        decile_val = decile_val.fillna(worst_fallback)
    else:
        raise ValueError(f"Unknown decile_source: {decile_source!r}")

    records = []
    for i, sid in enumerate(student_ids):
        matched_idx = int(matches[i])
        ranking = student_rankings[i]
        rank_pos = None
        if matched_idx >= 0:
            try:
                rank_pos = ranking.index(matched_idx) + 1
            except ValueError:
                pass
        records.append({
            'matched': matched_idx >= 0,
            'rank_pos': rank_pos,
            'decile_val': decile_val.get(i, np.nan),
        })
    df = pd.DataFrame(records).dropna(subset=['decile_val'])
    df['decile'] = pd.qcut(df['decile_val'], N_DECILES, labels=range(1, N_DECILES + 1))
    return df


def decile_metrics(df):
    rows = []
    for d in range(1, N_DECILES + 1):
        sub = df[df['decile'] == d]
        n = len(sub)
        if n == 0:
            continue
        rows.append({
            'decile': d,
            'pct_unmatched': 100.0 * (~sub['matched']).sum() / n,
            'top1_pct': 100.0 * (sub['rank_pos'] == 1).sum() / n,
            'top5_pct': 100.0 * (sub['rank_pos'] <= 5).sum() / n,
            'avg_rank': sub.loc[sub['matched'], 'rank_pos'].mean(),
        })
    return pd.DataFrame(rows).set_index('decile')


def run_replicates(get_indv_df_for_replicate, school_table, rng, n_runs, agg_names, topk):
    """
    Runs n_runs replicates once. Each replicate matches STB once and MTB once,
    then computes decile metrics for every name in agg_names from that SAME
    MTB match -- MTB is never re-run per aggregation. Returns
    (stats, decile_value_pool) where stats has keys 'STB' plus every agg_name,
    each {'mean': df, 'ci95': df}.
    """
    metrics_by_name = {'STB': [], **{name: [] for name in agg_names}}
    decile_value_pool = {'STB': [], **{name: [] for name in agg_names}}

    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}...")
        indv_df = get_indv_df_for_replicate(rng)
        applications_long = build_applications_long(indv_df)
        all_student_ids = sorted(applications_long['mrun'].unique().tolist())
        if run == 0:
            print(f"  Students: {len(all_student_ids):,}")

        student_lottery = {sid: float(rng.random()) for sid in all_student_ids}
        student_ids, rankings, matches, lottery_df = run_matching(
            applications_long, school_table, rng,
            student_lottery=student_lottery, return_lottery_df=True,
        )
        df = compute_student_records(student_ids, rankings, matches, lottery_df, 'single')
        metrics_by_name['STB'].append(decile_metrics(df))
        decile_value_pool['STB'].append(df['decile_val'].to_numpy())

        student_ids, rankings, matches, lottery_df = run_matching(
            applications_long, school_table, rng,
            student_lottery=None, return_lottery_df=True,
        )
        for name in agg_names:
            df = compute_student_records(student_ids, rankings, matches, lottery_df,
                                          AGG_TO_SOURCE[name], topk=topk)
            metrics_by_name[name].append(decile_metrics(df))
            decile_value_pool[name].append(df['decile_val'].to_numpy())

    stats = {}
    for name, records in metrics_by_name.items():
        grouped = pd.concat(records).groupby(level=0)
        mean = grouped.mean()
        std = grouped.std()
        stats[name] = {'mean': mean, 'ci95': 1.96 * std / np.sqrt(n_runs)}

    return stats, decile_value_pool


DECILE_COLORS = {'pct_unmatched': '#111111', 'top1_pct': '#1565C0', 'top5_pct': '#AD1457', 'avg_rank': '#D4A017'}
DECILE_LABELS = {'pct_unmatched': 'Unmatched', 'top1_pct': 'Top-1', 'top5_pct': 'Top-5', 'avg_rank': 'Avg Rank'}
DECILE_STYLES = {'STB': '-', 'MTB': '--'}
DECILE_MARKERS = {'pct_unmatched': 'o', 'top1_pct': 's', 'top5_pct': '^', 'avg_rank': 'D'}


def plot_decile_axes(ax, ax2, stats, mtb_name, deciles, markersize=6, linewidth=1.8, capsize=5, capthick=2):
    """Draw STB + stats[mtb_name] (MTB) errorbar lines onto ax (match rate) / ax2 (avg rank)."""
    for cond, name in [('STB', 'STB'), ('MTB', mtb_name)]:
        for metric in ['pct_unmatched', 'top1_pct', 'top5_pct']:
            mean_vals = stats[name]['mean'][metric].reindex(deciles)
            ci_vals = stats[name]['ci95'][metric].reindex(deciles)
            ax.errorbar(deciles, mean_vals, yerr=ci_vals,
                        color=DECILE_COLORS[metric], linestyle=DECILE_STYLES[cond], marker=DECILE_MARKERS[metric],
                        markersize=markersize, linewidth=linewidth,
                        ecolor='#000000', elinewidth=capthick, capsize=capsize, capthick=capthick, zorder=6)
        mean_vals = stats[name]['mean']['avg_rank'].reindex(deciles)
        ci_vals = stats[name]['ci95']['avg_rank'].reindex(deciles)
        ax2.errorbar(deciles, mean_vals, yerr=ci_vals,
                    color=DECILE_COLORS['avg_rank'], linestyle=DECILE_STYLES[cond], marker=DECILE_MARKERS['avg_rank'],
                    markersize=markersize, linewidth=linewidth,
                    ecolor='#000000', elinewidth=capthick, capsize=capsize, capthick=capthick, zorder=6)
    ax2.tick_params(axis='y', colors=DECILE_COLORS['avg_rank'])
    ax2.spines['right'].set_color(DECILE_COLORS['avg_rank'])
    ax.spines['top'].set_visible(False)


def plot_decile_value_distributions(stb_vals, mtb_vals, output_path):
    """Overlaid histograms of the raw, pre-qcut decile-bucketing value for STB vs MTB."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(
        min(stb_vals.min(), mtb_vals.min()),
        max(stb_vals.max(), mtb_vals.max()),
        80,
    )
    ax.hist(stb_vals, bins=bins, density=True, alpha=0.45, color='#1565C0', label='STB (single draw)')
    ax.hist(mtb_vals, bins=bins, density=True, alpha=0.45, color='#AD1457', label='MTB (current aggregation)')

    for vals, color in [(stb_vals, '#1565C0'), (mtb_vals, '#AD1457')]:
        boundaries = np.quantile(vals, np.arange(1, N_DECILES) / N_DECILES)
        for b in boundaries:
            ax.axvline(b, color=color, linewidth=0.6, alpha=0.5, linestyle=':')

    ax.set_xlabel('Decile-defining value (raw, pre-qcut)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('STB vs MTB: distribution of the value each condition is decile-bucketed by\n'
                 '(dotted lines = the 9 decile boundaries actually used for that condition)', fontsize=10)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--individual', default=None,
                         help='Fixed --individual CSV. Mutually exclusive with --params_pkl.')
    parser.add_argument('--params_pkl', default=None,
                         help='Fitted Mallows params.pkl; resamples preferences + lottery each replicate. '
                              'Mutually exclusive with --individual.')
    parser.add_argument('--real_individual', default=None,
                         help='Real indv_df, used with --params_pkl to calibrate synthetic resampling.')
    parser.add_argument('--subdivision_col', default='Region',
                         help="Only used with --params_pkl: 'Region' or 'Provincia'.")
    parser.add_argument('--capacity', required=True)
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--mtb_decile_agg',
                         choices=['min', 'median', 'mean', 'rank_weighted_avg', 'first_choice', 'matched_school'],
                         default='rank_weighted_avg',
                         help="How to collapse a student's per-school MTB lottery draws into one scalar.")
    parser.add_argument('--mtb_decile_topk', type=int, default=5,
                         help="Top-k preferences to aggregate over for min/median/mean/rank_weighted_avg. "
                              "Ignored by first_choice and matched_school.")
    parser.add_argument('--output', default=None,
                         help='Defaults to fig4_chile_lottery_decile_<agg tag>.png.')
    parser.add_argument('--output_dist', default=None,
                         help='Defaults to fig4_decile_value_distributions_<agg tag>.png.')
    parser.add_argument('--condensed', action='store_true',
                         help='Instead of the single --mtb_decile_agg figure, produce one multi-panel '
                              'figure with all OTHER aggregations (CONDENSED_AGGS) as subplots, sharing '
                              'one legend. Skips the distribution plot / CSV dump / console diagnostics.')
    parser.add_argument('--output_condensed', default='fig4_chile_lottery_decile_condensed.png')
    parser.add_argument('--seed', type=int, default=DATA_GENERATION_SEED)
    args = parser.parse_args()

    mtb_decile_source = AGG_TO_SOURCE[args.mtb_decile_agg]
    uses_topk = mtb_decile_source.endswith('_topk')
    agg_tag = f'{args.mtb_decile_agg}_k{args.mtb_decile_topk}' if uses_topk else args.mtb_decile_agg

    if args.output is None:
        args.output = f'fig4_chile_lottery_decile_{agg_tag}.png'
    if args.output_dist is None:
        args.output_dist = f'fig4_decile_value_distributions_{agg_tag}.png'

    if bool(args.individual) == bool(args.params_pkl):
        parser.error('Pass exactly one of --individual or --params_pkl.')
    if args.params_pkl and not args.real_individual:
        parser.error('--params_pkl requires --real_individual (for calibration).')

    print("Loading data...")
    capacity_df = load_df(args.capacity)
    school_table = _prepare_school_capacity_table(capacity_df)  # capacities are always real, never synthetic

    if args.individual:
        fixed_indv_df = load_df(args.individual)
        mallows_params = None
        real_indv_df_for_calibration = None
    else:
        fixed_indv_df = None
        real_indv_df_for_calibration = load_df(args.real_individual)
        with open(args.params_pkl, 'rb') as f:
            mallows_params = pickle.load(f)

    def get_indv_df_for_replicate(rng):
        if fixed_indv_df is not None:
            return fixed_indv_df
        from build_synthetic_chile_indv_df import build_synthetic_indv_df
        return build_synthetic_indv_df(
            mallows_params, real_indv_df_for_calibration, capacity_df, rng,
            subdivision_col=args.subdivision_col, n_jobs=1, verbose=False,
        )

    print(f"  Schools: {len(school_table):,}")
    print(f"  Mode: {'fixed --individual, lottery-only CI' if fixed_indv_df is not None else 'resampling preferences + lottery each replicate'}")

    rng = np.random.default_rng(args.seed)

    if args.condensed:
        print(f"  Condensed aggregations: {CONDENSED_AGGS}")
        print(f"  Output: {args.output_condensed}")
        stats, _ = run_replicates(get_indv_df_for_replicate, school_table, rng, args.n_runs,
                                   CONDENSED_AGGS, args.mtb_decile_topk)
        deciles = list(range(1, N_DECILES + 1))

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes_flat = axes.flatten()
        twin_axes = []
        for i, name in enumerate(CONDENSED_AGGS):
            ax = axes_flat[i]
            ax2 = ax.twinx()
            twin_axes.append(ax2)
            plot_decile_axes(ax, ax2, stats, name, deciles,
                              markersize=4, linewidth=1.2, capsize=3, capthick=1)
            ax.set_title(AGG_TITLES[name], fontsize=11)
            ax.set_xlabel('Lottery Decile', fontsize=9)
            ax.set_xticks(deciles)
            ax.tick_params(axis='both', labelsize=8)
            ax2.tick_params(axis='y', labelsize=8)
            ax.margins(y=0.08)
            ax2.margins(y=0.08)
        axes_flat[-1].axis('off')

        axes_flat[0].set_ylabel('Match Rate (%)', fontsize=10)
        axes_flat[3].set_ylabel('Match Rate (%)', fontsize=10)
        twin_axes[2].set_ylabel('Avg Rank (matched)', fontsize=10, color=DECILE_COLORS['avg_rank'])
        twin_axes[4].set_ylabel('Avg Rank (matched)', fontsize=10, color=DECILE_COLORS['avg_rank'])

        metric_handles = [
            plt.Line2D([], [], color=DECILE_COLORS[m], marker=DECILE_MARKERS[m], linestyle='-', label=DECILE_LABELS[m])
            for m in ['pct_unmatched', 'top1_pct', 'top5_pct', 'avg_rank']
        ]
        style_handles = [plt.Line2D([], [], color='gray', linestyle=DECILE_STYLES[c], label=c) for c in ['STB', 'MTB']]
        all_handles = metric_handles + style_handles
        fig.legend(handles=all_handles, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                   ncol=len(all_handles), fontsize=11, frameon=False)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(args.output_condensed, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {args.output_condensed}")
        return

    topk_note = f", topk={args.mtb_decile_topk}" if uses_topk else ""
    print(f"  MTB decile aggregation: {args.mtb_decile_agg} ({mtb_decile_source}{topk_note})")
    print(f"  Output: {args.output}")
    print(f"  Output (distribution diagnostic): {args.output_dist}")

    metrics_by_condition = {'STB': [], 'MTB': []}
    decile_value_pool = {'STB': [], 'MTB': []}
    for run in range(args.n_runs):
        print(f"Run {run + 1}/{args.n_runs}...")
        indv_df = get_indv_df_for_replicate(rng)
        applications_long = build_applications_long(indv_df)
        all_student_ids = sorted(applications_long['mrun'].unique().tolist())
        if run == 0:
            print(f"  Students: {len(all_student_ids):,}")

        student_lottery = {sid: float(rng.random()) for sid in all_student_ids}
        student_ids, rankings, matches, lottery_df = run_matching(
            applications_long, school_table, rng,
            student_lottery=student_lottery, return_lottery_df=True,
        )
        df = compute_student_records(student_ids, rankings, matches, lottery_df, 'single')
        metrics_by_condition['STB'].append(decile_metrics(df))
        decile_value_pool['STB'].append(df['decile_val'].to_numpy())

        student_ids, rankings, matches, lottery_df = run_matching(
            applications_long, school_table, rng,
            student_lottery=None, return_lottery_df=True,
        )
        df = compute_student_records(student_ids, rankings, matches, lottery_df, mtb_decile_source,
                                      topk=args.mtb_decile_topk)
        metrics_by_condition['MTB'].append(decile_metrics(df))
        decile_value_pool['MTB'].append(df['decile_val'].to_numpy())

    stats = {}
    for cond in ['STB', 'MTB']:
        grouped = pd.concat(metrics_by_condition[cond]).groupby(level=0)
        mean = grouped.mean()
        std = grouped.std()
        stats[cond] = {'mean': mean, 'ci95': 1.96 * std / np.sqrt(args.n_runs)}

    plot_decile_value_distributions(
        np.concatenate(decile_value_pool['STB']),
        np.concatenate(decile_value_pool['MTB']),
        args.output_dist,
    )

    dump_rows = []
    for cond in ['STB', 'MTB']:
        merged = stats[cond]['mean'].join(stats[cond]['ci95'], rsuffix='_ci95')
        merged['condition'] = cond
        dump_rows.append(merged.reset_index())
    dump_path = args.output.rsplit('.', 1)[0] + '.csv'
    pd.concat(dump_rows, ignore_index=True).to_csv(dump_path, index=False)
    print(f"Saved: {dump_path}")

    print("\nCI half-width (95%, pp) across deciles -- if these are all near-zero,")
    print("the shaded bands are legitimately too thin to see at this scale, not a bug:")
    for cond in ['STB', 'MTB']:
        for metric in ['pct_unmatched', 'top1_pct', 'top5_pct']:
            ci_series = stats[cond]['ci95'][metric]
            print(f"  {cond:>3} {metric:>13}: min={ci_series.min():.3f}pp  "
                  f"max={ci_series.max():.3f}pp  mean={ci_series.mean():.3f}pp")

    print("\nSTB - MTB gap per decile, with 95% CI on the DIFFERENCE ('*' = exceeds CI, real not noise):")
    for metric in ['pct_unmatched', 'top1_pct', 'top5_pct']:
        print(f"  {metric}:")
        for d in range(1, N_DECILES + 1):
            stb_mean = stats['STB']['mean'][metric].get(d, float('nan'))
            mtb_mean = stats['MTB']['mean'][metric].get(d, float('nan'))
            diff = stb_mean - mtb_mean
            diff_ci = float(np.sqrt(stats['STB']['ci95'][metric].get(d, 0) ** 2
                                     + stats['MTB']['ci95'][metric].get(d, 0) ** 2))
            flag = '*' if abs(diff) > diff_ci else ' '
            print(f"    decile {d:>2}: STB={stb_mean:6.2f}%  MTB={mtb_mean:6.2f}%  "
                  f"diff={diff:+6.2f}pp  ci95={diff_ci:5.2f}pp {flag}")

    # ── plot ──────────────────────────────────────────────────────────────
    COLORS = {'pct_unmatched': '#111111', 'top1_pct': '#1565C0', 'top5_pct': '#AD1457', 'avg_rank': '#D4A017'}
    LABELS = {'pct_unmatched': 'Unmatched', 'top1_pct': 'Top-1', 'top5_pct': 'Top-5', 'avg_rank': 'Avg Rank'}
    STYLES = {'STB': '-', 'MTB': '--'}
    MARKERS = {'pct_unmatched': 'o', 'top1_pct': 's', 'top5_pct': '^', 'avg_rank': 'D'}

    deciles = list(range(1, N_DECILES + 1))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax2 = ax.twinx()
    for cond in ['STB', 'MTB']:
        for metric in ['pct_unmatched', 'top1_pct', 'top5_pct']:
            mean_vals = stats[cond]['mean'][metric].reindex(deciles)
            ci_vals = stats[cond]['ci95'][metric].reindex(deciles)
            ax.errorbar(deciles, mean_vals, yerr=ci_vals,
                        color=COLORS[metric], linestyle=STYLES[cond], marker=MARKERS[metric],
                        markersize=6, linewidth=1.8, label=f'{LABELS[metric]} ({cond})',
                        ecolor='#000000', elinewidth=2, capsize=5, capthick=2, zorder=6)
        mean_vals = stats[cond]['mean']['avg_rank'].reindex(deciles)
        ci_vals = stats[cond]['ci95']['avg_rank'].reindex(deciles)
        ax2.errorbar(deciles, mean_vals, yerr=ci_vals,
                    color=COLORS['avg_rank'], linestyle=STYLES[cond], marker=MARKERS['avg_rank'],
                    markersize=6, linewidth=1.8, label=f"{LABELS['avg_rank']} ({cond})",
                    ecolor='#000000', elinewidth=2, capsize=5, capthick=2, zorder=6)

    ax.set_xlabel('Lottery Decile', fontsize=12)
    ax.set_ylabel('Match Rate (%)', fontsize=12)
    ax2.set_ylabel('Avg Rank (matched)', fontsize=12, color=COLORS['avg_rank'])
    ax.set_xticks(deciles)
    ax.margins(y=0.08)
    ax2.margins(y=0.08)
    ax2.tick_params(axis='y', colors=COLORS['avg_rank'])
    ax2.spines['right'].set_color(COLORS['avg_rank'])
    ax.spines['top'].set_visible(False)

    metric_handles = [
        plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle='-', label=LABELS[m])
        for m in ['pct_unmatched', 'top1_pct', 'top5_pct', 'avg_rank']
    ]
    style_handles = [plt.Line2D([], [], color='gray', linestyle=STYLES[c], label=c) for c in ['STB', 'MTB']]
    all_handles = metric_handles + style_handles
    fig.legend(handles=all_handles, loc='lower center', bbox_to_anchor=(0.5, 1.0),
               ncol=len(all_handles), fontsize=10, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
