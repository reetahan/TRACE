

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

from file_config import DATA_GENERATION_SEED
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from chile_priority_attributes import (
    _prepare_school_capacity_table,
    _assign_school_level_priority_tiers_and_dense_scores,
)
from gale_shapley import gale_shapley_per_school_numba_wrapper



def load_df(path: str) -> pd.DataFrame:
    if path.endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_excel(path)


def build_applications_long(indv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert indv_df to long-form applications with real priority flags.
    Uses rbd_program_code as school_id to match school_table.
    """
    df = indv_df.copy()
    df['mrun'] = df['mrun'].astype(str)
    df['rbd'] = df['rbd'].astype(str) + '_' + df['program_code'].astype(str)
    priority_cols = [
        'priority_already_registered',
        'priority_sibling',
        'priority_student',
        'priority_parent_civil_servant',
        'priority_ex_student',
    ]
    keep = ['mrun', 'rbd', 'preference_number'] + priority_cols
    return df[keep].copy()


def build_student_attrs(indv_df: pd.DataFrame) -> pd.DataFrame:
    """Per-student attributes including female flag, indexed by mrun string."""
    return (
        indv_df[['mrun', 'female']]
        .drop_duplicates(subset='mrun')
        .assign(mrun=lambda x: x['mrun'].astype(str))
        .set_index('mrun')
    )



def run_matching(
    applications_long: pd.DataFrame,
    school_table: pd.DataFrame,
    rng: np.random.Generator,
    student_lottery: Optional[Dict[str, float]] = None,
    return_lottery_df: bool = False,
):

    student_ids, student_rankings, dense_scores, lottery_df, capacities, _, _ = \
        _assign_school_level_priority_tiers_and_dense_scores(
            applications_long, school_table, rng,
            student_lottery=student_lottery,
        )
    matches = gale_shapley_per_school_numba_wrapper(
        student_rankings, dense_scores, capacities
    )
    if return_lottery_df:
        return student_ids, student_rankings, matches, lottery_df
    return student_ids, student_rankings, matches


def compute_top_p_curves(
    student_ids: np.ndarray,
    student_rankings: List[List[int]],
    matches: np.ndarray,
    student_attrs: pd.DataFrame,
    max_p: int,
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Returns nested dict:
      result[group][condition] = Series indexed 1..max_p with % matched to top-p
    group: 'all', 'female', 'nonfemale'
    condition: 'uncond' (all students), 'cond' (matched only)
    """
    records = []
    for i, sid in enumerate(student_ids):
        female = int(student_attrs.loc[sid, 'female']) if sid in student_attrs.index else -1
        matched_idx = int(matches[i])
        ranking = student_rankings[i]
        rank_pos = None
        if matched_idx >= 0:
            try:
                rank_pos = ranking.index(matched_idx) + 1  # 1-indexed
            except ValueError:
                pass
        records.append({'female': female, 'matched': matched_idx >= 0, 'rank_pos': rank_pos})

    df = pd.DataFrame(records)

    result = {}
    for group in ['all', 'female', 'nonfemale']:
        if group == 'female':
            sub = df[df['female'] == 1]
        elif group == 'nonfemale':
            sub = df[df['female'] == 0]
        else:
            sub = df

        sub_cond = sub[sub['matched']]
        uncond_vals, cond_vals = {}, {}
        for p in range(1, max_p + 1):
            in_top = lambda d, p=p: (d['rank_pos'].apply(
                lambda r: r is not None and r <= p)).sum()
            uncond_vals[p] = 100.0 * in_top(sub)      / len(sub)      if len(sub)      > 0 else 0.0
            cond_vals[p]   = 100.0 * in_top(sub_cond) / len(sub_cond) if len(sub_cond) > 0 else 0.0

        result[group] = {
            'uncond': pd.Series(uncond_vals),
            'cond':   pd.Series(cond_vals),
            'unmatched': 100.0 * (~sub['matched']).sum() / len(sub) if len(sub) > 0 else 0.0,
        }

    return result


def aggregate_runs(curve_list: List[pd.Series]) -> pd.DataFrame:
    """mean/std/ci95 (normal-approx 95% CI) across independent lottery-draw runs."""
    df = pd.concat(curve_list, axis=1)
    n = df.shape[1]
    # ddof=0: pandas' default ddof=1 returns NaN for a single run (n-1=0 degrees of
    # freedom), which then NaNs out ci95 and breaks downstream plot y-limits. ddof=0
    # matches aggregate_scalar_runs' numpy .std() (also ddof=0 by default) and gives
    # the correct std=0 for n=1, not NaN.
    std = df.std(axis=1, ddof=0)
    return pd.DataFrame({'mean': df.mean(axis=1), 'std': std, 'ci95': 1.96 * std / np.sqrt(n)})


def aggregate_scalar_runs(values: List[float]) -> Dict[str, float]:
    """Same as aggregate_runs, for a plain list of scalar per-run values (e.g. unmatched rate)."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    std = float(arr.std())
    return {'mean': float(arr.mean()), 'std': std, 'ci95': 1.96 * std / np.sqrt(n)}


def select_best_run(curve_list: List[pd.Series], best_i: int) -> pd.DataFrame:
    """Same return shape as aggregate_runs, but mean=curve_list[best_i] and std/ci95=0 --
    a single selected replicate, not an average."""
    mean = curve_list[best_i]
    zeros = pd.Series(0.0, index=mean.index)
    return pd.DataFrame({'mean': mean, 'std': zeros, 'ci95': zeros})


def select_best_scalar_run(values: List[float], best_i: int) -> Dict[str, float]:
    return {'mean': float(values[best_i]), 'std': 0.0, 'ci95': 0.0}

COLORS = {
    'all':       '#2c3e50',
    'female':    '#2980b9',
    'nonfemale': '#27ae60',
}
LABELS = {
    'all':       'All students',
    'female':    'Female',
    'nonfemale': 'Non-female',
}


def make_plot(
    mtb_results: Dict,
    stb_results: Dict,
    condition: str,
    max_p: int,
    output_path: str,
    n_stb_runs: int,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ps = list(range(1, max_p + 1))

    for group in ['all', 'female', 'nonfemale']:
        color = COLORS[group]
        label = LABELS[group]

        mtb = mtb_results[group][condition]
        ax.plot(ps, [mtb['mean'][p] for p in ps],
                linestyle='-', color=color, linewidth=1, marker='o', markersize=4,
                label=f'{label} — MTB')
        ax.fill_between(
            ps,
            [mtb['mean'][p] - mtb['ci95'][p] for p in ps],
            [mtb['mean'][p] + mtb['ci95'][p] for p in ps],
            alpha=0.28, color=color, edgecolor=color, linewidth=0.8,
        )

        stb = stb_results[group][condition]
        ax.plot(ps, [stb['mean'][p] for p in ps],
                linestyle='--', color=color, linewidth=1, marker='s', markersize=4,
                label=f'{label} — STB')
        ax.fill_between(
            ps,
            [stb['mean'][p] - stb['ci95'][p] for p in ps],
            [stb['mean'][p] + stb['ci95'][p] for p in ps],
            alpha=0.28, color=color, edgecolor=color, linewidth=0.8,
        )

    cond_label = 'matched students only' if condition == 'cond' else 'all students'
    ax.set_xlabel('p (top-p threshold)', fontsize=12)
    ax.set_ylabel(f'% matched to top-p choice ({cond_label})', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xticks(ps)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25, linestyle='--')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def make_diff_bar_plot(
    mtb_results: Dict,
    stb_results: Dict,
    condition: str,
    max_p: int,
    output_path: str,
    group: str = 'all',
):
    """Fig. 6 style plot: STB-MTB difference (pp) per top-p threshold, plus unmatched rate. Positive = STB better."""
    ps = list(range(1, max_p + 1))
    mtb = mtb_results[group][condition]
    stb = stb_results[group][condition]
    diffs = [stb['mean'][p] - mtb['mean'][p] for p in ps]
    diff_ci95 = [float(np.sqrt(stb['ci95'][p]**2 + mtb['ci95'][p]**2)) for p in ps]

    mtb_un = mtb_results[group]['unmatched']
    stb_un = stb_results[group]['unmatched']
    # Literal STB - MTB, matching the y-axis label and every other bar; real data has
    # STB's unmatched rate higher than MTB's, so this comes out positive like the rest.
    diffs.append(stb_un['mean'] - mtb_un['mean'])
    diff_ci95.append(float(np.sqrt(stb_un['ci95']**2 + mtb_un['ci95']**2)))

    labels = [str(p) for p in ps] + ['Unm.']
    x = np.arange(len(labels))

    dump = pd.DataFrame({
        'label':      labels,
        'stb_mean':   [stb['mean'][p] for p in ps] + [stb_un['mean']],
        'stb_ci95':   [stb['ci95'][p] for p in ps] + [stb_un['ci95']],
        'mtb_mean':   [mtb['mean'][p] for p in ps] + [mtb_un['mean']],
        'mtb_ci95':   [mtb['ci95'][p] for p in ps] + [mtb_un['ci95']],
        'diff':       diffs,
        'diff_ci95':  diff_ci95,
    })
    dump_path = output_path.rsplit('.', 1)[0] + '.csv'
    dump.to_csv(dump_path, index=False)
    print(f"Saved: {dump_path}")

    STB_COLOR = '#1a2f6e'   # navy: STB better
    MTB_COLOR = '#c1652f'   # orange: MTB better

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [STB_COLOR if d >= 0 else MTB_COLOR for d in diffs]
    colors[-1] = MTB_COLOR  # W_unmatch always orange, regardless of sign
    ax.bar(x, diffs, color=colors, width=0.65, yerr=diff_ci95, capsize=5,
           error_kw={'ecolor': '#000000', 'elinewidth': 2})
    ax.axhline(0, color='black', linewidth=0.8)

    # vertical dotted line at the first sign flip among the p=1..max_p bars
    flip_idx = next((i for i in range(1, len(ps)) if diffs[i] < 0 <= diffs[i - 1]), None)
    if flip_idx is not None:
        ax.axvline(flip_idx - 0.5, color='gray', linestyle=':', linewidth=1)

    tick_labels = [f'$W_{{{p}}}$' for p in ps] + ['$W_{\\mathrm{unmatch}}$']
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Matched Rank', fontsize=15)
    ax.set_ylabel('STB - MTB (pp)', fontsize=15)
    ax.tick_params(labelsize=12)

    bar_tops = [d + e for d, e in zip(diffs, diff_ci95)] + [0]
    bar_bottoms = [d - e for d, e in zip(diffs, diff_ci95)] + [0]
    y_max, y_min = max(bar_tops), min(bar_bottoms)
    pad = 0.22 * (y_max - y_min) if y_max > y_min else 1.0
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.text(0.02, 0.97, '← STB better', transform=ax.transAxes,
            ha='left', va='top', color=STB_COLOR, fontsize=13, fontweight='bold')
    ax.text(0.98, 0.03, 'MTB better →', transform=ax.transAxes,
            ha='right', va='bottom', color=MTB_COLOR, fontsize=13, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--individual',    default=None,
                         help='Fixed --individual CSV. Mutually exclusive with --params_pkl.')
    parser.add_argument('--params_pkl',    default=None,
                         help='Fitted Mallows params.pkl; resamples preferences + lottery each replicate. '
                              'Mutually exclusive with --individual.')
    parser.add_argument('--real_individual', default=None,
                         help='Real indv_df, used with --params_pkl to calibrate synthetic resampling.')
    parser.add_argument('--subdivision_col', default='Region',
                         help="Only used with --params_pkl: 'Region' or 'Provincia'.")
    parser.add_argument('--capacity',      required=True)
    parser.add_argument('--output_uncond', default='welfare_uncond.png')
    parser.add_argument('--output_cond',   default='welfare_cond.png')
    parser.add_argument('--output_diff_bar', default='welfare_diff_bar.png',
                         help='Fig. 6 style STB-MTB difference bar chart.')
    parser.add_argument('--n_stb_runs',   type=int, default=10,
                         help='Independent replicates averaged for BOTH MTB and STB.')
    parser.add_argument('--max_p',        type=int, default=10)
    parser.add_argument('--select_best_of_n', action='store_true',
                         help='Instead of averaging the n_stb_runs replicates, draw all of them and keep '
                              "only the single replicate whose overall unmatched rate is closest to the "
                              "real observed unmatched rate (from --real_individual's matched_first_round "
                              'column). No CI -- matches the "best of many, scored against real data" '
                              'regime as the real-profile Fig 2/16 figures. Requires --params_pkl.')
    parser.add_argument('--seed',         type=int, default=DATA_GENERATION_SEED)
    args = parser.parse_args()

    if bool(args.individual) == bool(args.params_pkl):
        parser.error('Pass exactly one of --individual or --params_pkl.')
    if args.params_pkl and not args.real_individual:
        parser.error('--params_pkl requires --real_individual (for calibration).')
    if args.select_best_of_n and not args.params_pkl:
        parser.error('--select_best_of_n requires --params_pkl (needs independent preference draws to select among).')

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
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'project_specific_scripts'))
        from build_synthetic_chile_indv_df import build_synthetic_indv_df
        return build_synthetic_indv_df(
            mallows_params, real_indv_df_for_calibration, capacity_df, rng,
            subdivision_col=args.subdivision_col, n_jobs=1, verbose=False,
        )

    print(f"  Schools: {len(school_table):,}")
    print(f"  Mode: {'fixed --individual, lottery-only CI' if fixed_indv_df is not None else 'resampling preferences + lottery each replicate'}")

    rng = np.random.default_rng(args.seed)

    real_observed_unmatched = None
    if args.select_best_of_n:
        real_matched = real_indv_df_for_calibration.groupby('mrun')['matched_first_round'].max()
        real_observed_unmatched = 100.0 * (1.0 - real_matched.mean())
        print(f"  Real observed unmatched rate (selection target): {real_observed_unmatched:.2f}%")

    print(f"\nRunning {args.n_stb_runs} MTB replicates (real priority, per-school lottery)...")
    mtb_curves: Dict = {g: {'uncond': [], 'cond': [], 'unmatched': []} for g in ['all', 'female', 'nonfemale']}
    mtb_scores = []
    for run in range(args.n_stb_runs):
        print(f"  MTB run {run+1}/{args.n_stb_runs}...")
        indv_df = get_indv_df_for_replicate(rng)
        applications_long = build_applications_long(indv_df)
        student_attrs = build_student_attrs(indv_df)
        if run == 0:
            print(f"    Students: {applications_long['mrun'].nunique():,}  "
                  f"Female: {(student_attrs['female'] == 1).sum():,}  "
                  f"Non-female: {(student_attrs['female'] == 0).sum():,}")
        student_ids_mtb, rankings_mtb, matches_mtb = run_matching(
            applications_long, school_table, rng, student_lottery=None
        )
        replicate_unmatched = (matches_mtb < 0).mean() * 100
        print(f"    Match rate: {(matches_mtb >= 0).mean() * 100:.1f}%")
        if real_observed_unmatched is not None:
            mtb_scores.append(abs(replicate_unmatched - real_observed_unmatched))
            print(f"    |unmatched diff from real|={mtb_scores[-1]:.2f}pp")
        run_curves = compute_top_p_curves(
            student_ids_mtb, rankings_mtb, matches_mtb, student_attrs, args.max_p
        )
        for g in ['all', 'female', 'nonfemale']:
            mtb_curves[g]['uncond'].append(run_curves[g]['uncond'])
            mtb_curves[g]['cond'].append(run_curves[g]['cond'])
            mtb_curves[g]['unmatched'].append(run_curves[g]['unmatched'])

    if real_observed_unmatched is not None:
        mtb_best_i = int(np.argmin(mtb_scores))
        print(f"  MTB: keeping run {mtb_best_i + 1} (closest to real, |diff|={mtb_scores[mtb_best_i]:.2f}pp)")
        mtb_results = {
            g: {
                'uncond': select_best_run(mtb_curves[g]['uncond'], mtb_best_i),
                'cond':   select_best_run(mtb_curves[g]['cond'], mtb_best_i),
                'unmatched': select_best_scalar_run(mtb_curves[g]['unmatched'], mtb_best_i),
            }
            for g in ['all', 'female', 'nonfemale']
        }
    else:
        mtb_results = {
            g: {
                'uncond': aggregate_runs(mtb_curves[g]['uncond']),
                'cond':   aggregate_runs(mtb_curves[g]['cond']),
                'unmatched': aggregate_scalar_runs(mtb_curves[g]['unmatched']),
            }
            for g in ['all', 'female', 'nonfemale']
        }

    # STB — n_stb_runs replicates with single per-student lottery
    print(f"\nRunning {args.n_stb_runs} STB replicates...")
    stb_curves: Dict = {g: {'uncond': [], 'cond': [], 'unmatched': []} for g in ['all', 'female', 'nonfemale']}
    stb_scores = []

    for run in range(args.n_stb_runs):
        print(f"  STB run {run+1}/{args.n_stb_runs}...")
        indv_df = get_indv_df_for_replicate(rng)
        applications_long = build_applications_long(indv_df)
        student_attrs = build_student_attrs(indv_df)
        all_student_ids = sorted(applications_long['mrun'].unique().tolist())
        student_lottery = {sid: float(rng.random()) for sid in all_student_ids}
        student_ids_stb, rankings_stb, matches_stb = run_matching(
            applications_long, school_table, rng, student_lottery=student_lottery
        )
        replicate_unmatched = (matches_stb < 0).mean() * 100
        print(f"    Match rate: {(matches_stb >= 0).mean() * 100:.1f}%")
        if real_observed_unmatched is not None:
            stb_scores.append(abs(replicate_unmatched - real_observed_unmatched))
            print(f"    |unmatched diff from real|={stb_scores[-1]:.2f}pp")
        run_curves = compute_top_p_curves(
            student_ids_stb, rankings_stb, matches_stb, student_attrs, args.max_p
        )
        for g in ['all', 'female', 'nonfemale']:
            stb_curves[g]['uncond'].append(run_curves[g]['uncond'])
            stb_curves[g]['cond'].append(run_curves[g]['cond'])
            stb_curves[g]['unmatched'].append(run_curves[g]['unmatched'])

    if real_observed_unmatched is not None:
        stb_best_i = int(np.argmin(stb_scores))
        print(f"  STB: keeping run {stb_best_i + 1} (closest to real, |diff|={stb_scores[stb_best_i]:.2f}pp)")
        stb_results = {
            g: {
                'uncond': select_best_run(stb_curves[g]['uncond'], stb_best_i),
                'cond':   select_best_run(stb_curves[g]['cond'], stb_best_i),
                'unmatched': select_best_scalar_run(stb_curves[g]['unmatched'], stb_best_i),
            }
            for g in ['all', 'female', 'nonfemale']
        }
    else:
        stb_results = {
            g: {
                'uncond': aggregate_runs(stb_curves[g]['uncond']),
                'cond':   aggregate_runs(stb_curves[g]['cond']),
                'unmatched': aggregate_scalar_runs(stb_curves[g]['unmatched']),
            }
            for g in ['all', 'female', 'nonfemale']
        }

    print("\nGenerating plots...")
    make_plot(mtb_results, stb_results, 'uncond', args.max_p, args.output_uncond, args.n_stb_runs)
    make_plot(mtb_results, stb_results, 'cond',   args.max_p, args.output_cond,   args.n_stb_runs)
    make_diff_bar_plot(mtb_results, stb_results, 'uncond', args.max_p, args.output_diff_bar)

    # Table 9 layout: Overall / Female / Non-female column groups, each with
    # MTB / STB / Diff (STB-MTB, pp) sub-columns, rows p=1..max_p then Unmatched.
    print("\n=== Table 9: Chile top-p match rates (%) under MTB and STB ===")
    col = f"{'':>4}  {'Overall':^20}  {'Female':^20}  {'Non-female':^20}"
    sub = f"{'p':>4}  {'MTB':>6} {'STB':>6} {'Diff':>7}  {'MTB':>6} {'STB':>6} {'Diff':>7}  {'MTB':>6} {'STB':>6} {'Diff':>7}"
    print(col)
    print(sub)
    for p in range(1, args.max_p + 1):
        mtb_v, stb_v = mtb_results['all']['uncond']['mean'][p], stb_results['all']['uncond']['mean'][p]
        mtb_f, stb_f = mtb_results['female']['uncond']['mean'][p], stb_results['female']['uncond']['mean'][p]
        mtb_nf, stb_nf = mtb_results['nonfemale']['uncond']['mean'][p], stb_results['nonfemale']['uncond']['mean'][p]
        print(f"{p:>4}  {mtb_v:>6.1f} {stb_v:>6.1f} {stb_v - mtb_v:>+6.1f}pp  "
              f"{mtb_f:>6.1f} {stb_f:>6.1f} {stb_f - mtb_f:>+6.1f}pp  "
              f"{mtb_nf:>6.1f} {stb_nf:>6.1f} {stb_nf - mtb_nf:>+6.1f}pp")

    mtb_un, stb_un = mtb_results['all']['unmatched']['mean'], stb_results['all']['unmatched']['mean']
    mtb_un_f, stb_un_f = mtb_results['female']['unmatched']['mean'], stb_results['female']['unmatched']['mean']
    mtb_un_nf, stb_un_nf = mtb_results['nonfemale']['unmatched']['mean'], stb_results['nonfemale']['unmatched']['mean']
    print(f"{'Unm.':>4}  {mtb_un:>6.1f} {stb_un:>6.1f} {stb_un - mtb_un:>+6.1f}pp  "
          f"{mtb_un_f:>6.1f} {stb_un_f:>6.1f} {stb_un_f - mtb_un_f:>+6.1f}pp  "
          f"{mtb_un_nf:>6.1f} {stb_un_nf:>6.1f} {stb_un_nf - mtb_un_nf:>+6.1f}pp")
    print("(Diff = STB - MTB, pp)")

if __name__ == '__main__':
    main()