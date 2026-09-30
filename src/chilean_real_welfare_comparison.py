"""
plot_chile_welfare.py

Compares welfare under MTB vs STB for:
  - All students
  - Female students
  - Non-female students

Uses real individual-level Chilean preference data (indv_df).
Preference lists and priority attributes are fixed and real.
Only the lottery mechanism changes between MTB and STB, and both are
averaged over --n_stb_runs independent lottery draws (95% CI shown as a
shaded band / error bars) -- the interval reflects uncertainty from the
lottery's own randomness, holding real preferences fixed, not sampling
uncertainty about the student population.

Usage:
    python plot_chile_welfare.py \
        --individual <path_to_indv_df> \
        --capacity   <path_to_capacity_df> \
        --output_uncond welfare_uncond.png \
        --output_cond   welfare_cond.png \
        --n_stb_runs 10 \
        --max_p      10 \
        --seed       42
"""

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


# ── data loading ───────────────────────────────────────────────────────────

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


# ── matching ───────────────────────────────────────────────────────────────

def run_matching(
    applications_long: pd.DataFrame,
    school_table: pd.DataFrame,
    rng: np.random.Generator,
    student_lottery: Optional[Dict[str, float]] = None,
    return_lottery_df: bool = False,
):
    """
    Run Chile DA with real priority attributes.
    student_lottery=None  -> MTB (independent per-school draws)
    student_lottery=dict  -> STB (single draw per student, shared across schools)
    Returns student_ids, student_rankings, matches_idx.
    If return_lottery_df=True, also returns the long-form per-(student, school)
    application table (with the per-application 'lottery' draw and
    'preference_number'), needed to derive per-student lottery-position stats.
    """
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


# ── welfare computation ────────────────────────────────────────────────────

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
    """
    mean/std/ci95 across independent lottery-draw runs, holding real
    preference data fixed. ci95 is the normal-approximation 95% CI
    half-width on the mean (1.96 * std/sqrt(n)) -- it answers "how much
    would this estimate move if the lottery were redrawn", not sampling
    uncertainty about the student population or model uncertainty.
    """
    df = pd.concat(curve_list, axis=1)
    n = df.shape[1]
    std = df.std(axis=1)
    return pd.DataFrame({'mean': df.mean(axis=1), 'std': std, 'ci95': 1.96 * std / np.sqrt(n)})


def aggregate_scalar_runs(values: List[float]) -> Dict[str, float]:
    """Same as aggregate_runs, for a plain list of scalar per-run values (e.g. unmatched rate)."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    std = float(arr.std())
    return {'mean': float(arr.mean()), 'std': std, 'ci95': 1.96 * std / np.sqrt(n)}


# ── plotting ───────────────────────────────────────────────────────────────

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
    """
    Fig. 6 style plot: STB-MTB difference (pp) in top-p match rate for
    p=1..max_p, plus the unmatched-rate difference as a final bar. Every bar
    uses "positive = STB better" -- for top-p match rate that's stb - mtb
    (higher match rate is better), but for the unmatched-rate bar it's
    mtb - stb (lower unmatched rate is better), so the sign is flipped there
    to keep the color/label convention consistent instead of silently
    inverted on just that one bar.

    Error bars are a 95% CI on the DIFFERENCE itself, combining both sides'
    independent lottery-draw variance: SE_diff = sqrt(SE_stb^2 + SE_mtb^2),
    since STB and MTB are redrawn independently each run.
    """
    ps = list(range(1, max_p + 1))
    mtb = mtb_results[group][condition]
    stb = stb_results[group][condition]
    diffs = [stb['mean'][p] - mtb['mean'][p] for p in ps]
    diff_ci95 = [float(np.sqrt(stb['ci95'][p]**2 + mtb['ci95'][p]**2)) for p in ps]

    mtb_un = mtb_results[group]['unmatched']
    stb_un = stb_results[group]['unmatched']
    # Every other bar is "higher match rate = STB better", so diff = stb - mtb
    # with diff >= 0 meaning STB wins. Unmatched rate is the opposite -- lower
    # is better -- so it's flipped here (mtb - stb) to keep that same
    # "positive = STB better" convention consistent across every bar, instead
    # of silently inverting the STB/MTB color and "better" direction only on
    # this one bar.
    diffs.append(mtb_un['mean'] - stb_un['mean'])
    diff_ci95.append(float(np.sqrt(stb_un['ci95']**2 + mtb_un['ci95']**2)))

    labels = [str(p) for p in ps] + ['Unm.']
    x = np.arange(len(labels))

    STB_COLOR = '#1a2f6e'   # navy: STB better
    MTB_COLOR = '#c1652f'   # orange: MTB better

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [STB_COLOR if d >= 0 else MTB_COLOR for d in diffs]
    ax.bar(x, diffs, color=colors, width=0.65, yerr=diff_ci95, capsize=5,
           error_kw={'ecolor': '#000000', 'elinewidth': 2})
    ax.axhline(0, color='black', linewidth=0.8)

    # vertical dotted line at the first sign flip among the p=1..max_p bars
    flip_idx = next((i for i in range(1, len(ps)) if diffs[i] < 0 <= diffs[i - 1]), None)
    if flip_idx is not None:
        ax.axvline(flip_idx - 0.5, color='gray', linestyle=':', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Top-p threshold / Match status', fontsize=12)
    # "STB - MTB" for the top-p bars; the Unm. bar is MTB - STB (see the sign
    # flip above) so that positive always means "STB better" on every bar.
    ax.set_ylabel('pp (positive = STB better)', fontsize=12)

    # Pad the y-range so the STB/MTB-better labels sit in dedicated
    # whitespace above/below the bars (including their error bars) instead
    # of landing on top of whichever bar happens to be tallest/shortest.
    bar_tops = [d + e for d, e in zip(diffs, diff_ci95)] + [0]
    bar_bottoms = [d - e for d, e in zip(diffs, diff_ci95)] + [0]
    y_max, y_min = max(bar_tops), min(bar_bottoms)
    pad = 0.22 * (y_max - y_min) if y_max > y_min else 1.0
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.text(0.02, 0.97, '← STB better', transform=ax.transAxes,
            ha='left', va='top', color=STB_COLOR, fontsize=11, fontweight='bold')
    ax.text(0.98, 0.03, 'MTB better →', transform=ax.transAxes,
            ha='right', va='bottom', color=MTB_COLOR, fontsize=11, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--individual',    default=None,
                         help='Fixed --individual CSV (real or a pre-built synthetic one). Mutually '
                              'exclusive with --params_pkl; when given, only the lottery is redrawn '
                              'each replicate (preferences stay fixed).')
    parser.add_argument('--params_pkl',    default=None,
                         help='Fitted Mallows params.pkl. When given (with --real_individual), each '
                              'replicate draws a FRESH synthetic preference sample from this model AND '
                              'a fresh lottery, together -- the CI then reflects both preference-sampling '
                              'and lottery randomness, not just the lottery. Mutually exclusive with '
                              '--individual.')
    parser.add_argument('--real_individual', default=None,
                         help='Real indv_df, used with --params_pkl to calibrate list lengths, female '
                              'rate, and priority-flag base rates for synthetic resampling.')
    parser.add_argument('--subdivision_col', default='Region',
                         help="Only used with --params_pkl: 'Region' or 'Provincia', matching how the "
                              "saved params were fit.")
    parser.add_argument('--capacity',      required=True)
    parser.add_argument('--output_uncond', default='welfare_uncond.png')
    parser.add_argument('--output_cond',   default='welfare_cond.png')
    parser.add_argument('--output_diff_bar', default='welfare_diff_bar.png',
                         help='Fig. 6 style STB-MTB difference bar chart.')
    parser.add_argument('--n_stb_runs',   type=int, default=10,
                         help='Independent replicates averaged for BOTH MTB and STB. With --individual, '
                              'only the lottery is redrawn each replicate; with --params_pkl, preferences '
                              'are also freshly resampled each replicate.')
    parser.add_argument('--max_p',        type=int, default=10)
    parser.add_argument('--seed',         type=int, default=DATA_GENERATION_SEED)
    args = parser.parse_args()

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
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'project_specific_scripts'))
        from build_synthetic_chile_indv_df import build_synthetic_indv_df
        return build_synthetic_indv_df(
            mallows_params, real_indv_df_for_calibration, capacity_df, rng,
            subdivision_col=args.subdivision_col, n_jobs=1, verbose=False,
        )

    print(f"  Schools: {len(school_table):,}")
    print(f"  Mode: {'fixed --individual, lottery-only CI' if fixed_indv_df is not None else 'resampling preferences + lottery each replicate'}")

    rng = np.random.default_rng(args.seed)

    # MTB — n_stb_runs independent replicates. MTB's per-school lottery is
    # just as random as STB's single draw, so it's averaged the same way STB
    # already is; when --params_pkl is given, each replicate also draws a
    # fresh synthetic preference sample, so the CI captures both sources of
    # randomness combined, not just the lottery.
    print(f"\nRunning {args.n_stb_runs} MTB replicates (real priority, per-school lottery)...")
    mtb_curves: Dict = {g: {'uncond': [], 'cond': [], 'unmatched': []} for g in ['all', 'female', 'nonfemale']}
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
        print(f"    Match rate: {(matches_mtb >= 0).mean() * 100:.1f}%")
        run_curves = compute_top_p_curves(
            student_ids_mtb, rankings_mtb, matches_mtb, student_attrs, args.max_p
        )
        for g in ['all', 'female', 'nonfemale']:
            mtb_curves[g]['uncond'].append(run_curves[g]['uncond'])
            mtb_curves[g]['cond'].append(run_curves[g]['cond'])
            mtb_curves[g]['unmatched'].append(run_curves[g]['unmatched'])

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
        print(f"    Match rate: {(matches_stb >= 0).mean() * 100:.1f}%")
        run_curves = compute_top_p_curves(
            student_ids_stb, rankings_stb, matches_stb, student_attrs, args.max_p
        )
        for g in ['all', 'female', 'nonfemale']:
            stb_curves[g]['uncond'].append(run_curves[g]['uncond'])
            stb_curves[g]['cond'].append(run_curves[g]['cond'])
            stb_curves[g]['unmatched'].append(run_curves[g]['unmatched'])

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

    print("\nMTB vs STB summary (all students, unconditional):")
    print(f"{'p':>4}  {'MTB, Overall%':>6}  {'STB, Overall%':>6}  {'Overall Diff':>7} {'MTB, Female%':>6}  {'STB, Female%':>6}  {'Female Diff':>7} {'MTB, Non-female%':>6}  {'STB, Non-female%':>6}  {'Non-female Diff':>7}")
    for p in range(1, args.max_p + 1):
        mtb_v = mtb_results['all']['uncond']['mean'][p]
        stb_v = stb_results['all']['uncond']['mean'][p]
        mtb_v_f = mtb_results['female']['uncond']['mean'][p]
        stb_v_f = stb_results['female']['uncond']['mean'][p]
        mtb_v_nf = mtb_results['nonfemale']['uncond']['mean'][p]
        stb_v_nf = stb_results['nonfemale']['uncond']['mean'][p]
        print(f"{p:>4}  {mtb_v:>6.1f}  {stb_v:>6.1f}  {stb_v - mtb_v:>+7.1f}pp {mtb_v_f:>6.1f}  {stb_v_f:>6.1f} {stb_v_f - mtb_v_f:>+7.1f}pp {mtb_v_nf:>6.1f}  {stb_v_nf:>6.1f}  {stb_v_nf - mtb_v_nf:>+7.1f}pp")

    print(f"\nUnmatched rates:")
    print(f"{'':>4}  {'MTB Overall':>12}  {'STB Overall':>12}  {'MTB Female':>12}  {'STB Female':>12}  {'MTB Non-f':>12}  {'STB Non-f':>12}")
    mtb_un    = mtb_results['all']['unmatched']['mean']
    stb_un    = stb_results['all']['unmatched']['mean']
    mtb_un_f  = mtb_results['female']['unmatched']['mean']
    stb_un_f  = stb_results['female']['unmatched']['mean']
    mtb_un_nf = mtb_results['nonfemale']['unmatched']['mean']
    stb_un_nf = stb_results['nonfemale']['unmatched']['mean']
    print(f"{'':>4}  {mtb_un:>12.1f}  {stb_un:>12.1f}  {mtb_un_f:>12.1f}  {stb_un_f:>12.1f}  {mtb_un_nf:>12.1f}  {stb_un_nf:>12.1f}")

if __name__ == '__main__':
    main()