"""
plot_chile_welfare.py

Compares welfare under MTB (real lottery) vs STB (counterfactual) for:
  - All students
  - Female students
  - Non-female students

Uses real individual-level Chilean preference data (indv_df).
Preference lists and priority attributes are fixed and real.
Only the lottery mechanism changes between MTB and STB.

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
):
    """
    Run Chile DA with real priority attributes.
    student_lottery=None  -> MTB (independent per-school draws)
    student_lottery=dict  -> STB (single draw per student, shared across schools)
    Returns student_ids, student_rankings, matches_idx.
    """
    student_ids, student_rankings, dense_scores, _, capacities, _, _ = \
        _assign_school_level_priority_tiers_and_dense_scores(
            applications_long, school_table, rng,
            student_lottery=student_lottery,
        )
    matches = gale_shapley_per_school_numba_wrapper(
        student_rankings, dense_scores, capacities
    )
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
    df = pd.concat(curve_list, axis=1)
    return pd.DataFrame({'mean': df.mean(axis=1), 'std': df.std(axis=1)})


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
        ax.plot(ps, [mtb[p] for p in ps],
                linestyle='-', color=color, linewidth=1, marker='o', markersize=4,
                label=f'{label} — MTB (real)')

        stb = stb_results[group][condition]
        ax.plot(ps, [stb['mean'][p] for p in ps],
                linestyle='--', color=color, linewidth=1, marker='s', markersize=4,
                label=f'{label} — STB')
        ax.fill_between(
            ps,
            [stb['mean'][p] - stb['std'][p] for p in ps],
            [stb['mean'][p] + stb['std'][p] for p in ps],
            alpha=0.12, color=color,
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


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--individual',    required=True)
    parser.add_argument('--capacity',      required=True)
    parser.add_argument('--output_uncond', default='welfare_uncond.png')
    parser.add_argument('--output_cond',   default='welfare_cond.png')
    parser.add_argument('--n_stb_runs',   type=int, default=10)
    parser.add_argument('--max_p',        type=int, default=10)
    parser.add_argument('--seed',         type=int, default=DATA_GENERATION_SEED)
    args = parser.parse_args()

    print("Loading data...")
    indv_df     = load_df(args.individual)
    capacity_df = load_df(args.capacity)

    applications_long = build_applications_long(indv_df)
    student_attrs     = build_student_attrs(indv_df)
    school_table      = _prepare_school_capacity_table(capacity_df)

    all_student_ids = sorted(applications_long['mrun'].unique().tolist())
    print(f"  Students:   {len(all_student_ids):,}")
    print(f"  Schools:    {len(school_table):,}")
    print(f"  Female:     {(student_attrs['female'] == 1).sum():,}")
    print(f"  Non-female: {(student_attrs['female'] == 0).sum():,}")

    rng = np.random.default_rng(args.seed)

    # MTB — single run with per-school independent lottery draws
    print("\nRunning MTB matching (real priority, per-school lottery)...")
    student_ids_mtb, rankings_mtb, matches_mtb = run_matching(
        applications_long, school_table, rng, student_lottery=None
    )
    print(f"  MTB match rate: {(matches_mtb >= 0).mean() * 100:.1f}%")
    mtb_results = compute_top_p_curves(
        student_ids_mtb, rankings_mtb, matches_mtb, student_attrs, args.max_p
    )

    # STB — n_stb_runs counterfactual runs with single per-student lottery
    print(f"\nRunning {args.n_stb_runs} STB counterfactual runs...")
    stb_curves: Dict = {g: {'uncond': [], 'cond': [], 'unmatched': []} for g in ['all', 'female', 'nonfemale']}

    for run in range(args.n_stb_runs):
        print(f"  STB run {run+1}/{args.n_stb_runs}...")
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
            'unmatched': {
                'mean': np.mean(stb_curves[g]['unmatched']),
                'std':  np.std(stb_curves[g]['unmatched']),
            },
        }
        for g in ['all', 'female', 'nonfemale']
    }

    print("\nGenerating plots...")
    make_plot(mtb_results, stb_results, 'uncond', args.max_p, args.output_uncond, args.n_stb_runs)
    make_plot(mtb_results, stb_results, 'cond',   args.max_p, args.output_cond,   args.n_stb_runs)

    print("\nMTB vs STB summary (all students, unconditional):")
    print(f"{'p':>4}  {'MTB, Overall%':>6}  {'STB, Overall%':>6}  {'Overall Diff':>7} {'MTB, Female%':>6}  {'STB, Female%':>6}  {'Female Diff':>7} {'MTB, Non-female%':>6}  {'STB, Non-female%':>6}  {'Non-female Diff':>7}")
    for p in range(1, args.max_p + 1):
        mtb_v = mtb_results['all']['uncond'][p]
        stb_v = stb_results['all']['uncond']['mean'][p]
        mtb_v_f = mtb_results['female']['uncond'][p]
        stb_v_f = stb_results['female']['uncond']['mean'][p]
        mtb_v_nf = mtb_results['nonfemale']['uncond'][p]
        stb_v_nf = stb_results['nonfemale']['uncond']['mean'][p]
        print(f"{p:>4}  {mtb_v:>6.1f}  {stb_v:>6.1f}  {stb_v - mtb_v:>+7.1f}pp {mtb_v_f:>6.1f}  {stb_v_f:>6.1f} {stb_v_f - mtb_v_f:>+7.1f}pp {mtb_v_nf:>6.1f}  {stb_v_nf:>6.1f}  {stb_v_nf - mtb_v_nf:>+7.1f}pp")

    print(f"\nUnmatched rates:")
    print(f"{'':>4}  {'MTB Overall':>12}  {'STB Overall':>12}  {'MTB Female':>12}  {'STB Female':>12}  {'MTB Non-f':>12}  {'STB Non-f':>12}")
    mtb_un    = mtb_results['all']['unmatched']
    stb_un    = stb_results['all']['unmatched']['mean']
    mtb_un_f  = mtb_results['female']['unmatched']
    stb_un_f  = stb_results['female']['unmatched']['mean']
    mtb_un_nf = mtb_results['nonfemale']['unmatched']
    stb_un_nf = stb_results['nonfemale']['unmatched']['mean']
    print(f"{'':>4}  {mtb_un:>12.1f}  {stb_un:>12.1f}  {mtb_un_f:>12.1f}  {stb_un_f:>12.1f}  {mtb_un_nf:>12.1f}  {stb_un_nf:>12.1f}")

if __name__ == '__main__':
    main()