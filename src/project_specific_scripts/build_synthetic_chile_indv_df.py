

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from list_length import return_chilean_list_params, sample_empirical_lengths
from mallows import sample_students_global_mixture
from chile_priority_attributes import prepare_chile_numba_inputs_from_rankings
from chilean_real_welfare_comparison import load_df

PRIORITY_COLS = [
    'priority_already_registered',
    'priority_sibling',
    'priority_student',
    'priority_parent_civil_servant',
    'priority_ex_student',
]

# Official Chile Provincia -> Region mapping (16 regions incl. Ñuble). Needed because
# some experiments are fit with --subdivision_col Provincia (a finer partition of Region),
# but the downstream figure scripts (e.g. chile_region_cumulative_stb_mtb.py) group by the
# coarser Region -- so the synthetic 'Region' column must be derived from Provincia, not
# copied from it verbatim. Santiago's macrozone labels ("Santiago_Sur", "Santiago_Centro",
# ...) are handled via a prefix match below rather than being listed individually.
PROVINCIA_TO_REGION = {
    'Iquique': 'Región de Tarapacá', 'Tamarugal': 'Región de Tarapacá',
    'Antofagasta': 'Región de Antofagasta', 'El Loa': 'Región de Antofagasta', 'Tocopilla': 'Región de Antofagasta',
    'Copiapó': 'Región de Atacama', 'Chañaral': 'Región de Atacama', 'Huasco': 'Región de Atacama',
    'Elqui': 'Región de Coquimbo', 'Choapa': 'Región de Coquimbo', 'Limarí': 'Región de Coquimbo',
    'Valparaíso': 'Región de Valparaíso', 'Isla de Pascua': 'Región de Valparaíso',
    'Los Andes': 'Región de Valparaíso', 'Petorca': 'Región de Valparaíso', 'Quillota': 'Región de Valparaíso',
    'San Antonio': 'Región de Valparaíso', 'San Felipe': 'Región de Valparaíso',
    'San Felipe de Aconcagua': 'Región de Valparaíso', 'Marga Marga': 'Región de Valparaíso',
    'Cachapoal': "Región del Libertador Bernardo O'Higgins",
    'Cardenal Caro': "Región del Libertador Bernardo O'Higgins",
    'Colchagua': "Región del Libertador Bernardo O'Higgins",
    'Talca': 'Región del Maule', 'Cauquenes': 'Región del Maule',
    'Curicó': 'Región del Maule', 'Linares': 'Región del Maule',
    'Concepción': 'Región del Bío-Bío', 'Arauco': 'Región del Bío-Bío', 'Bío-Bío': 'Región del Bío-Bío',
    'Cautín': 'Región de La Araucanía', 'Malleco': 'Región de La Araucanía',
    'Llanquihue': 'Región de Los Lagos', 'Chiloe': 'Región de Los Lagos', 'Chiloé': 'Región de Los Lagos',
    'Osorno': 'Región de Los Lagos', 'Palena': 'Región de Los Lagos',
    'Coyhaique': 'Región de Aysén del Gral.Ibañez del Campo',
    'Aysén': 'Región de Aysén del Gral.Ibañez del Campo',
    'Capitán Prat': 'Región de Aysén del Gral.Ibañez del Campo',
    'General Carrera': 'Región de Aysén del Gral.Ibañez del Campo',
    'Magallanes': 'Región de Magallanes y Antártica Chilena',
    'Antártica Chilena': 'Región de Magallanes y Antártica Chilena',
    'Tierra del Fuego': 'Región de Magallanes y Antártica Chilena',
    'Ultima Esperanza': 'Región de Magallanes y Antártica Chilena',
    'Última Esperanza': 'Región de Magallanes y Antártica Chilena',
    'Cordillera': 'Región Metropolitana de Santiago', 'Chacabuco': 'Región Metropolitana de Santiago',
    'Maipo': 'Región Metropolitana de Santiago', 'Melipilla': 'Región Metropolitana de Santiago',
    'Talagante': 'Región Metropolitana de Santiago', 'Santiago': 'Región Metropolitana de Santiago',
    'Valdivia': 'Región de Los Ríos', 'Ranco': 'Región de Los Ríos',
    'Arica': 'Región de Arica y Parinacota', 'Parinacota': 'Región de Arica y Parinacota',
    'Diguillín': 'Región de Ñuble', 'Itata': 'Región de Ñuble', 'Punilla': 'Región de Ñuble',
}


def provincia_to_region(provincia: str) -> str:
    if provincia.startswith('Santiago'):
        return 'Región Metropolitana de Santiago'
    if provincia not in PROVINCIA_TO_REGION:
        raise ValueError(f"Unrecognized Provincia value, update PROVINCIA_TO_REGION: {provincia!r}")
    return PROVINCIA_TO_REGION[provincia]


def estimate_priority_calibration(real_indv_df: pd.DataFrame) -> dict:
    """Per-student (any-application) empirical rate for each priority flag."""
    per_student = real_indv_df.groupby('mrun')[PRIORITY_COLS].max()
    return {
        'priority_student_student_rate': per_student['priority_student'].mean(),
        'priority_sibling_student_rate': per_student['priority_sibling'].mean(),
        'priority_parent_civil_servant_student_rate': per_student['priority_parent_civil_servant'].mean(),
        'priority_ex_student_student_rate': per_student['priority_ex_student'].mean(),
        'priority_already_registered_student_rate': per_student['priority_already_registered'].mean(),
    }


def sample_district_rankings(params, district, n_students, empirical_probs, rng, n_jobs=1):
    """
    Draw n_students rankings for one district, with per-student list length
    drawn from the real empirical distribution, and truncate accordingly.
    Returns list of lists of school keys (already 'rbd_program_code' strings,
    matching how params['districts'][district]['schools'] was built).
    """
    lengths = sample_empirical_lengths(n_students, empirical_probs, rng)
    n_schools_in_district = len(params['districts'][district]['schools'])
    mallows_k = min(int(lengths.max()), n_schools_in_district)

    rankings_idx = sample_students_global_mixture(
        params, district, n_students,
        n_jobs=n_jobs, random_seed=int(rng.integers(2**32)),
        k_ranking_length=mallows_k,
    )
    schools = params['districts'][district]['schools']
    rankings = [
        [schools[idx] for idx in ranking[:min(L, len(ranking))]]
        for ranking, L in zip(rankings_idx, lengths)
    ]
    return rankings, lengths


def build_synthetic_indv_df(
    params: dict,
    real_indv_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    rng: np.random.Generator,
    subdivision_col: str = 'Region',
    district: str = None,
    n_students: int = None,
    total_students: int = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Draw ONE fresh synthetic individual-level dataframe from a fitted Mallows
    mixture, in the exact schema chilean_real_welfare_comparison.py /
    chile_lottery_decile_stb_mtb.py / chile_region_cumulative_stb_mtb.py
    expect via --individual.

    Calibration (list lengths, female rate, priority-flag base rates, and
    per-district population shares) comes from real_indv_df and is fixed;
    only the preference lists -- and the priority-attribute simulation that
    depends on them -- are freshly sampled from `params` using `rng`. Calling
    this repeatedly with fresh rng state (rather than reusing one fixed
    output CSV) gives independent replicates suitable for a CI that captures
    preference-sampling randomness, not just lottery tie-break randomness.
    """
    list_length_params = return_chilean_list_params(real_indv_df)
    empirical_probs = list_length_params['list_length_empirical_probs']
    female_rate = real_indv_df.drop_duplicates('mrun')['female'].mean()
    calibration = estimate_priority_calibration(real_indv_df)
    if verbose:
        print(f"  Empirical list lengths: {sorted(empirical_probs.items())}")
        print(f"  Female rate: {female_rate:.3f}")
        print(f"  Priority calibration: {calibration}")

    if district is not None:
        if n_students is None:
            raise ValueError('n_students is required when district is given.')
        district_counts = {district: n_students}
    else:
        real_counts = real_indv_df.drop_duplicates('mrun')[subdivision_col].value_counts()
        total = total_students or int(real_counts.sum())
        fit_districts = [d for d in params['districts'] if d in real_counts.index]
        missing = set(params['districts']) - set(fit_districts)
        if missing and verbose:
            print(f"  Note: {len(missing)} district(s) in params have no match in "
                  f"real_indv_df's '{subdivision_col}' column, skipping: {sorted(missing)[:5]}...")
        if not fit_districts:
            raise ValueError(
                f"None of the {len(params['districts'])} district(s) in params match any value "
                f"in real_indv_df's '{subdivision_col}' column. This usually means the params were "
                f"fit against a different subdivision (e.g. 'Provincia' vs 'Region') -- pass the "
                f"matching subdivision_col. Sample params district keys: {sorted(params['districts'])[:5]}. "
                f"Sample '{subdivision_col}' values in real_indv_df: {sorted(real_counts.index)[:5]}."
            )
        shares = real_counts.loc[fit_districts] / real_counts.loc[fit_districts].sum()
        district_counts = {d: max(1, int(round(total * shares[d]))) for d in fit_districts}

    if verbose:
        print(f"Sampling {sum(district_counts.values()):,} students across {len(district_counts)} district(s)...")

    all_rankings, all_districts, all_lengths = [], [], []
    for d, n in district_counts.items():
        if verbose:
            print(f"  {d}: {n:,} students...")
        rankings, lengths = sample_district_rankings(params, d, n, empirical_probs, rng, n_jobs=n_jobs)
        all_rankings.extend(rankings)
        all_districts.extend([d] * n)
        all_lengths.extend(lengths.tolist())

    if verbose:
        print("Synthesizing priority attributes for sampled rankings...")
    prepared = prepare_chile_numba_inputs_from_rankings(
        truncated_rankings=all_rankings,
        capacity_rows=capacity_df,
        seed=int(rng.integers(2**32)),
        calibration=calibration,
    )
    app = prepared['application_table'][['mrun', 'rbd', 'preference_number'] + PRIORITY_COLS].copy()

    # mrun assigned as "0", "1", ... in the order of all_rankings -- same order as all_districts.
    district_by_mrun = {str(i): d for i, d in enumerate(all_districts)}
    if subdivision_col == 'Provincia':
        region_by_mrun = {mrun: provincia_to_region(d) for mrun, d in district_by_mrun.items()}
    elif subdivision_col == 'Region':
        region_by_mrun = district_by_mrun
    else:
        raise ValueError(
            f"Don't know how to derive 'Region' from subdivision_col={subdivision_col!r}; "
            "extend provincia_to_region() or add a matching case here."
        )
    app['Region'] = app['mrun'].map(region_by_mrun)

    unique_mruns = app['mrun'].unique()
    female_by_mrun = pd.Series(
        (rng.random(len(unique_mruns)) < female_rate).astype(int), index=unique_mruns
    )
    app['female'] = app['mrun'].map(female_by_mrun)

    split = app['rbd'].str.rsplit('_', n=1, expand=True)
    app['rbd'] = split[0]
    app['program_code'] = split[1]

    out = app[['mrun', 'rbd', 'program_code', 'preference_number', 'female', 'Region'] + PRIORITY_COLS]
    if verbose:
        achieved_lengths = pd.Series(all_lengths)
        print(f"\nAchieved list-length distribution:\n{achieved_lengths.value_counts(normalize=True).sort_index()}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_pkl', required=True)
    parser.add_argument('--real_individual', required=True,
                         help='Real indv_df, used only to calibrate list-length distribution, '
                              'female rate, and priority-flag base rates.')
    parser.add_argument('--capacity', required=True,
                         help='Real school capacity table (capacities themselves are not synthesized).')
    parser.add_argument('--subdivision_col', default='Region',
                         help="Column in --real_individual identifying districts. Use 'Provincia' "
                              'if the saved params were fit at province level.')
    parser.add_argument('--district', default=None,
                         help='Restrict to one district (requires --n_students). Omit to sample '
                              'every district in the params file, sized proportionally to the '
                              'real population.')
    parser.add_argument('--n_students', type=int, default=None,
                         help='Required with --district: students to sample for that district.')
    parser.add_argument('--total_students', type=int, default=None,
                         help='Total students across all districts when --district is omitted. '
                              'Defaults to the real data\'s total student count.')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='synthetic_indv_df.csv')
    args = parser.parse_args()

    if args.district is not None and args.n_students is None:
        parser.error('--n_students is required when --district is given.')

    print("Loading real data for calibration...")
    real_indv_df = load_df(args.real_individual)
    capacity_df = load_df(args.capacity)

    with open(args.params_pkl, 'rb') as f:
        params = pickle.load(f)

    rng = np.random.default_rng(args.seed)

    out = build_synthetic_indv_df(
        params, real_indv_df, capacity_df, rng,
        subdivision_col=args.subdivision_col, district=args.district,
        n_students=args.n_students, total_students=args.total_students,
        n_jobs=args.n_jobs, verbose=True,
    )
    out.to_csv(args.output, index=False)

    print(f"\nSaved {out['mrun'].nunique():,} students ({len(out):,} application rows) to {args.output}")
    print("This file can be passed directly as --individual to the Fig 4/6/8 scripts, "
          "alongside the same real --capacity file. Those scripts can also take --params_pkl/"
          "--real_individual directly instead of a pre-built --individual file, resampling a "
          "fresh synthetic replicate (like this one) each run for a CI that also captures "
          "preference-sampling randomness, not just the lottery.")


if __name__ == '__main__':
    main()
