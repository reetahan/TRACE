
import argparse
import pickle
import os
import json
import numpy as np
import pandas as pd 


from data_ingestion import read_data, nyc_preprocess_data
from welfare import evaluate_simulation_output
from file_config import *
from concurrent.futures import ProcessPoolExecutor
from mallows import _sample_students_chunk
from gale_shapley import  compute_aggregates 
from constants import DISTRICT_TO_BOROUGH_MAPPING
from nyc_priority_attributes import run_nyc_priority_matching, _sample_student_attributes, _get_tiers
from list_length import sample_truncated_normal_lengths


def sample_rankings(
    params,
    match_stats_df,
    sampling_n_jobs=32,
    sampling_chunk_size=2000,
    list_length_max=12,
    seed=DATA_GENERATION_SEED,
    executor=None,
):
    """
    Sample Mallows preference rankings for all students across all districts.
    Returns raw index-based rankings (before list length truncation),
    district assignments, and the rng used.
    """
    districts = list(params['districts'].keys())
    all_rankings = []
    all_district_assignments = []
    all_chunks = []
    rng = np.random.default_rng(seed=seed)

    
    for district in districts:
        n_students = int(
            match_stats_df[
                match_stats_df['Residential District'] == district
            ]['Total Applicants'].iloc[0]
        )
        sigma_d = params['districts'][district]['central_ranking']
        schools_list = params['districts'][district]['schools']
        school_to_idx = {s: i for i, s in enumerate(schools_list)}
        sigma_indices = np.array([school_to_idx[s] for s in sigma_d])

        component_indices = rng.choice(
            len(params['global_phis']),
            size=n_students,
            p=params['global_weights']
        )
        for start in range(0, n_students, sampling_chunk_size):
            chunk = component_indices[start:start + sampling_chunk_size]
            all_chunks.append(
                (district, schools_list, sigma_indices, chunk, rng.integers(2**32), list_length_max)
            )
        all_district_assignments.extend([district] * n_students)

    results_by_district = {d: [] for d in districts}

    if sampling_n_jobs > 1 and executor is not None:
        futures = []
        for district, schools_list, sigma_indices, chunk, seed, list_length_max in all_chunks:
            future = executor.submit(_sample_students_chunk, sigma_indices, params['global_phis'], chunk, seed, list_length_max)
            futures.append((district, future))
        for district, future in futures:
            results_by_district[district].extend(future.result())
    elif sampling_n_jobs > 1:
        with ProcessPoolExecutor(max_workers=sampling_n_jobs) as pool:
            futures = []
            for district, schools_list, sigma_indices, chunk, seed, list_length_max in all_chunks:
                future = pool.submit(_sample_students_chunk, sigma_indices, params['global_phis'], chunk, seed, list_length_max)
                futures.append((district, future))
            for district, future in futures:
                results_by_district[district].extend(future.result())
    else:
        for district, schools_list, sigma_indices, chunk, seed, list_length_max in all_chunks:
            results_by_district[district].extend(
                _sample_students_chunk(sigma_indices, params['global_phis'], chunk, seed, list_length_max)
            )

    # Convert to school DBNs — full lists, no truncation
    for district in districts:
        schools_list = params['districts'][district]['schools']
        rankings = results_by_district[district]
        rankings_as_schools = [[schools_list[idx] for idx in r] for r in rankings]
        all_rankings.extend(rankings_as_schools)

    return all_rankings, all_district_assignments, rng

def run_matching(
    all_rankings,
    all_district_assignments,
    df,
    school_info_df,
    lottery_global,
    list_length_min,
    list_length_mean,
    list_length_std,
    list_length_max,
    rng,
    priority_config=None,
    per_school_lottery=False,
    student_attrs=None
):
    """
    Given pre-sampled full rankings, applies list length truncation and runs DA.
    Isolates the matching step so the same preference profile can be reused
    across multiple list_length_min values.
    """
    

    all_schools = df['School DBN'].unique()
    school_to_idx = {s: i for i, s in enumerate(all_schools)}
    capacities_dict = school_info_df.set_index('School DBN')['Capacity'].to_dict()
    capacities = np.array([capacities_dict.get(s, 0) for s in all_schools])
    n_students = len(all_rankings)
    n_schools = len(all_schools)

    # Group rankings by district to apply per-district list length sampling
    district_list = list(dict.fromkeys(all_district_assignments))  # preserve order
    district_indices = {}
    for i, d in enumerate(all_district_assignments):
        district_indices.setdefault(d, []).append(i)

    truncated_rankings = [None] * n_students
    all_list_lengths = [None] * n_students

    for district, indices in district_indices.items():
        n_d = len(indices)
        schools_list = [s for r in [all_rankings[i] for i in indices] for s in r]
        max_schools = max(len(all_rankings[i]) for i in indices)
        max_len_here = min(list_length_max, max_schools)
        effective_min = min(list_length_min, max_len_here)

        list_lengths = sample_truncated_normal_lengths(
            n_students=n_d,
            mean=list_length_mean,
            std=list_length_std,
            min_len=effective_min,
            max_len=max_len_here,
            rng=rng,
        )
        for j, (idx, L) in enumerate(zip(indices, list_lengths)):
            truncated_rankings[idx] = all_rankings[idx][:L]
            all_list_lengths[idx] = L

    def expand_ranking(ranking):
        seen = set()
        expanded = []
        for dbn in ranking:
            if dbn in seen or dbn not in school_to_idx:
                continue
            seen.add(dbn)
            expanded.append(school_to_idx[dbn])
        return np.array(expanded, dtype=np.int32)

    rankings_as_indices = [expand_ranking(r) for r in truncated_rankings]

    if per_school_lottery:
        school_lotteries = rng.random((n_schools, n_students))
    else:
        lottery_1d = np.argsort(lottery_global[:n_students]).astype(np.float64) / n_students
        school_lotteries = np.tile(lottery_1d, (n_schools, 1))


    student_attrs = None
    matches_schools, student_attrs = run_nyc_priority_matching(
        truncated_rankings=truncated_rankings,
        district_assignments=all_district_assignments,
        all_schools=all_schools,
        priority_config=priority_config,
        district_to_borough=DISTRICT_TO_BOROUGH_MAPPING,
        school_lotteries=school_lotteries,
        rng=rng,
        student_attrs=student_attrs
    )
    matches_idx = np.array([
        school_to_idx.get(s, -1) if s != '-1' else -1
        for s in matches_schools
    ], dtype=np.int32)

    agg = compute_aggregates(
        truncated_rankings,
        matches_schools,
        np.array(all_district_assignments),
        all_schools,
    )

    return agg, truncated_rankings, rankings_as_indices, matches_idx, student_attrs


def run_sweep(params, lottery, df, match_stats_df, school_info_df,
              priority_config, district_to_borough,
              min_lengths, output_dir, seed, n_jobs, save_ranking=False):

    rng = np.random.default_rng(seed=seed)

    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []
    list_length_max = 15

    # Sample preferences and attributes ONCE
    print("Sampling preferences (fixed across all min_len values)...")
    all_rankings, all_district_assignments, rng = sample_rankings(
        params=params,
        match_stats_df=match_stats_df,
        sampling_n_jobs=n_jobs,
        list_length_max=list_length_max,
        seed=seed
    )
    print(f"  Sampled {len(all_rankings)} student rankings")

    all_schools = df['School DBN'].unique()
    borough_rows = [] 
    lottery_rows = []


    school_overrides = priority_config.get('school_overrides', {})
    borough_swd_rates = {}
    for prog_key, prog_data in school_overrides.items():
        borough = prog_data.get('borough', '')
        seats_ge = prog_data.get('seats_ge', 0)
        seats_swd = prog_data.get('seats_swd', 0)
        total = seats_ge + seats_swd
        if total > 0 and borough:
            if borough not in borough_swd_rates:
                borough_swd_rates[borough] = {'swd': 0, 'total': 0}
            borough_swd_rates[borough]['swd'] += seats_swd
            borough_swd_rates[borough]['total'] += total

    borough_swd_fractions = {
        b: v['swd'] / v['total']
        for b, v in borough_swd_rates.items()
        if v['total'] > 0
    }

    fixed_student_attrs = _sample_student_attributes(
        district_assignments=all_district_assignments,
        district_to_borough=DISTRICT_TO_BOROUGH_MAPPING,
        rng=rng,
        borough_swd_fractions=borough_swd_fractions,
        priority_config=priority_config,
    )




    for min_len in min_lengths:
        print(f"\n{'='*50}")
        print(f"Running list_length_min={min_len}")
        print(f"{'='*50}")

        agg, syn_rankings, syn_rankings_idx, matches_idx, syn_attrs = run_matching(
            all_rankings=all_rankings,
            all_district_assignments=all_district_assignments,
            df=df,
            school_info_df=school_info_df,
            lottery_global=lottery,
            list_length_min=min_len,
            list_length_mean=7,
            list_length_std=2,
            list_length_max=list_length_max,
            rng=rng,
            priority_config=priority_config,
            per_school_lottery=False,
            student_attrs=fixed_student_attrs
        )

        if save_ranking and min_len == 1:
            max_len = max(len(r) for r in syn_rankings)
            ranking_rows = []
            for i, (ranking, district) in enumerate(zip(syn_rankings, all_district_assignments)):
                row = {'student_id': i, 'district': district}
                for j in range(max_len):
                    row[f'choice_{j+1}'] = ranking[j] if j < len(ranking) else None
                ranking_rows.append(row)
            pd.DataFrame(ranking_rows).to_csv(
                os.path.join(output_dir, 'synthetic_rankings.csv'), index=False
            )
            print(f"Saved synthetic rankings to {output_dir}/synthetic_rankings.csv")



        n_students = len(all_district_assignments)
        lottery_1d = np.argsort(lottery[:n_students]).astype(np.float64) / n_students

        attr_df = syn_attrs if syn_attrs is not None else pd.DataFrame()
        attr_df['district'] = list(all_district_assignments)
        attr_df['lottery'] = lottery_1d
        attr_df['lottery_decile'] = pd.qcut(lottery_1d, q=10, labels=[f'D{i}' for i in range(1, 11)])


        welfare_results = evaluate_simulation_output(
            sim_output={
                'rankings_as_indices': syn_rankings_idx,
                'matches_idx':         matches_idx,
                'student_attributes':  attr_df,
            },
            categories=['district', 'borough', 'lottery_decile'],
            output_dir=os.path.join(output_dir, f'min_len_{min_len}'),
        )

        stats = welfare_results.rank_stats
        matched = (matches_idx >= 0).sum()
        n_total = len(matches_idx)

        print(f"  pct_matched: {stats['pct_matched']:.2f}%")
        print(f"  avg_rank:    {stats['avg_rank']:.3f}")
        print(f"  rank_var:    {stats['rank_variance']:.3f}")

        summary_rows.append({
            'list_length_min': min_len,
            'pct_matched':     round(stats['pct_matched'], 4),
            'avg_rank':        round(stats['avg_rank'], 4),
            'rank_variance':   round(stats['rank_variance'], 4),
            'n_matched':       int(matched),
            'n_total':         int(n_total),
        })

        borough_sweep = welfare_results.top_p_sweep_by_category.get('borough')
        if borough_sweep is not None:
            borough_sweep = borough_sweep.copy()
            borough_sweep['list_length_min'] = min_len
            borough_rows.append(borough_sweep)
        
        lottery_sweep = welfare_results.top_p_sweep_by_category.get('lottery_decile')
        if lottery_sweep is not None:
            lottery_sweep = lottery_sweep.copy()
            lottery_sweep['list_length_min'] = min_len
            lottery_rows.append(lottery_sweep)

    lottery_df = pd.concat(lottery_rows, ignore_index=True) if lottery_rows else None
    if lottery_df is not None:
        lottery_path = os.path.join(output_dir, 'sweep_lottery.csv')
        lottery_df.to_csv(lottery_path, index=False)
        print(f"Lottery sweep saved to {lottery_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, 'sweep_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    borough_df = pd.concat(borough_rows, ignore_index=True) if borough_rows else None
    if borough_df is not None:
        borough_path = os.path.join(output_dir, 'sweep_borough.csv')
        borough_df.to_csv(borough_path, index=False)
        print(f"\nBorough sweep saved to {borough_path}")
    print(f"\nSummary saved to {summary_path}")
    print(summary_df.to_string(index=False))
    return summary_df, borough_df, lottery_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params',      required=True)
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--min_lengths', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    parser.add_argument('--seed',        type=int, default=DATA_GENERATION_SEED)
    parser.add_argument('--n_jobs',      type=int, default=32)
    parser.add_argument('--df_filepath', type=str, default=None)
    parser.add_argument('--save_ranking', action='store_true')
    args = parser.parse_args()

    # Load data
    if args.df_filepath is None:
        args.df_filepath = f"{POLISHED_DATA_DIR}/master_data_03_residential_district.xlsx"

    print("Loading data...")
    df_raw = read_data(args.df_filepath)
    match_stats_df = read_data(
        f"{RAW_DATA_DIR}/DATA3_fall-2024-high-school-offer-results-website-1.xlsx",
        sheet='Match to Choice-District',
        is_first_row_header = True
    )
    school_info_df = read_data(
        f"{RAW_DATA_DIR}/DATA4_fall-2025---hs-directory-data.xlsx",
        sheet='Data'
    )
    addtl_school_info_df = read_data(
        f"{RAW_DATA_DIR}/DATA2_fall-2024-admissions_part-ii_suppressed.xlsx",
        sheet='School'
    )
    df, match_stats_df, school_info_df = nyc_preprocess_data(
        df_raw, match_stats_df, school_info_df, addtl_school_info_df
    )
    print(f"  Schools: {df['School DBN'].nunique()}, Students: {int(match_stats_df['Total Applicants'].sum())}")

    # Load priority config
    nyc_config_path = f"{POLISHED_DATA_DIR}/nyc_priority_config.json"
    priority_config = None
    if os.path.exists(nyc_config_path):
        with open(nyc_config_path) as f:
            priority_config = json.load(f)
        print(f"  Loaded priority config: {nyc_config_path}")

    # Load params
    print(f"Loading params from {args.params}...")
    with open(args.params, 'rb') as f:
        params = pickle.load(f)
    print(f"  K={len(params['global_phis'])} components")
    print(f"  phis: {params['global_phis']}")

    # Generate a fixed lottery
    n_students = int(match_stats_df['Total Applicants'].sum())
    lottery = np.random.default_rng(args.seed).permutation(n_students)

    run_sweep(
        params=params,
        lottery=lottery,
        df=df,
        match_stats_df=match_stats_df,
        school_info_df=school_info_df,
        priority_config=priority_config,
        district_to_borough=DISTRICT_TO_BOROUGH_MAPPING,
        min_lengths=args.min_lengths,
        output_dir=args.output_dir,
        seed=args.seed,
        n_jobs=args.n_jobs,
        save_ranking=args.save_ranking
    )


if __name__ == '__main__':
    main()