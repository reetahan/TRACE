from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import copy
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from util import log_and_print
from constants import *
from data_ingestion import extract_observed_aggregates
from gale_shapley import compute_aggregates, gale_shapley_per_school_numba_wrapper
from mallows import  _sample_students_chunk
from list_length import sample_truncated_normal_lengths, sample_empirical_lengths
from nyc_priority_attributes import run_nyc_priority_matching
from chile_priority_attributes import prepare_chile_numba_inputs_from_rankings

class ExperimentResult:
    def __init__(self):
        self.params = None 
        self.lottery = None
        self.log_likelihoods = None
        self.final_agg = None
        self.syn_rankings = None
        self.syn_rankings_idx = None
        self.matches_idx = None
        self.syn_districts = None
        self.syn_attrs = None
    
    def set_params(self, params):
        self.params = params

    def set_other_stats(self, lottery, log_likelihoods, final_agg):
        self.lottery = lottery
        self.log_likelihoods = log_likelihoods
        self.final_agg = final_agg

    def set_synthetic_output(self, syn_rankings, syn_rankings_idx, 
                             matches_idx, syn_districts, syn_attrs):
        self.syn_rankings = syn_rankings
        self.syn_rankings_idx = syn_rankings_idx
        self.matches_idx = matches_idx
        self.syn_districts = syn_districts
        self.syn_attrs = syn_attrs


def run_single_simulation(
    params,
    df,
    match_stats_df,
    school_info_df,
    lottery_fixed=None,
    mallows_seed=39,
    outfile=None,
    sampling_n_jobs=32,
    sampling_chunk_size=2000,
    executor=None,
    per_school_lottery=False,
    profile_timing=False,
    priority_config=None, 
    district_to_region=None, 
    list_length_params=None, 
    save_best_sample=False,
    run_priority_analysis=False,
    max_p=None
):
    t_total_start = time.perf_counter()
    timings = {}

    def _mark_timing(label, t_start):
        timings[label] = timings.get(label, 0.0) + (time.perf_counter() - t_start)

    all_rankings = []
    all_district_assignments = []
    all_list_lengths = []
    
    if(list_length_params is not None):
        list_length_mode = list_length_params.get('list_length_mode', 'fixed') if list_length_params else 'fixed'
        if(list_length_mode == 'fixed'):
            k_ranking_length = list_length_params.get('k_ranking_length', 5)
        if(list_length_mode == 'gaussian'):
            list_length_mean = list_length_params.get('list_length_mean', 7)
            list_length_std = list_length_params.get('list_length_std', 2)
            list_length_min = list_length_params.get('list_length_min', 1)

            list_length_max = list_length_params['list_length_max'] if 'list_length_max' in list_length_params else None

        if(list_length_mode == 'empirical'):
            list_length_empirical_probs = list_length_params.get('list_length_empirical_probs', None)

        if list_length_mode == 'gaussian_per_district':
            list_length_mean_per_district = list_length_params.get('list_length_mean_per_district', {})
            list_length_std = list_length_params.get('list_length_std', 2)
            list_length_min = list_length_params.get('list_length_min', 1)

    districts = list(params['districts'].keys())
    max_schools_in_any_district = max(len(params['districts'][d]['schools']) for d in districts)
    if list_length_mode == 'fixed':
        mallows_k = k_ranking_length
    elif list_length_mode == 'gaussian_per_district':
        max_mean = max(list_length_mean_per_district.values())
        mallows_k = min(max_schools_in_any_district, int(max_mean + 10 * list_length_std) + 1)
    elif list_length_mode == 'gaussian':
        max_stds_above = 10
        mallows_k = min(list_length_max, max_schools_in_any_district) if list_length_max is not None  \
            else min(max_schools_in_any_district, int(list_length_mean + max_stds_above * list_length_std) + 1)
    elif list_length_mode == 'empirical':
        mallows_k = len(list_length_empirical_probs)

    # Collect all chunks across all districts
    t_chunks_start = time.perf_counter()
    all_chunks = []  # (district, schools_list, sigma_indices, chunk_components, seed, mallows_kx)
    rng = np.random.default_rng(seed=mallows_seed)
    
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
                (district, schools_list, sigma_indices, chunk, rng.integers(2**32), mallows_k)
            )

        all_district_assignments.extend([district] * n_students)
    _mark_timing('build_chunks', t_chunks_start)

    # ONE pool for all districts
    results_by_district = {d: [] for d in districts}

    t_sampling_start = time.perf_counter()
    if sampling_n_jobs > 1 and executor is not None:
        futures = []
        for district, schools_list, sigma_indices, chunk, seed, mallows_k in all_chunks:
            future = executor.submit(
                _sample_students_chunk,
                sigma_indices,
                params['global_phis'],
                chunk,
                seed,
                mallows_k
            )
            futures.append((district, future))

        for district, future in futures:
            results_by_district[district].extend(future.result())

    elif sampling_n_jobs > 1:
        with ProcessPoolExecutor(max_workers=sampling_n_jobs) as pool:
            futures = []
            for district, schools_list, sigma_indices, chunk, seed, mallows_k in all_chunks:
                future = pool.submit(
                    _sample_students_chunk,
                    sigma_indices,
                    params['global_phis'],
                    chunk,
                    seed,
                    mallows_k
                )
                futures.append((district, future))

            for district, future in futures:
                results_by_district[district].extend(future.result())

    else:
        for district, schools_list, sigma_indices, chunk, seed, mallows_k in all_chunks:
            results_by_district[district].extend(
                _sample_students_chunk(
                    sigma_indices,
                    params['global_phis'],
                    chunk,
                    seed,
                    mallows_k
                )
            )
    _mark_timing('sample_preferences', t_sampling_start)

    # Convert to school DBNs and truncate
    t_convert_start = time.perf_counter()
    for district in districts:
        schools_list = params['districts'][district]['schools']
        rankings = results_by_district[district]
        n_students_d = len(rankings)

        if list_length_mode == "fixed":
            max_len_here = min(k_ranking_length, len(schools_list))
            list_lengths = np.full(n_students_d, max_len_here, dtype=int)

        elif list_length_mode == "gaussian":
            max_len_here = min(list_length_max, len(schools_list)) if list_length_max is not None else len(schools_list)
            list_lengths = sample_truncated_normal_lengths(
                n_students=n_students_d,
                mean=list_length_mean,
                std=list_length_std,
                min_len=list_length_min,
                max_len=max_len_here,
                rng=rng
            )
        elif list_length_mode == "gaussian_per_district":   
            mu_d = list_length_mean_per_district.get(str(district), list_length_std * 3)
            max_len_here = len(schools_list)
            list_lengths = sample_truncated_normal_lengths(
                n_students=n_students_d,
                mean=mu_d,
                std=list_length_std,
                min_len=list_length_min,
                max_len=max_len_here,
                rng=rng
            )
        elif list_length_mode == "empirical":
            if list_length_empirical_probs is None:
                raise ValueError("list_length_mode='empirical' requires list_length_empirical_probs")
            list_lengths = sample_empirical_lengths(n_students_d, list_length_empirical_probs, rng)
        else:
            raise ValueError(f"Unknown list_length_mode: {list_length_mode}")

        truncated_rankings = [r[:L] for r, L in zip(rankings, list_lengths)]
        rankings_as_schools = [[schools_list[idx] for idx in r] for r in truncated_rankings]

        all_rankings.extend(rankings_as_schools)
        all_list_lengths.extend(list_lengths.tolist())
    _mark_timing('convert_and_truncate', t_convert_start)

    log_and_print(
        f" Generated {len(all_rankings)} student rankings across {len(districts)} districts ({len(all_chunks)} chunks)",
        log_file=outfile
    )

    t_match_prep_start = time.perf_counter()
    all_schools = df['School DBN'].unique()
    school_to_idx = {s: i for i, s in enumerate(all_schools)}


    capacities_dict = school_info_df.set_index('School DBN')['Capacity'].to_dict()
    capacities = np.array([capacities_dict.get(s, 0) for s in all_schools])
    _mark_timing('prepare_matching_inputs', t_match_prep_start)

    def expand_ranking(ranking):
        seen = set()
        expanded = []
        for dbn in ranking:
            if dbn in seen or dbn not in school_to_idx:
                continue
            seen.add(dbn)
            expanded.append(school_to_idx[dbn])
        return np.array(expanded, dtype=np.int32)

    rankings_as_indices = [expand_ranking(r) for r in all_rankings]

    t_matching_start = time.perf_counter()
    n_students = len(rankings_as_indices)
    n_schools = len(all_schools)

    if per_school_lottery:
        school_lotteries = rng.random((n_schools, n_students))
    else:
        lottery_1d = np.argsort(lottery_fixed[:n_students]).astype(np.float64) / n_students
        school_lotteries = np.tile(lottery_1d, (n_schools, 1))

    system_name = priority_config.get('__meta__', {}).get('system_name', '') if priority_config else ''

    if system_name == 'NYC':
        matches_schools, student_attrs = run_nyc_priority_matching(
            truncated_rankings=all_rankings,
            district_assignments=all_district_assignments,
            all_schools=all_schools,
            priority_config=priority_config,
            district_to_borough=district_to_region,
            school_lotteries=school_lotteries,
            rng=rng,
            log_file=outfile if run_priority_analysis else None
        )
        matches_idx = np.array([
            school_to_idx.get(s, -1) if s != '-1' else -1
            for s in matches_schools
        ], dtype=np.int32)

    elif system_name == 'Chile':
        
        region_overrides = priority_config.get('region_overrides', {})
        if region_overrides:
            first_region = next(iter(region_overrides.values()))
            fracs = first_region.get('student_attribute_fractions', {})
            tiers = first_region.get('priority_tiers', [])
            calibration = {
                'priority_student_student_rate': fracs.get('disadvantaged', 0.6889),
                'priority_sibling_student_rate': next((t['fraction_eligible'] for t in tiers if t['group'] == 'sibling'), 0.1539),
                'priority_parent_civil_servant_student_rate': next((t['fraction_eligible'] for t in tiers if t['group'] == 'working_parent'), 0.0066),
                'priority_ex_student_student_rate': next((t['fraction_eligible'] for t in tiers if t['group'] == 'returning_student'), 0.0521),
                'priority_already_registered_student_rate': fracs.get('already_registered', 0.1398),
            }
        else:
            calibration = None

        prepared = prepare_chile_numba_inputs_from_rankings(
            truncated_rankings=all_rankings,
            capacity_rows=school_info_df,
            seed=int(rng.integers(2**32)),
            calibration=calibration,
        )
        matches_virtual = gale_shapley_per_school_numba_wrapper(
            prepared['student_rankings'],
            prepared['school_priority_scores'],
            prepared['school_capacities'],
        )
        school_ids = prepared['school_ids']
        matches_schools = np.array([
            school_ids[m] if m >= 0 else '-1'
            for m in matches_virtual
        ])
        matches_idx = np.array([
            school_to_idx.get(s, -1) if s != '-1' else -1
            for s in matches_schools
        ], dtype=np.int32)

        student_attrs = prepared['student_attributes']

    else:
        matches_idx = gale_shapley_per_school_numba_wrapper(
            rankings_as_indices, school_lotteries, capacities
        )
        matches_schools = np.array([all_schools[m] if m >= 0 else '-1' for m in matches_idx])
        student_attrs = None

    _mark_timing('matching', t_matching_start)

    t_agg_start = time.perf_counter()
    agg = compute_aggregates(
        all_rankings,
        matches_schools,
        np.array(all_district_assignments),
        all_schools, 
        max_p=max_p
    )
    _mark_timing('compute_aggregates', t_agg_start)

    if profile_timing:
        timings['total'] = time.perf_counter() - t_total_start
        log_and_print(
            (
                " [TIMING] run_single_simulation "
                f"total={timings['total']:.3f}s | "
                f"build_chunks={timings.get('build_chunks', 0.0):.3f}s | "
                f"sample_preferences={timings.get('sample_preferences', 0.0):.3f}s | "
                f"convert_and_truncate={timings.get('convert_and_truncate', 0.0):.3f}s | "
                f"prepare_matching_inputs={timings.get('prepare_matching_inputs', 0.0):.3f}s | "
                f"matching={timings.get('matching', 0.0):.3f}s | "
                f"compute_aggregates={timings.get('compute_aggregates', 0.0):.3f}s"
            ),
            log_file=outfile,
        )
        log_and_print(
            (
                " [TIMING] workload "
                f"districts={len(districts)} | students={len(all_rankings)} | "
                f"chunks={len(all_chunks)} | sampling_n_jobs={sampling_n_jobs} | "
                f"chunk_size={sampling_chunk_size}"
            ),
            log_file=outfile,
        )


    if save_best_sample:
        student_df = pd.DataFrame({
            "student_id": np.arange(len(all_rankings)),
            "district": np.array(all_district_assignments),
            "list_length": np.array(all_list_lengths),
            "match": matches_schools,
            "unmatched": (matches_schools == "-1").astype(int),
        })

        synth_info = {
            'all_rankings': all_rankings,
            'rankings_as_indices': rankings_as_indices,
            'matches_idx': matches_idx,
            'syn_districts': np.array(all_district_assignments),
            'student_attrs': student_attrs,
            'student_df': student_df
        }

        return agg, synth_info
    return agg


def EM_algorithm(df, match_stats_df, school_info_df,
                 max_iter=10, tol=0.01, K=1, M_simulations=20, seed=40, eta=LEARNING_RATE, outfile=None, 
                 sampling_n_jobs=32, max_iter_opt=5, per_school_lottery=False, 
                 profile_timing=True, priority_config=None, district_to_region=None, 
                 list_length_params=None, save_best_sample=False, max_p=None):


    cur_experiment_result = ExperimentResult()
    executor = ProcessPoolExecutor(max_workers=sampling_n_jobs)
    
    districts = sorted(df['Residential District'].unique())
    n_total_students = int(match_stats_df['Total Applicants'].sum())
    log_and_print(f"\nInitialization:", log_file=outfile)
    log_and_print(f"  Districts: {len(districts)}", log_file=outfile)
    log_and_print(f"  Total students: {n_total_students}", log_file=outfile)
    log_and_print(f"  Global mixture components: K={K}", log_file=outfile)
    log_and_print(f"  Max iterations of EM Algorithm: {max_iter}", log_file=outfile)
    log_and_print(f"  Max iterations of nonconvex optimizer: {max_iter_opt}", log_file=outfile)
    log_and_print(f"  Simulations per evaluation: M={M_simulations}\n", log_file=outfile)
    if profile_timing:
        log_and_print("  Timing instrumentation: ENABLED", log_file=outfile)
    
    rng_init = np.random.default_rng(seed)
    params = initialize_parameters_global_mixture(districts, df, K, rng=rng_init)

    observed_agg = extract_observed_aggregates(df, match_stats_df, max_p=max_p)
    
    log_likelihoods = []
    best_params = None
    best_log_like = -np.inf
    best_agg = None
    best_syn_data = None
    best_phis_seen = None

    all_schools = df['School DBN'].unique()
    obs_total_app, obs_true_app = _build_observed_app_matrices(df, districts, all_schools)
    
    for iteration in range(max_iter):
        log_and_print(f"\n{'='*30}\n EM ITERATION {iteration + 1}/{max_iter} \n{'='*30}", log_file=outfile)
        
        old_params = copy.deepcopy(params)
        
        log_and_print(f"Entering the optimization of the global mixture...", log_file=outfile)
        # M-STEP: Optimize global parameters
        params, final_agg, total_log_like, lottery, syn_data = optimize_global_mixture(
            params, observed_agg, df, match_stats_df, 
            school_info_df, M=M_simulations, seed=seed, 
            iteration=iteration, outfile=outfile, sampling_n_jobs=sampling_n_jobs,
            executor=executor, max_iter_em=max_iter, max_iter_opt=max_iter_opt,
            per_school_lottery=per_school_lottery, priority_config=priority_config,
            district_to_region=district_to_region, list_length_params=list_length_params, 
            save_best_sample=save_best_sample, best_phis_seen=best_phis_seen, max_p=max_p
        )

        log_and_print(f"Checking results of optimizing global mixture...", log_file=outfile)

        # Sort them to remove indexing ambiguity
        sorted_indices = np.argsort(params['global_phis'])
        params['global_phis'] = params['global_phis'][sorted_indices]
        params['global_weights'] = params['global_weights'][sorted_indices]

        
        log_likelihoods.append(total_log_like)
        log_and_print(f"\nTotal log-likelihood: {total_log_like:.2f}", log_file=outfile)
        if total_log_like > best_log_like:
            best_log_like = total_log_like
            best_params = copy.deepcopy(params)
            best_agg = copy.deepcopy(final_agg)
            best_phis_seen = params['global_phis'].copy()
            if(save_best_sample):
                best_syn_data = copy.deepcopy(syn_data)
            log_and_print(f"  New best log-likelihood! - {best_log_like:.2f}", log_file=outfile)
        
        max_phi_change = max(
            abs(params['global_phis'][k] - old_params['global_phis'][k])
            for k in range(K)
        )
        
        log_and_print(f"Max phi change: {max_phi_change:.4f}", log_file=outfile)
        
        if iteration > 0:
            delta_log_lik = log_likelihoods[-1] - log_likelihoods[-2]
            log_and_print(f"Log-likelihood change: {delta_log_lik:.4f}", log_file=outfile)
        
        if max_phi_change < tol:
            log_and_print("\n{'='*60}\nEM CONVERGED!\n{'='*60}", log_file=outfile)
            break

        # M-STEP: Nudge sigmas using the result of the simulation above
        log_and_print(f"Nudging sigmas...", log_file=outfile)
        params = nudge_district_sigmas(
            params,
            final_agg,
            school_info_df,
            obs_total=obs_total_app,
            obs_true=obs_true_app,
            eta_util=eta,
            eta_demand=eta,
            lambda_true=0.7,
            lambda_total=0.3,
            all_schools=all_schools,
            outfile=outfile
        )

       
    
    executor.shutdown()
    log_and_print(f"\nFinal global parameters:", log_file=outfile)
    log_and_print(f"  Global phis: {best_params['global_phis']}", log_file=outfile)
    log_and_print(f"  Global weights: {best_params['global_weights']}", log_file=outfile)
    log_and_print(f"\nEstimated central rankings (sigma) per district:", log_file=outfile)
    for district in sorted(best_params['districts'].keys()):
        sigma = best_params['districts'][district]['central_ranking']
        log_and_print(f"\n  District {district}: {sigma}", log_file=outfile)
    
    cur_experiment_result.set_params(best_params)
    cur_experiment_result.set_other_stats(lottery, log_likelihoods, best_agg)
    if(save_best_sample):
        cur_experiment_result.set_synthetic_output(
            syn_rankings=best_syn_data['all_rankings'],
            syn_rankings_idx=best_syn_data['rankings_as_indices'],
            matches_idx=best_syn_data['matches_idx'],
            syn_districts=best_syn_data['syn_districts'],
            syn_attrs=best_syn_data['student_attrs']
        )

    return cur_experiment_result

def initialize_parameters_global_mixture(districts, df, K=1, rng=None):
    """
    Initialize with global phis, district-specific sigmas
    """
    if rng is None:
        rng = np.random.default_rng()

    # Global mixture parameters (shared across districts)
    global_phis = rng.beta(6, 1, K)
    global_phis = np.clip(global_phis, 0.5, 0.99)
    
    global_weights = np.ones(K) / K  # Uniform initially
    
    params = {
        'global_phis': global_phis,
        'global_weights': global_weights,
        'districts': {}
    }
    
    # District-specific central rankings
    for district in districts:
        df_district = df[df['Residential District'] == district]
        schools_list = df_district['School DBN'].values
        
        obs_total = df_district.set_index('School DBN')['Ratio'].to_dict()
        
        central_ranking = sorted(schools_list, key=lambda s: obs_total[s], reverse=True)
        
        params['districts'][district] = {
            'schools': schools_list,
            'central_ranking': central_ranking
        }
    
    return params

def compute_log_likelihood_gaussian_all_districts(params_global, observed_agg,
                                                   df, match_stats_df, school_info_df,
                                                   M=1, seed=42, outfile=None, 
                                                   executor=None, sampling_n_jobs=32, per_school_lottery=False, 
                                                   priority_config=None, district_to_region=None, 
                                                   list_length_params=None, save_best_sample=False, 
                                                   profile_timing=True, lottery_fixed=None, max_p=None):


    t_total_start = time.perf_counter()
    districts = sorted(observed_agg.keys())
    simulated_samples = {d: [] for d in districts}

    # Initialize based on actual unique schools in df, not school_info_df rows
    all_schools = df['School DBN'].unique()
    capacities_dict = school_info_df.set_index('School DBN')['Capacity'].to_dict()
    total_filled = np.zeros(len(all_schools))
    match_stats_accum = None

    saved_synth_info = None
    all_synth_infos = [] if save_best_sample else None
    all_agg_vecs = [] if save_best_sample else None
    
    total_app_accum = None
    true_app_accum  = None
    for sim in range(M):
        log_and_print(f"      Simulation {sim+1}/{M}...", log_file=outfile)
        t_sim_start = time.perf_counter()
        # Simulate ALL districts together (do this ONCE per M iteration)
        res = run_single_simulation(
            params_global, df, match_stats_df, school_info_df, mallows_seed=seed + sim,
            lottery_fixed=lottery_fixed, outfile=outfile, executor=executor,
            sampling_n_jobs=sampling_n_jobs, per_school_lottery=per_school_lottery,
            priority_config=priority_config, district_to_region=district_to_region, 
            list_length_params=list_length_params, save_best_sample=save_best_sample,
            run_priority_analysis = (sim == M-1), max_p=max_p
        )
        if save_best_sample:
            agg, synth_info = res
            all_synth_infos.append(synth_info)
            all_agg_vecs.append(agg['match_stats'].copy())
        else:
            agg = res

        total_filled += agg['filled']
        if match_stats_accum is None:
            match_stats_accum = agg['match_stats'].copy()
        else:
            match_stats_accum += agg['match_stats']
        
        if total_app_accum is None:
            total_app_accum = agg['total_app'].copy()
            true_app_accum  = agg['true_app'].copy()
        else:
            total_app_accum += agg['total_app']
            true_app_accum  += agg['true_app']

        # Extract stats for EACH district from this single simulation
        for d_idx, district in enumerate(districts):
            agg_vec = agg['match_stats'][d_idx, :]
            simulated_samples[district].append(agg_vec)

        if profile_timing:
            log_and_print(
                f"      [TIMING] simulation {sim+1}/{M}: {time.perf_counter() - t_sim_start:.3f}s",
                log_file=outfile,
            )
    
    if save_best_sample and all_synth_infos:
        mean_stats = match_stats_accum / M
        dists = [np.sum((v - mean_stats) ** 2) for v in all_agg_vecs]
        saved_synth_info = all_synth_infos[int(np.argmin(dists))]

    mean_filled = total_filled / M
    # Get capacities in same order as all_schools
    capacities = np.array([capacities_dict.get(s, 0) for s in all_schools])
    sim_util = np.full_like(mean_filled, np.nan, dtype=float)
    np.divide(mean_filled, capacities, out=sim_util, where=capacities > 0)
    sim_util = sim_util * 100

    # Get observed utilization only for schools we have
    obs_util_dict = school_info_df.set_index('School DBN')['Utilization'].to_dict()
    obs_util = np.array([obs_util_dict.get(s, np.nan) for s in all_schools], dtype=float)
    util_valid_mask = np.isfinite(obs_util) & np.isfinite(sim_util)
    if np.any(util_valid_mask):
        util_penalty =  np.mean((obs_util[util_valid_mask] - sim_util[util_valid_mask])**2)
    else:
        util_penalty = 0.0
        log_and_print("Warning: No valid utilization pairs after NaN filtering.", log_file=outfile)
    
    log_and_print('') 
    
    log_and_print("\n" + "="*60, log_file=outfile)
    log_and_print(f"Fit Diagnostics", log_file=outfile)
    log_and_print("="*60, log_file=outfile)
    
    #metric_names = ["top3", "top5", "top10", "unmatched"]
    n_stats = len(next(iter(observed_agg.values()))['match_stats'])
    if n_stats == 4:
        metric_names = ["top3", "top5", "top10", "unmatched"]
    else:
        metric_names = [f"top{p}" for p in range(1, n_stats)] + ["unmatched"]

    for d_idx, district in enumerate(districts):
        obs = np.array(observed_agg[district]['match_stats'], dtype=float)
        sim = np.array(match_stats_accum[d_idx, :], dtype=float) / M
        valid_mask = np.isfinite(obs) & np.isfinite(sim)

        log_and_print(f"\nDistrict {district}:", log_file=outfile)
        if not np.any(valid_mask):
            log_and_print("  No valid observed/simulated pairs after NaN filtering.", log_file=outfile)
            continue

        obs_parts = [
            f"{metric_names[i]}={obs[i]:5.1f}%" for i in range(len(metric_names)) if valid_mask[i]
        ]
        sim_parts = [
            f"{metric_names[i]}={sim[i]:5.1f}%" for i in range(len(metric_names)) if valid_mask[i]
        ]
        diff_parts = [
            f"{metric_names[i]}={obs[i]-sim[i]:+5.1f}" for i in range(len(metric_names)) if valid_mask[i]
        ]

        log_and_print(f"  Observed:  {', '.join(obs_parts)}", log_file=outfile)
        log_and_print(f"  Simulated: {', '.join(sim_parts)}", log_file=outfile)
        log_and_print(f"  Difference: {', '.join(diff_parts)}", log_file=outfile)
    
    log_and_print("Global School Utilization (Top 5 Mismatches):", log_file=outfile)
    util_diff = obs_util - sim_util
    valid_indices = np.where(np.isfinite(util_diff) & util_valid_mask)[0]
    if len(valid_indices) > 0:
        sorted_valid = valid_indices[np.argsort(np.abs(util_diff[valid_indices]))[::-1]]
        mismatch_indices = sorted_valid[:5]
        for idx in mismatch_indices:
            s_name = all_schools[idx]
            log_and_print(f"  {s_name}: Obs={obs_util[idx]:5.1f}%, Sim={sim_util[idx]:5.1f}%, Diff={util_diff[idx]:+5.1f}%", log_file=outfile)
        log_and_print(
            f"  Mean Absolute Utilization Error: {np.mean(np.abs(util_diff[valid_indices])):.2f}%",
            log_file=outfile,
        )
    

    district_map = school_info_df.set_index('School DBN')['District'].to_dict()
    district_obs = defaultdict(list)
    district_sim = defaultdict(list)

    for idx, s_name in enumerate(all_schools):
        if not np.isfinite(obs_util[idx]) or not np.isfinite(sim_util[idx]):
            continue
        district = district_map.get(s_name)
        if district is None:
            continue
        district_obs[district].append(obs_util[idx])
        district_sim[district].append(sim_util[idx])

    log_and_print("District Utilization:", log_file=outfile)
    for d in sorted(district_obs.keys()):
        obs_mean = np.mean(district_obs[d])
        sim_mean = np.mean(district_sim[d])
        log_and_print(
            f"  DIST_UTIL {d}: Obs={obs_mean:5.1f}%, Sim={sim_mean:5.1f}%",
            log_file=outfile
        )

    log_and_print("="*60 + "\n", log_file=outfile)
    # Now compute likelihood for each district separately
    total_log_lik = 0
    
    for district in districts:
        X = np.array(simulated_samples[district])  # M × 4 array
        
        # Check for valid data
        if len(X) == 0 or np.any(np.isnan(X)) or np.any(np.isinf(X)):
            log_and_print(f"      Warning: Invalid data for district {district}", log_file=outfile)
            continue
        
        # Estimate mean and covariance
        mu = np.mean(X, axis=0)
        
        if M > 1:
            Sigma = np.cov(X, rowvar=False)
            
            # Handle different dimensionalities
            if Sigma.ndim == 0:  # Scalar
                Sigma = np.array([[Sigma]])
            elif Sigma.ndim == 1:  # 1D
                Sigma = np.diag(Sigma)
            
            # Add regularization for numerical stability
            regularization = 1e-3 * np.eye(len(Sigma))
            Sigma = Sigma + regularization
            
            # Check for singularity
            try:
                np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError:
                Sigma = Sigma + 1e-2 * np.eye(len(Sigma))
        else:
            # Not enough samples for covariance
            Sigma = 1e-2 * np.eye(4)
        
        # Get observed vector
        obs_vec = observed_agg[district]['match_stats']
        
        # Compute Mahalanobis distance
        try:
            diff = obs_vec - mu
            inv_Sigma = np.linalg.inv(Sigma)
            mahalanobis_sq = diff @ inv_Sigma @ diff

            sign, log_det = np.linalg.slogdet(Sigma)
            if sign <= 0:
                log_and_print(f"Non-positive definite Sigma for district {district}", log_file=outfile)
                log_lik = -1e10
            else:
                log_lik = -0.5 * mahalanobis_sq - 0.5 * log_det
            
            # Sanity check
            if np.isnan(log_lik) or np.isinf(log_lik):
                log_and_print(f"Warning: Invalid log-likelihood for district {district}", log_file=outfile)
                log_lik = -1e10
                
        except Exception as e:
            log_and_print(f" Warning: Likelihood computation failed for district {district}: {e}", log_file=outfile)
            
            # Fall back to simple MSE
            mse = np.mean((obs_vec - mu)**2)
            log_lik = -mse * 100
        
        total_log_lik += log_lik

    if profile_timing:
        log_and_print(
            f"  [TIMING] compute_log_likelihood_gaussian_all_districts total: {time.perf_counter() - t_total_start:.3f}s",
            log_file=outfile,
        )
    
    log_and_print(f"  Match stats log-likelihood: {total_log_lik:.2f}, Util penalty: {util_penalty:.2f}", log_file=outfile)
    #mean_agg = {'filled': total_filled / M, 'match_stats': match_stats_accum / M}
    mean_agg = {
        'filled':      total_filled / M,
        'match_stats': match_stats_accum / M,
        'total_app':   total_app_accum / M,
        'true_app':    true_app_accum  / M,
    }
    return total_log_lik , mean_agg, saved_synth_info

def optimize_global_mixture(params, observed_agg, df, match_stats_df, 
                            school_info_df, M=20, seed=42, iteration=1,
                            sampling_n_jobs=32, outfile=None, executor=None, 
                            max_iter_em=5, max_iter_opt=5, per_school_lottery=False, 
                            profile_timing=True, priority_config=None, 
                            district_to_region=None, list_length_params=None, 
                            save_best_sample=False, best_phis_seen=None, max_p=None):

    t_opt_start = time.perf_counter()
    K = len(params['global_phis'])
    last_agg = None
    last_syn_data = None
    best_log_like_returned = -np.inf

    n_students_total = int(match_stats_df['Total Applicants'].sum())
    rng_lottery = np.random.default_rng(seed=seed)
    lottery_fixed = None if per_school_lottery else rng_lottery.permutation(n_students_total)

    for k in range(K):
        best_log_like_seen = -np.inf
        eval_count = 0
        phi_k_initial = params['global_phis'][k]
        log_and_print(f"\n  [EM iter {iteration+1}/{max_iter_em}] Optimizing phi[{k+1}/{K}], starting at {phi_k_initial:.4f}", log_file=outfile)
        
        def objective_global_phi_k(phi):
            
            nonlocal  eval_count, last_agg, last_syn_data, best_log_like_seen, best_log_like_returned
            eval_count += 1
            t_eval_start = time.perf_counter()
            original_phi = params['global_phis'][k]
            params['global_phis'][k] = phi
            log_and_print(f"    [EM iter {iteration+1}/{max_iter_em}] | phi[{k+1}/{K}] eval #{eval_count}/{max_iter_opt}, trying phi={phi:.4f}", log_file=outfile)
            
           
            total_log_lik, mean_agg, synth_info  = compute_log_likelihood_gaussian_all_districts(
                params, observed_agg, df, match_stats_df, 
                school_info_df, M=M, seed=seed,  outfile=outfile, 
                executor=executor, sampling_n_jobs=sampling_n_jobs, per_school_lottery=per_school_lottery, 
                profile_timing=profile_timing, priority_config=priority_config, district_to_region=district_to_region, 
                list_length_params=list_length_params, save_best_sample=save_best_sample, lottery_fixed=lottery_fixed, max_p=max_p
            )
            if total_log_lik > best_log_like_seen:
                best_log_like_seen = total_log_lik
                last_agg = mean_agg
                last_syn_data = synth_info
            if total_log_lik > best_log_like_returned:
                best_log_like_returned = total_log_lik

            
            params['global_phis'][k] = original_phi
            if profile_timing:
                log_and_print(
                    (
                        f"    [TIMING] phi[{k+1}/{K}] eval #{eval_count} "
                        f"duration: {time.perf_counter() - t_eval_start:.3f}s"
                    ),
                    log_file=outfile,
                )

            return -total_log_lik
        
        
        #lo = max(0.01, phi_k_initial - 0.3)
        #hi = min(0.99, phi_k_initial + 0.15)
        phi_k_center = best_phis_seen[k] if best_phis_seen is not None else phi_k_initial
        drift = abs(phi_k_initial - phi_k_center)
        window = max(0.3, drift + 0.1)
        lo = max(0.01, phi_k_center - window)
        hi = min(0.99, phi_k_center + window)

        result = minimize_scalar(
            objective_global_phi_k,
            bounds=(lo, hi),
            method='bounded',
            options={'xatol': 0.01, 'maxiter': max_iter_opt}
        )

        phi_final = np.clip(result.x, 0.01, 0.99)
        params['global_phis'][k] = phi_final
        log_and_print(f"  [EM iter {iteration+1}/{max_iter_em}] phi[{k+1}/{K}] -> {phi_final:.4f} (took {eval_count} evals)", log_file=outfile)

    if profile_timing:
        log_and_print(
            f"  [TIMING] optimize_global_mixture total: {time.perf_counter() - t_opt_start:.3f}s",
            log_file=outfile,
        )
    
    return params, last_agg, best_log_like_returned, lottery_fixed, last_syn_data

def _build_observed_app_matrices(df, districts, all_schools):
    district_to_idx = {str(d): i for i, d in enumerate(districts)}
    school_to_idx   = {str(s): i for i, s in enumerate(all_schools)}
    n_d, n_s = len(districts), len(all_schools)
    obs_total = np.zeros((n_d, n_s))
    obs_true  = np.zeros((n_d, n_s))
    for _, row in df.iterrows():
        d_idx = district_to_idx.get(str(row['Residential District']))
        s_idx = school_to_idx.get(str(row['School DBN']))
        if d_idx is not None and s_idx is not None:
            obs_total[d_idx, s_idx] = row['Total Applicants by Residential District']
            obs_true[d_idx, s_idx]  = row['True Applicants by Residential District']
    return obs_total, obs_true

def nudge_district_sigmas(
    params,
    final_agg,
    school_info_df,
    obs_total=None,
    obs_true=None,
    eta_util=LEARNING_RATE,
    eta_demand=LEARNING_RATE,
    lambda_true=0.7,
    lambda_total=0.3,
    all_schools=None,
    outfile=None
):
    if all_schools is None:
        all_schools = school_info_df['School DBN'].values

    school_to_idx = {str(s): i for i, s in enumerate(all_schools)}
    districts     = sorted(params['districts'].keys())
    district_to_idx = {str(d): i for i, d in enumerate(districts)}

    # utilization error (school-level, same as before)
    sim_filled = pd.Series(final_agg['filled'], index=all_schools)
    real_util_counts = (
        school_info_df.set_index('School DBN')['Utilization'] / 100
    ) * school_info_df.set_index('School DBN')['Capacity']
    util_error = real_util_counts - sim_filled

    # demand error (district-school level)
    use_demand = (
        obs_total is not None and obs_true is not None
        and 'total_app' in final_agg and 'true_app' in final_agg
    )
    if use_demand:
        sim_total = final_agg['total_app']
        sim_true  = final_agg['true_app']
        sim_total_share = sim_total / sim_total.sum(axis=1, keepdims=True).clip(min=1e-9)
        sim_true_share  = sim_true  / sim_true.sum(axis=1, keepdims=True).clip(min=1e-9)
        obs_total_share = obs_total / obs_total.sum(axis=1, keepdims=True).clip(min=1e-9)
        obs_true_share  = obs_true  / obs_true.sum(axis=1, keepdims=True).clip(min=1e-9)
        demand_error = (
            lambda_true  * (obs_true_share  - sim_true_share)
          + lambda_total * (obs_total_share - sim_total_share)
        )

    for d_id, d_data in params['districts'].items():
        if 'pop_scores' not in d_data:
            d_data['pop_scores'] = {
                s: (len(d_data['schools']) - i)
                for i, s in enumerate(d_data['central_ranking'])
            }

        d_idx = district_to_idx.get(str(d_id))

        for s_dbn in list(d_data['pop_scores'].keys()):
            s_idx = school_to_idx.get(str(s_dbn))
            util_err = util_error.get(s_dbn, 0.0)
            util_contrib = eta_util * util_err if np.isfinite(util_err) else 0.0

            dem_contrib = 0.0
            if use_demand and d_idx is not None and s_idx is not None:
                dem_err = demand_error[d_idx, s_idx]
                dem_contrib = eta_demand * dem_err if np.isfinite(dem_err) else 0.0

            d_data['pop_scores'][s_dbn] += util_contrib + dem_contrib

        old_top3 = d_data['central_ranking'][:3] if 'central_ranking' in d_data else []
        new_sigma = sorted(d_data['pop_scores'].items(), key=lambda x: x[1], reverse=True)
        d_data['central_ranking'] = [s[0] for s in new_sigma]
        new_top3 = d_data['central_ranking'][:3]
        if old_top3 != new_top3:
            log_and_print(f"    District {d_id} sigma changed: {old_top3} -> {new_top3}", log_file=outfile)

    return params
