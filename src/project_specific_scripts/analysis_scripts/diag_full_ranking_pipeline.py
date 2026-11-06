import sys
sys.path.insert(0, 'src')
import pickle
import json
import numpy as np
import pandas as pd

from data_ingestion import read_data, preprocess_chilean_data
from em import run_single_simulation
from list_length import return_chilean_list_params
from file_config import CHILEAN_DATA_DIR

DISTRICT = 'Antofagasta'
PARAMS_PKL = 'experimental_output/chile_res_logs/20261012_123851/chilean_experiment_K=6_M=15_iter=13_opt=15_lr=0.05_maxp=10_loss=allTopP+unm_seed=40_20261012_123851_params.pkl'
INDV = f'{CHILEAN_DATA_DIR}/individual_level_preferences_and_result_province.xlsx'
MATCH = f'{CHILEAN_DATA_DIR}/matching_outcome_by_province.xlsx'
CAP = f'{CHILEAN_DATA_DIR}/school_capacity_with_region_and_province.xlsx'
PRIORITY_CONFIG = f'{CHILEAN_DATA_DIR}/chile_priority_config.json'

with open(PARAMS_PKL, 'rb') as f:
    params = pickle.load(f)
print(f"K={len(params['global_phis'])}, global_phis={params['global_phis']}")

indv_df = read_data(INDV)
match_df = read_data(MATCH)
school_cap_df = read_data(CAP)

df, match_stats_df, school_info_df = preprocess_chilean_data(
    indv_df, match_df, school_cap_df, is_province_level=True
)
list_length_params = return_chilean_list_params(indv_df=indv_df)
print("list_length_params:", list_length_params)

with open(PRIORITY_CONFIG) as f:
    priority_config = json.load(f)

district_to_region = {str(r): str(r) for r in df['Residential District'].unique()}

# Restrict to one district to keep this cheap enough for the login node.
params_sub = {
    'global_phis': params['global_phis'],
    'global_weights': params['global_weights'],
    'districts': {DISTRICT: params['districts'][DISTRICT]},
}
print(f"\nDistrict {DISTRICT}: {len(params_sub['districts'][DISTRICT]['schools'])} schools in its school list")
n_students_district = int(match_stats_df[match_stats_df['Residential District'] == DISTRICT]['Total Applicants'].iloc[0])
print(f"n students in {DISTRICT} (from match_stats_df): {n_students_district}")

rng_lottery = np.random.default_rng(seed=40)
lottery_fixed = rng_lottery.permutation(n_students_district)

print("\n=== Running WITHOUT priority_config (simple lottery branch) ===")
agg_simple = run_single_simulation(
    params=params_sub, df=df, match_stats_df=match_stats_df, school_info_df=school_info_df,
    mallows_seed=40, sampling_n_jobs=1, per_school_lottery=False, lottery_fixed=lottery_fixed,
    priority_config=None, district_to_region=district_to_region,
    list_length_params=list_length_params, raw_capacity_df=school_cap_df,
    max_p=10,
)
print("match_stats (simple):", agg_simple['match_stats'])

print("\n=== Running WITH priority_config (Chile priority branch) ===")
agg_priority = run_single_simulation(
    params=params_sub, df=df, match_stats_df=match_stats_df, school_info_df=school_info_df,
    mallows_seed=40, sampling_n_jobs=1, per_school_lottery=False, lottery_fixed=lottery_fixed,
    priority_config=priority_config, district_to_region=district_to_region,
    list_length_params=list_length_params, raw_capacity_df=school_cap_df,
    max_p=10,
)
print("match_stats (priority):", agg_priority['match_stats'])

obs = match_stats_df[match_stats_df['Residential District'] == DISTRICT]
print("\nobserved row:", obs.to_dict('records'))
