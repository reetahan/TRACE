import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / '.env')

EXP_OUT_FOLDER = os.getenv('EXP_OUT_FOLDER', '/scratch/rm6609/MatchingInferenceEngine/experimental_output/')
RAW_DATA_DIR = os.getenv(
    'RAW_DATA_DIR',
    '/scratch/rm6609/MatchingInferenceEngine/sample-data/raw-data'
)
POLISHED_DATA_DIR = os.getenv(
    'POLISHED_DATA_DIR',
    '/scratch/rm6609/MatchingInferenceEngine/sample-data/data'
)
CHILEAN_DATA_DIR = os.getenv(
    'CHILEAN_DATA_DIR',
    '/scratch/rm6609/MatchingInferenceEngine/sample-data/data/chilean_data_processed'
)

DATA_GENERATION_SEED = int(os.getenv('DATA_GENERATION_SEED', '44'))

MAIN_AGG_APP_STATS_FILEPATH = os.getenv("MAIN_AGG_APP_STATS_FILEPATH", "master_data_03_residential_district.xlsx")

MAIN_AGG_MATCH_STATS_FILEPATH = os.getenv("MAIN_AGG_MATCH_STATS_FILEPATH", "DATA3_fall-2024-high-school-offer-results-website-1.xlsx")
MAIN_AGG_MATCH_STATS_FILEPATH_SHEET = os.getenv("MAIN_AGG_MATCH_STATS_FILEPATH_SHEET", "Match to Choice-District")

SCHOOL_INFO_STATS_FILEPATH = os.getenv("SCHOOL_INFO_STATS_FILEPATH", "DATA4_fall-2025---hs-directory-data.xlsx")
SCHOOL_INFO_STATS_FILEPATH_SHEET = os.getenv("SCHOOL_INFO_STATS_FILEPATH_SHEET", "Data")

ADDTL_SCHOOL_INFO_STATS_FILEPATH = os.getenv("ADDTL_SCHOOL_INFO_STATS_FILEPATH", "DATA2_fall-2024-admissions_part-ii_suppressed.xlsx")
ADDTL_SCHOOL_INFO_STATS_FILEPATH_SHEET = os.getenv("ADDTL_SCHOOL_INFO_STATS_FILEPATH_SHEET", "School")

NYC_CONFIG_FILEPATH = os.getenv("NYC_CONFIG_FILEPATH", "nyc_priority_config.json")

CHILEAN_INDV_PREF_FILEPATH = os.getenv("CHILEAN_INDV_PREF_FILEPATH", "individual_level_preferences_and_result.xlsx")
CHILEAN_MATCH_OUTCOME_FILEPATH = os.getenv("CHILEAN_MATCH_OUTCOME_FILEPATH", "matching_outcome_by_region.xlsx")
CHILEAN_SCHOOL_CAPACITY_FILEPATH = os.getenv("CHILEAN_SCHOOL_CAPACITY_FILEPATH", "school_capacity.xlsx")
CHILEAN_SCHOOL_CAPACITY_BY_REGION_FILEPATH = os.getenv("CHILEAN_SCHOOL_CAPACITY_BY_REGION_FILEPATH", "school_capacity_by_region.xlsx")

CHILEAN_INDV_PREF_PROVINCE_FILEPATH = os.getenv("CHILEAN_INDV_PREF_PROVINCE_FILEPATH", "individual_level_preferences_and_result_province.xlsx")
CHILEAN_MATCH_OUTCOME_PROVINCE_FILEPATH = os.getenv("CHILEAN_MATCH_OUTCOME_PROVINCE_FILEPATH", "matching_outcome_by_province.xlsx")
CHILEAN_SCHOOL_CAPACITY_BY_REGION_PROVINCE_FILEPATH = os.getenv("CHILEAN_SCHOOL_CAPACITY_BY_REGION_PROVINCE_FILEPATH","school_capacity_with_region_and_province.xlsx")

CHILE_CONFIG_FILEPATH = os.getenv("CHILE_CONFIG_FILEPATH", "chile_priority_config.json")

CURRENT_EXPERIMENT = os.getenv("CURRENT_EXPERIMENT", "NYC")
CONFIG_FILEPATH = os.getenv("CONFIG_FILEPATH", "nyc_priority_config.json")
