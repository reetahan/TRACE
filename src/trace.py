from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from types_trace import DataKey, EvaluateConfig, MatchOutcomes, Metric, REQUIRED_COLUMNS, SweepResults
from welfare import evaluate_simulation_output, WelfareResults
from constants import *
from custom_user_functions import *

from em import EM_algorithm, run_single_simulation
from mallows import sample_students_global_mixture
from gale_shapley import gale_shapley_per_school_numba_wrapper
from list_length import sample_truncated_normal_lengths, sample_empirical_lengths
from priority_attributes import sample_student_attributes, build_composite_rank_matrix
from data_ingestion import nyc_preprocess_data, preprocess_chilean_data, to_generic


class TRACE:
    """
    School-assignment welfare simulation API.
    """

    def __init__(
        self,
        individual_data_fpath: str | None        = None,
        school_data_fpath: str | None            = None,
        matching_data_fpath: str | None          = None,
        mallows_params_fpath: str | None         = None,
        final_aggregates_fpath: str | None       = None,
        priority_config_fpath: str | None        = None,
        custom_preprocessing_function: Callable | None = None,
    ):

        self._individual_fpath       = individual_data_fpath
        self._school_fpath           = school_data_fpath
        self._match_stats_fpath      = matching_data_fpath
        self._mallows_params_fpath   = mallows_params_fpath
        self._final_aggregates_fpath = final_aggregates_fpath
        self._priority_config_fpath  = priority_config_fpath

        self._individual_df:   pd.DataFrame | None = None
        self._school_df:       pd.DataFrame | None = None
        self._match_stats_df:  pd.DataFrame | None = None
        self._final_agg_df:    pd.DataFrame | None = None
        self._priority_config: dict | None          = None
        self._mallows_params:  dict | None          = None

        self._fitted_params:           dict | None            = None
        self._preference_lists:        list[list[str]] | None = None
        self._subdivision_assignments: list | None            = None
        self._list_length_params:      dict | None            = None
        self._subdivision_to_region:   dict | None            = None

        self._last_match_outcomes: MatchOutcomes | None = None

        self._load_raw()

        if custom_preprocessing_function is not None:
            self.preprocess(fn=custom_preprocessing_function)

    @property
    def mode(self) -> int:
        if self._final_agg_df is not None and self._match_stats_df is not None:
            return 1
        if self._mallows_params is not None:
            return 2
        if self._individual_df is not None:
            return 3
        raise ValueError(
            "Cannot determine mode. Provide:\n"
            "  Mode 1: final_aggregates_fpath + match_stats_fpath (+ school_data_fpath)\n"
            "  Mode 2: mallows_params_fpath\n"
            "  Mode 3: individual_data_fpath (+ school_data_fpath)"
        )

    @property
    def _system_name(self) -> str:
        if self._priority_config is None:
            return ''
        return self._priority_config.get('__meta__', {}).get('system_name', '')


    def _load_raw(self):
        def _load(fpath, key):
            df = self._read_file(fpath)
            m = COLUMN_MAPS.get(key, {})
            return df.rename(columns=m) if m else df

        if self._individual_fpath:
            self._individual_df  = _load(self._individual_fpath,      DataKey.INDIVIDUAL)
        if self._school_fpath:
            self._school_df      = _load(self._school_fpath,          DataKey.SCHOOL)
        if self._match_stats_fpath:
            self._match_stats_df = _load(self._match_stats_fpath,     DataKey.MATCH_STATS)
        if self._final_aggregates_fpath:
            self._final_agg_df   = _load(self._final_aggregates_fpath, DataKey.FINAL_AGGREGATES)
        if self._mallows_params_fpath:
            with open(self._mallows_params_fpath, 'rb') as f:
                self._mallows_params = pickle.load(f)
        if self._priority_config_fpath:
            with open(self._priority_config_fpath) as f:
                self._priority_config = json.load(f)

    @staticmethod
    def _read_file(fpath: str) -> pd.DataFrame:
        path = Path(fpath)
        if path.suffix == '.csv':
            return pd.read_csv(fpath)
        if path.suffix in ('.xlsx', '.xls'):
            return pd.read_excel(fpath)
        raise ValueError(f"Unsupported file format: {path.suffix}")


    def get_df(self, key: DataKey) -> pd.DataFrame | None:
        return {
            DataKey.INDIVIDUAL:       self._individual_df,
            DataKey.SCHOOL:           self._school_df,
            DataKey.MATCH_STATS:      self._match_stats_df,
            DataKey.FINAL_AGGREGATES: self._final_agg_df,
        }[key]

    def set_df(self, key: DataKey, df: pd.DataFrame,
               column_map: dict[str, str] | None = None):
        m = column_map or COLUMN_MAPS.get(key, {})
        if m:
            df = df.rename(columns=m)
        self._validate_df(key, df)
        if key == DataKey.INDIVIDUAL:
            self._individual_df = df
        elif key == DataKey.SCHOOL:
            self._school_df = df
        elif key == DataKey.MATCH_STATS:
            self._match_stats_df = df
        elif key == DataKey.FINAL_AGGREGATES:
            self._final_agg_df = df
            

    def get_priority_config(self) -> dict | None:
        return self._priority_config

    def set_priority_config(self, config: dict):
        self._priority_config = config

    def get_mallows_params(self) -> dict | None:
        return self._fitted_params or self._mallows_params

    def set_mallows_params(self, params: dict):
        self._mallows_params = params

    def set_list_length_params(self, params: dict):
        self._list_length_params = params

    @staticmethod
    def _validate_df(key: DataKey, df: pd.DataFrame):
        missing = [c for c in REQUIRED_COLUMNS.get(key, []) if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame for {key.value} is missing columns: {missing}")

    def _to_em_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Rename generic columns to em.py's internal schema before any pipeline call.
        """
        if self._final_agg_df is None or self._match_stats_df is None or self._school_df is None:
            raise ValueError("_to_em_dataframes requires final_agg_df, match_stats_df, and school_df.")

        def _rename(df: pd.DataFrame) -> pd.DataFrame:
            renamed = df.rename(columns=EM_COLUMN_MAP)
            pct_renames = {
                col: f"% Matches to Choice 1-{col[len('pct_top_'):]}"
                for col in df.columns if col.startswith('pct_top_')
            }
            return renamed.rename(columns=pct_renames)

        return (
            _rename(self._final_agg_df),
            _rename(self._match_stats_df),
            _rename(self._school_df),
        )

    @staticmethod
    def _to_em_list_length_params(lp: dict | None) -> dict | None:
        """
        Translate TRACE generic list_length_params keys to em.py's expected keys.
        """
        if lp is None:
            return None
        result = dict(lp)
        if result.get('list_length_mode') == 'gaussian_per_subdivision':
            result['list_length_mode'] = 'gaussian_per_district'
        if 'mean_per_subdivision' in result:
            result['list_length_mean_per_district'] = result.pop('mean_per_subdivision')
        return result


    def preprocess(self, fn=None) -> TRACE:
        """
        fn: which preprocessing pipeline to run.
              None       -> NYC (default, logged)
             'nyc'       -> NYC pipeline
             'chile'     -> Chilean pipeline
              callable   -> user-provided function from custom_user_functions.py.
                           Signature: (final_agg_df, match_stats_df, school_df, addtl_df=None)
                                    -> (final_agg_df, match_stats_df, school_df) in TRACE generic format

        Skip entirely if DataFrames are already preprocessed and loaded via set_df().
        """
        self._run_internal_preprocessing(fn)
        return self

    def _run_internal_preprocessing(self, fn=None):

        if fn is None:
            print("[TRACE] No preprocessing function specified — defaulting to NYC pipeline.")
            fn = 'nyc'

        if fn == 'nyc':
            df_em, match_em, school_em = nyc_preprocess_data(
                self._final_agg_df, self._match_stats_df, self._school_df, None
            )
            result = to_generic(df_em), to_generic(match_em), to_generic(school_em)

        elif fn == 'chile':
            df_em, match_em, school_em = preprocess_chilean_data(
                self._final_agg_df, self._match_stats_df, self._school_df
            )
            result = to_generic(df_em), to_generic(match_em), to_generic(school_em)

        elif callable(fn):
            result = fn(self._final_agg_df, self._match_stats_df, self._school_df)
            if not (isinstance(result, tuple) and len(result) == 3):
                raise ValueError(
                    "Preprocessing function must return (final_agg_df, match_stats_df, school_df)."
                )
        else:
            raise ValueError(
                f"Unknown preprocessing option {fn!r}. Use 'nyc', 'chile', or a callable."
            )

        self._final_agg_df, self._match_stats_df, self._school_df = result

    def fit(
        self,
        K: int                              = 6,
        M: int                              = 15,
        lr: float                           = LEARNING_RATE,
        max_iter: int                       = 20,
        max_opt: int                        = 5,
        max_p: int | None                   = None,
        n_jobs: int                         = 32,
        seed: int                           = 40,
        verbose: bool                       = False,
        subdivision_to_region: dict | None  = None,
        list_length_params: dict | None     = None,
        outfile: str | None                 = None,
        custom_matching_fn: Callable | None = None,
    ) -> TRACE:
        """
        Run EM to infer Mallows mixture parameters from aggregate match statistics (Mode 1).
        """
        if self.mode != 1:
            raise ValueError("fit() requires Mode 1 (final_aggregates_fpath + match_stats_fpath).")
        if self._final_agg_df is None or self._match_stats_df is None or self._school_df is None:
            raise ValueError("fit() requires final_agg_df, match_stats_df, and school_df.")

        if list_length_params is not None:
            self._list_length_params = list_length_params
        if subdivision_to_region is not None:
            self._subdivision_to_region = subdivision_to_region

        individual_em, match_stats_em, school_em = self._to_em_dataframes()
        self._custom_matching_fn = custom_matching_fn

        experiment_results = EM_algorithm(
            df=individual_em,
            match_stats_df=match_stats_em,
            school_info_df=school_em,
            max_iter=max_iter,
            K=K,
            M_simulations=M,
            seed=seed,
            eta=lr,
            outfile=outfile,
            sampling_n_jobs=n_jobs,
            max_iter_opt=max_opt,
            priority_config=self._priority_config,
            district_to_region=self._subdivision_to_region,
            list_length_params=self._to_em_list_length_params(self._list_length_params),
            save_best_sample=True,
            max_p=max_p,
            profile_timing=verbose,
            custom_matching_fn=custom_matching_fn
        )

        self._fitted_params = experiment_results.params

        if experiment_results.syn_rankings is not None:
            self._preference_lists        = experiment_results.syn_rankings
            self._subdivision_assignments = list(experiment_results.syn_districts)

            attr_df = (
                pd.DataFrame(experiment_results.syn_attrs)
                if experiment_results.syn_attrs is not None
                else pd.DataFrame()
            )
            attr_df['subdivision'] = list(experiment_results.syn_districts)

            self._last_match_outcomes = MatchOutcomes(
                rankings_as_indices=experiment_results.syn_rankings_idx,
                matches_idx=experiment_results.matches_idx,
                student_attributes=attr_df,
            )

        return self


    def sample(
        self,
        mallows_params: dict | None     = None,
        n_jobs: int                     = 1,
        seed: int | None                = None,
        list_length_params: dict | None = None,
    ) -> list[list[str]]:
        """
        Sample synthetic preference lists from the Mallows mixture.

        list_length_params formats:
          gaussian:                 {list_length_mode, list_length_mean, list_length_std, list_length_min}
          gaussian_per_subdivision: {list_length_mode, mean_per_subdivision: {id: float}, list_length_std, list_length_min}
          empirical:                {list_length_mode, list_length_empirical_probs: {length: prob}}
          fixed:                    {list_length_mode, k_ranking_length}

        """
        params = mallows_params or self._fitted_params or self._mallows_params
        if params is None:
            raise ValueError("No Mallows parameters. Run fit() or set via set_mallows_params().")
        if self._match_stats_df is None:
            raise ValueError("sample() requires match_stats_df for per-subdivision student counts.")

        lp = list_length_params or self._list_length_params
        if lp is None:
            raise ValueError("No list_length_params. Call set_list_length_params() or pass directly.")


        rng = np.random.default_rng(seed)
        all_rankings: list[list[str]]  = []
        all_subdivisions: list         = []

        for subdivision, d_data in params['districts'].items():
            rows = self._match_stats_df[self._match_stats_df['subdivision'] == subdivision]
            if rows.empty:
                continue
            n_students   = int(rows['n_students'].iloc[0])
            schools_list = d_data['schools']

            raw_rankings = sample_students_global_mixture(
                params=params,
                district=subdivision,
                n_students=n_students,
                n_jobs=n_jobs,
                random_seed=int(rng.integers(2**32)),
                k_ranking_length=self._compute_mallows_k(lp, params)
            )

            effective_lp = lp
            if lp.get('list_length_mode') == 'gaussian_per_subdivision':
                mean_s = lp.get('mean_per_subdivision', {}).get(str(subdivision), 7)
                effective_lp = {**lp, 'list_length_mean': mean_s}

            lengths = self._sample_list_lengths(effective_lp, n_students, schools_list, rng)

            for raw, L in zip(raw_rankings, lengths):
                all_rankings.append([schools_list[i] for i in raw[:L]])
            all_subdivisions.extend([subdivision] * n_students)

        self._preference_lists        = all_rankings
        self._subdivision_assignments = all_subdivisions
        return all_rankings

    @staticmethod
    def _compute_mallows_k(lp: dict, params: dict) -> int:
        """Max ranking length implied by list_length_params; mirrors run_single_simulation's mallows_k."""
        mode     = lp.get('list_length_mode', 'gaussian')
        max_sch  = max(len(d['schools']) for d in params['districts'].values())
        if mode == 'fixed':
            return int(lp.get('k_ranking_length', 5))
        if mode in ('gaussian', 'gaussian_per_subdivision'):
            if mode == 'gaussian_per_subdivision':
                mean = max(lp.get('mean_per_subdivision', {'_': 7}).values())
            else:
                mean = lp.get('list_length_mean', 7)
            std     = lp.get('list_length_std', 2)
            max_len = lp.get('list_length_max', None)
            raw     = int(mean + 10 * std) + 1
            return min(raw, max_len) if max_len is not None else min(raw, max_sch)
        if mode == 'empirical':
            return max(lp.get('list_length_empirical_probs', {1: 1}).keys())
        raise ValueError(f"Unknown list_length_mode: {mode}")

    @staticmethod
    def _sample_list_lengths(
        lp: dict,
        n_students: int,
        schools_list: list,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample per-student list lengths. gaussian_per_subdivision callers must pre-set list_length_mean."""
        mode    = lp.get('list_length_mode', 'gaussian')
        max_len = len(schools_list)
        if mode in ('gaussian', 'gaussian_per_subdivision'):
            return sample_truncated_normal_lengths(
                n_students=n_students,
                mean=lp.get('list_length_mean', 7),
                std=lp.get('list_length_std', 2),
                min_len=lp.get('list_length_min', 1),
                max_len=min(max_len, lp['list_length_max']) if 'list_length_max' in lp else max_len,
                rng=rng,
            )
        if mode == 'empirical':
            return sample_empirical_lengths(n_students, lp['list_length_empirical_probs'], rng)
        if mode == 'fixed':
            return np.full(n_students, min(int(lp.get('k_ranking_length', 5)), max_len), dtype=int)
        raise ValueError(f"Unknown list_length_mode: {mode}")

    def run_matching(
        self,
        preference_lists: list[list[str]] | None  = None,
        subdivision_assignments: list | None       = None,
        custom_matching_fn: Callable | None        = None,
        seed: int                                  = 42,
        subdivision_to_region: dict | None         = None,
        priority_attribute_cols: list[str] | None  = None,
        student_attribute_cols: list[str] | None   = None,
    ) -> MatchOutcomes:
        """
        Run matching on the given (or stored) preference lists.
        """
        lists        = preference_lists
        subdivisions = subdivision_assignments
        observed_attrs: pd.DataFrame | None = None

        if lists is None:
            if self._preference_lists is not None:
                lists        = self._preference_lists
                subdivisions = subdivisions or self._subdivision_assignments
            elif self._individual_df is not None:
                lists, subdivisions, observed_attrs = self._extract_individual_data(
                    priority_attribute_cols=priority_attribute_cols,
                    student_attribute_cols=student_attribute_cols,
                )
            else:
                raise ValueError("No preference lists. Run sample() or fit() first, or provide individual_data_fpath.")

        if subdivisions is None:
            subdivisions = self._subdivision_assignments or [None] * len(lists)

        if custom_matching_fn is not None:
            raw = custom_matching_fn(lists, self._individual_df, self._school_df)
            self._last_match_outcomes = self._validate_match_outcomes(raw)
        else:
            if subdivision_to_region is not None:
                self._subdivision_to_region = subdivision_to_region
            self._last_match_outcomes = self._run_internal_matching(
                lists, subdivisions, seed=seed,
                observed_student_attrs=observed_attrs,
            )

        return self._last_match_outcomes

    def _run_internal_matching(
        self,
        preference_lists: list[list[str]],
        subdivision_assignments: list,
        seed: int = 42,
        observed_student_attrs: pd.DataFrame | None = None,
    ) -> MatchOutcomes:
        """
        Dispatches on _system_name:
          '' (empty) — plain STB Gale-Shapley
          any other  — priority-aware matching via priority_attributes.py

        For the priority path:
          Modes 1/2 (no observed_student_attrs): calls sample_student_attributes to
              synthesise per-student priority flags, then build_composite_rank_matrix.
          Mode 3 (observed_student_attrs provided): converts the observed DataFrame to
              the list-of-dicts format expected by build_composite_rank_matrix. Binary
              priority flags from the raw data are treated as school-independent (applied
              at all schools), since we do not know which specific school each flag
              corresponds to.
        """
        if self._school_df is None:
            raise ValueError("Matching requires school_df.")

        rng = np.random.default_rng(seed)

        all_schools     = self._school_df['school_id'].values
        school_to_idx   = {s: i for i, s in enumerate(all_schools)}
        capacities_dict = self._school_df.set_index('school_id')['capacity'].to_dict()
        capacities      = np.array([capacities_dict.get(s, 0) for s in all_schools], dtype=np.int32)

        def expand_ranking(ranking: list[str]) -> np.ndarray:
            seen, expanded = set(), []
            for sid in ranking:
                if sid in seen or sid not in school_to_idx:
                    continue
                seen.add(sid)
                expanded.append(school_to_idx[sid])
            return np.array(expanded, dtype=np.int32)

        rankings_as_indices = [expand_ranking(r) for r in preference_lists]
        n_students = len(rankings_as_indices)
        n_schools  = len(all_schools)
        system     = self._system_name

        # STB lottery — same permutation tiled across all schools.
        # Used as the base lottery for build_composite_rank_matrix and for plain GS.
        lottery_1d       = np.argsort(rng.permutation(n_students)).astype(np.float64) / n_students
        school_lotteries = np.tile(lottery_1d, (n_schools, 1))

        if system == '':
            matches_idx   = gale_shapley_per_school_numba_wrapper(
                rankings_as_indices, school_lotteries, capacities
            )
            student_attrs = observed_student_attrs

        else:
            subdivision_to_region = (
                self._subdivision_to_region
                or {str(s): str(s) for s in set(subdivision_assignments) if s is not None}
            )
            # Generic: one school = one program (no program-level split).
            dbn_to_progs = {str(s): [str(s)] for s in all_schools}

            if observed_student_attrs is not None:
                # Mode 3: observed binary flags -> list-of-dicts.
                # School-dependent keys (sibling_school, continuing_school, etc.) are
                # absent; those priority groups will not apply per-school.
                student_attrs_list = observed_student_attrs.reset_index(drop=True).to_dict('records')
            else:
                # Modes 1/2: sample synthetic attributes from priority_config fractions.
                student_attrs_list = sample_student_attributes(
                    district_assignments=list(subdivision_assignments),
                    all_schools=list(all_schools),
                    dbn_to_progs=dbn_to_progs,
                    priority_config=self._priority_config,
                    district_to_region=subdivision_to_region,
                    rng=rng,
                    district_to_borough=subdivision_to_region,
                )

            score_matrix = build_composite_rank_matrix(
                all_schools=list(all_schools),
                student_attrs=student_attrs_list,
                priority_config=self._priority_config,
                school_lotteries=school_lotteries,
                district_to_region=subdivision_to_region,
                district_assignments=list(subdivision_assignments),
            )
            matches_idx   = gale_shapley_per_school_numba_wrapper(
                rankings_as_indices, score_matrix, capacities
            )
            student_attrs = pd.DataFrame(student_attrs_list)

        attr_df = pd.DataFrame(student_attrs) if student_attrs is not None else pd.DataFrame()
        if 'subdivision' not in attr_df.columns:
            attr_df['subdivision'] = list(subdivision_assignments)

        return MatchOutcomes(
            rankings_as_indices=rankings_as_indices,
            matches_idx=matches_idx,
            student_attributes=attr_df,
        )

    def _extract_individual_data(
        self,
        priority_attribute_cols: list[str] | None = None,
        student_attribute_cols:  list[str] | None = None,
    ) -> tuple[list[list[str]], list, pd.DataFrame]:
        """
        Convert _individual_df (long format) into preference lists, subdivision
        assignments, and a student attributes DataFrame.

        Long-format input: one row per (student_id, school_id, preference_number).

        Returns (preference_lists, subdivision_assignments, student_attrs_df).
        """
        if self._individual_df is None:
            raise ValueError("_extract_individual_data requires individual_df (Mode 3).")

        df = self._individual_df.copy()
        df['student_id'] = df['student_id'].astype(str)
        df['school_id']  = df['school_id'].astype(str)

        grouped = (
            df.sort_values('preference_number')
              .groupby('student_id', sort=True)['school_id']
              .apply(list)
        )
        student_ids      = grouped.index.tolist()
        preference_lists = grouped.tolist()

        per_student = df.drop_duplicates(subset='student_id').set_index('student_id').loc[student_ids]

        subdivision_assignments = (
            per_student['subdivision'].tolist()
            if 'subdivision' in per_student.columns
            else [None] * len(student_ids)
        )

        attr_cols = list(priority_attribute_cols or []) + list(student_attribute_cols or [])
        attr_cols = [c for c in attr_cols if c in per_student.columns]
        student_attrs_df = per_student[attr_cols].copy() if attr_cols else pd.DataFrame(index=per_student.index)
        if 'subdivision' in per_student.columns:
            student_attrs_df['subdivision'] = per_student['subdivision']

        return preference_lists, subdivision_assignments, student_attrs_df


    @staticmethod
    def validate_priority_config(config: dict) -> list[str]:
        """
        Validate a priority config for use with priority_attributes.py.
        Returns a list of warning strings (empty = OK).

        Expected schema:
          __meta__.system_name       string, non-empty
          system_defaults            dict with student_attribute_fractions and priority_tiers
          priority_tiers             list of {group, tier (int), school_dependent (bool)}
          student_attribute_fractions  dict of {attr: fraction in [0,1]}
          region_overrides           optional dict of {region: {student_attribute_fractions, priority_tiers}}
          school_overrides           optional dict of {school_id: {priority_tiers, reserves, borough, region}}
        """
        warnings: list[str] = []
        system = config.get('__meta__', {}).get('system_name', '')
        if not system:
            warnings.append(
                "__meta__.system_name is empty; run_matching() will use plain STB GS "
                "and ignore priority_tiers."
            )

        def _check_fracs(fracs: dict, context: str):
            for attr, frac in fracs.items():
                try:
                    f = float(frac)
                except (TypeError, ValueError):
                    warnings.append(f"{context}: fraction for '{attr}' = {frac!r} is not numeric.")
                    continue
                if not (0.0 <= f <= 1.0):
                    warnings.append(f"{context}: fraction for '{attr}' = {f} is outside [0, 1].")

        def _check_tiers(tiers: list, context: str):
            for t in tiers:
                if 'tier' not in t:
                    warnings.append(f"{context}: tier entry {t} is missing required 'tier' (int) field.")
                if 'group' not in t:
                    warnings.append(f"{context}: tier entry {t} is missing required 'group' field.")

        defaults = config.get('system_defaults', {})
        has_per_school_entries = any(
            k not in ('__meta__', 'system_defaults', 'region_overrides', 'school_overrides')
            for k in config
        )
        if not defaults and not has_per_school_entries:
            warnings.append("system_defaults is missing; system-level fractions and tiers not set.")
        elif defaults:
            _check_fracs(defaults.get('student_attribute_fractions', {}), 'system_defaults')
            _check_tiers(defaults.get('priority_tiers', []), 'system_defaults.priority_tiers')

        for region, rdata in config.get('region_overrides', {}).items():
            _check_fracs(rdata.get('student_attribute_fractions', {}), f"region_overrides['{region}']")
            _check_tiers(rdata.get('priority_tiers', []), f"region_overrides['{region}'].priority_tiers")

        for sid, sdata in config.get('school_overrides', {}).items():
            _check_tiers(sdata.get('priority_tiers', []), f"school_overrides['{sid}'].priority_tiers")
            reserves = sdata.get('reserves', {})
            for group, frac in reserves.items():
                try:
                    f = float(frac)
                except (TypeError, ValueError):
                    warnings.append(f"school_overrides['{sid}'].reserves['{group}'] = {frac!r} is not numeric.")
                    continue
                if not (0.0 <= f <= 1.0):
                    warnings.append(f"school_overrides['{sid}'].reserves['{group}'] = {f} is outside [0, 1].")

        return warnings

    @staticmethod
    def _validate_match_outcomes(result) -> MatchOutcomes:
        if isinstance(result, MatchOutcomes):
            return result
        if isinstance(result, pd.DataFrame):
            missing = [c for c in ['student_id', 'matched_school_id', 'rank'] if c not in result.columns]
            if missing:
                raise ValueError(f"Custom matching output missing columns: {missing}")
            raise NotImplementedError(
                "DataFrame -> MatchOutcomes conversion not implemented. "
                "Return a MatchOutcomes directly from custom_matching_fn."
            )
        raise TypeError(f"custom_matching_fn must return MatchOutcomes or DataFrame, got {type(result)}.")


    def evaluate(
        self,
        metrics: list[Metric] | None              = None,
        custom_function_list: list[Callable] | None = None,
        config: EvaluateConfig | None             = None,
        match_outcomes: MatchOutcomes | None      = None,
    ) -> WelfareResults | SweepResults:
        """
        Run welfare evaluation.

        metrics: list of Metric enum values to compute; None = all built-in metrics.
        custom_function_list: list of user-defined evaluation functions to run in
            addition to (or instead of) built-in metrics. Each receives (MatchOutcomes,
            EvaluateConfig) and returns a result dict.
        config: EvaluateConfig controlling max_p, stratification, output paths, sweep params.
        match_outcomes: override the stored _last_match_outcomes.
        """
        outcomes = match_outcomes or self._last_match_outcomes
        cfg      = config or EvaluateConfig()
        metrics_to_run = metrics or list(Metric)

        needs_matching = {
            Metric.GLOBAL_TOP_P_SWEEP, Metric.RANK_DISTRIBUTION, Metric.RANK_STATS,
            Metric.TOP_P_BY_CATEGORY, Metric.RANK_DISTRIBUTION_BY_CATEGORY,
            Metric.RANK_STATS_BY_CATEGORY, Metric.RANK_VARIANCE_BY_CATEGORY,
            Metric.TOP_P_BY_LIST_LENGTH, Metric.AVG_RANK_BY_LIST_LENGTH,
            Metric.TOP_P_BY_PRIORITY_PERCENTILE, Metric.TOP_P_BY_CONJUNCTION,
            Metric.MTB_VS_STB,
        }

        if outcomes is None:
            blocked = [m for m in metrics_to_run if m in needs_matching]
            if blocked:
                raise ValueError(
                    f"Metrics {[m.value for m in blocked]} require match outcomes. "
                    "Run run_matching() or fit() first."
                )

        if Metric.LIST_LENGTH_SWEEP in (metrics or []):
            return self._run_sweep(cfg)

        results = evaluate_simulation_output(
            sim_output={
                'rankings_as_indices': outcomes.rankings_as_indices,
                'matches_idx':         outcomes.matches_idx,
                'student_attributes':  outcomes.student_attributes,
            },
            max_p=cfg.max_p,
            categories=cfg.stratify_by,
            conjunctions=cfg.conjunctions or None,
            priority_col=cfg.priority_col,
            priority_matrix=cfg.priority_matrix,
            priority_higher_is_better=cfg.priority_higher_is_better,
            output_dir=cfg.output_dir,
            n_priority_bins=cfg.n_priority_bins,
            show=cfg.show_plots,
            compute_by_category=any(
                m in metrics_to_run for m in {
                    Metric.TOP_P_BY_CATEGORY, Metric.RANK_DISTRIBUTION_BY_CATEGORY,
                    Metric.RANK_STATS_BY_CATEGORY, Metric.RANK_VARIANCE_BY_CATEGORY,
                }
            ),
            compute_by_list_length=Metric.TOP_P_BY_LIST_LENGTH in metrics_to_run
                                   or Metric.AVG_RANK_BY_LIST_LENGTH in metrics_to_run,
            compute_by_priority=Metric.TOP_P_BY_PRIORITY_PERCENTILE in metrics_to_run,
        )

        if custom_function_list:
            results.custom_results = {
                fn.__name__: fn(outcomes, cfg)
                for fn in custom_function_list
            }

        return results

    def _run_sweep(self, config: EvaluateConfig) -> SweepResults:
        """
        Sweep welfare over config.sweep_min_lengths.

        For each min_len: runs config.n_stb_runs simulations with fixed Mallows preferences
        (mallows_seed constant) and varied STB lotteries (one permutation per run).
        config.sweep_n_jobs controls sampling parallelism within each run_single_simulation call.

        Requires fit() to have been called first.
        Not meaningful with list_length_mode='empirical' (no min_len parameter to vary).
        """
        if self._fitted_params is None:
            raise ValueError("_run_sweep requires fit() first.")
        if self._final_agg_df is None or self._match_stats_df is None or self._school_df is None:
            raise ValueError("_run_sweep requires final_agg_df, match_stats_df, and school_df.")
        if self._list_length_params is None:
            raise ValueError("_run_sweep requires list_length_params; set via fit() or set_list_length_params().")
        if self._list_length_params.get('list_length_mode') == 'empirical':
            raise ValueError(
                "LIST_LENGTH_SWEEP is not meaningful with list_length_mode='empirical'. "
                "Use 'gaussian' or 'gaussian_per_subdivision'."
            )

        _SWEEP_MALLOWS_SEED = 42
        n_students_total = int(self._match_stats_df['n_students'].sum())
        rng_lotteries = np.random.default_rng(0)
        lottery_arrays = [
            rng_lotteries.permutation(n_students_total)
            for _ in range(config.n_stb_runs)
        ]

        individual_em, match_stats_em, school_em = self._to_em_dataframes()
        results_by_min_len: dict[int, list[WelfareResults]] = {}

        for min_len in config.sweep_min_lengths:
            lp_sweep   = {**self._list_length_params, 'list_length_min': min_len}
            lp_sweep_em = self._to_em_list_length_params(lp_sweep)
            run_results: list[WelfareResults] = []

            for lottery_fixed in lottery_arrays:
                agg, synth_info = run_single_simulation(
                    params=self._fitted_params,
                    df=individual_em,
                    match_stats_df=match_stats_em,
                    school_info_df=school_em,
                    lottery_fixed=lottery_fixed,
                    mallows_seed=_SWEEP_MALLOWS_SEED,
                    outfile=None,
                    sampling_n_jobs=config.sweep_n_jobs,
                    per_school_lottery=False,
                    priority_config=self._priority_config,
                    district_to_region=self._subdivision_to_region,
                    list_length_params=lp_sweep_em,
                    save_best_sample=True,
                    max_p=config.max_p,
                    custom_matching_fn=self._custom_matching_fn
                )

                attr_df = (
                    pd.DataFrame(synth_info['student_attrs'])
                    if synth_info['student_attrs'] is not None
                    else pd.DataFrame()
                )
                attr_df['subdivision'] = list(synth_info['syn_districts'])

                welfare = evaluate_simulation_output(
                    sim_output={
                        'rankings_as_indices': synth_info['rankings_as_indices'],
                        'matches_idx':         synth_info['matches_idx'],
                        'student_attributes':  attr_df,
                    },
                    max_p=config.max_p,
                    categories=config.stratify_by,
                    conjunctions=config.conjunctions or None,
                    priority_col=config.priority_col,
                    priority_matrix=config.priority_matrix,
                    priority_higher_is_better=config.priority_higher_is_better,
                    output_dir=None,
                    n_priority_bins=config.n_priority_bins,
                    show=False,
                )
                run_results.append(welfare)

            results_by_min_len[min_len] = run_results

        return SweepResults(results_by_min_len=results_by_min_len)