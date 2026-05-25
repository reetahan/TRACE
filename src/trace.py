from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from types import DataKey, EvaluateConfig, MatchOutcomes, Metric, REQUIRED_COLUMNS, SweepResults
from welfare import evaluate_simulation_output, WelfareResults
from constants import *
from custom_user_functions import *

from em import EM_algorithm, run_single_simulation
from mallows import sample_students_global_mixture
from gale_shapley import gale_shapley_per_school_numba_wrapper
from list_length import sample_truncated_normal_lengths, sample_empirical_lengths
from priority_attributes import sample_student_attributes, build_composite_rank_matrix
from data_ingestion import preprocess_data


class TRACE:
    """
    School-assignment welfare simulation API.

    Mode 1  full EM pipeline
            Inputs: final_aggregates_fpath (aggregate ratio stats per subdivision×school),
                    match_stats_fpath (observed top-p match rates per subdivision),
                    school_data_fpath
            Workflow: preprocess() → fit() → sample() → run_matching() → evaluate()

    Mode 2  sampling from pre-fitted parameters
            Inputs: mallows_params_fpath
            Workflow: sample() → run_matching() → evaluate()

    Mode 3  raw individual preferences
            Inputs: individual_data_fpath (long format: one row per student×preference),
                    school_data_fpath
            Workflow: run_matching() → evaluate()
            TRACE reconstructs preference lists from the long-format data, reads
            priority flags directly from the observed columns (no sampling needed),
            runs DA, and computes match rank.
            Long-format columns: student_id, school_id, preference_number (required);
            subdivision, binary priority flags, and any stratification attributes
            (e.g. female) are optional but used when present.

    DataFrame slots:
      _final_agg_df   aggregate application stats, one row per (subdivision × school) —
                      em.py's 'df' argument (Mode 1)
      _match_stats_df observed top-p match outcome rates, one row per subdivision —
                      em.py's fitting target (Mode 1)
      _school_df      school list with capacities (Modes 1 and 3)
      _individual_df  raw individual student preference lists, one row per student (Mode 3)

    fit() and _run_sweep() use plain GS internally regardless of priority_config;
    priority tiers are applied only in run_matching().
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
        """
        matching_data_fpath: aggregate match outcome statistics (Mode 1) —
            the observed top-p rates the EM fits against.
        custom_preprocessing_function: if provided, called immediately after raw
            files are loaded. Signature: (dict[DataKey, DataFrame]) -> dict[DataKey, DataFrame].
            further_processing is set to False (no built-in preprocessing runs after it).
        """
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
            self.preprocess(custom_fn=custom_preprocessing_function, further_processing=False)


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
        if self._individual_fpath:
            self._individual_df = self._read_file(self._individual_fpath)
        if self._school_fpath:
            self._school_df = self._read_file(self._school_fpath)
        if self._match_stats_fpath:
            self._match_stats_df = self._read_file(self._match_stats_fpath)
        if self._final_aggregates_fpath:
            self._final_agg_df = self._read_file(self._final_aggregates_fpath)
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

    def set_df(self, key: DataKey, df: pd.DataFrame):
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

    @staticmethod
    def _validate_custom_preprocessing_output(result: dict):
        if not isinstance(result, dict):
            raise TypeError(f"Custom preprocessing fn must return dict, got {type(result)}.")
        for key, df in result.items():
            if not isinstance(key, DataKey):
                raise TypeError(f"Keys must be DataKey members, got {type(key)}.")
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Values must be DataFrames, got {type(df)} for {key}.")
            TRACE._validate_df(key, df)


    def _to_em_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Rename generic columns to em.py's internal schema before any pipeline call.

        Returns (final_agg_em, match_stats_em, school_em), which map to
        em.py's (df, match_stats_df, school_info_df) arguments respectively.
        Static mappings in _EM_COLUMN_MAP; pct_top_{k} → '% Matches to Choice 1-{k}'.
        """
        if self._final_agg_df is None or self._match_stats_df is None or self._school_df is None:
            raise ValueError("_to_em_dataframes requires final_agg_df, match_stats_df, and school_df.")

        def _rename(df: pd.DataFrame) -> pd.DataFrame:
            renamed = df.rename(columns=self._EM_COLUMN_MAP)
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


    def preprocess(
        self,
        custom_fn: Callable[[dict[DataKey, pd.DataFrame]], dict[DataKey, pd.DataFrame]] | None = None,
        further_processing: bool = True,
    ) -> TRACE:
        """
        Apply optional custom_fn first, then _run_internal_preprocessing if further_processing=True.

        custom_fn signature: (dict[DataKey, DataFrame]) -> dict[DataKey, DataFrame].
        Set further_processing=False if custom_fn already returns fully processed DataFrames.

        """
        if custom_fn is not None:
            raw = {k: v for k, v in {
                DataKey.INDIVIDUAL:       self._individual_df,
                DataKey.SCHOOL:           self._school_df,
                DataKey.MATCH_STATS:      self._match_stats_df,
                DataKey.FINAL_AGGREGATES: self._final_agg_df,
            }.items() if v is not None}
            result = custom_fn(raw)
            self._validate_custom_preprocessing_output(result)
            for key, df in result.items():
                self.set_df(key, df)

        if further_processing:
            self._run_internal_preprocessing()

        return self

    def _run_internal_preprocessing(self):
        """
        Calls preprocess_data() from data_ingestion.py.
        """
        

        result = preprocess_data(
            self._final_agg_df,
            self._match_stats_df,
            self._school_df,
        )
        if result is None:
            raise NotImplementedError(
                "preprocess_data() in data_ingestion.py returned None. "
                "Implement it for your system, or pass custom_fn to preprocess() "
                "with further_processing=False."
            )
        self._final_agg_df, self._match_stats_df, self._school_df = result

    # ------------------------------------------------------------------ #
    # Fit (Mode 3)                                                         #
    # ------------------------------------------------------------------ #

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

        Requires _final_agg_df, _match_stats_df, and _school_df.

        subdivision_to_region: optional subdivision → region mapping.
        list_length_params: stored internally for subsequent sample() calls.
        custom_matching_fn: override the Gale-Shapley step used inside the EM loop.
            TODO: run_single_simulation in em.py must accept a matching hook before
            this is wired through. Accepted here and stored but not yet active.

        After fit(): get_mallows_params() returns best params; _preference_lists and
        _last_match_outcomes hold the best synthetic sample from the EM run.
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

    # ------------------------------------------------------------------ #
    # Sample (Modes 2 & 3)                                                 #
    # ------------------------------------------------------------------ #

    def sample(
        self,
        mallows_params: dict | None     = None,
        n_jobs: int                     = 1,
        seed: int | None                = None,
        list_length_params: dict | None = None,
    ) -> list[list[str]]:
        """
        Sample synthetic preference lists from the Mallows mixture.

        Requires match_stats_df (for per-subdivision student counts).
        Stores results in _preference_lists and _subdivision_assignments.

        list_length_params formats:
          gaussian:                 {list_length_mode, list_length_mean, list_length_std, list_length_min}
          gaussian_per_subdivision: {list_length_mode, mean_per_subdivision: {id: float}, list_length_std, list_length_min}
          empirical:                {list_length_mode, list_length_empirical_probs: {length: prob}}
          fixed:                    {list_length_mode, k_ranking_length}

        Raises ValueError if the implied max list length exceeds 10 (internal cap in
        sample_students_global_mixture). Use fit() for list lengths above 10.
        """
        params = mallows_params or self._fitted_params or self._mallows_params
        if params is None:
            raise ValueError("No Mallows parameters. Run fit() or set via set_mallows_params().")
        if self._match_stats_df is None:
            raise ValueError("sample() requires match_stats_df for per-subdivision student counts.")

        lp = list_length_params or self._list_length_params
        if lp is None:
            raise ValueError("No list_length_params. Call set_list_length_params() or pass directly.")

        mallows_k = self._compute_mallows_k(lp, params)
        if mallows_k > 10:
            raise ValueError(
                f"list_length_params implies max list length {mallows_k}, but "
                "sample_students_global_mixture caps at 10. Use fit() for longer lists."
            )

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
        Run Deferred Acceptance on the given (or stored) preference lists.

        In Mode 3, if preference_lists is not provided, they are extracted
        automatically from _individual_df via _extract_individual_data().
        priority_attribute_cols names columns in _individual_df whose values are
        observed binary priority flags; these are used directly rather than sampled.
        student_attribute_cols names additional per-student columns to carry into
        MatchOutcomes.student_attributes (e.g. 'female').

        Dispatches on priority_config.__meta__.system_name:
          '' (empty) — plain STB Gale-Shapley
          any other  — priority-aware matching via priority_attributes.py
                       (or observed attributes if priority_attribute_cols provided)
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

        school_lotteries are generated once and serve as the within-tier tiebreaker for
        both the plain GS path and the priority path.
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
                # Mode 3: observed binary flags → list-of-dicts.
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
        Optional columns carried through:
          subdivision             — used as subdivision_assignments
          priority attribute cols — binary flags that match priority_config.priority_tiers
                                    group names; passed directly to run_matching() instead
                                    of sampling from fractions
          student attribute cols  — any other per-student columns to carry into
                                    MatchOutcomes.student_attributes (e.g. 'female')

        Returns (preference_lists, subdivision_assignments, student_attrs_df).
        preference_lists is ordered by student_id (sorted ascending).
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
        if not defaults:
            warnings.append("system_defaults is missing; system-level fractions and tiers not set.")
        else:
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
            TODO: custom function execution not yet wired; results are currently ignored.
            Convention: define custom functions in custom_user_functions.py and import.
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

        if Metric.LIST_LENGTH_SWEEP in metrics_to_run:
            return self._run_sweep(cfg)

        return evaluate_simulation_output(
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
        )

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