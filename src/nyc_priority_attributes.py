from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from util import log_and_print
from gale_shapley import gale_shapley_per_school_numba_wrapper

NYC_SWD_RATE = 0.20
SCREENED_ACADEMIC_THRESHOLDS = (2.0 / 3.0, 1.0 / 3.0)

def analyze_priority_vs_lottery(
    matches_virtual: np.ndarray,
    score_matrix: np.ndarray,
    virtual_program_table: pd.DataFrame,
    capacities: np.ndarray,
) -> pd.DataFrame:
    n_students = len(matches_virtual)
    vp_table = virtual_program_table.reset_index(drop=True)
    
    results = []
    # For each virtual program, find accepted students and worst tier
    vp_accepted = {}
    for i in range(n_students):
        v = int(matches_virtual[i])
        if v >= 0:
            vp_accepted.setdefault(v, []).append(i)
    
    vp_worst_tier = {}
    for v, students in vp_accepted.items():
        method_type = vp_table.iloc[v]['method_type']
        C1 = 1e8 if method_type == 'screened' else 1e4
        tiers = [int(score_matrix[v, i] // C1) for i in students]
        vp_worst_tier[v] = max(tiers)
    
    for i in range(n_students):
        v = int(matches_virtual[i])
        if v < 0:
            results.append({'determination': 'unmatched', 'tier': None, 'seat_type': None})
            continue
        method_type = vp_table.iloc[v]['method_type']
        C1 = 1e8 if method_type == 'screened' else 1e4
        tier = int(score_matrix[v, i] // C1)
        worst = vp_worst_tier[v]
        determination = 'priority' if tier < worst else 'lottery'
        results.append({
            'determination': determination,
            'tier': tier,
            'seat_type': vp_table.iloc[v]['seat_type'],
            'method_type': method_type,
        })
    
    return pd.DataFrame(results)

@dataclass(frozen=True)
class _VirtualProgram:
    key: str
    parent_key: str
    capacity: int
    method_type: str
    seat_type: str
    edopt_group: Optional[str]
    priority_tiers: List[Dict[str, Any]]
    borough: Optional[str]


@dataclass(frozen=True)
class _PreparedVirtualInputs:
    student_rankings: List[List[int]]
    score_matrix: np.ndarray
    capacities: np.ndarray
    virtual_program_table: pd.DataFrame
    student_attrs: pd.DataFrame


def _get_tiers(priority_config: Dict[str, Any], region: Optional[str]) -> List[Dict[str, Any]]:
    region_data = priority_config.get("region_overrides", {}).get(region, {}) if region else {}
    system_data = priority_config.get("system_defaults", {})
    return region_data.get("priority_tiers") or system_data.get("priority_tiers", [])


def _classify_method(method_str: str) -> str:
    m = " ".join(method_str.lower().replace(".", " ").split())
    if any(k in m for k in ("ed opt", "educational option", "edopt")):
        return "ed_opt"
    if "audition" in m:
        return "audition"
    if any(k in m for k in ("screened", "language")):
        return "screened"
    return "unscreened"


def _vp_key(parent_key: str, seat_type: str, edopt_group: Optional[str]) -> str:
    parts = [parent_key, seat_type]
    if edopt_group:
        parts.append(edopt_group)
    return "_".join(parts)


def _split_evenly(total: int, groups: Sequence[str]) -> Dict[str, int]:
    base = total // len(groups)
    rem = total % len(groups)
    return {g: base + (1 if i < rem else 0) for i, g in enumerate(groups)}


def _sample_student_attributes(
    district_assignments: Sequence[Any],
    district_to_borough: Dict[str, str],
    rng: np.random.Generator,
    priority_config: Optional[Dict[str, Any]] = None,
    borough_swd_fractions: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    n_students = len(district_assignments)
    edopt_vals = rng.random(n_students)
    edopt_groups = np.where(
        edopt_vals < 1.0 / 3.0, "low",
        np.where(edopt_vals < 2.0 / 3.0, "mid", "high"),
    )
    academic_scores = rng.random(n_students)
    boroughs = [district_to_borough.get(str(d), "") for d in district_assignments]
    swd = np.array([
        rng.random() < borough_swd_fractions.get(b, NYC_SWD_RATE)
        for b in boroughs
    ], dtype=bool)

    continuing_schools = np.full(n_students, None, dtype=object)
    if priority_config is not None:
        school_overrides = priority_config.get('school_overrides', {})
        prog_keys = list(school_overrides.keys())
        prog_cont_p = []
        for pk in prog_keys:
            tiers = school_overrides[pk].get('priority_tiers', [])
            cont = next((t for t in tiers if t['group'] == 'continuing'), None)
            prog_cont_p.append(cont.get('fraction_eligible', 0.0) if cont else 0.0)
        prog_cont_p = np.array(prog_cont_p)
        total = prog_cont_p.sum()
        if total > 0:
            for i in range(n_students):
                if rng.random() < total:
                    chosen = rng.choice(len(prog_keys), p=prog_cont_p / total)
                    continuing_schools[i] = prog_keys[chosen]

    return pd.DataFrame({
        "borough": boroughs,
        "swd": swd,
        "edopt_group": edopt_groups,
        "academic_score": academic_scores,
        "continuing_school": continuing_schools,
    })


def _add_pool(
    result: List[_VirtualProgram],
    parent_key: str,
    seat_type: str,
    capacity: int,
    method_type: str,
    priority_tiers: List[Dict[str, Any]],
    borough: Optional[str],
    edopt_group: Optional[str],
) -> None:
    if capacity <= 0:
        return
    result.append(_VirtualProgram(
        key=_vp_key(parent_key, seat_type, edopt_group),
        parent_key=parent_key,
        capacity=capacity,
        method_type=method_type,
        seat_type=seat_type,
        edopt_group=edopt_group,
        priority_tiers=priority_tiers,
        borough=borough,
    ))


def _expand_programs(
    all_schools: Sequence[str],
    priority_config: Dict[str, Any],
) -> List[_VirtualProgram]:
    #school_overrides = priority_config["school_overrides"]
    if 'school_overrides' not in priority_config:
        if any(k.startswith('0') for k in priority_config.keys()):
            # config is a flat school_overrides dict
            school_overrides = priority_config
        else:
            raise ValueError("priority_config must contain 'school_overrides'.")
    else:
        school_overrides = priority_config['school_overrides']
    default_fallback_tiers = _get_tiers(priority_config, None)

    result: List[_VirtualProgram] = []
    for prog_key_raw in all_schools:
        prog_key = str(prog_key_raw)
        so = school_overrides.get(prog_key, {})
        school_region = so.get("region")
        fallback_tiers = _get_tiers(priority_config, school_region) if school_region else default_fallback_tiers

        method_type = _classify_method(str(so.get("method", "")))
        seats_ge = int(so.get("seats_ge", 0) or 0)
        seats_swd = int(so.get("seats_swd", 0) or 0)
        priority_tiers = so.get("priority_tiers") or fallback_tiers
        borough = so.get("borough") or None

        if seats_ge > 0:
            if method_type == "ed_opt":
                for group, group_seats in _split_evenly(seats_ge, ["high", "mid", "low"]).items():
                    _add_pool(result, prog_key, "GE", group_seats, method_type, priority_tiers, borough, group)
            else:
                _add_pool(result, prog_key, "GE", seats_ge, method_type, priority_tiers, borough, None)

        if seats_swd > 0:
            _add_pool(result, prog_key, "SWD", seats_swd, "unscreened", priority_tiers, borough, None)

    return result


def _normalize_parent_rankings(
    truncated_rankings: Sequence[Sequence[str]],
    known_parent_keys: set[str],
) -> List[List[str]]:
    normalized: List[List[str]] = []
    for ranking in truncated_rankings:
        seen: set[str] = set()
        row: List[str] = []
        for prog in ranking:
            key = str(prog)
            if key in known_parent_keys and key not in seen:
                row.append(key)
                seen.add(key)
        normalized.append(row)
    return normalized


def _expand_student_rankings(
    parent_rankings: Sequence[Sequence[str]],
    virtual_programs: List[_VirtualProgram],
    student_attrs: pd.DataFrame,
    vp_key_to_idx: Dict[str, int],
) -> List[List[int]]:
    parent_to_vps: Dict[str, List[_VirtualProgram]] = {}
    for vp in virtual_programs:
        parent_to_vps.setdefault(vp.parent_key, []).append(vp)

    student_rankings: List[List[int]] = []
    for student_idx, ranking in enumerate(parent_rankings):
        attrs = student_attrs.iloc[student_idx]
        is_swd = bool(attrs["swd"])
        edopt_group = str(attrs["edopt_group"])
        expanded: List[int] = []

        for prog_key in ranking:
            vps = parent_to_vps.get(str(prog_key), [])
            if not vps:
                continue

            if is_swd:
                candidate_vps = [vp for vp in vps if vp.seat_type == "SWD"]
                if not candidate_vps:
                    candidate_vps = [vp for vp in vps if vp.seat_type == "GE"]
            else:
                candidate_vps = [vp for vp in vps if vp.seat_type == "GE"]
                if any(vp.edopt_group is not None for vp in candidate_vps):
                    candidate_vps = [vp for vp in candidate_vps if vp.edopt_group == edopt_group]

            for vp in candidate_vps:
                idx = vp_key_to_idx[vp.key]
                if idx not in expanded:
                    expanded.append(idx)

        student_rankings.append(expanded)

    return student_rankings


def _build_score_matrix(
    virtual_programs: List[_VirtualProgram],
    student_attrs: pd.DataFrame,
    school_lotteries: np.ndarray,
    parent_key_to_idx: Dict[str, int],
) -> np.ndarray:
    n_students = len(student_attrs)
    a_borough = student_attrs["borough"].to_numpy(dtype=object)
    a_academic = student_attrs["academic_score"].to_numpy(dtype=float)

    hi_thresh, lo_thresh = SCREENED_ACADEMIC_THRESHOLDS
    acad_group = np.where(a_academic > hi_thresh, 1.0, np.where(a_academic > lo_thresh, 2.0, 3.0))

    scores = np.empty((len(virtual_programs), n_students), dtype=np.float64)
    for vp_idx, vp in enumerate(virtual_programs):
        tiers = sorted(vp.priority_tiers, key=lambda t: t["tier"])
        max_tier = max((t["tier"] for t in tiers), default=1)
        prio_tier = np.full(n_students, float(max_tier + 1))
        assigned = np.zeros(n_students, dtype=bool)

        for tier in tiers:
            group = tier["group"]
            tier_value = float(tier["tier"])

            if group in ("all", "all_nyc"):
                prio_tier[~assigned] = tier_value
                break

            if group == "borough":
                target_borough = tier.get("borough_code") or vp.borough
                if target_borough:
                    mask = (~assigned) & (a_borough == target_borough)
                    prio_tier[mask] = tier_value
                    assigned[mask] = True
            
            elif group == "continuing":
                a_continuing = student_attrs["continuing_school"] if "continuing_school" in student_attrs.columns else pd.Series([None] * n_students)
                if hasattr(a_continuing, 'to_numpy'):
                    cont_arr = a_continuing.to_numpy(dtype=object)
                else:
                    cont_arr = np.array(a_continuing, dtype=object)
                mask = (~assigned) & (cont_arr == vp.parent_key)
                prio_tier[mask] = tier_value
                assigned[mask] = True

        parent_idx = parent_key_to_idx[vp.parent_key]
        lottery = school_lotteries[parent_idx]

        if vp.method_type == "screened":
            scores[vp_idx] = prio_tier * 1e8 + acad_group * 1e4 + lottery
        else:
            scores[vp_idx] = prio_tier * 1e4 + lottery

    return scores


def _prepare_virtual_inputs(
    truncated_rankings: Sequence[Sequence[str]],
    district_assignments: Sequence[Any],
    all_schools: Sequence[str],
    priority_config: Dict[str, Any],
    district_to_borough: Dict[str, str],
    school_lotteries: np.ndarray,
    rng: np.random.Generator,
    student_attrs=None
) -> _PreparedVirtualInputs:

    if student_attrs is None:
        student_attrs = _sample_student_attributes(
            district_assignments=district_assignments,
            district_to_borough=district_to_borough,
            rng=rng,
            borough_swd_fractions=borough_swd_fractions,
            priority_config=priority_config
        )

    school_overrides = priority_config.get("school_overrides", {})
    if not school_overrides:
        school_overrides = {k: v for k, v in priority_config.items() if k != '__meta__'}
        if not school_overrides:
            raise ValueError("priority_config must contain 'school_overrides'.")

    # Compute borough-specific SWD rates from seat data
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

    student_attrs = _sample_student_attributes(
        district_assignments=district_assignments,
        district_to_borough=district_to_borough,
        rng=rng,
        borough_swd_fractions=borough_swd_fractions,
        priority_config=priority_config
    )

    n_students = len(district_assignments)

    if len(truncated_rankings) != n_students:
        raise ValueError("truncated_rankings must have the same length as district_assignments.")

    all_schools = [str(s) for s in all_schools]
    if school_lotteries.shape != (len(all_schools), n_students):
        raise ValueError("school_lotteries must have shape (len(all_schools), len(district_assignments)).")

    parent_key_to_idx = {pk: i for i, pk in enumerate(all_schools)}
    parent_rankings = _normalize_parent_rankings(truncated_rankings, set(all_schools))

    virtual_programs = _expand_programs(
        all_schools=all_schools,
        priority_config=priority_config,
    )
    if not virtual_programs:
        raise ValueError("No virtual programs generated from priority_config['school_overrides'].")

    vp_key_to_idx = {vp.key: i for i, vp in enumerate(virtual_programs)}
    student_rankings = _expand_student_rankings(parent_rankings, virtual_programs, student_attrs, vp_key_to_idx)
    score_matrix = _build_score_matrix(virtual_programs, student_attrs, school_lotteries, parent_key_to_idx)
    capacities = np.array([vp.capacity for vp in virtual_programs], dtype=np.int32)

    virtual_program_table = pd.DataFrame([{
        "key": vp.key,
        "parent_key": vp.parent_key,
        "capacity": vp.capacity,
        "method_type": vp.method_type,
        "seat_type": vp.seat_type,
        "edopt_group": vp.edopt_group,
        "borough": vp.borough,
    } for vp in virtual_programs])


    return _PreparedVirtualInputs(
        student_rankings=student_rankings,
        score_matrix=score_matrix,
        capacities=capacities,
        virtual_program_table=virtual_program_table,
        student_attrs=student_attrs,
    )


def _to_parent_matches(matches_virtual: Sequence[int], virtual_program_table: pd.DataFrame) -> np.ndarray:
    parent_matches: List[str] = []
    vp_table = virtual_program_table.reset_index(drop=True)

    for match_idx in matches_virtual:
        m = int(match_idx)
        if m < 0:
            parent_matches.append("-1")
        elif m >= len(vp_table):
            raise IndexError(f"Virtual match index out of range: {m}")
        else:
            parent_matches.append(str(vp_table.iloc[m]["parent_key"]))

    return np.asarray(parent_matches, dtype=object)


def run_nyc_priority_matching(
    truncated_rankings: Sequence[Sequence[str]],
    district_assignments: Sequence[Any],
    all_schools: Sequence[str],
    priority_config: Dict[str, Any],
    district_to_borough: Dict[str, str],
    school_lotteries: np.ndarray,
    rng: np.random.Generator,
    log_file: Optional[str] = None,
    student_attrs=None
) -> np.ndarray:
    """
    NYC priority matching entry point, accepting exactly what em.py already produces.

    Expected input:
    - truncated_rankings : Mallows rankings after list-length truncation, as school DBN strings.
                           Produced by the convert_and_truncate step in em.py.
    - district_assignments: list of residential district per student, in the same order.
                            Produced by all_district_assignments in em.py.
    - all_schools         : ordered list of school DBNs.
                            Produced by df['School DBN'].unique() in em.py.
    - priority_config     : dict with a 'school_overrides' key containing per-program
                            seats_ge, seats_swd, method, and borough.
    - district_to_borough : dict mapping district string -> borough code (e.g. "9" -> "X").
                            Already passed as parameter to run_single_simulation in em.py.
    - school_lotteries    : lottery matrix of shape (len(all_schools), n_students).
                            Already generated in em.py (per-school or tiled global).
    - rng                 : NumPy random generator. Already present in em.py.

    Returned output:
    - np.ndarray of length n_students.
    - Each entry is the matched parent school DBN, or '-1' if unmatched.

    Replaces in em.py:
    - the call to sample_student_attributes() from priority_attributes.py
    - the call to build_composite_rank_matrix() from priority_attributes.py
    - the call to gale_shapley_per_school_numba_wrapper()
    All three are handled internally.

    Branching point in em.py (run_matching / run_single_simulation):
        BEFORE:
            if priority_config is not None and student_attrs is not None:
                school_lotteries = build_composite_rank_matrix(...)
            matches_idx = gale_shapley_per_school_numba_wrapper(
                rankings_as_indices, school_lotteries, capacities)

        AFTER:
            matches_schools = run_nyc_priority_matching(
                truncated_rankings=all_rankings,
                district_assignments=all_district_assignments,
                all_schools=all_schools,
                priority_config=priority_config,
                district_to_borough=district_to_borough,
                school_lotteries=school_lotteries,
                rng=rng,
            )
    """

    prepared = _prepare_virtual_inputs(
        truncated_rankings=truncated_rankings,
        district_assignments=district_assignments,
        all_schools=all_schools,
        priority_config=priority_config,
        district_to_borough=district_to_borough,
        school_lotteries=school_lotteries,
        rng=rng,
        student_attrs=student_attrs
    )

    matches_virtual = gale_shapley_per_school_numba_wrapper(
        prepared.student_rankings,
        prepared.score_matrix,
        prepared.capacities,
    )

    ### Understand lottery vs priority breakdown
    if(log_file is not None):
        analysis_df = analyze_priority_vs_lottery(
            matches_virtual, prepared.score_matrix,
            prepared.virtual_program_table, prepared.capacities,
        )

        total = len(analysis_df)
        n_matched = (analysis_df['determination'] != 'unmatched').sum()
        n_priority = (analysis_df['determination'] == 'priority').sum()
        n_lottery = (analysis_df['determination'] == 'lottery').sum()
        n_unmatched = (analysis_df['determination'] == 'unmatched').sum()

        log_and_print(f"\n  Priority vs Lottery Analysis:", log_file=log_file)
        log_and_print(f"    Matched:   {n_matched}/{total} ({100*n_matched/total:.1f}%)", log_file=log_file)
        if n_matched > 0:
            log_and_print(f"    Priority-determined: {n_priority}/{n_matched} ({100*n_priority/n_matched:.1f}% of matched)", log_file=log_file)
            log_and_print(f"    Lottery-determined:  {n_lottery}/{n_matched} ({100*n_lottery/n_matched:.1f}% of matched)", log_file=log_file)
        else:
            log_and_print(f"    No matched students.", log_file=log_file)
        log_and_print(f"    Unmatched: {n_unmatched}/{total} ({100*n_unmatched/total:.1f}%)", log_file=log_file)

        tier_breakdown = analysis_df[analysis_df['determination']=='priority'].groupby('tier').size()
        log_and_print(f"    Priority tier breakdown: {dict(tier_breakdown)}", log_file=log_file)

        seat_breakdown = analysis_df[analysis_df['determination'] != 'unmatched'].groupby('seat_type').size()
        log_and_print(f"    Seat type breakdown: {dict(seat_breakdown)}", log_file=log_file)
    ####

    return _to_parent_matches(matches_virtual, prepared.virtual_program_table), prepared.student_attrs
