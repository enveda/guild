"""
Rank percentile scoring functions.

Per protein, per method, each molecule's raw score becomes the fraction of
molecules (same protein, itself included) it scores at least as well as:

      rp_score = rank / denominator

**Convention: 0 = best, 1 = worst**, bounded to ``(0, 1]``: a uniquely best
binder scores ``1 / denominator`` and the uniquely worst ``1.0``. Ties share
their average rank, so tied extremes fall short of those endpoints -- an
all-tied group of ``n`` scores ``(n + 1) / 2n`` throughout. This is the
orientation of the published Guild results, pinned by
``test_orientation_contract_zero_is_best`` and
``test_orientation_contract_ties_share_average_rank``.

Raw ``*_score`` columns keep their own native directions, which is exactly
what the rank percentile exists to normalise away.
"""

import numpy as np
import pandas as pd

from guild.constants.bulk import (
    AGGREGATION_FLAT,
    AGGREGATION_MODES,
    AGGREGATION_POSE_SOURCE,
    CONFIDENCE_ONLY_METHODS,
    DENOMINATOR_ATTEMPTED,
    DENOMINATOR_MODES,
    DENOMINATOR_VALID,
    GLOBAL_RP_SCORE,
    POSE_SOURCE_DICTIONARY,
    RANKS_DICTIONARY,
    RP_SCORES_DICTIONARY,
    SCORES_DIRECTION_DICTIONARY,
    VINA_FAMILY_PLAUSIBLE_SCORE_RANGE,
    VINA_FAMILY_SCORE_METHODS,
)
from guild.constants.guild import PROTEIN_CONF_ID


# ---------------------------------------------------------------------------
# Per-protein scoring
# ---------------------------------------------------------------------------
def _score_one_protein(
    current_protein_group: pd.DataFrame,
    methods: list[str],
    protein_col: str,
    denominator: str = DENOMINATOR_VALID,
) -> pd.DataFrame:
    """
    Rank all molecules per protein and convert to a percentile score.

    rank 1 = best binder → rp_score = 1 / denominator.
    Ties share their average rank.

    :param current_protein_group: DataFrame rows for one protein.
    :param methods: Docking methods to score.
    :param protein_col: Column name identifying the protein.
    :param denominator: ``valid`` or ``attempted``; see
        :func:`compute_rank_percentile_scores`.
    :returns: DataFrame with rank percentile score and rank columns added.
    """
    current_protein_group = current_protein_group.copy()

    for method in methods:
        raw_score_column = f"{method}_score"
        rank_column = RANKS_DICTIONARY[method]
        rp_score_column = RP_SCORES_DICTIONARY[method]
        lower_is_better = SCORES_DIRECTION_DICTIONARY[method] == "minimum"

        if (
            raw_score_column not in current_protein_group.columns
            or current_protein_group[raw_score_column].isna().all()
        ):
            current_protein_group[rank_column] = np.nan
            current_protein_group[rp_score_column] = np.nan
            continue

        n_valid = int(current_protein_group[raw_score_column].notna().sum())
        if n_valid == 0:
            current_protein_group[rank_column] = np.nan
            current_protein_group[rp_score_column] = np.nan
            continue

        # Rank: 1 = best binder.
        ranks = current_protein_group[raw_score_column].rank(
            method="average",
            ascending=lower_is_better,
            na_option="keep",
        )

        # A failed pair still counts under DENOMINATOR_ATTEMPTED, so the
        # worst-ranked molecule falls short of 1.0 by the failure rate.
        divisor = len(current_protein_group) if denominator == DENOMINATOR_ATTEMPTED else n_valid

        current_protein_group[rank_column] = ranks
        current_protein_group[rp_score_column] = ranks / divisor

    return current_protein_group


def _combine_percentiles(
    result: pd.DataFrame,
    voting_methods: list[str],
    aggregation: str,
) -> pd.Series:
    """
    Average the per-method rank percentiles into one global score.

    Under ``pose_source``, tracks that judge the same engine's pose are averaged
    together first, so the cross-method mean is over pose hypotheses rather than
    over scoring passes. Missing scores are skipped rather than imputed at both
    levels: a track with no score for a row drops out of its pose source's mean,
    and a pose source with no scores at all drops out of the outer mean.

    :param result: Frame carrying the per-method rank percentile columns.
    :param voting_methods: Contributing methods, confidence-only ones already
        removed.
    :param aggregation: ``pose_source`` or ``flat``.
    :returns: One global score per row, sharing the 0 = best orientation.
    """
    if aggregation == AGGREGATION_FLAT:
        return result[[RP_SCORES_DICTIONARY[method] for method in voting_methods]].mean(axis=1)

    columns_by_source: dict[str, list[str]] = {}
    for method in voting_methods:
        # A method with no explicit mapping is its own pose source, so a newly
        # added engine behaves sensibly before anyone touches the dictionary.
        source = POSE_SOURCE_DICTIONARY.get(method, method)
        columns_by_source.setdefault(source, []).append(RP_SCORES_DICTIONARY[method])

    per_source_means = pd.DataFrame(
        {source: result[columns].mean(axis=1) for source, columns in columns_by_source.items()},
        index=result.index,
    )
    return per_source_means.mean(axis=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_rank_percentile_scores(
    df: pd.DataFrame,
    methods: list[str] | None = None,
    protein_col: str = PROTEIN_CONF_ID,
    denominator: str = DENOMINATOR_VALID,
    aggregation: str = AGGREGATION_POSE_SOURCE,
) -> pd.DataFrame:
    """
    Compute rank percentile scores per protein for one or more docking methods.

    Rank percentile ``rank / denominator``, where rank 1 = best binder.

    Convention: **0 = best**, bounded to ``(0, 1]``. A uniquely best binder
    scores ``1 / denominator`` and the uniquely worst ``1.0``; ties share their
    average rank, so tied extremes fall short of those endpoints.
    :data:`GLOBAL_RP_SCORE` excludes :data:`CONFIDENCE_ONLY_METHODS`
    (``diffdock``, ``boltz`` — pose-confidence values, not affinity estimates)
    and combines the rest per ``aggregation``; methods can have different
    denominators for the same protein, so it is not bounded by any single
    method's endpoints.

    :param df: Input DataFrame with protein IDs and raw score columns.
    :param methods: Docking methods to score. Defaults to all available.
    :param protein_col: Protein identifier column.
    :param denominator: ``valid`` divides by the molecules that scored,
        ``attempted`` by every row in the protein group including the ones
        whose raw score is null. The published case-study results were
        generated with ``attempted``, where about 5.6% of pairs failed to
        score and still counted, so reproducing the paper's numbers
        requires it. Defaults to ``valid``.
    :param aggregation: ``pose_source`` averages the rescore tracks with their
        upstream engine before averaging across engines, so DiffDock and Boltz
        contribute one vote each instead of three (their own confidence plus
        two auto-added rescores). ``flat`` is the original unweighted mean over
        every voting track, kept to reproduce scores computed before this
        grouping existed. Defaults to ``pose_source``.
    :raises ValueError: If ``denominator`` or ``aggregation`` is not one of
        those modes.
    :return: Copy of df with rank percentile and rank columns added.
    """
    if denominator not in DENOMINATOR_MODES:
        raise ValueError(f"denominator must be one of {DENOMINATOR_MODES}, got {denominator!r}")
    if aggregation not in AGGREGATION_MODES:
        raise ValueError(f"aggregation must be one of {AGGREGATION_MODES}, got {aggregation!r}")

    result = df.copy()

    if methods is None:
        methods = [
            method
            for method in SCORES_DIRECTION_DICTIONARY
            if f"{method}_score" in result.columns
        ]

    if not methods:
        return result

    result = (
        result.groupby(protein_col, group_keys=False)
        .apply(
            _score_one_protein,
            methods=methods,
            protein_col=protein_col,
            denominator=denominator,
        )
        .reset_index(drop=True)
    )

    voting_methods = [
        method
        for method in methods
        if method not in CONFIDENCE_ONLY_METHODS
        and RP_SCORES_DICTIONARY[method] in result.columns
    ]
    if voting_methods:
        result[GLOBAL_RP_SCORE] = _combine_percentiles(result, voting_methods, aggregation)

    return result


# ---------------------------------------------------------------------------
# Score plausibility
# ---------------------------------------------------------------------------
def is_physical_score(value, method: str) -> bool:
    """
    Is this raw score a plausible value for its method?

    Vina-family methods (:data:`VINA_FAMILY_SCORE_METHODS`) report a binding
    free-energy estimate in kcal/mol, which is negative by convention; a
    positive value, or one whose magnitude is absurdly large, is a red flag
    (numerical overflow, a parsing bug, a failed pose scored as a sentinel)
    rather than a real weak binder. Measured on a large Vina case study,
    6.06% of scored values were non-negative and 0.79% exceeded 1,000,000 in
    magnitude, against an observed maximum of 43,851,078.

    Every other method returns True unconditionally: a "maximum"-direction
    method (karmadock, diffdock, boltz — see
    :data:`SCORES_DIRECTION_DICTIONARY`) has no such sign convention, so a
    positive score there is exactly what "better" looks like, and no
    plausible numeric range is documented for methods outside the Vina
    family (including Nesso, which shares the "minimum" direction but is a
    different physical quantity — log10(IC50/uM), not a docking energy).

    This is a plausibility check, not enforcement: it never nulls, clamps or
    drops a value, so a caller can only use it to *count* how suspicious a
    scored value looks, the same way a failed docking attempt is counted.

    :param value: Raw score value. NaN/None is treated as physical — "no
        score" is a distinct, already-tracked failure mode.
    :param method: Method prefix key into :data:`SCORES_DIRECTION_DICTIONARY`.
    :return: False only for a numeric Vina-family score outside
        :data:`VINA_FAMILY_PLAUSIBLE_SCORE_RANGE`.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    if method not in VINA_FAMILY_SCORE_METHODS:
        return True

    low, high = VINA_FAMILY_PLAUSIBLE_SCORE_RANGE
    return low <= value <= high
