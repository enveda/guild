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
    AGGREGATION_POSE_SOURCE_MEDIAN,
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
def _rank_and_score_one_method(
    result: pd.DataFrame,
    method: str,
    protein_col: str,
    denominator: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Rank one method's raw scores per protein and convert to a percentile score.

    rank 1 = best binder → rp_score = 1 / denominator. Ties share their
    average rank. Implemented as groupby transforms over the whole frame,
    rather than a per-group ``.apply()``, so every row keeps its original
    position and every column (including ``protein_col`` itself) survives by
    construction -- ``DataFrameGroupBy.apply`` relies on the grouping column
    being passed through to the callable and back out, which stopped being
    the default on pandas 2.2+ and is gone on 3.x.

    :param result: Frame carrying ``protein_col`` and the raw score column.
    :param method: Docking method to score.
    :param protein_col: Column name identifying the protein.
    :param denominator: ``valid`` or ``attempted``; see
        :func:`compute_rank_percentile_scores`.
    :returns: ``(rank, rp_score)``, each a Series aligned to ``result``'s index.
    """
    raw_score_column = f"{method}_score"
    lower_is_better = SCORES_DIRECTION_DICTIONARY[method] == "minimum"

    grouped_raw_scores = result.groupby(protein_col)[raw_score_column]

    # Rank: 1 = best binder. A protein group with no valid score for this
    # method ranks as all-NaN (na_option="keep" on an all-NaN input), the
    # same outcome as the old per-group early exit.
    ranks = grouped_raw_scores.rank(method="average", ascending=lower_is_better, na_option="keep")

    if denominator == DENOMINATOR_ATTEMPTED:
        # A failed pair still counts, so the worst-ranked molecule falls
        # short of 1.0 by the failure rate.
        divisor = grouped_raw_scores.transform("size")
    else:
        divisor = result[raw_score_column].notna().groupby(result[protein_col]).transform("sum")

    # A wholly-unscored group has ranks already all-NaN, so guarding the
    # divisor here only avoids a spurious 0/0 warning, not a value change.
    rp_score = ranks / divisor.replace(0, np.nan)

    return ranks, rp_score


def _combine_percentiles(
    result: pd.DataFrame,
    voting_methods: list[str],
    aggregation: str,
) -> pd.Series:
    """
    Combine the per-method rank percentiles into one global score.

    Under ``pose_source`` and ``pose_source_median``, tracks that judge the same
    engine's pose are averaged together first (a mean, regardless of the outer
    mode — DiffDock and Boltz have only two rescores apiece, where mean and
    median are identical anyway), so the cross-source combination is over pose
    hypotheses rather than over scoring passes. Missing scores are skipped
    rather than imputed at both levels: a track with no score for a row drops
    out of its pose source's mean, and a pose source with no scores at all
    drops out of the outer combination.

    :param result: Frame carrying the per-method rank percentile columns.
    :param voting_methods: Contributing methods, confidence-only ones already
        removed.
    :param aggregation: ``pose_source_median`` (median across sources),
        ``pose_source`` (mean across sources) or ``flat`` (unweighted mean
        over every voting track, no pose-source grouping at all).
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
    if aggregation == AGGREGATION_POSE_SOURCE_MEDIAN:
        # skipna=True by default, same missing-vote handling as the mean below.
        return per_source_means.median(axis=1)
    return per_source_means.mean(axis=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_rank_percentile_scores(
    df: pd.DataFrame,
    methods: list[str] | None = None,
    protein_col: str = PROTEIN_CONF_ID,
    denominator: str = DENOMINATOR_VALID,
    aggregation: str = AGGREGATION_POSE_SOURCE_MEDIAN,
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
    :param aggregation: Every mode averages the rescore tracks with their
        upstream engine before combining across engines, so DiffDock and Boltz
        contribute one vote each instead of three (their own confidence plus
        two auto-added rescores). ``pose_source_median`` then takes the
        *median* across those per-engine votes — measured on the three-target
        benchmark this beats the flat unweighted mean (0.824 vs 0.781 AUC)
        because it has no defence-free failure mode against a single
        aberrant vote (there, DiffDock alone), unlike a mean, while adding no
        fitted parameters (unlike performance-weighting) and hard-coding no
        per-engine judgement (unlike dropping a track outright). This is a
        directional result at n=15 known binders, not a significant one.
        ``pose_source`` takes the mean across engines instead (the previous
        default). ``flat`` is the original unweighted mean over every voting
        track with no pose-source grouping at all, kept to reproduce scores
        computed before that grouping existed. Defaults to
        ``pose_source_median``.
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

    for method in methods:
        raw_score_column = f"{method}_score"
        rank_column = RANKS_DICTIONARY[method]
        rp_score_column = RP_SCORES_DICTIONARY[method]

        if raw_score_column not in result.columns:
            result[rank_column] = np.nan
            result[rp_score_column] = np.nan
            continue

        rank, rp_score = _rank_and_score_one_method(result, method, protein_col, denominator)
        result[rank_column] = rank
        result[rp_score_column] = rp_score

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
