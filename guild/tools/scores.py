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
    Rank one method's raw scores per protein into a percentile score.

    rank 1 = best binder -> rp_score = 1 / denominator; ties share average
    rank. Uses groupby transforms rather than a per-group ``.apply()``, since
    ``DataFrameGroupBy.apply`` no longer reliably passes the grouping column
    through on pandas 2.2+/3.x.

    :param denominator: ``valid`` or ``attempted``; see
        :func:`compute_rank_percentile_scores`.
    :returns: ``(rank, rp_score)`` Series aligned to ``result``'s index.
    """
    raw_score_column = f"{method}_score"
    lower_is_better = SCORES_DIRECTION_DICTIONARY[method] == "minimum"

    grouped_raw_scores = result.groupby(protein_col)[raw_score_column]

    # Rank: 1 = best binder. An all-NaN group ranks as all-NaN
    # (na_option="keep"), same as the old per-group early exit.
    ranks = grouped_raw_scores.rank(method="average", ascending=lower_is_better, na_option="keep")

    if denominator == DENOMINATOR_ATTEMPTED:
        # A failed pair still counts, so the worst-ranked molecule falls
        # short of 1.0 by the failure rate.
        divisor = grouped_raw_scores.transform("size")
    else:
        divisor = result[raw_score_column].notna().groupby(result[protein_col]).transform("sum")

    # Ranks are already all-NaN for a wholly-unscored group; this only avoids
    # a spurious 0/0 warning.
    rp_score = ranks / divisor.replace(0, np.nan)

    return ranks, rp_score


def _combine_percentiles(
    result: pd.DataFrame,
    voting_methods: list[str],
    aggregation: str,
) -> pd.Series:
    """
    Combine per-method rank percentiles into one global score.

    Tracks judging the same engine's pose are averaged together first, so the
    cross-source combination is over pose hypotheses, not scoring passes.
    Missing scores are skipped (not imputed) at both levels.

    :param aggregation: ``pose_source_median`` (median across sources),
        ``pose_source`` (mean across sources), or ``flat`` (mean over every
        voting track, no pose-source grouping).
    :returns: One global score per row, 0 = best.
    """
    if aggregation == AGGREGATION_FLAT:
        return result[[RP_SCORES_DICTIONARY[method] for method in voting_methods]].mean(axis=1)

    columns_by_source: dict[str, list[str]] = {}
    for method in voting_methods:
        # Unmapped method is its own pose source, so a new engine behaves
        # sensibly by default.
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
    :param aggregation: Each mode averages rescore tracks into their upstream
        engine first, so DiffDock/Boltz contribute one vote, not three.
        ``pose_source_median`` (default) then takes the median across
        engines — beat the flat mean 0.824 vs 0.781 AUC on a 3-target/n=15
        benchmark by not being dragged by one aberrant vote (directional, not
        significant). ``pose_source`` is the plain mean across engines.
        ``flat`` is the original mean over every track, kept to reproduce
        pre-grouping scores.
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
    Is this raw score plausible for its method?

    Vina-family scores are a binding free energy in kcal/mol (negative by
    convention); positive or absurdly large values are a red flag (overflow,
    parsing bug, sentinel) rather than a real weak binder — 6.06% of a large
    Vina case study was non-negative, 0.79% exceeded 1e6 in magnitude. Every
    other method (Nesso included, a different quantity) returns True.

    Plausibility only: never nulls, clamps or drops a value, just flags it
    for logging. NaN/None counts as physical (a separate failure mode).

    :return: False only for a Vina-family score outside
        :data:`VINA_FAMILY_PLAUSIBLE_SCORE_RANGE`.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    if method not in VINA_FAMILY_SCORE_METHODS:
        return True

    low, high = VINA_FAMILY_PLAUSIBLE_SCORE_RANGE
    return low <= value <= high
