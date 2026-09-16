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
    DENOMINATOR_ATTEMPTED,
    DENOMINATOR_MODES,
    DENOMINATOR_VALID,
    GLOBAL_RP_SCORE,
    RANKS_DICTIONARY,
    RP_SCORES_DICTIONARY,
    SCORES_DIRECTION_DICTIONARY,
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_rank_percentile_scores(
    df: pd.DataFrame,
    methods: list[str] | None = None,
    protein_col: str = PROTEIN_CONF_ID,
    denominator: str = DENOMINATOR_VALID,
) -> pd.DataFrame:
    """
    Compute rank percentile scores per protein for one or more docking methods.

    Rank percentile ``rank / denominator``, where rank 1 = best binder.

    Convention: **0 = best**, bounded to ``(0, 1]``. A uniquely best binder
    scores ``1 / denominator`` and the uniquely worst ``1.0``; ties share their
    average rank, so tied extremes fall short of those endpoints.
    :data:`GLOBAL_RP_SCORE` is the unweighted mean of the per-method
    percentiles, and methods can have different denominators for the same
    protein, so it is not bounded by any single method's endpoints.

    :param df: Input DataFrame with protein IDs and raw score columns.
    :param methods: Docking methods to score. Defaults to all available.
    :param protein_col: Protein identifier column.
    :param denominator: ``valid`` divides by the molecules that scored,
        ``attempted`` by every row in the protein group including the ones
        whose raw score is null. The published case-study results were
        generated with ``attempted``, where about 5.6% of pairs failed to
        score and still counted, so reproducing the paper's numbers
        requires it. Defaults to ``valid``.
    :raises ValueError: If ``denominator`` is neither of those.
    :return: Copy of df with rank percentile and rank columns added.
    """
    if denominator not in DENOMINATOR_MODES:
        raise ValueError(f"denominator must be one of {DENOMINATOR_MODES}, got {denominator!r}")

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

    rp_score_columns = [
        RP_SCORES_DICTIONARY[method]
        for method in methods
        if RP_SCORES_DICTIONARY[method] in result.columns
    ]
    if rp_score_columns:
        result[GLOBAL_RP_SCORE] = result[rp_score_columns].mean(axis=1)

    return result
