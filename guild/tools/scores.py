"""
Rank percentile scoring functions.

Simple rank percentile per protein, per method:
  For each molecule, compute what fraction of ALL molecules (same protein)
  have a worse docking score.

      rp_score = rank / N     (0 ≈ best, 1 ≈ worst)

Convention: 0 = best, 1 = worst.
"""

import numpy as np
import pandas as pd

from guild.constants.bulk import (
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
) -> pd.DataFrame:
    """
    Rank all molecules per protein and convert to a percentile score.

    rank 1 = best binder → rp_score ≈ 1/N (near 0).
    Ties share their average rank.

    :param current_protein_group: DataFrame rows for one protein.
    :param methods: Docking methods to score.
    :param protein_col: Column name identifying the protein.
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

        current_protein_group[rank_column] = ranks
        current_protein_group[rp_score_column] = ranks / n_valid

    return current_protein_group


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_rank_percentile_scores(
    df: pd.DataFrame,
    methods: list[str] | None = None,
    protein_col: str = PROTEIN_CONF_ID,
) -> pd.DataFrame:
    """
    Compute rank percentile scores per protein for one or more docking methods.

    Simple rank percentile: rank / N, where rank 1 = best binder.

    Convention: rank percentile score  0 ≈ best, 1 ≈ worst.

    :param df: Input DataFrame with protein IDs and raw score columns.
    :param methods: Docking methods to score. Defaults to all available.
    :param protein_col: Protein identifier column.
    :return: Copy of df with rank percentile and rank columns added.
    """
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
