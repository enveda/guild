"""
Tests for guild.tools.scores — rank percentile scoring.

Convention: 0 = best, 1 = worst.
  rank 1 = best binder → rp_score = 1 / n_valid; the worst scores 1.0.
"""

import numpy as np
import pandas as pd
import pytest

from guild.constants.bulk import (
    DENOMINATOR_ATTEMPTED,
    DENOMINATOR_VALID,
    GLOBAL_RP_SCORE,
    RANKS_DICTIONARY,
    RP_SCORES_DICTIONARY,
)
from guild.constants.guild import PROTEIN_CONF_ID
from guild.tools.scores import compute_rank_percentile_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_df(protein_ids, vina_scores, karmadock_scores=None):
    """Build a minimal DataFrame for testing."""
    data = {
        PROTEIN_CONF_ID: protein_ids,
        "vina_score": vina_scores,
    }
    if karmadock_scores is not None:
        data["karmadock_score"] = karmadock_scores
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 1. Basic single-method scoring (vina — lower is better)
# ---------------------------------------------------------------------------
class TestSingleMethodVina:
    """Test rank percentile with Vina (ascending = lower is better)."""

    def test_basic_ranking_order(self):
        """Best Vina (lowest) gets rank 1 → rp_score = 1/N (near 0)."""
        df = _make_df(
            protein_ids=["P1"] * 4,
            vina_scores=[-10.0, -8.0, -6.0, -4.0],  # -10 is best
        )
        result = compute_rank_percentile_scores(df, methods=["vina"])

        rp_col = RP_SCORES_DICTIONARY["vina"]
        rank_col = RANKS_DICTIONARY["vina"]

        # -10 → rank 1, -8 → rank 2, -6 → rank 3, -4 → rank 4
        assert result[rank_col].tolist() == [1.0, 2.0, 3.0, 4.0]

        # rp_score = rank / N → 0.25, 0.50, 0.75, 1.00
        expected_rp = [1 / 4, 2 / 4, 3 / 4, 4 / 4]
        np.testing.assert_allclose(result[rp_col].values, expected_rp)

    def test_best_molecule_has_lowest_score(self):
        """Convention: rp_score near 0 = best binder."""
        df = _make_df(
            protein_ids=["P1"] * 5,
            vina_scores=[-12.0, -9.0, -7.0, -5.0, -3.0],
        )
        result = compute_rank_percentile_scores(df, methods=["vina"])

        rp_col = RP_SCORES_DICTIONARY["vina"]
        best_idx = df["vina_score"].idxmin()  # -12.0
        worst_idx = df["vina_score"].idxmax()  # -3.0

        assert result.loc[best_idx, rp_col] < result.loc[worst_idx, rp_col]
        assert result.loc[best_idx, rp_col] == pytest.approx(1 / 5)
        assert result.loc[worst_idx, rp_col] == pytest.approx(1.0)

    def test_two_molecules(self):
        """Simplest non-trivial case: 2 molecules → scores 0.5 and 1.0."""
        df = _make_df(
            protein_ids=["P1", "P1"],
            vina_scores=[-5.0, -3.0],
        )
        result = compute_rank_percentile_scores(df, methods=["vina"])

        rp_col = RP_SCORES_DICTIONARY["vina"]
        assert result[rp_col].tolist() == pytest.approx([0.5, 1.0])


# ---------------------------------------------------------------------------
# 2. Score direction: karmadock (higher is better)
# ---------------------------------------------------------------------------
class TestScoreDirectionKarmadock:
    """KarmaDock uses 'maximum' — higher raw score = better binder."""

    def test_higher_karmadock_gets_rank_1(self):
        """Highest karmadock_score should get rank 1 → rp_score = 1/N."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[None, None, None],
            karmadock_scores=[2.0, 5.0, 8.0],  # 8.0 is best
        )
        result = compute_rank_percentile_scores(df, methods=["karmadock"])

        rp_col = RP_SCORES_DICTIONARY["karmadock"]
        rank_col = RANKS_DICTIONARY["karmadock"]

        # 8.0 → rank 1 (best), 5.0 → rank 2, 2.0 → rank 3 (worst)
        assert result[rank_col].tolist() == [3.0, 2.0, 1.0]
        np.testing.assert_allclose(result[rp_col].values, [3 / 3, 2 / 3, 1 / 3])


# ---------------------------------------------------------------------------
# 3. Multiple proteins scored independently
# ---------------------------------------------------------------------------
class TestMultipleProteins:
    """Rankings should be computed per protein, not globally."""

    def test_proteins_ranked_independently(self):
        """Each protein group gets its own ranks from 1..N."""
        df = _make_df(
            protein_ids=["P1", "P1", "P2", "P2", "P2"],
            vina_scores=[-10.0, -5.0, -3.0, -6.0, -9.0],
        )
        result = compute_rank_percentile_scores(df, methods=["vina"])

        rp_col = RP_SCORES_DICTIONARY["vina"]
        rank_col = RANKS_DICTIONARY["vina"]

        # P1: -10 → rank 1, -5 → rank 2 → rp_scores: 0.5, 1.0
        p1 = result[result[PROTEIN_CONF_ID] == "P1"]
        assert p1[rank_col].tolist() == [1.0, 2.0]
        assert p1[rp_col].tolist() == pytest.approx([0.5, 1.0])

        # P2: -9 → rank 1, -6 → rank 2, -3 → rank 3 → rp_scores: 1/3, 2/3, 1.0
        p2 = result[result[PROTEIN_CONF_ID] == "P2"].sort_values("vina_score")
        expected_ranks = [1.0, 2.0, 3.0]
        expected_rp = [1 / 3, 2 / 3, 1.0]
        assert p2[rank_col].tolist() == expected_ranks
        np.testing.assert_allclose(p2[rp_col].values, expected_rp)


# ---------------------------------------------------------------------------
# 4. Global score = mean of per-method scores
# ---------------------------------------------------------------------------
class TestGlobalScore:
    """GLOBAL_RP_SCORE should be the mean across all method rp_scores."""

    def test_global_score_is_mean_of_methods(self):
        """With two methods, global = (vina_rp + karmadock_rp) / 2."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
            karmadock_scores=[1.0, 5.0, 3.0],
        )
        result = compute_rank_percentile_scores(
            df, methods=["vina", "karmadock"]
        )

        vina_rp = result[RP_SCORES_DICTIONARY["vina"]]
        karma_rp = result[RP_SCORES_DICTIONARY["karmadock"]]
        expected_global = (vina_rp + karma_rp) / 2

        np.testing.assert_allclose(
            result[GLOBAL_RP_SCORE].values, expected_global.values
        )

    def test_single_method_global_equals_method_score(self):
        """With one method, global score = that method's rp_score."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
        )
        result = compute_rank_percentile_scores(df, methods=["vina"])

        np.testing.assert_allclose(
            result[GLOBAL_RP_SCORE].values,
            result[RP_SCORES_DICTIONARY["vina"]].values,
        )


# ---------------------------------------------------------------------------
# 5. Tied scores get average rank
# ---------------------------------------------------------------------------
class TestTiedScores:
    """Molecules with identical raw scores should share the average rank."""

    def test_two_way_tie(self):
        """Two molecules with the same score share their average rank."""
        df = _make_df(
            protein_ids=["P1"] * 4,
            vina_scores=[-10.0, -7.0, -7.0, -4.0],
        )
        result = compute_rank_percentile_scores(df, methods=["vina"])

        rank_col = RANKS_DICTIONARY["vina"]
        rp_col = RP_SCORES_DICTIONARY["vina"]

        # -10 → rank 1, two -7s share ranks 2,3 → avg 2.5, -4 → rank 4
        expected_ranks = [1.0, 2.5, 2.5, 4.0]
        assert result[rank_col].tolist() == expected_ranks

        expected_rp = [1 / 4, 2.5 / 4, 2.5 / 4, 4 / 4]
        np.testing.assert_allclose(result[rp_col].values, expected_rp)

    def test_all_tied(self):
        """All identical scores → all share the same average rank."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-5.0, -5.0, -5.0],
        )
        result = compute_rank_percentile_scores(df, methods=["vina"])

        rank_col = RANKS_DICTIONARY["vina"]
        rp_col = RP_SCORES_DICTIONARY["vina"]

        # All share average of ranks 1,2,3 = 2.0
        assert result[rank_col].tolist() == [2.0, 2.0, 2.0]
        np.testing.assert_allclose(result[rp_col].values, [2 / 3, 2 / 3, 2 / 3])


# ---------------------------------------------------------------------------
# 6. NaN handling
# ---------------------------------------------------------------------------
class TestNaNHandling:
    """Missing raw scores should propagate NaN to rank and rp_score."""

    def test_nan_scores_produce_nan_ranks(self):
        """Rows with NaN raw score get NaN rank and rp_score."""
        df = _make_df(
            protein_ids=["P1"] * 4,
            vina_scores=[-10.0, np.nan, -6.0, -4.0],
        )
        result = compute_rank_percentile_scores(df, methods=["vina"])

        rp_col = RP_SCORES_DICTIONARY["vina"]
        rank_col = RANKS_DICTIONARY["vina"]

        # Row 1 (NaN input) → NaN output
        assert np.isnan(result.loc[1, rank_col])
        assert np.isnan(result.loc[1, rp_col])

        # Valid rows: N = 3 valid scores → ranked 1..3
        valid = result.dropna(subset=[rp_col])
        assert len(valid) == 3
        np.testing.assert_allclose(
            sorted(valid[rp_col].values), [1 / 3, 2 / 3, 1.0]
        )

    def test_all_nan_produces_all_nan(self):
        """If all scores are NaN, all outputs should be NaN."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[np.nan, np.nan, np.nan],
        )
        result = compute_rank_percentile_scores(df, methods=["vina"])

        rp_col = RP_SCORES_DICTIONARY["vina"]
        rank_col = RANKS_DICTIONARY["vina"]

        assert result[rp_col].isna().all()
        assert result[rank_col].isna().all()


# ---------------------------------------------------------------------------
# 7. Auto-detection of methods
# ---------------------------------------------------------------------------
class TestAutoDetection:
    """When methods=None, score all methods whose columns are present."""

    def test_auto_detects_vina(self):
        """With only vina_score column, auto-detects and scores vina."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
        )
        result = compute_rank_percentile_scores(df, methods=None)

        rp_col = RP_SCORES_DICTIONARY["vina"]
        assert rp_col in result.columns
        assert GLOBAL_RP_SCORE in result.columns

    def test_auto_detects_multiple_methods(self):
        """With vina + karmadock columns, both get scored."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
            karmadock_scores=[1.0, 5.0, 3.0],
        )
        result = compute_rank_percentile_scores(df, methods=None)

        assert RP_SCORES_DICTIONARY["vina"] in result.columns
        assert RP_SCORES_DICTIONARY["karmadock"] in result.columns
        assert GLOBAL_RP_SCORE in result.columns


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases: no methods, missing columns, single molecule."""

    def test_no_matching_methods_returns_unchanged(self):
        """If no score columns match, return the DataFrame unchanged."""
        df = pd.DataFrame({
            PROTEIN_CONF_ID: ["P1", "P1"],
            "some_other_column": [1.0, 2.0],
        })
        result = compute_rank_percentile_scores(df, methods=None)

        # No rp_score columns should be added
        assert RP_SCORES_DICTIONARY["vina"] not in result.columns
        assert GLOBAL_RP_SCORE not in result.columns
        assert len(result) == 2

    def test_explicit_missing_method_column_produces_nan(self):
        """Explicitly requesting a method whose column is missing → NaN."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
        )
        result = compute_rank_percentile_scores(df, methods=["karmadock"])

        rp_col = RP_SCORES_DICTIONARY["karmadock"]
        assert rp_col in result.columns
        assert result[rp_col].isna().all()

    def test_single_molecule(self):
        """Single molecule per protein → rank 1 → rp_score = 1.0."""
        df = _make_df(
            protein_ids=["P1"],
            vina_scores=[-7.5],
        )
        result = compute_rank_percentile_scores(df, methods=["vina"])

        rp_col = RP_SCORES_DICTIONARY["vina"]
        rank_col = RANKS_DICTIONARY["vina"]

        assert result[rank_col].iloc[0] == 1.0
        assert result[rp_col].iloc[0] == pytest.approx(1.0)  # 1/1

    def test_does_not_mutate_input(self):
        """The original DataFrame should not be modified."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
        )
        original_cols = set(df.columns)
        _ = compute_rank_percentile_scores(df, methods=["vina"])

        assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# 9. Pinned public contract: orientation
# ---------------------------------------------------------------------------
class TestOrientationContract:
    """The one property every downstream consumer of rp_* depends on."""

    @pytest.mark.parametrize(
        ("method", "raw_scores", "lower_is_better"),
        [
            ("vina", [-4.0, -12.0, -7.0, -9.0, -6.0], True),
            ("karmadock", [3.0, 9.0, 5.0, 1.0, 7.0], False),
        ],
    )
    def test_orientation_contract_zero_is_best(self, method, raw_scores, lower_is_better):
        """PINNED CONTRACT: the best-scoring ligand's rank percentile is nearest 0.

        rp_score = rank / N with rank 1 = best, so the best ligand scores 1/N and the
        worst scores 1.0. This matches the published Guild results; inverting it
        silently flips every downstream ranking. Do not change it without updating
        the manuscript and regenerating the published figures.
        """
        n = len(raw_scores)
        raw_score_column = f"{method}_score"
        df = pd.DataFrame(
            {
                PROTEIN_CONF_ID: ["P1"] * n,
                raw_score_column: raw_scores,
            }
        )
        result = compute_rank_percentile_scores(df, methods=[method])
        rp_col = RP_SCORES_DICTIONARY[method]

        ranks = result[RANKS_DICTIONARY[method]]
        assert result.loc[ranks.idxmin(), rp_col] == pytest.approx(1 / n)
        assert result.loc[ranks.idxmax(), rp_col] == pytest.approx(1.0)

        # Bounded to (0, 1].
        assert (result[rp_col] > 0).all()
        assert (result[rp_col] <= 1.0).all()

        # Non-decreasing as the raw score gets worse.
        best_first = result.sort_values(raw_score_column, ascending=lower_is_better)
        assert (np.diff(best_first[rp_col].values) >= 0).all()

    def test_global_score_shares_the_orientation(self):
        """GLOBAL_RP_SCORE averages rp_* values, so 0 = best holds there too."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
            karmadock_scores=[9.0, 5.0, 1.0],
        )
        result = compute_rank_percentile_scores(df, methods=["vina", "karmadock"])

        # Row 0 is best under both methods, row 2 worst under both.
        assert result[GLOBAL_RP_SCORE].iloc[0] == pytest.approx(1 / 3)
        assert result[GLOBAL_RP_SCORE].iloc[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 10. Denominator: molecules that scored vs pairs attempted
# ---------------------------------------------------------------------------
class TestDenominator:
    """The published case-study results divide by pairs attempted, not by
    molecules that scored. A group with a failed pair separates the two."""

    # 5 rows, 1 of them unscored.
    RAW_SCORES = [-10.0, -8.0, np.nan, -6.0, -4.0]

    def _rp(self, denominator):
        df = _make_df(protein_ids=["P1"] * 5, vina_scores=self.RAW_SCORES)
        result = compute_rank_percentile_scores(df, methods=["vina"], denominator=denominator)
        return result[RP_SCORES_DICTIONARY["vina"]]

    def test_valid_divides_by_the_molecules_that_scored(self):
        """Default: 4 valid scores → ranks 1..4 over 4."""
        rp = self._rp(DENOMINATOR_VALID)

        assert np.isnan(rp.iloc[2])
        np.testing.assert_allclose(rp.dropna().values, [1 / 4, 2 / 4, 3 / 4, 4 / 4])

    def test_attempted_divides_by_every_pair_in_the_group(self):
        """The unscored row still counts, so the worst never reaches 1.0."""
        rp = self._rp(DENOMINATOR_ATTEMPTED)

        assert np.isnan(rp.iloc[2])
        np.testing.assert_allclose(rp.dropna().values, [1 / 5, 2 / 5, 3 / 5, 4 / 5])
        assert rp.max() < 1.0

    def test_the_two_modes_differ(self):
        """Guards against the parameter being silently ignored."""
        assert not np.allclose(
            self._rp(DENOMINATOR_VALID).dropna().values,
            self._rp(DENOMINATOR_ATTEMPTED).dropna().values,
        )

    def test_modes_agree_when_every_pair_scored(self):
        """With no failures the denominators are the same number."""
        df = _make_df(protein_ids=["P1"] * 4, vina_scores=[-10.0, -8.0, -6.0, -4.0])
        valid = compute_rank_percentile_scores(df, methods=["vina"], denominator=DENOMINATOR_VALID)
        attempted = compute_rank_percentile_scores(
            df, methods=["vina"], denominator=DENOMINATOR_ATTEMPTED
        )

        rp_col = RP_SCORES_DICTIONARY["vina"]
        np.testing.assert_allclose(valid[rp_col].values, attempted[rp_col].values)

    def test_default_is_valid(self):
        """Adding the option must not change what existing callers get."""
        df = _make_df(protein_ids=["P1"] * 5, vina_scores=self.RAW_SCORES)
        rp_col = RP_SCORES_DICTIONARY["vina"]

        np.testing.assert_allclose(
            compute_rank_percentile_scores(df, methods=["vina"])[rp_col].values,
            self._rp(DENOMINATOR_VALID).values,
        )

    def test_unknown_denominator_raises(self):
        """Fail loud rather than silently falling back to a default."""
        df = _make_df(protein_ids=["P1"] * 3, vina_scores=[-10.0, -8.0, -6.0])

        with pytest.raises(ValueError, match="denominator must be one of"):
            compute_rank_percentile_scores(df, methods=["vina"], denominator="n_valid")
