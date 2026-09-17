"""
Tests for guild.tools.scores — rank percentile scoring.

Convention: 0 = best, 1 = worst.
  rank 1 = best binder → rp_score = 1 / n_valid; the worst scores 1.0.
"""

import numpy as np
import pandas as pd
import pytest

from guild.constants.bulk import (
    AGGREGATION_FLAT,
    AGGREGATION_POSE_SOURCE,
    AGGREGATION_POSE_SOURCE_MEDIAN,
    DENOMINATOR_ATTEMPTED,
    DENOMINATOR_VALID,
    GLOBAL_RP_SCORE,
    RANKS_DICTIONARY,
    RP_SCORES_DICTIONARY,
    VINA_FAMILY_PLAUSIBLE_SCORE_RANGE,
)
from guild.constants.guild import PROTEIN_CONF_ID
from guild.tools.scores import compute_rank_percentile_scores, is_physical_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_df(protein_ids, vina_scores, karmadock_scores=None, **extra_scores):
    """Build a minimal DataFrame; extra_scores accepts any <method>_scores=[...] kwarg."""
    data = {
        PROTEIN_CONF_ID: protein_ids,
        "vina_score": vina_scores,
    }
    if karmadock_scores is not None:
        data["karmadock_score"] = karmadock_scores
    for name, values in extra_scores.items():
        assert name.endswith("_scores"), f"expected a '<method>_scores' kwarg, got {name!r}"
        data[f"{name[: -len('_scores')]}_score"] = values
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
        """All identical scores → all share rank (n + 1) / 2, so rp = (n + 1) / 2n.

        Pins the tie half of the orientation contract; see
        :meth:`TestOrientationContract.test_orientation_contract_ties_share_average_rank`.
        """
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

    def test_orientation_contract_ties_share_average_rank(self):
        """PINNED CONTRACT: tied extremes do not reach the endpoints.

        Ranks use method="average", so tied best values share a rank and score
        above 1 / denominator. Documented alongside the endpoint contract so the
        two cannot drift apart.
        """
        # Two molecules tied at the best score share ranks 1 and 2 -> 1.5.
        tied_best = _make_df(
            protein_ids=["P1"] * 4,
            vina_scores=[-10.0, -10.0, -7.0, -4.0],
        )
        rp_col = RP_SCORES_DICTIONARY["vina"]
        rp = compute_rank_percentile_scores(tied_best, methods=["vina"])[rp_col]

        assert rp.iloc[0] == pytest.approx(1.5 / 4)
        assert rp.iloc[1] == pytest.approx(1.5 / 4)
        assert rp.min() > 1 / 4

        # Tied worst likewise falls short of 1.0.
        tied_worst = _make_df(
            protein_ids=["P1"] * 4,
            vina_scores=[-10.0, -7.0, -4.0, -4.0],
        )
        rp = compute_rank_percentile_scores(tied_worst, methods=["vina"])[rp_col]

        assert rp.iloc[2] == pytest.approx(3.5 / 4)
        assert rp.iloc[3] == pytest.approx(3.5 / 4)
        assert rp.max() < 1.0

        # All tied: every molecule gets (n + 1) / 2n, neither endpoint.
        n = 3
        all_tied = _make_df(protein_ids=["P1"] * n, vina_scores=[-5.0] * n)
        rp = compute_rank_percentile_scores(all_tied, methods=["vina"])[rp_col]

        np.testing.assert_allclose(rp.values, [(n + 1) / (2 * n)] * n)

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


# ---------------------------------------------------------------------------
# 11. Aggregation: confidence exclusion and pose-source grouping
# ---------------------------------------------------------------------------
class TestGlobalScoreAggregation:
    """GLOBAL_RP_SCORE excludes confidence-only tracks and groups rescores
    with the engine whose pose they scored, instead of voting flat."""

    def test_confidence_track_scored_but_excluded_from_global(self):
        """diffdock_score gets its rp_* column but never enters the global mean."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
            diffdock_scores=[1.0, 2.0, 3.0],
        )
        result = compute_rank_percentile_scores(df, methods=["vina", "diffdock"])

        assert RP_SCORES_DICTIONARY["diffdock"] in result.columns
        vina_rp = result[RP_SCORES_DICTIONARY["vina"]]
        diffdock_rp = result[RP_SCORES_DICTIONARY["diffdock"]]

        np.testing.assert_allclose(result[GLOBAL_RP_SCORE].values, vina_rp.values)

        # Had diffdock voted, the (flat) mean of the two would differ from vina alone.
        would_be_flat = (vina_rp + diffdock_rp) / 2
        assert not np.allclose(result[GLOBAL_RP_SCORE].values, would_be_flat.values)

    def test_only_confidence_methods_adds_no_global_column(self):
        """Requesting only diffdock/boltz produces no GLOBAL_RP_SCORE at all."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[None, None, None],
            diffdock_scores=[1.0, 2.0, 3.0],
            boltz_scores=[0.1, 0.5, 0.9],
        )
        result = compute_rank_percentile_scores(df, methods=["diffdock", "boltz"])

        assert RP_SCORES_DICTIONARY["diffdock"] in result.columns
        assert RP_SCORES_DICTIONARY["boltz"] in result.columns
        assert GLOBAL_RP_SCORE not in result.columns

    def test_diffdock_rescores_count_once_not_flat(self):
        """DiffDock's two rescores average to one pose-source vote, not two."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
            vina_rescore_diffdock_scores=[-1.0, -2.0, -3.0],
            gnina_rescore_diffdock_scores=[5.0, 10.0, 1.0],
        )
        result = compute_rank_percentile_scores(
            df, methods=["vina", "vina_rescore_diffdock", "gnina_rescore_diffdock"]
        )

        vina_rp = result[RP_SCORES_DICTIONARY["vina"]]
        vrd_rp = result[RP_SCORES_DICTIONARY["vina_rescore_diffdock"]]
        grd_rp = result[RP_SCORES_DICTIONARY["gnina_rescore_diffdock"]]

        expected_pose_source = (vina_rp + (vrd_rp + grd_rp) / 2) / 2
        expected_flat = (vina_rp + vrd_rp + grd_rp) / 3

        np.testing.assert_allclose(result[GLOBAL_RP_SCORE].values, expected_pose_source.values)
        assert not np.allclose(result[GLOBAL_RP_SCORE].values, expected_flat.values)

    def test_flat_mode_reproduces_the_old_flat_mean(self):
        """aggregation='flat' is the unweighted mean over every voting column."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
            vina_rescore_diffdock_scores=[-1.0, -2.0, -3.0],
            gnina_rescore_diffdock_scores=[5.0, 10.0, 1.0],
        )
        result = compute_rank_percentile_scores(
            df,
            methods=["vina", "vina_rescore_diffdock", "gnina_rescore_diffdock"],
            aggregation=AGGREGATION_FLAT,
        )

        vina_rp = result[RP_SCORES_DICTIONARY["vina"]]
        vrd_rp = result[RP_SCORES_DICTIONARY["vina_rescore_diffdock"]]
        grd_rp = result[RP_SCORES_DICTIONARY["gnina_rescore_diffdock"]]
        expected_flat = (vina_rp + vrd_rp + grd_rp) / 3

        np.testing.assert_allclose(result[GLOBAL_RP_SCORE].values, expected_flat.values)

    def test_default_aggregation_is_pose_source_median(self):
        """Default must be pose_source_median, not the older pose_source mean."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],  # row 0 best (rank 1 -> rp 1/3)
            karmadock_scores=[9.0, 5.0, 1.0],  # row 0 best (rank 1 -> rp 1/3)
            gnina_scores=[10.0, 8.0, 6.0],  # row 0 worst (rank 3 -> rp 1.0) -- the aberrant vote
        )
        default = compute_rank_percentile_scores(df, methods=["vina", "karmadock", "gnina"])
        explicit_median = compute_rank_percentile_scores(
            df,
            methods=["vina", "karmadock", "gnina"],
            aggregation=AGGREGATION_POSE_SOURCE_MEDIAN,
        )
        explicit_mean = compute_rank_percentile_scores(
            df, methods=["vina", "karmadock", "gnina"], aggregation=AGGREGATION_POSE_SOURCE
        )

        np.testing.assert_allclose(
            default[GLOBAL_RP_SCORE].values, explicit_median[GLOBAL_RP_SCORE].values
        )
        # Confirms default is the median: with 3 sources disagreeing, mean would differ.
        assert not np.allclose(
            default[GLOBAL_RP_SCORE].values, explicit_mean[GLOBAL_RP_SCORE].values
        )

    def test_unknown_aggregation_raises(self):
        """Fail loud rather than silently falling back to a default."""
        df = _make_df(protein_ids=["P1"] * 3, vina_scores=[-10.0, -8.0, -6.0])

        with pytest.raises(ValueError, match="aggregation must be one of"):
            compute_rank_percentile_scores(df, methods=["vina"], aggregation="weighted")

    def test_one_rescore_nan_still_votes_via_its_sibling(self):
        """A NaN in one rescore track still lets the pose source vote via its sibling."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
            vina_rescore_diffdock_scores=[np.nan, -2.0, -3.0],
            gnina_rescore_diffdock_scores=[5.0, 10.0, 1.0],
        )
        result = compute_rank_percentile_scores(
            df, methods=["vina", "vina_rescore_diffdock", "gnina_rescore_diffdock"]
        )

        vina_rp = result[RP_SCORES_DICTIONARY["vina"]]
        vrd_rp = result[RP_SCORES_DICTIONARY["vina_rescore_diffdock"]]
        grd_rp = result[RP_SCORES_DICTIONARY["gnina_rescore_diffdock"]]

        assert np.isnan(vrd_rp.iloc[0])  # premise: row 0's vina rescore is missing
        assert not np.isnan(grd_rp.iloc[0])  # its gnina sibling still scored

        # Row 0's diffdock pose source has only gnina_rescore_diffdock to go on.
        expected_row0 = (vina_rp.iloc[0] + grd_rp.iloc[0]) / 2
        assert result[GLOBAL_RP_SCORE].iloc[0] == pytest.approx(expected_row0)

    def test_pose_source_with_every_track_nan_drops_out_of_outer_mean(self):
        """A wholly-missing pose source is excluded from the outer mean, not imputed."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
            vina_rescore_diffdock_scores=[np.nan, -2.0, -3.0],
            gnina_rescore_diffdock_scores=[np.nan, 10.0, 1.0],
        )
        result = compute_rank_percentile_scores(
            df, methods=["vina", "vina_rescore_diffdock", "gnina_rescore_diffdock"]
        )

        vina_rp = result[RP_SCORES_DICTIONARY["vina"]]
        vrd_rp = result[RP_SCORES_DICTIONARY["vina_rescore_diffdock"]]
        grd_rp = result[RP_SCORES_DICTIONARY["gnina_rescore_diffdock"]]

        assert np.isnan(vrd_rp.iloc[0]) and np.isnan(grd_rp.iloc[0])  # premise: source fully missing

        # Row 0's diffdock pose source is entirely NaN, so global falls back to vina alone.
        assert result[GLOBAL_RP_SCORE].iloc[0] == pytest.approx(vina_rp.iloc[0])


# ---------------------------------------------------------------------------
# 12. Median cross-source aggregation (the new default)
# ---------------------------------------------------------------------------
class TestMedianAggregation:
    """pose_source_median combines per-source votes with a median instead of
    a mean, so one aberrant pose source cannot drag the consensus as far."""

    def test_median_of_five_pose_source_votes_resists_one_aberrant_vote(self):
        """4 sources agree row 0 is best, 1 (diffdock) calls it worst; the mean
        gets dragged toward the outlier, the median ignores it."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],  # row 0 best -> rp 1/3
            karmadock_scores=[9.0, 5.0, 1.0],  # row 0 best -> rp 1/3
            gnina_scores=[-10.0, -8.0, -6.0],  # row 0 best -> rp 1/3
            nesso_scores=[-10.0, -8.0, -6.0],  # row 0 best -> rp 1/3
            # diffdock pose source (rescore tracks, mean of the two): row 0 worst -> rp 1.0
            vina_rescore_diffdock_scores=[-1.0, -2.0, -3.0],
            gnina_rescore_diffdock_scores=[-1.0, -2.0, -3.0],
        )
        methods = [
            "vina",
            "karmadock",
            "gnina",
            "nesso",
            "vina_rescore_diffdock",
            "gnina_rescore_diffdock",
        ]
        median_result = compute_rank_percentile_scores(
            df, methods=methods, aggregation=AGGREGATION_POSE_SOURCE_MEDIAN
        )
        mean_result = compute_rank_percentile_scores(
            df, methods=methods, aggregation=AGGREGATION_POSE_SOURCE
        )

        # 5 pose-source votes for row 0: [1/3, 1/3, 1/3, 1/3, 1.0].
        expected_median = 1 / 3
        expected_mean = (4 * (1 / 3) + 1.0) / 5

        assert median_result[GLOBAL_RP_SCORE].iloc[0] == pytest.approx(expected_median)
        assert mean_result[GLOBAL_RP_SCORE].iloc[0] == pytest.approx(expected_mean)
        assert median_result[GLOBAL_RP_SCORE].iloc[0] < mean_result[GLOBAL_RP_SCORE].iloc[0]

    def test_missing_pose_source_takes_median_of_the_survivors(self):
        """Median of 3 surviving pose sources; the 2 missing ones are excluded, not imputed."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],  # row 0 -> rp 1/3
            karmadock_scores=[9.0, 5.0, 1.0],  # row 0 -> rp 1/3
            gnina_scores=[10.0, 8.0, 6.0],  # row 0 -> rp 1.0
            nesso_scores=[np.nan, -8.0, -6.0],  # row 0 missing entirely
            vina_rescore_diffdock_scores=[np.nan, -2.0, -3.0],  # row 0 missing
            gnina_rescore_diffdock_scores=[np.nan, 10.0, 1.0],  # row 0 missing (whole source gone)
        )
        methods = [
            "vina",
            "karmadock",
            "gnina",
            "nesso",
            "vina_rescore_diffdock",
            "gnina_rescore_diffdock",
        ]
        result = compute_rank_percentile_scores(
            df, methods=methods, aggregation=AGGREGATION_POSE_SOURCE_MEDIAN
        )

        # Row 0 has 3 surviving votes: vina=1/3, karmadock=1/3, gnina=1.0.
        assert result[GLOBAL_RP_SCORE].iloc[0] == pytest.approx(1 / 3)

    def test_two_surviving_votes_give_their_mean(self):
        """The even-count case: median of exactly 2 votes equals their mean."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],  # row 0 -> rp 1/3
            karmadock_scores=[1.0, 5.0, 9.0],  # row 0 -> rp 1.0
        )
        result = compute_rank_percentile_scores(
            df, methods=["vina", "karmadock"], aggregation=AGGREGATION_POSE_SOURCE_MEDIAN
        )

        vina_rp = result[RP_SCORES_DICTIONARY["vina"]].iloc[0]
        karma_rp = result[RP_SCORES_DICTIONARY["karmadock"]].iloc[0]
        assert result[GLOBAL_RP_SCORE].iloc[0] == pytest.approx((vina_rp + karma_rp) / 2)

    def test_explicit_pose_source_still_reproduces_the_plain_mean(self):
        """aggregation='pose_source' must still mean the mean, not the new default."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
            karmadock_scores=[9.0, 5.0, 1.0],
            gnina_scores=[10.0, 8.0, 6.0],
        )
        result = compute_rank_percentile_scores(
            df, methods=["vina", "karmadock", "gnina"], aggregation=AGGREGATION_POSE_SOURCE
        )

        vina_rp = result[RP_SCORES_DICTIONARY["vina"]]
        karma_rp = result[RP_SCORES_DICTIONARY["karmadock"]]
        gnina_rp = result[RP_SCORES_DICTIONARY["gnina"]]
        expected_mean = (vina_rp + karma_rp + gnina_rp) / 3

        np.testing.assert_allclose(result[GLOBAL_RP_SCORE].values, expected_mean.values)

    def test_flat_mode_unaffected_by_the_new_default(self):
        """aggregation='flat' still reproduces the unweighted mean over every voting track."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
            karmadock_scores=[9.0, 5.0, 1.0],
            gnina_scores=[10.0, 8.0, 6.0],
            nesso_scores=[-10.0, -8.0, -6.0],
        )
        methods = ["vina", "karmadock", "gnina", "nesso"]
        result = compute_rank_percentile_scores(df, methods=methods, aggregation=AGGREGATION_FLAT)

        expected_flat = sum(result[RP_SCORES_DICTIONARY[m]] for m in methods) / len(methods)
        np.testing.assert_allclose(result[GLOBAL_RP_SCORE].values, expected_flat.values)


# ---------------------------------------------------------------------------
# 13. Boltz-2's affinity head votes as part of Boltz's pose source
# ---------------------------------------------------------------------------
class TestBoltzAffinityVote:
    """boltz_affinity_score is ranked and voted, joining Boltz's existing
    vote as a third estimate rather than counting as its own."""

    def test_boltz_affinity_is_ranked(self):
        """rp_boltz_affinity_score / rank_boltz_affinity_score get produced."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[None, None, None],
            boltz_affinity_scores=[-9.0, -7.0, -5.0],
        )
        result = compute_rank_percentile_scores(df, methods=["boltz_affinity"])

        assert RP_SCORES_DICTIONARY["boltz_affinity"] in result.columns
        assert RANKS_DICTIONARY["boltz_affinity"] in result.columns

    def test_direction_is_minimum_most_negative_is_best(self):
        """Lower log10(IC50/uM) = more potent = best -> lowest rp_score."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[None, None, None],
            boltz_affinity_scores=[-9.0, -7.0, -5.0],
        )
        result = compute_rank_percentile_scores(df, methods=["boltz_affinity"])

        rp_col = RP_SCORES_DICTIONARY["boltz_affinity"]
        assert result.loc[0, rp_col] == pytest.approx(1 / 3)  # -9.0, most negative
        assert result.loc[2, rp_col] == pytest.approx(1.0)  # -5.0, least negative

    def test_boltz_affinity_joins_boltz_group_instead_of_voting_alone(self):
        """Vina and Boltz's group must combine as two pose-source votes, not four independent ones."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
            vina_rescore_boltz_scores=[-1.0, -5.0, -9.0],
            gnina_rescore_boltz_scores=[-2.0, -5.0, -8.0],
            boltz_affinity_scores=[-1.0, -4.0, -9.0],
        )
        methods = [
            "vina",
            "vina_rescore_boltz",
            "gnina_rescore_boltz",
            "boltz_affinity",
        ]
        result = compute_rank_percentile_scores(df, methods=methods)

        vina_rp = result[RP_SCORES_DICTIONARY["vina"]]
        vrb_rp = result[RP_SCORES_DICTIONARY["vina_rescore_boltz"]]
        grb_rp = result[RP_SCORES_DICTIONARY["gnina_rescore_boltz"]]
        ba_rp = result[RP_SCORES_DICTIONARY["boltz_affinity"]]

        # vina votes once, Boltz's 3 tracks average to one more vote (2 total).
        boltz_group = (vrb_rp + grb_rp + ba_rp) / 3
        expected_two_votes = (vina_rp + boltz_group) / 2

        # What it would be if boltz_affinity voted independently, flat, ungrouped.
        expected_four_independent_votes = (vina_rp + vrb_rp + grb_rp + ba_rp) / 4

        np.testing.assert_allclose(result[GLOBAL_RP_SCORE].values, expected_two_votes.values)
        assert not np.allclose(
            result[GLOBAL_RP_SCORE].values, expected_four_independent_votes.values
        )

    def test_missing_affinity_for_one_row_leaves_boltz_vote_intact(self):
        """A missing boltz_affinity_score is excluded, not imputed; the Boltz
        vote still comes from its two rescore siblings."""
        df = _make_df(
            protein_ids=["P1"] * 3,
            vina_scores=[-10.0, -8.0, -6.0],
            vina_rescore_boltz_scores=[-1.0, -5.0, -9.0],
            gnina_rescore_boltz_scores=[-2.0, -5.0, -8.0],
            boltz_affinity_scores=[np.nan, -4.0, -9.0],
        )
        methods = [
            "vina",
            "vina_rescore_boltz",
            "gnina_rescore_boltz",
            "boltz_affinity",
        ]
        result = compute_rank_percentile_scores(df, methods=methods)

        vina_rp = result[RP_SCORES_DICTIONARY["vina"]]
        vrb_rp = result[RP_SCORES_DICTIONARY["vina_rescore_boltz"]]
        grb_rp = result[RP_SCORES_DICTIONARY["gnina_rescore_boltz"]]
        ba_rp = result[RP_SCORES_DICTIONARY["boltz_affinity"]]

        assert np.isnan(ba_rp.iloc[0])  # premise: row 0's affinity head is missing

        expected_boltz_group_row0 = (vrb_rp.iloc[0] + grb_rp.iloc[0]) / 2
        expected_row0 = (vina_rp.iloc[0] + expected_boltz_group_row0) / 2
        assert result[GLOBAL_RP_SCORE].iloc[0] == pytest.approx(expected_row0)


# ---------------------------------------------------------------------------
# 14. Score plausibility
# ---------------------------------------------------------------------------
class TestIsPhysicalScore:
    def test_plausible_vina_energy_is_physical(self):
        assert is_physical_score(-9.5, "vina") is True

    def test_range_endpoints_are_inclusive(self):
        low, high = VINA_FAMILY_PLAUSIBLE_SCORE_RANGE
        assert is_physical_score(low, "vina") is True
        assert is_physical_score(high, "vina") is True

    def test_positive_vina_score_is_not_physical(self):
        assert is_physical_score(5.0, "vina") is False

    def test_absurdly_large_magnitude_vina_score_is_not_physical(self):
        assert is_physical_score(-43_851_078, "vina") is False

    def test_gnina_and_its_rescore_tracks_use_the_same_range(self):
        assert is_physical_score(-9.0, "gnina") is True
        assert is_physical_score(5.0, "gnina_rescore_diffdock") is False
        assert is_physical_score(-9.0, "vina_rescore_boltz") is True

    def test_maximising_method_treats_a_positive_score_as_physical(self):
        """karmadock_score is 'maximum'-direction — positive is correct, not suspicious."""
        assert is_physical_score(5.0, "karmadock") is True
        assert is_physical_score(1_000_000.0, "karmadock") is True

    def test_nesso_is_not_checked_against_the_vina_range(self):
        """Nesso is log10(IC50/uM), not a docking energy, so the vina range doesn't apply."""
        assert is_physical_score(5.0, "nesso") is True
        assert is_physical_score(-43_851_078, "nesso") is True

    def test_missing_score_is_treated_as_physical(self):
        """No score is a distinct, already-tracked failure mode, not a physicality one."""
        assert is_physical_score(np.nan, "vina") is True
        assert is_physical_score(None, "vina") is True


# ---------------------------------------------------------------------------
# 15. Regression: the grouping column (and row order) must survive
# ---------------------------------------------------------------------------
class TestGroupingColumnSurvives:
    """Regression: groupby(...).apply(...) used to drop protein_col on
    pandas 2.2+/3.x. Rows are interleaved out of alphabetical order here so a
    fix that re-attaches columns by position, not by index, would still fail."""

    def _interleaved_df(self):
        return pd.DataFrame(
            {
                PROTEIN_CONF_ID: ["P2", "P1", "P2", "P1"],
                "ligand_id": ["lig_p2_a", "lig_p1_a", "lig_p2_b", "lig_p1_b"],
                "vina_score": [-9.0, -10.0, -7.0, -8.0],
            }
        )

    def test_protein_col_survives(self):
        df = self._interleaved_df()
        result = compute_rank_percentile_scores(df, methods=["vina"])
        assert PROTEIN_CONF_ID in result.columns

    def test_row_count_is_unchanged(self):
        df = self._interleaved_df()
        result = compute_rank_percentile_scores(df, methods=["vina"])
        assert len(result) == len(df)

    def test_unrelated_column_and_protein_col_stay_aligned_to_their_own_row(self):
        """Still attached to the right row, in original order, not group-sorted order."""
        df = self._interleaved_df()
        result = compute_rank_percentile_scores(df, methods=["vina"])

        assert result["ligand_id"].tolist() == df["ligand_id"].tolist()
        assert result[PROTEIN_CONF_ID].tolist() == df[PROTEIN_CONF_ID].tolist()

    def test_computed_scores_stay_aligned_to_their_own_row(self):
        """The computed rp_score must land on its own row, not get shuffled by a positional re-attach."""
        df = self._interleaved_df()
        result = compute_rank_percentile_scores(df, methods=["vina"])

        rp_col = RP_SCORES_DICTIONARY["vina"]
        # Row order in: P2(-9, best of its pair), P1(-10, best), P2(-7, worst), P1(-8, worst).
        expected_rp = [0.5, 0.5, 1.0, 1.0]
        np.testing.assert_allclose(result[rp_col].values, expected_rp)
