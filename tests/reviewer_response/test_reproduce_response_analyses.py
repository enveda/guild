"""
Tests for direction_aware_auc (reproduce_response_analyses.py), the helper behind
r2_5_training_overlap. Getting a track's direction backwards silently inverts its AUC.

Imported via sys.path since the script runs standalone, with no __init__.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks" / "analysis" / "reviewer_response"))

from reproduce_response_analyses import (  # noqa: E402
    auc_lower_better,
    direction_aware_auc,
    jaccard_overlap,
    spearman_rho,
)


class TestDirectionAwareAuc:
    def test_minimum_direction_passes_through_auc_lower_better(self):
        actives = [1.0, 2.0, 3.0]
        decoys = [4.0, 5.0, 6.0]
        expected = auc_lower_better(actives, decoys)
        assert direction_aware_auc(actives, decoys, "minimum") == pytest.approx(expected)

    def test_maximum_direction_is_the_complement(self):
        # Same values, but under "maximum" the actives (the smaller numbers)
        # are the WORSE class, so the AUC should be low, not high.
        actives = [1.0, 2.0, 3.0]
        decoys = [4.0, 5.0, 6.0]
        minimum_auc = direction_aware_auc(actives, decoys, "minimum")
        maximum_auc = direction_aware_auc(actives, decoys, "maximum")
        assert maximum_auc == pytest.approx(1 - minimum_auc)

    def test_direction_flip_changes_which_side_looks_good(self):
        # Actives genuinely have the higher raw values (as for a real
        # "maximum" track that actually discriminates) -- getting the
        # direction backwards must not silently report this as a bad track.
        actives = [8.0, 9.0, 10.0]
        decoys = [1.0, 2.0, 3.0]
        assert direction_aware_auc(actives, decoys, "maximum") == pytest.approx(1.0)
        assert direction_aware_auc(actives, decoys, "minimum") == pytest.approx(0.0)

    def test_matches_the_verified_boltz_affinity_and_karmadock_rows(self):
        # Regression values from the 3-target rerun's 8GDC target (see HISTORY.rst):
        # karmadock_score is "maximum" and needs the 1 - auc correction, pinned here
        # so this helper can't silently reintroduce that sign error.
        boltz_affinity_active = [-2.844814, -2.536119]
        boltz_affinity_decoy = [-2.1, -1.8, -0.9, -0.5, -0.2]
        assert direction_aware_auc(
            boltz_affinity_active, boltz_affinity_decoy, "minimum"
        ) == pytest.approx(1.0)

        karmadock_active = [90.0, 85.0, 80.0]
        karmadock_decoy = [10.0, 20.0, 30.0, 40.0]
        # Actives genuinely score higher on this maximising track, so the
        # direction-correct AUC must be high, not its complement.
        assert direction_aware_auc(karmadock_active, karmadock_decoy, "maximum") == pytest.approx(1.0)
        assert direction_aware_auc(karmadock_active, karmadock_decoy, "minimum") == pytest.approx(0.0)

    def test_nan_handling_matches_auc_lower_better(self):
        actives = [1.0, np.nan, 3.0]
        decoys = [4.0, 5.0]
        expected = auc_lower_better(actives, decoys)
        assert direction_aware_auc(actives, decoys, "minimum") == pytest.approx(expected)

    def test_too_few_points_is_nan_regardless_of_direction(self):
        assert np.isnan(direction_aware_auc([1.0], [2.0, 3.0], "minimum"))
        assert np.isnan(direction_aware_auc([1.0], [2.0, 3.0], "maximum"))


class TestSpearmanRho:
    def test_perfect_agreement_is_one(self):
        assert spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_perfect_inversion_is_minus_one(self):
        assert spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_tied_values_use_average_rank_not_arbitrary_order(self):
        # x has a three-way tie in the middle, ranked [1, 3, 3, 3, 5] under
        # "average" rather than an arbitrary tie-break order; y is untied
        # [1, 2, 3, 4, 5]. Pearson correlation of those two rank series.
        x = [1, 5, 5, 5, 9]
        y = [1, 2, 3, 4, 5]
        assert spearman_rho(x, y) == pytest.approx(0.8944272, abs=1e-6)

    def test_a_constant_series_is_nan_not_a_zero_division(self):
        assert np.isnan(spearman_rho([1, 1, 1], [1, 2, 3]))

    def test_ranking_a_monotone_transform_gives_the_same_rho(self):
        # rp_vina_score is 1 - a monotone function of vina_score within a
        # target, so ranking either column must agree exactly.
        raw = [-9.0, -7.0, -5.0, -3.0]
        rp = [v / 4 for v in [1, 2, 3, 4]]  # already rank/N, same order as raw ascending
        assert spearman_rho(raw, rp) == pytest.approx(1.0)


class TestJaccardOverlap:
    def test_identical_sets_is_one(self):
        assert jaccard_overlap({"a", "b", "c"}, {"a", "b", "c"}) == pytest.approx(1.0)

    def test_disjoint_sets_is_zero(self):
        assert jaccard_overlap({"a", "b"}, {"c", "d"}) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # intersection {b, c} = 2, union {a, b, c, d} = 4 -> 0.5
        assert jaccard_overlap({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)

    def test_two_empty_sets_is_nan_not_a_division_by_zero(self):
        assert np.isnan(jaccard_overlap(set(), set()))
