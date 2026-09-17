"""
Tests for notebooks/analysis/reviewer_response/reproduce_response_analyses.py's
direction_aware_auc, the helper behind r2_5_training_overlap. Getting a
track's direction backwards silently inverts its AUC (1 - auc instead of
auc), which would invert R2-5's argument, so this is worth a real test rather
than only eyeballing the printed table.

Imported via sys.path, not a package -- this script is meant to run
standalone from notebooks/analysis/reviewer_response/, same as its sibling
scripts, so it has no __init__.py to import through normally.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks" / "analysis" / "reviewer_response"))

from reproduce_response_analyses import (  # noqa: E402
    auc_lower_better,
    direction_aware_auc,
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
        # Concrete regression values from the 3-target rerun's 8GDC target
        # (see HISTORY.rst / the reviewer_response README for the full
        # verification): boltz_affinity_score is "minimum" and matches a
        # hand-computed reference table directly; karmadock_score is
        # "maximum" and needed the 1 - auc correction to match sklearn's
        # roc_auc_score rather than that same reference table, which had it
        # backwards. Both are pinned here so a future change to this helper
        # cannot silently reintroduce that sign error.
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
