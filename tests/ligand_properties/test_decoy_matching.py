"""
Tests for guild.tools.decoy_matching: property-matched decoy selection and
the descriptor-only baseline behind R2-7's requested Figure 3 control.
"""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from guild.tools.decoy_matching import (
    DEFAULT_MATCH_TOLERANCES,
    assign_matching_properties,
    combined_descriptor_auc,
    descriptor_only_auc,
    ecfp4_fingerprint,
    match_decoys_to_binders,
    max_tanimoto_similarity,
)

IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
# Property-matched to ibuprofen under DEFAULT_MATCH_TOLERANCES, ECFP4 Tanimoto
# 0.243 to it -- a genuinely distinct scaffold, not a close analogue.
PHENYLBUTYRIC_ACID = "OC(=O)CCCc1ccccc1"
# Ibuprofen's methyl ester: also property-matched, but ECFP4 Tanimoto 0.656 --
# a close structural analogue that should be rejected regardless of tolerance.
IBUPROFEN_METHYL_ESTER = "CC(C)Cc1ccc(cc1)C(C)C(=O)OC"
# Far outside DEFAULT_MATCH_TOLERANCES on every descriptor (much smaller,
# fewer HBA/HBD/rotatable bonds) and structurally unrelated.
BENZENE = "c1ccccc1"
NH4 = "[NH4+]"
ACETATE = "CC(=O)[O-]"


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------
class TestAssignMatchingProperties:
    def test_reuses_assign_properties_columns(self):
        df = pd.DataFrame({"id": ["ibuprofen"], "smiles": [IBUPROFEN]})
        result = assign_matching_properties(df)
        # molecular_weight/logp/n_hba/n_hbd come from assign_properties itself.
        for col in ["molecular_weight", "logp", "n_hba", "n_hbd", "scaffold_smiles"]:
            assert col in result.columns

    def test_adds_rotatable_bonds_matching_rdkit_directly(self):
        df = pd.DataFrame({"id": ["ibuprofen"], "smiles": [IBUPROFEN]})
        result = assign_matching_properties(df)
        expected = Descriptors.NumRotatableBonds(Chem.MolFromSmiles(IBUPROFEN))
        assert result["n_rotatable_bonds"].iloc[0] == expected

    def test_net_charge_of_charged_species(self):
        df = pd.DataFrame({"id": ["nh4", "acetate", "benzene"], "smiles": [NH4, ACETATE, BENZENE]})
        result = assign_matching_properties(df).set_index("id")
        assert result.loc["nh4", "net_charge"] == 1
        assert result.loc["acetate", "net_charge"] == -1
        assert result.loc["benzene", "net_charge"] == 0

    def test_invalid_smiles_gives_none_not_a_crash(self):
        df = pd.DataFrame({"id": ["bad"], "smiles": ["not a smiles"]})
        result = assign_matching_properties(df)
        assert result["n_rotatable_bonds"].iloc[0] is None
        assert result["net_charge"].iloc[0] is None


class TestFingerprints:
    def test_identical_molecule_has_similarity_one(self):
        fp = ecfp4_fingerprint(IBUPROFEN)
        assert max_tanimoto_similarity(fp, [ecfp4_fingerprint(IBUPROFEN)]) == pytest.approx(1.0)

    def test_close_analogue_scores_higher_than_distinct_scaffold(self):
        ibu_fp = ecfp4_fingerprint(IBUPROFEN)
        ester_similarity = max_tanimoto_similarity(ibu_fp, [ecfp4_fingerprint(IBUPROFEN_METHYL_ESTER)])
        distinct_similarity = max_tanimoto_similarity(ibu_fp, [ecfp4_fingerprint(PHENYLBUTYRIC_ACID)])
        assert ester_similarity > distinct_similarity
        assert ester_similarity >= 0.35
        assert distinct_similarity < 0.35

    def test_invalid_smiles_returns_none(self):
        assert ecfp4_fingerprint("not a smiles") is None

    def test_max_tanimoto_similarity_empty_reference_is_nan(self):
        assert np.isnan(max_tanimoto_similarity(ecfp4_fingerprint(IBUPROFEN), []))
        assert np.isnan(max_tanimoto_similarity(None, [ecfp4_fingerprint(IBUPROFEN)]))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
class TestMatchDecoysToBinders:
    """
    One target, one binder (ibuprofen), three decoy candidates exercising the
    three ways a decoy can be handled:

    - property-matched AND structurally distinct -> kept
    - property-matched BUT a close structural analogue -> rejected
    - not property-matched at all -> rejected regardless of structure
    """

    def _frames(self):
        binders = pd.DataFrame(
            {"protein_config_id": ["t1"], "ligand_id": ["ibuprofen"], "smiles": [IBUPROFEN]}
        )
        decoys = pd.DataFrame(
            {
                "protein_config_id": ["t1", "t1", "t1"],
                "ligand_id": ["distinct_match", "close_analogue", "size_mismatch"],
                "smiles": [PHENYLBUTYRIC_ACID, IBUPROFEN_METHYL_ESTER, BENZENE],
            }
        )
        return binders, decoys

    def test_keeps_only_the_property_matched_structurally_distinct_decoy(self):
        binders, decoys = self._frames()
        matched = match_decoys_to_binders(binders, decoys)
        assert list(matched["ligand_id"]) == ["distinct_match"]
        assert matched["matched_binder_id"].iloc[0] == "ibuprofen"

    def test_default_tolerances_and_threshold_are_the_documented_module_defaults(self):
        # Guards against a silent change to the shipped defaults.
        assert DEFAULT_MATCH_TOLERANCES["molecular_weight"] == 50.0
        assert DEFAULT_MATCH_TOLERANCES["net_charge"] == 0

    def test_no_matches_returns_empty_frame_with_expected_columns(self):
        binders = pd.DataFrame(
            {"protein_config_id": ["t1"], "ligand_id": ["ibuprofen"], "smiles": [IBUPROFEN]}
        )
        decoys = pd.DataFrame(
            {"protein_config_id": ["t1"], "ligand_id": ["benzene"], "smiles": [BENZENE]}
        )
        matched = match_decoys_to_binders(binders, decoys)
        assert len(matched) == 0
        assert "matched_binder_id" in matched.columns

    def test_decoys_for_a_different_target_are_never_matched(self):
        binders, decoys = self._frames()
        decoys = decoys.copy()
        decoys.loc[decoys["ligand_id"] == "distinct_match", "protein_config_id"] = "t2"
        matched = match_decoys_to_binders(binders, decoys)
        assert "distinct_match" not in list(matched["ligand_id"])

    def test_precomputed_descriptor_columns_are_not_recomputed(self):
        # If the caller already supplies MATCH_DESCRIPTORS, they are trusted
        # as-is rather than silently overwritten -- a wrong precomputed value
        # for the binder makes an otherwise-in-range decoy fail to match.
        binders, decoys = self._frames()
        computed = assign_matching_properties(binders[["ligand_id", "smiles"]]).rename(
            columns={"id": "ligand_id"}
        )
        binders = binders.merge(computed, on="ligand_id")
        binders["molecular_weight"] = 1.0  # implausible, forces every tolerance check to fail
        matched = match_decoys_to_binders(binders, decoys)
        assert len(matched) == 0


# ---------------------------------------------------------------------------
# Descriptor-only baseline
# ---------------------------------------------------------------------------
class TestDescriptorOnlyAuc:
    def test_detects_a_pure_size_confound(self):
        # Binders are all large, decoys all small, on molecular_weight alone --
        # exactly the heavy-atom-count-style confound R2-7 is checking for.
        large = ["CCCCCCCCCCCCCCCCCCCCC(=O)O"] * 20  # long fatty acid, MW ~ high
        small = ["CC(=O)O"] * 20  # acetic acid, MW ~ low
        binders = pd.DataFrame({"ligand_id": [f"b{i}" for i in range(20)], "smiles": large})
        decoys = pd.DataFrame({"ligand_id": [f"d{i}" for i in range(20)], "smiles": small})
        result = descriptor_only_auc(binders, decoys)
        mw_auc = result.loc[result["descriptor"] == "molecular_weight", "auc"].iloc[0]
        assert mw_auc > 0.95

    def test_no_separation_gives_auc_near_half(self):
        same_smiles = [IBUPROFEN] * 10
        binders = pd.DataFrame({"ligand_id": [f"b{i}" for i in range(10)], "smiles": same_smiles})
        decoys = pd.DataFrame({"ligand_id": [f"d{i}" for i in range(10)], "smiles": same_smiles})
        result = descriptor_only_auc(binders, decoys)
        # Identical descriptor values on both sides -- roc_auc_score's tie
        # handling gives exactly 0.5, not "near" it.
        assert (result["auc"] == 0.5).all()

    def test_returns_one_row_per_requested_descriptor(self):
        binders = pd.DataFrame({"ligand_id": ["b1"], "smiles": [IBUPROFEN]})
        decoys = pd.DataFrame({"ligand_id": ["d1"], "smiles": [BENZENE]})
        result = descriptor_only_auc(binders, decoys, descriptors=("molecular_weight", "logp"))
        assert list(result["descriptor"]) == ["molecular_weight", "logp"]


class TestCombinedDescriptorAuc:
    def test_returns_value_in_valid_range(self):
        large_like = ["CCCCCCCCCCCCCCCCCCCCC(=O)O", "CCCCCCCCCCCCCCCCCCC(=O)O"] * 10
        small_like = ["CC(=O)O", "CCC(=O)O"] * 10
        binders = pd.DataFrame({"ligand_id": [f"b{i}" for i in range(20)], "smiles": large_like})
        decoys = pd.DataFrame({"ligand_id": [f"d{i}" for i in range(20)], "smiles": small_like})
        auc = combined_descriptor_auc(binders, decoys, n_splits=4)
        assert 0.0 <= auc <= 1.0

    def test_too_few_rows_returns_nan(self):
        binders = pd.DataFrame({"ligand_id": ["b1"], "smiles": [IBUPROFEN]})
        decoys = pd.DataFrame({"ligand_id": ["d1"], "smiles": [BENZENE]})
        assert np.isnan(combined_descriptor_auc(binders, decoys, n_splits=5))
