"""
E2E tests for PLIP and ProLIF interaction analysis.

Uses a minimal complex PDB fixture (tests/test_data/3pbl_lig_complex.pdb)
built from the 3PBL structure with the ETQ ligand renamed to LIG:Z:1, so no
docking run is required.  Each test calls the public analysis functions
directly and asserts on the DataFrame schema and interaction semantics.
"""

from pathlib import Path

import pandas as pd

from guild.analysis.plip import (
    analyze_batch_interactions,
    analyze_protein_ligand_interactions,
    get_batch_detailed_interactions,
    get_detailed_interactions,
)
from guild.analysis.prolif import (
    analyze_prolif_interactions,
    get_batch_prolif_interactions,
)
from guild.constants.interactions import (
    DETAIL_COLUMNS,
    DETAIL_DOCKING_METHOD,
    DETAIL_INTERACTION_TYPE,
    DETAIL_SOURCE,
    INTERACTION_COMBINATION_ID,
    INTERACTION_TYPE_HBOND,
    INTERACTION_TYPE_HYDROPHOBIC,
    N_HBONDS,
    N_HYDROPHOBIC,
    N_UNIQUE_RESIDUES,
    TOTAL_INTERACTIONS,
)

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
COMPLEX_PDB = str(TEST_DATA_DIR / "3pbl_lig_complex.pdb")

COMBO_ID = "3pbl-A_lig1"
PCONF_ID = "3pbl-A"
SMILES = "CC(=COC=O)CCC1=C(C)CCCC1(C)C"
METHOD = "vina"

VALID_INTERACTION_TYPES = {
    "hbond",
    "hydrophobic",
    "pistacking",
    "pication",
    "saltbridge",
    "halogen",
    "waterbridge",
    "metal",
}


# ---------------------------------------------------------------------------
# PLIP summary
# ---------------------------------------------------------------------------


class TestPlipSummary:
    def test_returns_dict_for_valid_complex(self):
        result = analyze_protein_ligand_interactions(COMPLEX_PDB)
        assert result is not None
        assert isinstance(result, dict)

    def test_detects_hydrophobic_contacts(self):
        result = analyze_protein_ligand_interactions(COMPLEX_PDB)
        assert result[N_HYDROPHOBIC] > 0

    def test_detects_hbond(self):
        result = analyze_protein_ligand_interactions(COMPLEX_PDB)
        assert result[N_HBONDS] > 0

    def test_total_interactions_consistent(self):
        result = analyze_protein_ligand_interactions(COMPLEX_PDB)
        counted = sum(
            result[k]
            for k in [
                "n_hbonds",
                "n_hydrophobic",
                "n_pistacking",
                "n_pication",
                "n_saltbridges",
                "n_halogen",
                "n_waterbridges",
                "n_metal",
            ]
        )
        assert result[TOTAL_INTERACTIONS] == counted

    def test_unique_residues_positive(self):
        result = analyze_protein_ligand_interactions(COMPLEX_PDB)
        assert result[N_UNIQUE_RESIDUES] > 0

    def test_returns_none_for_missing_file(self):
        result = analyze_protein_ligand_interactions("/nonexistent/path.pdb")
        assert result is None

    def test_batch_returns_dataframe_with_combination_id(self):
        df = analyze_batch_interactions(
            complex_pdb_paths=[COMPLEX_PDB],
            combination_ids=[COMBO_ID],
        )
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert INTERACTION_COMBINATION_ID in df.columns
        assert df[INTERACTION_COMBINATION_ID].iloc[0] == COMBO_ID

    def test_batch_empty_paths_returns_empty_df(self):
        df = analyze_batch_interactions(complex_pdb_paths=[], combination_ids=[])
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ---------------------------------------------------------------------------
# PLIP detail
# ---------------------------------------------------------------------------


class TestPlipDetail:
    def test_returns_nonempty_list(self):
        records = get_detailed_interactions(COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD)
        assert isinstance(records, list)
        assert len(records) > 0

    def test_all_records_have_required_keys(self):
        records = get_detailed_interactions(COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD)
        for rec in records:
            assert rec[DETAIL_SOURCE] == "plip"
            assert rec[DETAIL_DOCKING_METHOD] == METHOD
            assert rec[INTERACTION_COMBINATION_ID] == COMBO_ID
            assert rec[DETAIL_INTERACTION_TYPE] in VALID_INTERACTION_TYPES

    def test_hbond_record_has_distance_fields(self):
        records = get_detailed_interactions(COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD)
        hbonds = [r for r in records if r[DETAIL_INTERACTION_TYPE] == INTERACTION_TYPE_HBOND]
        assert len(hbonds) > 0
        for hb in hbonds:
            assert hb.get("distance_ah") is not None
            assert hb.get("distance_ad") is not None
            assert hb.get("angle") is not None

    def test_hydrophobic_record_has_distance(self):
        records = get_detailed_interactions(COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD)
        hydro = [r for r in records if r[DETAIL_INTERACTION_TYPE] == INTERACTION_TYPE_HYDROPHOBIC]
        assert len(hydro) > 0
        for h in hydro:
            assert h.get("distance") is not None

    def test_batch_returns_full_schema_dataframe(self):
        metadata = [(COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD)]
        df = get_batch_detailed_interactions(metadata)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        for col in DETAIL_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_batch_empty_input_returns_schema_df(self):
        df = get_batch_detailed_interactions([])
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == DETAIL_COLUMNS

    def test_returns_empty_list_for_missing_file(self):
        records = get_detailed_interactions(
            "/nonexistent/path.pdb", COMBO_ID, PCONF_ID, SMILES, METHOD
        )
        assert records == []


# ---------------------------------------------------------------------------
# ProLIF detail
# ---------------------------------------------------------------------------


class TestProlifDetail:
    def test_returns_nonempty_list(self):
        records = analyze_prolif_interactions(COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD)
        assert isinstance(records, list)
        assert len(records) > 0

    def test_all_records_have_required_keys(self):
        records = analyze_prolif_interactions(COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD)
        for rec in records:
            assert rec[DETAIL_SOURCE] == "prolif"
            assert rec[DETAIL_DOCKING_METHOD] == METHOD
            assert rec[INTERACTION_COMBINATION_ID] == COMBO_ID
            assert rec[DETAIL_INTERACTION_TYPE] in VALID_INTERACTION_TYPES

    def test_records_include_distance(self):
        records = analyze_prolif_interactions(COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD)
        with_dist = [r for r in records if r.get("distance") is not None]
        assert len(with_dist) > 0

    def test_batch_returns_full_schema_dataframe(self):
        metadata = [(COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD)]
        df = get_batch_prolif_interactions(metadata)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        for col in DETAIL_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_batch_empty_input_returns_schema_df(self):
        df = get_batch_prolif_interactions([])
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == DETAIL_COLUMNS

    def test_returns_empty_list_for_missing_file(self):
        records = analyze_prolif_interactions(
            "/nonexistent/path.pdb", COMBO_ID, PCONF_ID, SMILES, METHOD
        )
        assert records == []


# ---------------------------------------------------------------------------
# Cross-source consistency
# ---------------------------------------------------------------------------


class TestSourceConsistency:
    def test_plip_and_prolif_share_interaction_type_vocabulary(self):
        plip_records = get_detailed_interactions(COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD)
        prolif_records = analyze_prolif_interactions(
            COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD
        )

        plip_types = {r[DETAIL_INTERACTION_TYPE] for r in plip_records}
        prolif_types = {r[DETAIL_INTERACTION_TYPE] for r in prolif_records}

        assert plip_types <= VALID_INTERACTION_TYPES
        assert prolif_types <= VALID_INTERACTION_TYPES

    def test_plip_source_tag_is_plip(self):
        records = get_detailed_interactions(COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD)
        assert all(r[DETAIL_SOURCE] == "plip" for r in records)

    def test_prolif_source_tag_is_prolif(self):
        records = analyze_prolif_interactions(COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD)
        assert all(r[DETAIL_SOURCE] == "prolif" for r in records)
