"""
PoseBusters pose-validity tests.

Covers:
  * the constants contract — PoseBusters must never be registered as a scoring
    method, and the two check groups must partition PB_CHECK_COLUMNS
  * PoseBusters' hyphenated output column names normalising onto guild constants
  * bond orders coming from the combination's SMILES rather than PDB geometry —
    the single most breakable property of this feature
  * detection of a *partial* SMILES template match, which
    AssignBondOrdersFromTemplate accepts silently
  * multi-model PDBQT splitting and DiffDock confidence-ordered pose discovery
  * every degradation path producing a row rather than raising
  * pose_scope best / escalate / all
  * validate_batch_poses covering every input combination, in input order, so
    row identity never has to be recovered positionally
  * _collect_complex_metadata discovery
  * run_pose_validity_analysis honouring the "file always exists" contract and
    merging additively into guild_scores.txt
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from guild.analysis.posebusters import (
    _normalise_check_name,
    _pose_mols,
    validate_batch_poses,
    validate_pose,
)
from guild.bulk import _COMPLEX_PDB_FOLDER_BY_METHOD, BulkRun, _collect_complex_metadata
from guild.constants.bulk import (
    BATCH_FOLDER,
    COMBINATIONS_TABLE_KEY,
    RANKS_DICTIONARY,
    RP_SCORES_DICTIONARY,
    SMILES_NAMES_DICTIONARY_KEY,
)
from guild.constants.bulk import (
    COMBINATION_ID as BULK_COMBINATION_ID,
)
from guild.constants.guild import (
    ALL_AVAILABLE_METHODS,
    BOLTZ_PREFIX,
    DIFFDOCK_PREFIX,
    GNINA_PREFIX,
    KARMADOCK_PREFIX,
    LIGAND_ID,
    NESSO_PREFIX,
    PROTEIN_CONF_ID,
    SCORES_DICTIONARY,
    VINA_PREFIX,
)
from guild.constants.guild import (
    SMILES as SMILES_COLUMN,
)
from guild.constants.posebusters import (
    CHECK_INTERNAL_ENERGY,
    PB_CHECK_COLUMNS,
    PB_COMBINATION_ID,
    PB_DOCKING_METHOD,
    PB_ERROR,
    PB_FAILED_CHECKS,
    PB_INTERMOLECULAR_CHECK_COLUMNS,
    PB_INTERMOLECULAR_VALID,
    PB_INTRAMOLECULAR_CHECK_COLUMNS,
    PB_INTRAMOLECULAR_VALID,
    PB_N_CHECKS_FAILED,
    PB_POSE,
    PB_STATUS,
    PB_STATUS_LIGAND_BUILD_FAILED,
    PB_STATUS_MISSING_COMPLEX,
    PB_STATUS_NO_POSES,
    PB_STATUS_OK,
    PB_STATUS_PROTEIN_BUILD_FAILED,
    PB_STATUS_TEMPLATE_FALLBACK,
    PB_STATUS_UNAVAILABLE,
    PB_VALID,
    POSE_SCOPE_ALL,
    POSE_SCOPE_BEST,
    POSE_SCOPE_ESCALATE,
    POSEBUSTERS_COLUMNS,
    POSEBUSTERS_SUPPORTED_METHODS,
)
from guild.tools.pose_molecules import (
    mol_from_pdb_block,
    split_complex_records,
    split_pdbqt_models,
)

TEST_DIR = Path(__file__).parent.parent
TEST_DATA_DIR = TEST_DIR / "test_data"

COMPLEX_PDB = str(TEST_DATA_DIR / "3pbl_lig_complex.pdb")
COMBO_ID = "3pbl-A_lig1"
PCONF_ID = "3pbl-A"
# Eticlopride — the ligand actually present in the fixture, as resname LIG.
SMILES = "O=C(NCC1CCCN1CC)c1c(O)c(CC)cc(Cl)c1OC"
METHOD = VINA_PREFIX
METADATA = (COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD)

# The 22 column names posebusters' "dock" config actually emits, verbatim
# (hyphens included). Guards the normaliser against an upstream relabelling.
POSEBUSTERS_RAW_CHECK_NAMES = [
    "mol_pred_loaded",
    "mol_cond_loaded",
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "no_radicals",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness",
    "double_bond_flatness",
    "internal_energy",
    "protein-ligand_maximum_distance",
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
]


def _all_pass_frame(overrides=None):
    """A stand-in for PoseBusters' returned frame with every check passing."""
    row = dict.fromkeys(POSEBUSTERS_RAW_CHECK_NAMES, True)
    row.update(overrides or {})
    return pd.DataFrame([row])


def _fake_buster(frames):
    """
    A PoseBusters stand-in returning ``frames`` in order, then repeating the last.

    Returned object records every call so pose-scope tests can assert how many
    poses were actually validated.
    """

    class _Buster:
        def __init__(self):
            self.calls = []

        def bust(self, mol_pred=None, mol_true=None, mol_cond=None, full_report=False):
            index = min(len(self.calls), len(frames) - 1)
            self.calls.append(mol_pred)
            result = frames[index]
            if isinstance(result, Exception):
                raise result
            return result

    return _Buster()


def _ligand_block():
    ligand_lines, _ = split_complex_records(open(COMPLEX_PDB).read(), "LIG")
    return "".join(ligand_lines) + "END\n"


@pytest.fixture
def test_input_table():
    df = pd.read_csv(TEST_DATA_DIR / "bulk_dummy.csv")
    df["protein_path"] = str(TEST_DATA_DIR / df["protein_path"].iloc[0])
    return df


@pytest.fixture
def cleanup():
    yield
    test_project = Path.cwd() / "data" / "test-posebusters"
    if test_project.exists():
        shutil.rmtree(test_project, ignore_errors=True)


def _batch_dict_for(combinations_df, batch_folder, smiles_by_ligand=None):
    if smiles_by_ligand is None:
        smiles_by_ligand = {row[LIGAND_ID]: SMILES for _, row in combinations_df.iterrows()}
    return {
        BATCH_FOLDER: str(batch_folder),
        COMBINATIONS_TABLE_KEY: combinations_df,
        SMILES_NAMES_DICTIONARY_KEY: smiles_by_ligand,
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
class TestConstants:
    def test_posebusters_is_not_a_scoring_method(self):
        """
        PoseBusters is a validator, not a docking method: registering it would
        give it a rank / rank-percentile column and let it pollute the score
        aggregation.
        """
        for registry in (
            ALL_AVAILABLE_METHODS,
            SCORES_DICTIONARY,
            RANKS_DICTIONARY,
            RP_SCORES_DICTIONARY,
        ):
            assert "posebusters" not in registry

    def test_supported_methods_are_exactly_the_complex_pdb_emitters(self):
        assert set(POSEBUSTERS_SUPPORTED_METHODS) == set(_COMPLEX_PDB_FOLDER_BY_METHOD)

    def test_methods_without_a_pose_are_excluded(self):
        """karmadock writes no complex PDB; nesso produces no 3D output at all."""
        assert KARMADOCK_PREFIX not in POSEBUSTERS_SUPPORTED_METHODS
        assert NESSO_PREFIX not in POSEBUSTERS_SUPPORTED_METHODS

    def test_check_groups_partition_all_checks(self):
        assert PB_INTRAMOLECULAR_CHECK_COLUMNS + PB_INTERMOLECULAR_CHECK_COLUMNS == PB_CHECK_COLUMNS
        assert not set(PB_INTRAMOLECULAR_CHECK_COLUMNS) & set(PB_INTERMOLECULAR_CHECK_COLUMNS)
        assert len(set(PB_CHECK_COLUMNS)) == len(PB_CHECK_COLUMNS)

    def test_every_check_column_appears_in_the_schema(self):
        assert set(PB_CHECK_COLUMNS).issubset(POSEBUSTERS_COLUMNS)

    def test_normaliser_maps_every_real_posebusters_column(self):
        """
        Every column posebusters' "dock" config emits must land on a declared
        check. A name that does not is silently dropped from the verdict, which
        makes pb_valid more permissive — the dangerous direction.
        """
        normalised = {_normalise_check_name(name) for name in POSEBUSTERS_RAW_CHECK_NAMES}
        assert normalised == set(PB_CHECK_COLUMNS)

    def test_normaliser_folds_hyphens_and_spaces(self):
        assert (
            _normalise_check_name("non-aromatic_ring_non-flatness")
            == "non_aromatic_ring_non_flatness"
        )
        assert _normalise_check_name("MOL_PRED loaded") == "mol_pred_loaded"


# ---------------------------------------------------------------------------
# Ligand construction — bond orders
# ---------------------------------------------------------------------------
class TestLigandMol:
    def test_bond_orders_come_from_the_smiles_template(self):
        """
        The complex PDB has no ligand CONECT records, so bond orders must come
        from the SMILES. Without this the aromatic ring is perceived as single
        bonds and bond_lengths / bond_angles / aromatic_ring_flatness /
        internal_energy all judge the wrong molecule.
        """
        from rdkit import Chem

        mol, reason, used_fallback = mol_from_pdb_block(_ligand_block(), SMILES)
        assert mol is not None
        assert reason is None
        assert used_fallback is False

        aromatic_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
        assert aromatic_bonds > 0, "template bond orders were not applied"

        template = Chem.MolFromSmiles(SMILES)
        assert mol.GetNumHeavyAtoms() == template.GetNumHeavyAtoms()

    def test_partial_template_match_is_detected(self):
        """
        AssignBondOrdersFromTemplate does NOT require the template to describe
        the whole molecule — given a template matching only part of the pose it
        succeeds silently and leaves the rest geometry-inferred. That is
        indistinguishable from success at the call site, so the heavy-atom
        counts have to be compared explicitly.
        """
        mol, reason, used_fallback = mol_from_pdb_block(_ligand_block(), "CCO")
        assert mol is not None, "the pose is still usable, just weakened"
        assert used_fallback is True
        assert "heavy atoms" in reason

    def test_missing_smiles_falls_back_without_raising(self):
        mol, reason, used_fallback = mol_from_pdb_block(_ligand_block(), "")
        assert mol is not None
        assert used_fallback is True
        assert "template" in reason

    def test_unparseable_block_returns_none(self):
        mol, reason, used_fallback = mol_from_pdb_block("not a pdb\n", SMILES)
        assert mol is None
        assert reason

    def test_empty_block_returns_none(self):
        mol, reason, _ = mol_from_pdb_block("   \n", SMILES)
        assert mol is None
        assert "empty" in reason


# ---------------------------------------------------------------------------
# Pose discovery
# ---------------------------------------------------------------------------
class TestPoseSources:
    def _write_pdbqt(self, path, n_models):
        """A minimal multi-model PDBQT, score-sorted as Vina writes them."""
        lines = []
        for model in range(1, n_models + 1):
            lines.append(f"MODEL {model}\n")
            lines.append(f"REMARK VINA RESULT: -{10 - model}.0\n")
            # z-coordinate encodes the model so blocks are distinguishable.
            lines.append(
                f"ATOM      1  C   LIG Z   1       0.000   0.000  "
                f"{float(model):6.3f}  1.00  0.00     0.000 C \n"
            )
            lines.append("ENDMDL\n")
        path.write_text("".join(lines))

    def test_pdbqt_models_split_in_order(self, tmp_path):
        pdbqt = tmp_path / "combo.pdbqt"
        self._write_pdbqt(pdbqt, 3)
        blocks = split_pdbqt_models(str(pdbqt))
        assert len(blocks) == 3
        # Vina writes best-first, so block order is the ranking.
        assert "1.000" in blocks[0]
        assert "3.000" in blocks[2]

    def test_pdbqt_lines_are_truncated_for_the_pdb_parser(self, tmp_path):
        pdbqt = tmp_path / "combo.pdbqt"
        self._write_pdbqt(pdbqt, 1)
        block = split_pdbqt_models(str(pdbqt))[0]
        for line in block.splitlines():
            if line.startswith("ATOM"):
                assert len(line) <= 66

    def test_single_pose_pdbqt_without_model_records(self, tmp_path):
        pdbqt = tmp_path / "combo.pdbqt"
        pdbqt.write_text(
            "ATOM      1  C   LIG Z   1       0.000   0.000   0.000  1.00  0.00     0.000 C \n"
        )
        assert len(split_pdbqt_models(str(pdbqt))) == 1

    def test_boltz_falls_back_to_the_complex_pdb_ligand(self, tmp_path):
        """Boltz predicts one complex, so its best pose is its only pose."""
        poses = _pose_mols(COMPLEX_PDB, str(tmp_path), BOLTZ_PREFIX, COMBO_ID, SMILES, "LIG", 9)
        assert len(poses) == 1
        assert poses[0][0] is not None

    def test_missing_native_pose_file_falls_back_to_the_complex_pdb(self, tmp_path):
        """A pruned or partially-synced tree still yields the top pose."""
        poses = _pose_mols(COMPLEX_PDB, str(tmp_path), VINA_PREFIX, COMBO_ID, SMILES, "LIG", 9)
        assert len(poses) == 1

    def test_vina_pdbqt_supplies_every_pose(self, tmp_path):
        vina_dir = tmp_path / VINA_PREFIX
        vina_dir.mkdir()
        self._write_pdbqt(vina_dir / f"{COMBO_ID}.pdbqt", 4)
        poses = _pose_mols(COMPLEX_PDB, str(tmp_path), VINA_PREFIX, COMBO_ID, SMILES, "LIG", 9)
        assert len(poses) == 4

    def test_max_poses_caps_discovery(self, tmp_path):
        vina_dir = tmp_path / VINA_PREFIX
        vina_dir.mkdir()
        self._write_pdbqt(vina_dir / f"{COMBO_ID}.pdbqt", 8)
        poses = _pose_mols(COMPLEX_PDB, str(tmp_path), VINA_PREFIX, COMBO_ID, SMILES, "LIG", 3)
        assert len(poses) == 3

    def test_diffdock_poses_are_ordered_by_confidence_not_rank_string(self, tmp_path):
        """
        Ranking on the confidence in the filename avoids the lexicographic trap
        where "rank10" sorts before "rank2", and matches how
        generate_diffdock_complex_pdbs picks the pose that went into the
        complex PDB.
        """
        from rdkit import Chem

        results = tmp_path / DIFFDOCK_PREFIX / "results" / COMBO_ID
        results.mkdir(parents=True)
        # Deliberately adversarial: highest confidence has the largest rank
        # number, so a lexicographic or rank-based sort gets it wrong.
        for rank, confidence in ((1, -2.5), (2, -0.5), (10, 3.5)):
            mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            Chem.rdDistGeom.EmbedMolecule(mol, randomSeed=0xC0FFEE)
            with Chem.SDWriter(str(results / f"rank{rank}_confidence{confidence}.sdf")) as w:
                w.write(mol)

        with patch("guild.analysis.posebusters.mols_from_sdf") as mocked:
            mocked.side_effect = lambda path, smiles: [(path, None, False)]
            poses = _pose_mols(
                COMPLEX_PDB, str(tmp_path), DIFFDOCK_PREFIX, COMBO_ID, SMILES, "LIG", 9
            )
        order = [Path(p[0]).name for p in poses]
        assert order == [
            "rank10_confidence3.5.sdf",
            "rank2_confidence-0.5.sdf",
            "rank1_confidence-2.5.sdf",
        ]

    def test_sdf_poses_skip_the_smiles_template(self, tmp_path):
        """
        An SDF already carries bond orders, so those poses are immune to the
        template-mismatch weakening. Passing a deliberately wrong SMILES must
        not flag them.
        """
        from rdkit import Chem

        gnina_dir = tmp_path / GNINA_PREFIX
        gnina_dir.mkdir()
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
        Chem.rdDistGeom.EmbedMolecule(mol, randomSeed=0xC0FFEE)
        with Chem.SDWriter(str(gnina_dir / f"{COMBO_ID}.sdf")) as writer:
            writer.write(mol)

        poses = _pose_mols(COMPLEX_PDB, str(tmp_path), GNINA_PREFIX, COMBO_ID, "CCO", "LIG", 9)
        assert len(poses) == 1
        assert poses[0][0] is not None
        assert poses[0][2] is False, "SDF bond orders should not need a template"


# ---------------------------------------------------------------------------
# Degradation — every failure is a row, never an exception
# ---------------------------------------------------------------------------
class TestDegradation:
    def _only_row(self, summary_rows):
        assert len(summary_rows) == 1
        return summary_rows[0]

    def _assert_fails_closed(self, row):
        """A row that could not be evaluated must never survive df[df.pb_valid]."""
        assert row[PB_VALID] is False
        assert row[PB_INTRAMOLECULAR_VALID] is False
        assert row[PB_INTERMOLECULAR_VALID] is False
        for column in PB_CHECK_COLUMNS:
            assert row.get(column) is None, f"{column} must not read as passed"

    def test_missing_complex_pdb(self, tmp_path):
        summary, report = validate_pose(
            "/nonexistent/x_complex.pdb", COMBO_ID, PCONF_ID, SMILES, METHOD, str(tmp_path)
        )
        row = self._only_row(summary)
        assert row[PB_STATUS] == PB_STATUS_MISSING_COMPLEX
        self._assert_fails_closed(row)
        assert report == []

    def test_unparseable_complex_pdb(self, tmp_path):
        bad = tmp_path / "bad_complex.pdb"
        bad.write_text("not a pdb at all\n")
        summary, _ = validate_pose(str(bad), COMBO_ID, PCONF_ID, SMILES, METHOD, str(tmp_path))
        row = self._only_row(summary)
        assert row[PB_STATUS] == PB_STATUS_NO_POSES
        self._assert_fails_closed(row)

    def test_protein_only_complex_has_no_pose(self, tmp_path):
        protein_only = tmp_path / "prot_complex.pdb"
        protein_only.write_text("".join(split_complex_records(open(COMPLEX_PDB).read(), "LIG")[1]))
        summary, _ = validate_pose(
            str(protein_only), COMBO_ID, PCONF_ID, SMILES, METHOD, str(tmp_path)
        )
        assert self._only_row(summary)[PB_STATUS] == PB_STATUS_NO_POSES

    def test_ligand_only_complex_has_no_receptor(self, tmp_path):
        ligand_only = tmp_path / "lig_complex.pdb"
        ligand_only.write_text(_ligand_block())
        summary, _ = validate_pose(
            str(ligand_only), COMBO_ID, PCONF_ID, SMILES, METHOD, str(tmp_path)
        )
        row = self._only_row(summary)
        assert row[PB_STATUS] == PB_STATUS_PROTEIN_BUILD_FAILED
        self._assert_fails_closed(row)

    def test_posebusters_raising_is_caught(self, tmp_path):
        """
        PoseBusters raises on a bad molecule rather than recording a failure row
        (an unusable pose surfaces as ValueError: Bad Conformer Id), so the call
        must be wrapped.
        """
        buster = _fake_buster([RuntimeError("boom")])
        summary, report = validate_pose(
            COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD, str(tmp_path), buster=buster
        )
        row = self._only_row(summary)
        assert row[PB_STATUS] == "posebusters_raised"
        assert "RuntimeError: boom" in row["posebusters_error"]
        self._assert_fails_closed(row)
        assert report == []

    def test_empty_frame_from_posebusters_is_caught(self, tmp_path):
        buster = _fake_buster([pd.DataFrame()])
        summary, _ = validate_pose(
            COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD, str(tmp_path), buster=buster
        )
        assert self._only_row(summary)[PB_STATUS] == "posebusters_raised"

    def test_unbuildable_ligand_is_reported_per_pose(self, tmp_path):
        with patch("guild.analysis.posebusters._pose_mols") as mocked:
            mocked.return_value = [(None, "synthetic failure", False)]
            summary, _ = validate_pose(
                COMPLEX_PDB,
                COMBO_ID,
                PCONF_ID,
                SMILES,
                METHOD,
                str(tmp_path),
                buster=_fake_buster([_all_pass_frame()]),
            )
        row = self._only_row(summary)
        assert row[PB_STATUS] == PB_STATUS_LIGAND_BUILD_FAILED
        self._assert_fails_closed(row)

    def test_missing_posebusters_degrades_to_a_row(self, tmp_path, caplog):
        """
        The step is on by default, so an image built before the dependency was
        added must warn rather than break every run.
        """
        import builtins

        real_import = builtins.__import__

        def _fail_posebusters(name, *args, **kwargs):
            if name == "posebusters":
                raise ImportError("No module named 'posebusters'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", _fail_posebusters):
            summary, report = validate_pose(
                COMPLEX_PDB, COMBO_ID, PCONF_ID, SMILES, METHOD, str(tmp_path)
            )
        row = self._only_row(summary)
        assert row[PB_STATUS] == PB_STATUS_UNAVAILABLE
        assert "ImportError" in row["posebusters_error"]
        self._assert_fails_closed(row)
        assert report == []


# ---------------------------------------------------------------------------
# Verdict aggregation
# ---------------------------------------------------------------------------
class TestVerdictAggregation:
    def _row(self, tmp_path, overrides=None, frames=None, config="dock"):
        buster = _fake_buster(frames or [_all_pass_frame(overrides)])
        summary, _ = validate_pose(
            COMPLEX_PDB,
            COMBO_ID,
            PCONF_ID,
            SMILES,
            METHOD,
            str(tmp_path),
            config=config,
            buster=buster,
        )
        return summary[0]

    def test_all_checks_passing(self, tmp_path):
        row = self._row(tmp_path)
        assert row[PB_VALID] is True
        assert row[PB_STATUS] == PB_STATUS_OK
        assert row[PB_N_CHECKS_FAILED] == 0
        assert row[PB_FAILED_CHECKS] == ""

    def test_intramolecular_failure_isolates_to_its_group(self, tmp_path):
        row = self._row(tmp_path, {"bond_angles": False})
        assert row[PB_INTRAMOLECULAR_VALID] is False
        assert row[PB_INTERMOLECULAR_VALID] is True
        assert row[PB_VALID] is False
        assert row[PB_FAILED_CHECKS] == "bond_angles"
        assert row[PB_N_CHECKS_FAILED] == 1

    def test_intermolecular_failure_isolates_to_its_group(self, tmp_path):
        row = self._row(tmp_path, {"minimum_distance_to_protein": False})
        assert row[PB_INTRAMOLECULAR_VALID] is True
        assert row[PB_INTERMOLECULAR_VALID] is False
        assert row[PB_VALID] is False
        assert row[PB_FAILED_CHECKS] == "minimum_distance_to_protein"

    def test_failed_checks_are_sorted_and_semicolon_joined(self, tmp_path):
        row = self._row(tmp_path, {"volume_overlap_with_protein": False, "bond_lengths": False})
        assert row[PB_FAILED_CHECKS] == "bond_lengths;volume_overlap_with_protein"
        assert row[PB_N_CHECKS_FAILED] == 2

    def test_none_valued_check_counts_as_not_passed(self, tmp_path):
        """
        PoseBusters returns None for the intermolecular checks when the receptor
        fails to load. Nothing about placement was verified, so the pose must
        not be reported valid.
        """
        row = self._row(tmp_path, {"minimum_distance_to_protein": None})
        assert row[PB_VALID] is False
        assert "minimum_distance_to_protein" in row[PB_FAILED_CHECKS]

    def test_check_absent_from_the_config_is_not_counted_against_the_pose(self, tmp_path, caplog):
        """
        dock_fast legitimately omits internal_energy, so its absence must not
        count against the pose — and must not trigger the renamed-check warning.
        """
        frame = _all_pass_frame().drop(columns=[CHECK_INTERNAL_ENERGY])
        with caplog.at_level("WARNING", logger="guild.analysis.posebusters"):
            row = self._row(tmp_path, frames=[frame], config="dock_fast")

        assert row[PB_VALID] is True
        assert row[PB_N_CHECKS_FAILED] == 0
        # Absent at the record level; _frame_with_schema fills it as NA so the
        # TSV keeps a stable schema across configs.
        assert CHECK_INTERNAL_ENERGY not in row
        assert "did not emit these declared checks" not in caplog.text

    def test_missing_check_warns_under_a_config_that_should_emit_it(self, tmp_path, caplog):
        """
        The upgrade-safety valve. A renamed check silently drops out of the
        verdict, making pb_valid *more* permissive — the dangerous direction —
        so its absence has to be loud.
        """
        frame = _all_pass_frame().drop(columns=["bond_angles"])
        with caplog.at_level("WARNING", logger="guild.analysis.posebusters"):
            self._row(tmp_path, frames=[frame], config="dock")
        assert "did not emit these declared checks" in caplog.text
        assert "bond_angles" in caplog.text

    def test_absent_check_becomes_na_in_the_frame(self, tmp_path):
        """The declared schema is stable even when a config omits a check."""
        pytest.importorskip("posebusters")
        frame = _all_pass_frame().drop(columns=[CHECK_INTERNAL_ENERGY])
        buster = _fake_buster([frame])
        # PoseBusters is imported inside the functions (the graceful-degradation
        # guard), so there is no module attribute to patch — patch the package.
        with patch("posebusters.PoseBusters", return_value=buster):
            summary, _ = validate_batch_poses(
                [METADATA], batch_folder=str(tmp_path), config="dock_fast"
            )
        assert list(summary.columns) == POSEBUSTERS_COLUMNS
        assert pd.isna(summary.iloc[0][CHECK_INTERNAL_ENERGY])

    def test_template_fallback_gets_its_own_status(self, tmp_path):
        """
        The fallback keeps intermolecular checks fully valid but leaves the
        intramolecular ones on inferred bonds, so it must be visible in the
        table rather than only in the log.
        """
        buster = _fake_buster([_all_pass_frame()])
        summary, _ = validate_pose(
            COMPLEX_PDB, COMBO_ID, PCONF_ID, "CCO", METHOD, str(tmp_path), buster=buster
        )
        assert summary[0][PB_STATUS] == PB_STATUS_TEMPLATE_FALLBACK


# ---------------------------------------------------------------------------
# Pose scope
# ---------------------------------------------------------------------------
class TestPoseScope:
    def _vina_batch(self, tmp_path, n_models):
        vina_dir = tmp_path / VINA_PREFIX
        vina_dir.mkdir(exist_ok=True)
        lines = []
        for model in range(1, n_models + 1):
            lines.append(f"MODEL {model}\n")
            lines.append(
                f"ATOM      1  C   LIG Z   1       0.000   0.000  "
                f"{float(model):6.3f}  1.00  0.00     0.000 C \n"
            )
            lines.append("ENDMDL\n")
        (vina_dir / f"{COMBO_ID}.pdbqt").write_text("".join(lines))

    def _run(self, tmp_path, scope, frames, n_models=5):
        self._vina_batch(tmp_path, n_models)
        buster = _fake_buster(frames)
        summary, _ = validate_pose(
            COMPLEX_PDB,
            COMBO_ID,
            PCONF_ID,
            SMILES,
            METHOD,
            str(tmp_path),
            pose_scope=scope,
            buster=buster,
        )
        return summary, buster

    def test_best_checks_exactly_one_pose(self, tmp_path):
        summary, buster = self._run(
            tmp_path, POSE_SCOPE_BEST, [_all_pass_frame({"bond_angles": False})]
        )
        assert len(buster.calls) == 1
        assert len(summary) == 1
        assert summary[0][PB_POSE] == 1

    def test_escalate_stops_at_the_first_valid_pose(self, tmp_path):
        """The question is whether a valid pose exists, so stop once one does."""
        frames = [
            _all_pass_frame({"bond_angles": False}),
            _all_pass_frame({"bond_angles": False}),
            _all_pass_frame(),
            _all_pass_frame(),
        ]
        summary, buster = self._run(tmp_path, POSE_SCOPE_ESCALATE, frames)
        assert len(buster.calls) == 3
        assert [row[PB_POSE] for row in summary] == [1, 2, 3]
        assert [row[PB_VALID] for row in summary] == [False, False, True]

    def test_escalate_stops_immediately_when_the_best_pose_passes(self, tmp_path):
        summary, buster = self._run(tmp_path, POSE_SCOPE_ESCALATE, [_all_pass_frame()])
        assert len(buster.calls) == 1
        assert summary[0][PB_VALID] is True

    def test_escalate_over_all_failing_poses_reports_each(self, tmp_path):
        summary, buster = self._run(
            tmp_path, POSE_SCOPE_ESCALATE, [_all_pass_frame({"bond_angles": False})], n_models=4
        )
        assert len(buster.calls) == 4
        assert len(summary) == 4
        assert not any(row[PB_VALID] for row in summary)

    def test_all_checks_every_pose_even_after_one_passes(self, tmp_path):
        summary, buster = self._run(tmp_path, POSE_SCOPE_ALL, [_all_pass_frame()], n_models=4)
        assert len(buster.calls) == 4
        assert len(summary) == 4

    def test_max_poses_bounds_the_worst_case(self, tmp_path):
        """
        A check that fails on every pose would otherwise make escalate cost the
        full pose set for every combination.
        """
        self._vina_batch(tmp_path, 9)
        buster = _fake_buster([_all_pass_frame({"bond_angles": False})])
        summary, _ = validate_pose(
            COMPLEX_PDB,
            COMBO_ID,
            PCONF_ID,
            SMILES,
            METHOD,
            str(tmp_path),
            pose_scope=POSE_SCOPE_ESCALATE,
            max_poses=2,
            buster=buster,
        )
        assert len(buster.calls) == 2
        assert len(summary) == 2


# ---------------------------------------------------------------------------
# Batch aggregation
# ---------------------------------------------------------------------------
class TestValidateBatchPoses:
    def test_empty_input_yields_header_only_frames(self):
        summary, report = validate_batch_poses([], batch_folder="/nowhere")
        assert summary.empty
        assert list(summary.columns) == POSEBUSTERS_COLUMNS
        assert report.empty

    def test_summary_columns_are_exactly_the_declared_schema(self, tmp_path):
        summary, _ = validate_batch_poses([METADATA], batch_folder=str(tmp_path))
        assert list(summary.columns) == POSEBUSTERS_COLUMNS

    def test_every_input_is_represented_in_input_order(self, tmp_path):
        """
        Identity travels inside each row, so it never has to be recovered by
        position — the failure mode that makes a positional column graft
        misalign as soon as the analyser drops a row.
        """
        mixed = [
            METADATA,
            ("/nonexistent/x_complex.pdb", "missing-one", "p", SMILES, VINA_PREFIX),
            (COMPLEX_PDB, "third-one", "p3", SMILES, GNINA_PREFIX),
        ]
        summary, _ = validate_batch_poses(mixed, batch_folder=str(tmp_path))
        assert list(summary[PB_COMBINATION_ID]) == [COMBO_ID, "missing-one", "third-one"]
        assert list(summary[PB_DOCKING_METHOD]) == [VINA_PREFIX, VINA_PREFIX, GNINA_PREFIX]

    def test_mixed_valid_and_broken_input_does_not_raise(self, tmp_path):
        mixed = [METADATA, ("/nonexistent/x_complex.pdb", "gone", "p", SMILES, VINA_PREFIX)]
        summary, _ = validate_batch_poses(mixed, batch_folder=str(tmp_path))
        statuses = set(summary[PB_STATUS])
        assert PB_STATUS_OK in statuses
        assert PB_STATUS_MISSING_COMPLEX in statuses

    def test_multiprocessing_path_matches_the_serial_path(self, tmp_path):
        """
        n_processes defaults to BulkRun.n_workers, so the pool path is what runs
        in production whenever N_WORKERS > 1 — it has to produce the same rows,
        and the tasks have to stay picklable.
        """
        pytest.importorskip("posebusters")
        inputs = [
            METADATA,
            (COMPLEX_PDB, "second", "p2", SMILES, GNINA_PREFIX),
            ("/nonexistent/x_complex.pdb", "gone", "p3", SMILES, VINA_PREFIX),
        ]
        serial, _ = validate_batch_poses(inputs, batch_folder=str(tmp_path), n_processes=1)
        parallel, _ = validate_batch_poses(inputs, batch_folder=str(tmp_path), n_processes=3)

        assert list(parallel[PB_COMBINATION_ID]) == list(serial[PB_COMBINATION_ID])
        assert list(parallel[PB_STATUS]) == list(serial[PB_STATUS])
        assert list(parallel[PB_VALID]) == list(serial[PB_VALID])

    def test_report_carries_the_identity_columns(self, tmp_path):
        _, report = validate_batch_poses([METADATA], batch_folder=str(tmp_path))
        assert not report.empty
        for column in (PB_COMBINATION_ID, PROTEIN_CONF_ID, PB_DOCKING_METHOD, PB_POSE):
            assert column in report.columns
        # Deliberately open-ended: a posebusters upgrade that adds a
        # measurement should widen the file, not trip an assertion.
        assert report.shape[1] > len(POSEBUSTERS_COLUMNS)


# ---------------------------------------------------------------------------
# Complex discovery
# ---------------------------------------------------------------------------
class TestCollectComplexMetadata:
    def _combinations(self):
        return pd.DataFrame([{PROTEIN_CONF_ID: "P1", LIGAND_ID: "lig1"}])

    def test_finds_only_methods_with_an_existing_file(self, tmp_path):
        (tmp_path / VINA_PREFIX).mkdir()
        (tmp_path / VINA_PREFIX / "P1_lig1_complex.pdb").write_text("x")
        (tmp_path / GNINA_PREFIX).mkdir()  # folder present, file absent

        collected = _collect_complex_metadata(
            _batch_dict_for(self._combinations(), tmp_path), [VINA_PREFIX, GNINA_PREFIX]
        )
        assert set(collected) == {VINA_PREFIX}
        assert collected[VINA_PREFIX] == [
            (
                f"{tmp_path}/{VINA_PREFIX}/P1_lig1_complex.pdb",
                "P1_lig1",
                "P1",
                SMILES,
                VINA_PREFIX,
            )
        ]

    def test_respects_methods_to_analyze(self, tmp_path):
        for method in (VINA_PREFIX, GNINA_PREFIX):
            (tmp_path / method).mkdir()
            (tmp_path / method / "P1_lig1_complex.pdb").write_text("x")

        collected = _collect_complex_metadata(
            _batch_dict_for(self._combinations(), tmp_path), [VINA_PREFIX]
        )
        assert set(collected) == {VINA_PREFIX}

    def test_methods_without_complex_pdbs_never_appear(self, tmp_path):
        collected = _collect_complex_metadata(
            _batch_dict_for(self._combinations(), tmp_path),
            [KARMADOCK_PREFIX, NESSO_PREFIX],
        )
        assert collected == {}

    def test_smiles_comes_from_the_batch_dictionary(self, tmp_path):
        """
        A wrong SMILES silently degrades bond orders rather than erroring, so
        the lookup is worth pinning.
        """
        (tmp_path / VINA_PREFIX).mkdir()
        (tmp_path / VINA_PREFIX / "P1_lig1_complex.pdb").write_text("x")

        batch = _batch_dict_for(
            self._combinations(), tmp_path, smiles_by_ligand={"lig1": "c1ccccc1"}
        )
        collected = _collect_complex_metadata(batch, [VINA_PREFIX])
        assert collected[VINA_PREFIX][0][3] == "c1ccccc1"

    def test_empty_when_nothing_was_produced(self, tmp_path):
        collected = _collect_complex_metadata(
            _batch_dict_for(self._combinations(), tmp_path), POSEBUSTERS_SUPPORTED_METHODS
        )
        assert collected == {}


# ---------------------------------------------------------------------------
# BulkRun orchestration
# ---------------------------------------------------------------------------
class TestRunPoseValidityAnalysis:
    def _bulk(self, test_input_table):
        return BulkRun(
            input_table=test_input_table,
            project_name="test-posebusters",
            methods_to_run=[VINA_PREFIX],
            use_decoys=False,
            use_known_binders=False,
            use_gpu=False,
            n_workers=1,
        )

    def test_header_only_tsvs_written_when_nothing_was_produced(self, test_input_table, cleanup):
        """
        The contract is that both files exist after every run so downstream
        pd.read_csv is unconditional.
        """
        bulk = self._bulk(test_input_table)
        result = bulk.run_pose_validity_analysis()

        assert result is not None, "must return an empty frame, not None"
        assert result.empty

        for path, expected in (
            (bulk.posebusters_path, POSEBUSTERS_COLUMNS),
            (bulk.posebusters_report_path, None),
        ):
            assert Path(path).exists()
            frame = pd.read_csv(path, sep="\t")
            assert len(frame) == 0
            if expected is not None:
                assert list(frame.columns) == expected

    def test_rows_written_and_attribute_set(self, test_input_table, cleanup):
        bulk = self._bulk(test_input_table)
        fake_summary = pd.DataFrame(
            [
                {
                    **dict.fromkeys(POSEBUSTERS_COLUMNS),
                    PB_COMBINATION_ID: COMBO_ID,
                    PROTEIN_CONF_ID: PCONF_ID,
                    SMILES_COLUMN: SMILES,
                    PB_DOCKING_METHOD: VINA_PREFIX,
                    PB_POSE: 1,
                    PB_VALID: True,
                    PB_STATUS: PB_STATUS_OK,
                    PB_FAILED_CHECKS: "",
                }
            ]
        )[POSEBUSTERS_COLUMNS]

        with (
            patch("guild.bulk._collect_complex_metadata") as collect,
            patch("guild.bulk.validate_batch_poses") as validate,
        ):
            collect.return_value = {VINA_PREFIX: [METADATA]}
            validate.return_value = (fake_summary, pd.DataFrame([{"x": 1}]))
            result = bulk.run_pose_validity_analysis()

        assert len(result) == 1
        assert bulk.posebusters_df.equals(result)
        written = pd.read_csv(bulk.posebusters_path, sep="\t")
        assert len(written) == 1
        assert written.iloc[0][PB_COMBINATION_ID] == COMBO_ID

    def test_progress_logged_to_the_batch_log(self, test_input_table, cleanup):
        bulk = self._bulk(test_input_table)
        bulk.run_pose_validity_analysis()

        for batch in bulk.batched_dictionary.values():
            log = Path(batch[BATCH_FOLDER]) / "output.log"
            assert log.exists()
            text = log.read_text()
            assert "Starting PoseBusters analysis" in text
            assert "Completed PoseBusters analysis" in text

    def test_scope_and_config_are_forwarded(self, test_input_table, cleanup):
        bulk = self._bulk(test_input_table)
        with (
            patch("guild.bulk._collect_complex_metadata") as collect,
            patch("guild.bulk.validate_batch_poses") as validate,
        ):
            collect.return_value = {VINA_PREFIX: [METADATA]}
            validate.return_value = (
                pd.DataFrame(columns=POSEBUSTERS_COLUMNS),
                pd.DataFrame(),
            )
            bulk.run_pose_validity_analysis(config="dock_fast", pose_scope=POSE_SCOPE_ALL)

        kwargs = validate.call_args.kwargs
        assert kwargs["config"] == "dock_fast"
        assert kwargs["pose_scope"] == POSE_SCOPE_ALL


# ---------------------------------------------------------------------------
# guild_scores.txt merge
# ---------------------------------------------------------------------------
class TestScoresMerge:
    def _bulk_with_scores(self, test_input_table, scores_df):
        bulk = BulkRun(
            input_table=test_input_table,
            project_name="test-posebusters",
            methods_to_run=[VINA_PREFIX],
            use_decoys=False,
            use_known_binders=False,
            use_gpu=False,
            n_workers=1,
        )
        bulk.rp_scores_df = scores_df
        return bulk

    def _summary(self, rows):
        frame = pd.DataFrame([{**dict.fromkeys(POSEBUSTERS_COLUMNS), **row} for row in rows])
        return frame[POSEBUSTERS_COLUMNS]

    @staticmethod
    def _flag(frame, row, column):
        """
        Read a flag cell as a Python bool.

        pandas stores an all-present boolean column as numpy bool_ but falls
        back to object dtype (Python bools) as soon as one value is NA, so
        identity comparison is not stable across these cases.
        """
        return bool(frame.loc[row, column])

    def test_adds_flag_columns_without_changing_rows(self, test_input_table, cleanup):
        scores = pd.DataFrame({BULK_COMBINATION_ID: ["c1", "c2"], "vina_score": [-9.0, -8.0]})
        bulk = self._bulk_with_scores(test_input_table, scores)
        bulk.posebusters_df = self._summary(
            [
                {
                    PB_COMBINATION_ID: "c1",
                    PB_DOCKING_METHOD: VINA_PREFIX,
                    PB_POSE: 1,
                    PB_VALID: True,
                },
                {
                    PB_COMBINATION_ID: "c2",
                    PB_DOCKING_METHOD: VINA_PREFIX,
                    PB_POSE: 1,
                    PB_VALID: False,
                },
            ]
        )
        bulk._merge_posebusters_into_scores()

        merged = bulk.rp_scores_df
        assert len(merged) == 2, "flag-only: no row may be dropped"
        assert list(merged["vina_score"]) == [-9.0, -8.0], "existing columns untouched"
        assert list(merged["vina_pb_valid"]) == [True, False]
        assert merged.loc[0, "vina_pb_pose"] == 1

    def test_combination_without_a_result_is_na_not_false(self, test_input_table, cleanup):
        """Absence of evidence is not invalidity."""
        scores = pd.DataFrame(
            {BULK_COMBINATION_ID: ["c1", "unchecked"], "vina_score": [-9.0, -8.0]}
        )
        bulk = self._bulk_with_scores(test_input_table, scores)
        bulk.posebusters_df = self._summary(
            [{PB_COMBINATION_ID: "c1", PB_DOCKING_METHOD: VINA_PREFIX, PB_POSE: 1, PB_VALID: True}]
        )
        bulk._merge_posebusters_into_scores()

        merged = bulk.rp_scores_df.set_index(BULK_COMBINATION_ID)
        assert bool(merged.loc["c1", "vina_pb_valid"]) is True
        assert pd.isna(merged.loc["unchecked", "vina_pb_valid"])

    def test_first_valid_pose_is_the_lowest_ranked_passing_pose(self, test_input_table, cleanup):
        scores = pd.DataFrame({BULK_COMBINATION_ID: ["c1"], "vina_score": [-9.0]})
        bulk = self._bulk_with_scores(test_input_table, scores)
        bulk.posebusters_df = self._summary(
            [
                {
                    PB_COMBINATION_ID: "c1",
                    PB_DOCKING_METHOD: VINA_PREFIX,
                    PB_POSE: 1,
                    PB_VALID: False,
                },
                {
                    PB_COMBINATION_ID: "c1",
                    PB_DOCKING_METHOD: VINA_PREFIX,
                    PB_POSE: 2,
                    PB_VALID: False,
                },
                {
                    PB_COMBINATION_ID: "c1",
                    PB_DOCKING_METHOD: VINA_PREFIX,
                    PB_POSE: 3,
                    PB_VALID: True,
                },
            ]
        )
        bulk._merge_posebusters_into_scores()

        merged = bulk.rp_scores_df
        assert self._flag(merged, 0, "vina_pb_valid") is True
        assert merged.loc[0, "vina_pb_pose"] == 3

    def test_pose_column_is_na_when_no_pose_passed(self, test_input_table, cleanup):
        scores = pd.DataFrame({BULK_COMBINATION_ID: ["c1"], "vina_score": [-9.0]})
        bulk = self._bulk_with_scores(test_input_table, scores)
        bulk.posebusters_df = self._summary(
            [{PB_COMBINATION_ID: "c1", PB_DOCKING_METHOD: VINA_PREFIX, PB_POSE: 1, PB_VALID: False}]
        )
        bulk._merge_posebusters_into_scores()
        assert self._flag(bulk.rp_scores_df, 0, "vina_pb_valid") is False
        assert pd.isna(bulk.rp_scores_df.loc[0, "vina_pb_pose"])

    def test_one_column_pair_per_method(self, test_input_table, cleanup):
        scores = pd.DataFrame({BULK_COMBINATION_ID: ["c1"], "vina_score": [-9.0]})
        bulk = self._bulk_with_scores(test_input_table, scores)
        bulk.posebusters_df = self._summary(
            [
                {
                    PB_COMBINATION_ID: "c1",
                    PB_DOCKING_METHOD: VINA_PREFIX,
                    PB_POSE: 1,
                    PB_VALID: True,
                },
                {
                    PB_COMBINATION_ID: "c1",
                    PB_DOCKING_METHOD: BOLTZ_PREFIX,
                    PB_POSE: 1,
                    PB_VALID: False,
                },
            ]
        )
        bulk._merge_posebusters_into_scores()

        merged = bulk.rp_scores_df
        assert self._flag(merged, 0, "vina_pb_valid") is True
        assert self._flag(merged, 0, "boltz_pb_valid") is False

    def test_missing_scores_table_is_skipped_not_fatal(self, test_input_table, cleanup):
        """A --posebusters-only run over a tree with no scores file must not die."""
        bulk = self._bulk_with_scores(test_input_table, pd.DataFrame())
        bulk.posebusters_df = self._summary(
            [{PB_COMBINATION_ID: "c1", PB_DOCKING_METHOD: VINA_PREFIX, PB_POSE: 1, PB_VALID: True}]
        )
        if Path(bulk.rp_scores_path).exists():
            Path(bulk.rp_scores_path).unlink()
        bulk._merge_posebusters_into_scores()  # must not raise

    def test_scores_read_from_disk_when_scoring_was_skipped(self, test_input_table, cleanup):
        bulk = self._bulk_with_scores(test_input_table, pd.DataFrame())
        pd.DataFrame({BULK_COMBINATION_ID: ["c1"], "vina_score": [-9.0]}).to_csv(
            bulk.rp_scores_path, sep="\t", index=False
        )
        bulk.posebusters_df = self._summary(
            [{PB_COMBINATION_ID: "c1", PB_DOCKING_METHOD: VINA_PREFIX, PB_POSE: 1, PB_VALID: True}]
        )
        bulk._merge_posebusters_into_scores()

        written = pd.read_csv(bulk.rp_scores_path, sep="\t")
        assert self._flag(written, 0, "vina_pb_valid") is True


# ---------------------------------------------------------------------------
# Every pose-producing method actually gets validated
# ---------------------------------------------------------------------------
@pytest.mark.e2e
class TestAllMethodsAreCovered:
    """
    The whole point of the feature is that no method escapes validation.

    Each method reaches PoseBusters through its own pose-discovery branch in
    ``_pose_mols``, so "vina works" does not imply "boltz works". These tests
    drive the real orchestrator with a real complex PDB staged for all four
    pose-producing methods at once, and assert every one of them ends up with a
    validated row and a flag column pair.
    """

    METHODS = [VINA_PREFIX, GNINA_PREFIX, BOLTZ_PREFIX, DIFFDOCK_PREFIX]

    def _bulk(self, test_input_table):
        return BulkRun(
            input_table=test_input_table,
            project_name="test-posebusters",
            methods_to_run=list(self.METHODS),
            use_decoys=False,
            use_known_binders=False,
            use_gpu=False,
            n_workers=1,
        )

    def _stage_complexes(self, bulk):
        """Put the fixture complex PDB where each method would have written it."""
        staged = []
        for batch_dict in bulk.batched_dictionary.values():
            batch_folder = Path(batch_dict[BATCH_FOLDER])
            combinations_df = batch_dict[COMBINATIONS_TABLE_KEY]
            for method in self.METHODS:
                method_folder = batch_folder / _COMPLEX_PDB_FOLDER_BY_METHOD[method]
                method_folder.mkdir(parents=True, exist_ok=True)
                for _, row in combinations_df.iterrows():
                    combination_id = f"{row[PROTEIN_CONF_ID]}_{row[LIGAND_ID]}"
                    shutil.copy(COMPLEX_PDB, method_folder / f"{combination_id}_complex.pdb")
                    staged.append((method, combination_id))
            # The fixture ligand is eticlopride regardless of the CSV's SMILES;
            # without this the template check would report a heavy-atom mismatch
            # and every row would land on ligand_template_fallback.
            for ligand_id in batch_dict[SMILES_NAMES_DICTIONARY_KEY]:
                batch_dict[SMILES_NAMES_DICTIONARY_KEY][ligand_id] = SMILES
        return staged

    def test_every_pose_producing_method_is_validated(self, test_input_table, cleanup):
        pytest.importorskip("posebusters")
        bulk = self._bulk(test_input_table)
        staged = self._stage_complexes(bulk)
        combination_ids = sorted({combo for _, combo in staged})
        bulk.rp_scores_df = pd.DataFrame({BULK_COMBINATION_ID: combination_ids})

        result = bulk.run_pose_validity_analysis()

        assert set(result[PB_DOCKING_METHOD]) == set(
            self.METHODS
        ), "a pose-producing method was skipped entirely"
        for method in self.METHODS:
            rows = result[result[PB_DOCKING_METHOD] == method]
            assert set(rows[PB_COMBINATION_ID]) == set(combination_ids)
            assert set(rows[PB_STATUS]) == {PB_STATUS_OK}, (
                f"{method} did not reach posebusters cleanly: "
                f"{rows[[PB_STATUS, PB_ERROR]].to_dict('records')}"
            )
            # Sourcing receptor and ligand from the same complex PDB is what
            # makes the intermolecular checks meaningful for every engine,
            # Boltz's recentred frame included.
            assert rows[PB_INTERMOLECULAR_VALID].all()

    def test_flag_columns_land_for_every_method(self, test_input_table, cleanup):
        pytest.importorskip("posebusters")
        bulk = self._bulk(test_input_table)
        staged = self._stage_complexes(bulk)
        combination_ids = sorted({combo for _, combo in staged})
        scores = pd.DataFrame({BULK_COMBINATION_ID: combination_ids, "vina_score": -9.0})
        bulk.rp_scores_df = scores

        bulk.run_pose_validity_analysis()

        merged = pd.read_csv(bulk.rp_scores_path, sep="\t")
        assert len(merged) == len(combination_ids), "flag-only: no row may be dropped"
        for method in self.METHODS:
            assert f"{method}_pb_valid" in merged.columns
            assert f"{method}_pb_pose" in merged.columns
        # Score-only tracks reuse another method's pose, so they must not
        # acquire a validity column of their own.
        for method in bulk.methods_to_run:
            if method not in self.METHODS:
                assert f"{method}_pb_valid" not in merged.columns

    def test_rescore_tracks_are_not_validated(self, test_input_table, cleanup):
        """
        boltz/diffdock auto-add four rescore tracks. They are score-only, and a
        validity row for them would double-count another method's pose.
        """
        bulk = self._bulk(test_input_table)
        assert len(bulk.methods_to_run) > len(self.METHODS), "auto-add did not fire"
        collected = _collect_complex_metadata(
            _batch_dict_for(
                pd.DataFrame([{PROTEIN_CONF_ID: "P1", LIGAND_ID: "lig1"}]),
                Path.cwd(),
            ),
            bulk.methods_to_run,
        )
        assert set(collected) <= set(self.METHODS)


# ---------------------------------------------------------------------------
# End-to-end against the real posebusters
# ---------------------------------------------------------------------------
@pytest.mark.e2e
class TestRealPoseBusters:
    def test_crystal_pose_passes_the_intermolecular_checks(self, tmp_path):
        """
        A deposited crystal pose must have no clash and no absurd distance.

        Asserted on the intermolecular group only, deliberately: internal_energy
        and bond_angles on a deposited ligand are genuinely borderline, so a
        pb_valid assertion would be flaky across rdkit/posebusters versions.
        """
        pytest.importorskip("posebusters")
        summary, report = validate_batch_poses([METADATA], batch_folder=str(tmp_path))

        assert len(summary) == 1
        row = summary.iloc[0]
        assert row[PB_STATUS] == PB_STATUS_OK
        assert bool(row[PB_INTERMOLECULAR_VALID]) is True
        assert not report.empty

    def test_every_declared_check_is_populated(self, tmp_path):
        """
        Guards against posebusters renaming a check: a renamed column silently
        drops out of the verdict, making pb_valid more permissive.
        """
        pytest.importorskip("posebusters")
        summary, _ = validate_batch_poses([METADATA], batch_folder=str(tmp_path))
        row = summary.iloc[0]
        for column in PB_CHECK_COLUMNS:
            assert row[column] is not None, f"{column} was not emitted by posebusters"
