"""
Tests for DiffDock pose selection (auto / box / blind), the shared selection
record, and the DiffDock rescore inputs (prepared receptor, ligand hydrogens,
minimisation, no-contact guard).
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from guild.constants.bulk import BATCH_FOLDER, COMBINATIONS_TO_RUN_KEY
from guild.constants.diffdock import (
    DIFFDOCK_POCKET_AUTO,
    DIFFDOCK_POCKET_BLIND,
    DIFFDOCK_POCKET_BOX,
    DIFFDOCK_POCKET_KEY,
    DIFFDOCK_POSE_SELECTION,
    POSE_NO_BOX,
    POSE_NO_SAMPLE_IN_BOX,
    POSE_NO_SAMPLES,
    POSE_SELECTED_BLIND,
    POSE_SELECTED_BOX,
    SELECTED_POSE_FILE,
)
from guild.constants.guild import DIFFDOCK_SCORE, LIGAND_ID, PROTEIN_CONF_ID, SMILES
from guild.docking.diffdock import (
    _has_receptor_contact,
    diffdock_combo_dir,
    diffdock_guild_scoring,
    find_pocket_box,
    read_selected_pose,
    resolve_diffdock_pose,
    select_diffdock_pose,
    write_diffdock_combinations_table,
)
from guild.docking.vina import compute_box_from_sdf, vina_score_pose
from guild.transformers.converters import sdf_to_pdbqt

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
PROTEIN = "prot-A-LIG-A"
LIGAND = "lig1"
COMBO = f"{PROTEIN}_{LIGAND}"


def _write_sample(combo_dir: Path, rank: int, confidence: float, centroid) -> Path:
    """An ethanol pose translated so its heavy-atom centroid sits at ``centroid``."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    mol = Chem.RemoveHs(mol)
    conformer = mol.GetConformer()
    shift = np.asarray(centroid) - conformer.GetPositions().mean(axis=0)
    for idx in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(idx, (conformer.GetPositions()[idx] + shift).tolist())
    path = combo_dir / f"rank{rank}_confidence{confidence:.2f}.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)
    return path


def _write_box(path: Path, center=(0.0, 0.0, 0.0), size=(10.0, 10.0, 10.0)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"center_{a} = {c}\n" for a, c in zip("xyz", center, strict=True))
        + "".join(f"size_{a} = {s}\n" for a, s in zip("xyz", size, strict=True))
    )
    return path


@pytest.fixture
def batch(tmp_path):
    """A batch with one combination: a high-confidence sample outside the box
    and a lower-confidence one inside it."""
    combo_dir = Path(diffdock_combo_dir(str(tmp_path), PROTEIN, LIGAND))
    combo_dir.mkdir(parents=True)
    outside = _write_sample(combo_dir, 1, 0.80, (30.0, 0.0, 0.0))
    inside = _write_sample(combo_dir, 2, -0.40, (1.0, 0.0, 0.0))
    box = _write_box(tmp_path / "boxes" / "vina_boxes" / f"{COMBO}.txt")
    return {
        "folder": tmp_path,
        "combo_dir": combo_dir,
        "outside": outside,
        "inside": inside,
        "box": box,
    }


class TestSelectDiffdockPose:
    def test_box_mode_prefers_in_box_sample_over_higher_confidence(self, batch):
        selection = select_diffdock_pose(
            str(batch["combo_dir"]), str(batch["box"]), DIFFDOCK_POCKET_BOX
        )
        assert selection["sdf"] == str(batch["inside"])
        assert selection["confidence"] == pytest.approx(-0.40)
        assert selection["reason"] == POSE_SELECTED_BOX

    def test_auto_with_box_behaves_like_box(self, batch):
        selection = select_diffdock_pose(
            str(batch["combo_dir"]), str(batch["box"]), DIFFDOCK_POCKET_AUTO
        )
        assert selection["sdf"] == str(batch["inside"])
        assert selection["reason"] == POSE_SELECTED_BOX

    def test_auto_without_box_falls_back_to_blind(self, batch):
        selection = select_diffdock_pose(str(batch["combo_dir"]), None, DIFFDOCK_POCKET_AUTO)
        assert selection["sdf"] == str(batch["outside"])
        assert selection["reason"] == POSE_SELECTED_BLIND

    def test_blind_ignores_box(self, batch):
        selection = select_diffdock_pose(
            str(batch["combo_dir"]), str(batch["box"]), DIFFDOCK_POCKET_BLIND
        )
        assert selection["sdf"] == str(batch["outside"])
        assert selection["confidence"] == pytest.approx(0.80)
        assert selection["reason"] == POSE_SELECTED_BLIND

    def test_box_mode_without_box_is_a_failure(self, batch):
        selection = select_diffdock_pose(str(batch["combo_dir"]), None, DIFFDOCK_POCKET_BOX)
        assert selection["sdf"] is None
        assert np.isnan(selection["confidence"])
        assert selection["reason"] == POSE_NO_BOX

    def test_no_sample_in_box_is_a_failure(self, batch):
        batch["inside"].unlink()
        selection = select_diffdock_pose(
            str(batch["combo_dir"]), str(batch["box"]), DIFFDOCK_POCKET_BOX
        )
        assert selection["sdf"] is None
        assert np.isnan(selection["confidence"])
        assert selection["reason"] == POSE_NO_SAMPLE_IN_BOX

    def test_no_samples(self, tmp_path):
        selection = select_diffdock_pose(str(tmp_path / "missing"), None, DIFFDOCK_POCKET_BLIND)
        assert selection["sdf"] is None
        assert selection["reason"] == POSE_NO_SAMPLES

    def test_invalid_mode_raises(self, batch):
        with pytest.raises(ValueError):
            select_diffdock_pose(str(batch["combo_dir"]), None, "pocket")


class TestPocketBoxAndRecord:
    def test_find_pocket_box_exact(self, batch):
        assert find_pocket_box(str(batch["folder"]), PROTEIN, LIGAND) == str(batch["box"])

    def test_find_pocket_box_falls_back_to_same_protein(self, batch):
        sibling = batch["box"].with_name(f"{PROTEIN}_other.txt")
        batch["box"].rename(sibling)
        assert find_pocket_box(str(batch["folder"]), PROTEIN, LIGAND) == str(sibling)

    def test_find_pocket_box_none_when_no_pocket(self, tmp_path):
        assert find_pocket_box(str(tmp_path), PROTEIN, LIGAND) is None

    def test_resolve_writes_record_that_round_trips(self, batch):
        selection = resolve_diffdock_pose(
            str(batch["folder"]), PROTEIN, LIGAND, DIFFDOCK_POCKET_AUTO
        )
        assert (batch["combo_dir"] / SELECTED_POSE_FILE).is_file()
        record = read_selected_pose(str(batch["combo_dir"]))
        assert record["sdf"] == selection["sdf"] == str(batch["inside"])
        assert record["reason"] == POSE_SELECTED_BOX

    def test_resolve_records_failures_with_null_pose(self, batch):
        batch["inside"].unlink()
        resolve_diffdock_pose(str(batch["folder"]), PROTEIN, LIGAND, DIFFDOCK_POCKET_BOX)
        record = read_selected_pose(str(batch["combo_dir"]))
        assert record["sdf"] is None
        assert record["confidence"] is None
        assert record["reason"] == POSE_NO_SAMPLE_IN_BOX


class TestDiffdockScoring:
    def _batch_dictionary(self, folder, mode):
        return {
            BATCH_FOLDER: str(folder),
            COMBINATIONS_TO_RUN_KEY: [(PROTEIN, LIGAND)],
            DIFFDOCK_POCKET_KEY: mode,
        }

    def test_score_is_confidence_of_selected_pose(self, batch):
        scores = diffdock_guild_scoring(
            self._batch_dictionary(batch["folder"], DIFFDOCK_POCKET_BOX)
        )
        assert scores[DIFFDOCK_SCORE].iloc[0] == pytest.approx(-0.40)
        assert scores[DIFFDOCK_POSE_SELECTION].iloc[0] == POSE_SELECTED_BOX

    def test_blind_score_is_top_confidence(self, batch):
        scores = diffdock_guild_scoring(
            self._batch_dictionary(batch["folder"], DIFFDOCK_POCKET_BLIND)
        )
        assert scores[DIFFDOCK_SCORE].iloc[0] == pytest.approx(0.80)
        assert scores[DIFFDOCK_POSE_SELECTION].iloc[0] == POSE_SELECTED_BLIND

    def test_failure_scores_nan_not_zero(self, batch):
        batch["inside"].unlink()
        scores = diffdock_guild_scoring(
            self._batch_dictionary(batch["folder"], DIFFDOCK_POCKET_BOX)
        )
        assert np.isnan(scores[DIFFDOCK_SCORE].iloc[0])
        assert scores[DIFFDOCK_POSE_SELECTION].iloc[0] == POSE_NO_SAMPLE_IN_BOX


def test_combinations_table_points_at_prepared_receptor(tmp_path):
    table = pd.DataFrame({PROTEIN_CONF_ID: [PROTEIN], LIGAND_ID: [LIGAND], SMILES: ["CCO"]})
    write_diffdock_combinations_table(table, str(tmp_path))
    written = pd.read_csv(tmp_path / "diffdock_combinations.csv")
    assert written["protein_path"].iloc[0] == str(
        tmp_path / "proteins" / f"{PROTEIN}_single_chain_clean.pdb"
    )


def _split_complex(tmp_path):
    receptor = tmp_path / "receptor.pdb"
    ligand_pdb = tmp_path / "ligand.pdb"
    lines = (TEST_DATA_DIR / "3pbl_lig_complex.pdb").read_text().splitlines(keepends=True)
    receptor.write_text("".join(line for line in lines if line.startswith("ATOM")) + "END\n")
    ligand_pdb.write_text("".join(line for line in lines if line.startswith("HETATM")) + "END\n")
    return receptor, ligand_pdb


def test_receptor_contact_guard(tmp_path):
    receptor, ligand_pdb = _split_complex(tmp_path)
    bound = tmp_path / "bound.sdf"
    subprocess.run(["obabel", str(ligand_pdb), "-O", str(bound)], check=True, capture_output=True)
    assert _has_receptor_contact(str(bound), str(receptor))

    far = Chem.MolFromMolFile(str(bound), sanitize=False)
    conformer = far.GetConformer()
    for idx in range(far.GetNumAtoms()):
        conformer.SetAtomPosition(idx, (conformer.GetPositions()[idx] + 200.0).tolist())
    far_path = tmp_path / "far.sdf"
    Chem.MolToMolFile(far, str(far_path), kekulize=False)
    assert not _has_receptor_contact(str(far_path), str(receptor))


def test_sdf_to_pdbqt_adds_polar_hydrogens_on_request(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    sdf = tmp_path / "ethanol.sdf"
    Chem.MolToMolFile(Chem.RemoveHs(mol), str(sdf))

    without = sdf_to_pdbqt(str(sdf), pdbqt=str(tmp_path / "without.pdbqt"))
    with_h = sdf_to_pdbqt(str(sdf), pdbqt=str(tmp_path / "with.pdbqt"), add_hydrogens=True)

    def atom_types(path):
        lines = Path(path).read_text().splitlines()
        return [line.split()[-1] for line in lines if line.startswith(("ATOM", "HETATM"))]

    assert "HD" not in atom_types(without)
    assert "HD" in atom_types(with_h)


def test_vina_minimised_score_is_not_worse_than_raw(tmp_path):
    receptor, ligand_pdb = _split_complex(tmp_path)
    receptor_pdbqt = tmp_path / "receptor.pdbqt"
    subprocess.run(
        ["obabel", str(receptor), "-xr", "-h", "-O", str(receptor_pdbqt)],
        check=True,
        capture_output=True,
    )
    ligand_sdf = tmp_path / "ligand.sdf"
    subprocess.run(
        ["obabel", str(ligand_pdb), "-O", str(ligand_sdf)], check=True, capture_output=True
    )
    ligand_pdbqt = sdf_to_pdbqt(
        str(ligand_sdf), pdbqt=str(tmp_path / "ligand.pdbqt"), add_hydrogens=True
    )

    center, size = compute_box_from_sdf(str(ligand_sdf), padding=6.0)
    raw = vina_score_pose(str(receptor_pdbqt), ligand_pdbqt, center, size)
    minimised_pose = tmp_path / "minimised.pdbqt"
    minimised = vina_score_pose(
        str(receptor_pdbqt),
        ligand_pdbqt,
        center,
        size,
        minimize=True,
        output_pdbqt=str(minimised_pose),
    )
    assert minimised <= raw + 1e-6
    assert minimised_pose.is_file() and minimised_pose.stat().st_size > 0
