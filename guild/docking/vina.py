"""
Autodock Vina tools
"""

import logging

import numpy as np
import pandas as pd
from rdkit import Chem
from vina import Vina

from guild.constants.bulk import (
    BATCH_FOLDER,
    COMBINATION_ID,
    COMBINATIONS_TO_RUN_KEY,
)
from guild.constants.general import RANDOM_SEED
from guild.constants.guild import (
    LIGAND_ID,
    PROTEIN_CONF_ID,
    VINA_FOLDER,
    VINA_SCORE,
)
from guild.constants.vina import (
    VINA_DEFAULT_EXHAUSTIVENESS,
    VINA_DEFAULT_NUMBER_OF_POSES,
)
from guild.tools.ligand_properties import (
    radius_of_gyration_from_smiles,
    vina_box_edge_from_radius_of_gyration,
)

logger = logging.getLogger(__name__)


def _validate_pdbqt(pdbqt_path: str, file_type: str = "file"):
    """
    Validate a PDBQT file to catch malformed files before they crash Vina's C++ layer.
    :param pdbqt_path: path to the PDBQT file
    :param file_type: descriptor for error messages (e.g., 'receptor', 'ligand')
    :raises ValueError: if the file is invalid
    """
    with open(pdbqt_path, "r") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"PDBQT {file_type} file is empty: {pdbqt_path}")

    atom_count = 0
    for line_num, line in enumerate(lines, 1):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_count += 1
            # PDBQT ATOM lines must be at least 78 characters
            if len(line.strip()) < 30:  # Very conservative minimum
                raise ValueError(
                    f"PDBQT {file_type} has malformed ATOM line {line_num} (too short): {line.strip()[:50]}"
                )

    if atom_count == 0:
        raise ValueError(f"PDBQT {file_type} has no ATOM records: {pdbqt_path}")

    logger.debug(f"PDBQT {file_type} validation passed: {atom_count} atoms found")


def deploy_vina(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    exhaustiveness: int = VINA_DEFAULT_EXHAUSTIVENESS,
    n_poses: int = VINA_DEFAULT_NUMBER_OF_POSES,
    seed: int = RANDOM_SEED,
    output_pdbqt: str = None,
    output_scores: str = None,
) -> dict:
    """
    Run docking using the python Vina API.
    :param receptor_pdbqt: path to the receptor pdbqt file
    :param ligand_pdbqt: path to the ligand pdbqt file
    :param center: tuple of x, y, z coordinates of the center of the box
    :param size: tuple of x, y, z sizes of the box
    :param exhaustiveness: exhaustiveness of the docking
    :param n_poses: number of poses to generate
    :param seed: seed for the random number generator
    :param out_pdbqt: path to the output pdbqt file
    :param output_scores_file: path to the output scores file
    Returns a dict: {'scores': [kcal/mol list], 'out_pdbqt': ..., 'log': ...}
    """
    # Validate PDBQT files before passing to Vina to prevent C++ crashes
    _validate_pdbqt(receptor_pdbqt, "receptor")
    _validate_pdbqt(ligand_pdbqt, "ligand")

    vina_object = Vina(sf_name="vina", seed=seed, verbosity=False)

    try:
        vina_object.set_receptor(receptor_pdbqt)
    except Exception as e:
        raise RuntimeError(f"Failed to set receptor from {receptor_pdbqt}: {e}") from e

    try:
        vina_object.set_ligand_from_file(ligand_pdbqt)
    except Exception as e:
        raise RuntimeError(f"Failed to set ligand from {ligand_pdbqt}: {e}") from e

    vina_object.compute_vina_maps(center=center, box_size=size)

    # default output names
    if output_pdbqt is None:
        output_pdbqt = ligand_pdbqt.replace(".pdbqt", "_docked.pdbqt")

    vina_object.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    vina_object.write_poses(output_pdbqt, n_poses=n_poses, overwrite=True)

    # parse scores
    energies = vina_object.energies(n_poses=n_poses)
    # energies is list of (score, RMSD_lb, RMSD_ub)
    scores = [e[0] for e in energies]
    with open(output_scores, "w") as scores_file:
        for index, score in enumerate(scores):
            scores_file.write(f"{index}: {score}\n")
    logger.info(f"Vina: docking completed. Scores saved to {output_scores}")

    return {"scores": scores, "out_pdbqt": output_pdbqt, "output_scores": output_scores}


def generate_vina_box(
    input_x: float,
    input_y: float,
    input_z: float,
    box_size: float = None,
    ligand_smiles: str = None,
    ratio: float = 0.35,
    output_file: str = "autodock_box.txt",
):
    """
    Generate the autodock vina box for docking.
    :param input_x: x coordinate of the center of the box
    :param input_y: y coordinate of the center of the box
    :param input_z: z coordinate of the center of the box
    :param box_size: size of the box
    :param ligand_smiles: SMILES string of the ligand
    :param ratio: ratio to calculate the box size from the radius of gyration
    :param output_file: output file to save the box
    """
    if box_size is None:
        if ligand_smiles is None:
            raise ValueError("Provide either box_size or ligand_smiles")
        try:
            rg = radius_of_gyration_from_smiles(ligand_smiles, heavy_only=True, mass_weighted=False)
            if rg is None or np.isnan(rg) or rg <= 0:
                logger.warning(
                    f"Invalid radius of gyration for SMILES {ligand_smiles}, using default box size 20"
                )
                box_size = 20.0
            else:
                box_size = vina_box_edge_from_radius_of_gyration(rg, ratio=ratio)
        except Exception as e:
            logger.warning(
                f"Failed to calculate radius of gyration for SMILES {ligand_smiles}: {e}, using default box size 20"
            )
            box_size = 20.0
    string_to_write = f"""center_x = {str(input_x)}
center_y = {str(input_y)}
center_z = {str(input_z)}
size_x = {str(box_size)}
size_y = {str(box_size)}
size_z = {str(box_size)}
"""
    with open(output_file, "w") as outfile:
        outfile.write(string_to_write)


def get_center_and_size_from_box_file(box_file: str):
    """
    Get the center of the box from the box file
    :param box_file: path to the box file
    :return: tuple of x, y, z coordinates of the center of the box
    """
    with open(box_file, "r") as box_file:
        lines = box_file.readlines()
        center = (
            float(lines[0].split("=")[1].strip()),
            float(lines[1].split("=")[1].strip()),
            float(lines[2].split("=")[1].strip()),
        )
        size = (
            float(lines[3].split("=")[1].strip()),
            float(lines[4].split("=")[1].strip()),
            float(lines[5].split("=")[1].strip()),
        )
        return center, size


def process_vina_output(input_file):
    """
    Process the autodock output file to retrieve the binding affinity
    """
    afinity_df = pd.read_csv(input_file, sep=":", header=None).sort_values(by=1, ascending=True)

    if afinity_df.empty:
        return np.nan

    return afinity_df[1].astype(float).iloc[0]


def vina_guild_scoring(batch_dictionary):
    """
    Perform the scoring of the docking results for AutoDock Vina.

    :param batch_dictionary: Dictionary containing the batch information.
    :return: DataFrame containing the docking scores.
    """

    # Create DataFrame from combinations
    combinations_df = pd.DataFrame(
        batch_dictionary[COMBINATIONS_TO_RUN_KEY],
        columns=[PROTEIN_CONF_ID, LIGAND_ID],
    )

    def score_combination(row):
        try:
            return process_vina_output(
                f"{batch_dictionary[BATCH_FOLDER]}/{VINA_FOLDER}/{row[PROTEIN_CONF_ID]}_{row[LIGAND_ID]}.txt"
            )
        except Exception as e:
            logger.info(f"Failed for {(row[PROTEIN_CONF_ID], row[LIGAND_ID])} with error {e}")
            return np.nan

    combinations_df[VINA_SCORE] = combinations_df.apply(score_combination, axis=1)
    combinations_df[COMBINATION_ID] = (
        combinations_df[PROTEIN_CONF_ID] + "_" + combinations_df[LIGAND_ID]
    )

    return combinations_df[[COMBINATION_ID, VINA_SCORE, PROTEIN_CONF_ID, LIGAND_ID]]


# ── Vina score-only re-scoring of pre-docked poses ──────────────────────────
# NOTE: Method-specific orchestration (rescore_boltz_pose, rescore_diffdock_pose,
# vina_rescore_*_batch, vina_rescore_*_guild_scoring) lives in
# guild/docking/boltz.py and guild/docking/diffdock.py — they know the output
# layout of their respective methods. This module keeps only the Vina-grid
# primitives that any pose source can reuse (compute_box_from_sdf,
# _compute_box_from_pdb_atoms, _extract_ligand_records,
# _extract_protein_from_complex, vina_score_pose).


def compute_box_from_sdf(sdf_path: str, padding: float = 4.0):
    """
    Compute a Vina box centered on the ligand pose in an SDF file.

    The center is the centroid of heavy atoms. The box size is the bounding-box
    span along each axis plus *padding* Angstrom on each side.

    :param sdf_path: Path to the SDF file with 3D coordinates.
    :param padding: Extra space (Å) added to each side of the bounding box.
    :return: (center, size) — two tuples of three floats each.
    """
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"Could not read a valid molecule from {sdf_path}")

    conf = mol.GetConformer()
    positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

    center = tuple(positions.mean(axis=0).tolist())
    span = positions.max(axis=0) - positions.min(axis=0)
    size = tuple((span + 2.0 * padding).tolist())

    return center, size


def vina_score_pose(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    seed: int = RANDOM_SEED,
) -> float:
    """
    Score a single pre-docked ligand pose with Vina (no re-docking).

    Uses ``Vina.score()`` which evaluates the current coordinates against the
    Vina scoring function and returns binding energy (kcal/mol, lower = better).
    The pose is not modified in any way.

    :param receptor_pdbqt: Path to the receptor PDBQT file.
    :param ligand_pdbqt: Path to the ligand PDBQT file (already in the docked pose).
    :param center: Box center (x, y, z).
    :param size: Box size (x, y, z).
    :param seed: Random seed (for reproducibility).
    :return: Vina binding energy in kcal/mol.
    """
    _validate_pdbqt(receptor_pdbqt, "receptor")
    _validate_pdbqt(ligand_pdbqt, "ligand")

    v = Vina(sf_name="vina", seed=seed, verbosity=False)

    try:
        v.set_receptor(receptor_pdbqt)
    except Exception as e:
        raise RuntimeError(f"Failed to set receptor from {receptor_pdbqt}: {e}") from e

    try:
        v.set_ligand_from_file(ligand_pdbqt)
    except Exception as e:
        raise RuntimeError(f"Failed to set ligand from {ligand_pdbqt}: {e}") from e

    v.compute_vina_maps(center=center, box_size=size)
    energy = v.score()

    return float(energy[0])


# ── Vina score-only re-scoring of Boltz-predicted complexes ─────────────────


def _compute_box_from_pdb_atoms(pdb_path: str, padding: float = 4.0):
    """
    Compute a Vina box (center + size) from the coordinates of all ATOM/HETATM
    records in a PDB file. Used to size the Vina scoring grid around a
    pre-docked ligand pose extracted from a Boltz complex.
    """
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
    if not coords:
        raise ValueError(f"No coordinates parsed from {pdb_path}")
    arr = np.array(coords)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    center = tuple(((mins + maxs) / 2.0).tolist())
    span = maxs - mins
    size = tuple((span + 2.0 * padding).tolist())
    return center, size


def _extract_ligand_records(input_pdb: str, output_pdb: str, resname: str = "LIG"):
    """
    Write a PDB containing only ATOM/HETATM records whose residue name matches
    ``resname``. Used to isolate the ligand of a Boltz complex (whose ligand is
    written under residue name ``LIG`` by :func:`relabel_ligand_chain_in_pdb` —
    the chain ID is not a reliable marker because ``cif_to_pdb`` may rename it).
    """
    kept = 0
    resname_padded = resname.ljust(3)[:3]
    with open(input_pdb) as fin, open(output_pdb, "w") as fout:
        for line in fin:
            if (
                line.startswith(("ATOM", "HETATM"))
                and len(line) > 20
                and line[17:20] == resname_padded
            ):
                fout.write(line)
                kept += 1
        fout.write("END\n")
    if kept == 0:
        raise ValueError(f"No atoms with resname '{resname}' found in {input_pdb}")
    return output_pdb


def _extract_protein_from_complex(complex_pdb: str, output_pdb: str, ligand_resname: str = "LIG"):
    """
    Write a PDB containing all ATOM records of the complex EXCEPT those that
    are the ligand (resname == ``ligand_resname``). Used to obtain a receptor
    PDB in Boltz's predicted coordinate frame — critical because Boltz often
    re-centres the whole complex, so the template-frame receptor and the
    Boltz-output ligand are not in the same physical space.
    """
    kept = 0
    resname_padded = ligand_resname.ljust(3)[:3]
    with open(complex_pdb) as fin, open(output_pdb, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")) and len(line) > 20:
                if line[17:20] == resname_padded:
                    continue
                fout.write(line)
                kept += 1
            elif line.startswith(("TER", "END")):
                fout.write(line)
        fout.write("END\n")
    if kept == 0:
        raise ValueError(
            f"No protein atoms left in {complex_pdb} after excluding resname '{ligand_resname}'"
        )
    return output_pdb


