"""
Autodock Vina tools
"""

import logging
import os
import subprocess

import numpy as np
import pandas as pd
from rdkit import Chem
from vina import Vina

from guild.constants.bulk import (
    BATCH_FOLDER,
    COMBINATION_ID,
    COMBINATIONS_TO_RUN_KEY,
)
from guild.constants.diffdock import DIFFDOCK_RESULTS_FOLDER
from guild.constants.general import RANDOM_SEED
from guild.constants.guild import (
    DIFFDOCK_FOLDER,
    LIGAND_ID,
    PROTEIN_CONF_ID,
    VINA_FOLDER,
    VINA_RESCORE_FOLDER,
    VINA_RESCORE_SCORE,
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
from guild.transformers.converters import sdf_to_pdbqt

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


def vina_rescore_guild_scoring(batch_dictionary):
    """
    Perform Vina re-scoring of DiffDock poses for a batch.

    Thin wrapper around :func:`vina_rescore_diffdock_batch` that accepts a
    *batch_dictionary* (the same structure every other ``*_guild_scoring``
    function receives) and returns a DataFrame compatible with the bulk
    scoring merge logic.

    :param batch_dictionary: Dictionary containing the batch information.
    :return: DataFrame with COMBINATION_ID, PROTEIN_CONF_ID, LIGAND_ID,
        and VINA_RESCORE_SCORE columns.
    """
    df = vina_rescore_diffdock_batch(
        batch_folder=batch_dictionary[BATCH_FOLDER],
        combinations=batch_dictionary[COMBINATIONS_TO_RUN_KEY],
    )
    # Keep only the columns the merge expects (drop 'diffdock_sdf')
    keep = [COMBINATION_ID, PROTEIN_CONF_ID, LIGAND_ID, VINA_RESCORE_SCORE]
    return df[[c for c in keep if c in df.columns]]


# ── Vina score-only re-scoring of pre-docked poses ──────────────────────────


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


def _find_best_diffdock_sdf(diffdock_results_dir: str, protein_conf_id: str, ligand_id: str) -> str:
    """
    Locate the highest-confidence SDF file for a given combination in a
    DiffDock results directory.

    DiffDock names output files like ``rank1_confidence-1.23.sdf``.

    :return: Absolute path to the best-confidence SDF.
    :raises FileNotFoundError: If no SDF files are found for the combination.
    """
    folder_name = f"{protein_conf_id}_{ligand_id}"
    combo_dir = os.path.join(diffdock_results_dir, folder_name)

    if not os.path.isdir(combo_dir):
        raise FileNotFoundError(f"DiffDock results folder not found: {combo_dir}")

    sdf_scores = {}
    for fname in os.listdir(combo_dir):
        if "_confidence" in fname and fname.endswith(".sdf"):
            try:
                score = float(fname.split("_confidence")[1].replace(".sdf", ""))
                sdf_scores[fname] = score
            except ValueError:
                continue

    if not sdf_scores:
        raise FileNotFoundError(f"No DiffDock SDF files found in {combo_dir}")

    best_fname = max(sdf_scores, key=sdf_scores.get)
    return os.path.join(combo_dir, best_fname)


def rescore_diffdock_pose(
    receptor_pdbqt: str,
    diffdock_results_dir: str,
    protein_conf_id: str,
    ligand_id: str,
    output_dir: str = None,
    box_padding: float = 4.0,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Re-score a DiffDock pose with Vina's physics-based scoring function.

    Pipeline:
    1. Find the highest-confidence SDF in ``diffdock_results_dir/{protein}_{ligand}/``
    2. Convert SDF → PDBQT (preserving 3D coordinates)
    3. Compute a Vina box centred on the SDF pose
    4. Call ``vina_score_pose()`` (score-only, no re-docking)

    :param receptor_pdbqt: Path to the receptor PDBQT file.
    :param diffdock_results_dir: Path to the DiffDock results directory.
    :param protein_conf_id: Protein configuration ID (e.g. ``8gut-R-KO8-R``).
    :param ligand_id: Ligand identifier.
    :param output_dir: Directory for intermediate PDBQT files (defaults to
        ``diffdock_results_dir/{protein}_{ligand}/``).
    :param box_padding: Padding around the ligand bounding box (Å).
    :param seed: Random seed for Vina.
    :return: Dict with ``combination``, ``protein_config_id``, ``ligand_id``,
        ``vina_rescore_score``, ``diffdock_sdf``.
    """
    sdf_path = _find_best_diffdock_sdf(diffdock_results_dir, protein_conf_id, ligand_id)

    if output_dir is None:
        output_dir = os.path.dirname(sdf_path)

    # SDF → PDBQT (preserves 3D coordinates)
    ligand_pdbqt = os.path.join(output_dir, f"{protein_conf_id}_{ligand_id}_rescore.pdbqt")
    sdf_to_pdbqt(sdf_path, pdbqt=ligand_pdbqt)

    # Compute box from the DiffDock pose coordinates
    center, size = compute_box_from_sdf(sdf_path, padding=box_padding)

    # Score-only evaluation
    score = vina_score_pose(
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        center=center,
        size=size,
        seed=seed,
    )

    combination_id = f"{protein_conf_id}_{ligand_id}"
    logger.info(f"Vina rescore: {combination_id} → {score:.3f} kcal/mol")

    return {
        COMBINATION_ID: combination_id,
        PROTEIN_CONF_ID: protein_conf_id,
        LIGAND_ID: ligand_id,
        VINA_RESCORE_SCORE: score,
        "diffdock_sdf": sdf_path,
    }


def _prepare_receptor_pdbqt_from_raw(
    raw_pdb: str,
    chain_id: str,
    output_pdbqt: str,
) -> str:
    """
    Extract a single chain (ATOM records only) from a raw PDB and convert to
    PDBQT via OpenBabel.  This preserves the original crystal coordinates so
    the receptor is in the same frame as DiffDock's output SDF files.

    :param raw_pdb: Path to the raw (multi-chain) PDB file.
    :param chain_id: Chain letter to extract (e.g. ``"R"``).
    :param output_pdbqt: Where to write the receptor PDBQT.
    :return: *output_pdbqt* path.
    :raises FileNotFoundError: If *raw_pdb* does not exist.
    :raises RuntimeError: If ``obabel`` is not available or conversion fails.
    """
    if not os.path.isfile(raw_pdb):
        raise FileNotFoundError(f"Raw PDB not found: {raw_pdb}")

    # Write a single-chain PDB (ATOM records only, no waters/HETATM)
    chain_pdb = output_pdbqt.replace(".pdbqt", "_chain.pdb")
    kept = 0
    with open(raw_pdb) as fin, open(chain_pdb, "w") as fout:
        for line in fin:
            if line.startswith("ATOM") and len(line) > 21 and line[21] == chain_id:
                fout.write(line)
                kept += 1
        fout.write("END\n")

    if kept == 0:
        raise ValueError(f"No ATOM records found for chain {chain_id} in {raw_pdb}")

    # Convert PDB → PDBQT using OpenBabel (receptor mode: -xr)
    result = subprocess.run(
        ["obabel", "-ipdb", chain_pdb, "-opdbqt", "-O", output_pdbqt, "-xr"],
        capture_output=True,
        text=True,
    )
    if not os.path.isfile(output_pdbqt) or os.path.getsize(output_pdbqt) == 0:
        raise RuntimeError(
            f"obabel PDB→PDBQT failed for {chain_pdb}: {result.stderr}"
        )

    logger.info(
        f"Prepared receptor PDBQT from raw PDB chain {chain_id}: "
        f"{kept} atoms → {output_pdbqt}"
    )
    return output_pdbqt


def vina_rescore_diffdock_batch(
    batch_folder: str,
    combinations: list[tuple[str, str]],
    receptor_pdbqt_dir: str = None,
    box_padding: float = 4.0,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Batch re-score DiffDock poses for all combinations in a batch.

    DiffDock produces ligand poses in the coordinate frame of the **raw**
    input PDB.  The guild pipeline's ``_single_chain_clean`` PDB is
    typically re-centred, so it cannot be used directly as the Vina
    receptor.  This function therefore looks for a receptor PDBQT in the
    following order:

    1. ``{proteins_dir}/{protein_conf_id}_raw.pdbqt`` — single-chain
       PDBQT already in the raw (crystal) coordinate frame.
    2. If (1) does not exist, it auto-generates it from
       ``{protein_conf_id}_raw.pdb`` by extracting the chain letter
       encoded in *protein_conf_id* and converting via OpenBabel.

    :param batch_folder: Path to the batch folder (e.g. ``data/project/batches/batch_1``).
    :param combinations: List of (protein_conf_id, ligand_id) tuples.
    :param receptor_pdbqt_dir: Directory containing receptor PDBQT files.
        Defaults to ``{batch_folder}/proteins/``.
    :param box_padding: Padding around the ligand bounding box (Å).
    :param seed: Random seed for Vina.
    :return: DataFrame with columns: combination, protein_config_id, ligand_id,
        vina_rescore_score.
    """
    diffdock_results_dir = os.path.join(batch_folder, DIFFDOCK_FOLDER, DIFFDOCK_RESULTS_FOLDER)
    if receptor_pdbqt_dir is None:
        receptor_pdbqt_dir = os.path.join(batch_folder, "proteins")

    rescore_output_dir = os.path.join(batch_folder, VINA_RESCORE_FOLDER)
    os.makedirs(rescore_output_dir, exist_ok=True)

    results = []
    for protein_conf_id, ligand_id in combinations:
        # --- Resolve receptor PDBQT in the DiffDock (raw) coordinate frame ---
        receptor_pdbqt = os.path.join(
            receptor_pdbqt_dir, f"{protein_conf_id}_raw.pdbqt"
        )
        if not os.path.exists(receptor_pdbqt):
            # Auto-generate from the raw PDB (extract chain, convert)
            raw_pdb = os.path.join(receptor_pdbqt_dir, f"{protein_conf_id}_raw.pdb")
            if os.path.isfile(raw_pdb):
                # Extract chain letter from protein_conf_id (e.g. "8gut-R-KO8-R" → "R")
                parts = protein_conf_id.split("-")
                chain_id = parts[1] if len(parts) >= 2 else "A"
                try:
                    _prepare_receptor_pdbqt_from_raw(raw_pdb, chain_id, receptor_pdbqt)
                except Exception as e:
                    logger.warning(
                        f"Could not prepare receptor PDBQT from raw PDB "
                        f"for {protein_conf_id}: {e}"
                    )
                    receptor_pdbqt = None
            else:
                logger.warning(
                    f"Neither {receptor_pdbqt} nor {raw_pdb} found for "
                    f"{protein_conf_id}, skipping combination"
                )
                receptor_pdbqt = None

        if receptor_pdbqt is None or not os.path.exists(receptor_pdbqt):
            results.append({
                COMBINATION_ID: f"{protein_conf_id}_{ligand_id}",
                PROTEIN_CONF_ID: protein_conf_id,
                LIGAND_ID: ligand_id,
                VINA_RESCORE_SCORE: np.nan,
                "diffdock_sdf": None,
            })
            continue

        try:
            result = rescore_diffdock_pose(
                receptor_pdbqt=receptor_pdbqt,
                diffdock_results_dir=diffdock_results_dir,
                protein_conf_id=protein_conf_id,
                ligand_id=ligand_id,
                output_dir=rescore_output_dir,
                box_padding=box_padding,
                seed=seed,
            )
            results.append(result)
        except Exception as e:
            logger.warning(
                f"Vina rescore failed for {protein_conf_id}_{ligand_id}: {e}"
            )
            results.append({
                COMBINATION_ID: f"{protein_conf_id}_{ligand_id}",
                PROTEIN_CONF_ID: protein_conf_id,
                LIGAND_ID: ligand_id,
                VINA_RESCORE_SCORE: np.nan,
                "diffdock_sdf": None,
            })

    return pd.DataFrame(results)
