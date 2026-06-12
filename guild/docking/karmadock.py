"""
KarmaDock tools
"""

import logging
import os
import subprocess
import time

import numpy as np
import pandas as pd
from rdkit import Chem

from guild.constants.bulk import BATCH_FOLDER, COMBINATION_ID, COMBINATIONS_TO_RUN_KEY
from guild.constants.guild import (
    KARMADOCK_FOLDER,
    KARMADOCK_SCORE,
    LIGAND_ID,
    PROTEIN_CONF_ID,
)
from guild.constants.karmadock import (
    KARMADOCK_COLUMNS,
    KARMADOCK_COLUMNS_TO_DROP,
    KARMADOCK_LIGAND,
    KARMADOCK_PDB_ID,
    KARMADOCK_PROTEIN,
    KARMADOCK_RAW_SCORE,
    KARMADOCK_RESULTS_FOLDER,
    KARMADOCK_RUN,
)
from guild.constants.system import PYTHON_EXECUTABLE, SHELL_SILENCER

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Pocket support
#
# KarmaDock does not accept a binding pocket directly — its ``get_pocket()``
# preprocessing step derives the pocket from residues within 12 Å of the
# *input ligand's* coordinates. By default the input ligand is whatever 3D
# conformer RDKit/OpenBabel happens to embed (typically near the origin),
# which means KarmaDock's pocket lands wherever that conformer falls. To
# control the pocket without modifying KarmaDock's codebase, we translate
# the ligand SDF (and mol2) so its centroid sits at the centre of the user-
# supplied Vina box. KarmaDock's get_pocket() then selects the correct
# residues around that point.
# ────────────────────────────────────────────────────────────────────────────


def _translate_sdf(sdf_path: str, target_center) -> bool:
    """Translate every conformer in ``sdf_path`` so its centroid sits at
    ``target_center``. Returns True on success."""
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    mols = [m for m in supplier if m is not None]
    if not mols:
        return False

    target = np.asarray(target_center, dtype=float)
    writer = Chem.SDWriter(sdf_path + ".tmp")
    for mol in mols:
        conf = mol.GetConformer()
        positions = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        )
        shift = target - positions.mean(axis=0)
        for i in range(mol.GetNumAtoms()):
            old = conf.GetAtomPosition(i)
            conf.SetAtomPosition(
                i, (old.x + shift[0], old.y + shift[1], old.z + shift[2])
            )
        writer.write(mol)
    writer.close()
    os.replace(sdf_path + ".tmp", sdf_path)
    return True


def _translate_mol2(mol2_path: str, target_center) -> bool:
    """Translate ATOM coordinates in a TRIPOS mol2 file in place so the
    centroid sits at ``target_center``. Returns True on success."""
    with open(mol2_path) as f:
        lines = f.readlines()

    in_atoms = False
    atom_indices = []
    coords = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if stripped.startswith("@<TRIPOS>") and in_atoms:
            in_atoms = False
            continue
        if in_atoms and stripped:
            parts = line.split()
            try:
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            except (IndexError, ValueError):
                continue
            atom_indices.append(idx)
            coords.append((x, y, z))

    if not coords:
        return False

    arr = np.array(coords)
    target = np.asarray(target_center, dtype=float)
    shift = target - arr.mean(axis=0)

    for idx, (x, y, z) in zip(atom_indices, coords, strict=True):
        nx, ny, nz = x + shift[0], y + shift[1], z + shift[2]
        parts = lines[idx].split()
        # mol2 ATOM line: id name x y z atom_type [subst_id] [subst_name] [charge]
        parts[2] = f"{nx:.4f}"
        parts[3] = f"{ny:.4f}"
        parts[4] = f"{nz:.4f}"
        # Preserve original spacing-ish — single space join is fine for KarmaDock
        lines[idx] = " ".join(parts) + "\n"

    with open(mol2_path, "w") as f:
        f.writelines(lines)
    return True


def center_karmadock_ligand_on_box(sdf_path: str, mol2_path: str, box_file: str) -> bool:
    """
    Re-centre the karmadock ligand files so their centroid sits at the centre
    of the supplied Vina box file. Both the SDF and the mol2 are translated
    because KarmaDock's ``get_pocket()`` reads SDF first but falls back to mol2.

    No-op (returns False) if the box file is missing or unreadable.
    """
    # Lazy import to avoid a circular load order between karmadock and vina.
    from guild.docking.vina import get_center_and_size_from_box_file

    if not os.path.isfile(box_file):
        return False
    try:
        center, _ = get_center_and_size_from_box_file(box_file)
    except Exception as e:
        logger.warning(f"Could not parse Vina box {box_file}: {e}")
        return False

    sdf_ok = _translate_sdf(sdf_path, center) if os.path.isfile(sdf_path) else False
    mol2_ok = _translate_mol2(mol2_path, center) if os.path.isfile(mol2_path) else False
    if sdf_ok or mol2_ok:
        logger.info(
            f"Re-centred karmadock ligand on Vina box {box_file} "
            f"(SDF={sdf_ok}, mol2={mol2_ok})"
        )
    return sdf_ok or mol2_ok


def process_karmadock_files(
    input_folder, ligand_name=KARMADOCK_LIGAND, protein_name=KARMADOCK_PROTEIN, mode="single"
):
    """
    Process the karmadock csv files to retrieve the highest scores for each pdb
    :param input_folder: folder containing the karmadock results
    :param ligand_name: name of the ligand
    :param protein_name: name of the protein
    :param mode: single or bulk, depending on the input. If single, the ligand and protein names are the same for all rows. If bulk, the ligand and protein names are extracted from the pdb_id
    """
    results = []
    for files in os.listdir(input_folder):
        file_id = files.split(".")[0]
        if files.endswith(".csv"):
            opened_table = pd.read_csv(f"{input_folder}/{files}")
            opened_table[KARMADOCK_RUN] = file_id
            results.append(opened_table)
    if len(results) == 0:
        return pd.DataFrame(columns=KARMADOCK_COLUMNS)
    results_df = pd.concat(results, axis=0)
    results_df.rename(columns={KARMADOCK_RAW_SCORE: KARMADOCK_SCORE}, inplace=True)
    # For each pdb_id, get the index of the row with the maximum karmadock_score
    max_scores = []
    for unique_pdb_id in results_df[KARMADOCK_PDB_ID].unique():
        max_score = (
            results_df[results_df[KARMADOCK_PDB_ID] == unique_pdb_id]
            .sort_values(by=KARMADOCK_SCORE, ascending=False)
            .iloc[0]
        )
        max_scores.append(max_score)
    results_df = pd.DataFrame(max_scores)

    if mode == "single":
        results_df[[KARMADOCK_PROTEIN, KARMADOCK_LIGAND]] = (ligand_name, protein_name)
        results_df = results_df.rename(columns={KARMADOCK_PDB_ID: COMBINATION_ID})

    elif mode == "bulk":
        results_df[[KARMADOCK_PROTEIN, KARMADOCK_LIGAND]] = results_df[KARMADOCK_PDB_ID].str.split(
            "_", expand=True, n=1
        )

    return results_df


def wait_for_graphs(graphs_dir, timeout=1):
    """
    Wait until at least one .dgl file appears in the graphs directory.
    :param graphs_dir: directory containing the graphs
    :param timeout: timeout in seconds
    """
    for i in range(timeout):
        if any(f.endswith(".dgl") for f in os.listdir(graphs_dir)):
            logger.info(f"KarmaDock: graphs visible after {i}s")
            return True
        time.sleep(1)
    logger.warning(f"KarmaDock: no .dgl files found after {timeout}s in {graphs_dir}")
    return False


def deploy_karmadock(
    home_path: str, karmadock_results_dir: str, karmadock_graphs_dir: str, karmadock_data_dir: str
):
    """
    Run KarmaDock for docking the ligand to the protein. Output is saved in the karmadock directory, inside the project folder.
    :param home_path: Path to the home directory.
    :param karmadock_results_dir: Path to the karmadock results directory.
    :param karmadock_graphs_dir: Path to the karmadock graphs directory.
    :param karmadock_data_dir: Path to the karmadock data directory.
    :return: Failed steps.
    """
    failed = 0
    # Resolve KarmaDock install dir. run_guild.py patches WORKING_DIR_PATH to
    # /workspace (so guild's own output lands in the mounted host folder), but
    # KarmaDock itself is baked into the image at /app/KarmaDock. Prefer the
    # home_path location if it exists; otherwise fall back to /app/KarmaDock —
    # same pattern as deploy_diffdock.
    karmadock_root = os.path.join(home_path, "KarmaDock")
    if not os.path.isdir(karmadock_root):
        karmadock_root = "/app/KarmaDock"
    if not os.path.isdir(karmadock_root):
        logger.error(
            f"KarmaDock install not found at {os.path.join(home_path, 'KarmaDock')} or /app/KarmaDock"
        )
        return failed

    pre_processing_command = f"""
    {PYTHON_EXECUTABLE} -u {karmadock_root}/utils/pre_processing.py --complex_file_dir {karmadock_data_dir} {SHELL_SILENCER}
    """

    try:
        subprocess.run(
            pre_processing_command,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("KarmaDock: pre-processing completed.")
    except Exception as e:
        logger.error(f"KarmaDock: error in pre-processing: {e}")
        # Note: failed_steps tracking removed - not returned by this function

    graph_generation_command = f"""
    {PYTHON_EXECUTABLE} -u {karmadock_root}/utils/generate_graph.py --complex_file_dir {karmadock_data_dir} --graph_file_dir {karmadock_graphs_dir} {SHELL_SILENCER}
    """

    try:
        subprocess.run(
            graph_generation_command,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("KarmaDock: graph generation completed.")
    except Exception as e:
        logger.error(f"KarmaDock: error in graph generation: {e}")
        # Note: failed_steps tracking removed - not returned by this function

    docking_command = f"""
    {PYTHON_EXECUTABLE} -u {karmadock_root}/utils/ligand_docking.py  \
        --graph_file_dir {karmadock_graphs_dir} \
        --model_file {karmadock_root}/trained_models/karmadock_screening.pkl \
        --out_dir {karmadock_results_dir} \
        --docking True --scoring True --correct True --batch_size 64 --random_seed 2023 {SHELL_SILENCER}
    """
    graph_generated = False
    max_attempts = 10
    attempt = 0
    while not graph_generated:
        graph_generated = wait_for_graphs(karmadock_graphs_dir)
        if graph_generated:
            logger.info(
                f"KarmaDock: graphs found after {attempt} attempts in {karmadock_graphs_dir}"
            )
            break
        attempt += 1
        if attempt >= max_attempts:
            logger.error(
                f"KarmaDock: graphs not found after {max_attempts} attempts in {karmadock_graphs_dir}"
            )
            return failed

    try:
        subprocess.run(
            docking_command,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("KarmaDock: docking completed.")
    except Exception as e:
        logger.error(f"KarmaDock: error in docking: {e}. Command: {docking_command}")
        failed += 1
    return failed


def karmadock_guild_scoring(batch_dictionary):
    """
    Perform the scoring of the docking results for KarmaDock.

    Returns a DataFrame with columns
        ``combination``, ``protein_config_id``, ``ligand_id``, ``karmadock_score``
    matching what the bulk rank/percentile engine expects (groups by
    ``protein_config_id``, not ``protein_id``).
    """

    df = (
        process_karmadock_files(
            f"{batch_dictionary[BATCH_FOLDER]}/{KARMADOCK_FOLDER}/{KARMADOCK_RESULTS_FOLDER}",
            mode="bulk",
        )
        .rename(columns={KARMADOCK_RAW_SCORE: KARMADOCK_SCORE, KARMADOCK_PDB_ID: COMBINATION_ID})
        .drop(columns=KARMADOCK_COLUMNS_TO_DROP, errors="ignore")
    )

    # KarmaDock's result CSV only carries the combined ``pdb_id`` string
    # (``{protein_config_id}_{ligand_id}``); unlike vina/gnina it has no separate
    # protein_config_id / ligand_id fields. Naively splitting on "_" is ambiguous
    # whenever EITHER part contains an underscore (e.g. protein_config_id
    # "6C97_P0", or ligand_id "pos_3"), which produces wrong keys → the row fails
    # the [COMBINATION_ID, PROTEIN_CONF_ID, LIGAND_ID] merge against the other
    # methods and gets its scores dropped downstream. Recover the authoritative
    # (protein_config_id, ligand_id) by matching the full combined string against
    # COMBINATIONS_TO_RUN_KEY — the same tuples vina/gnina use to build their keys.
    combo_to_keys = {
        f"{conf_id}_{lig_id}": (conf_id, lig_id)
        for conf_id, lig_id in batch_dictionary[COMBINATIONS_TO_RUN_KEY]
    }

    def split_combination(cid: str):
        """Resolve ``{protein_config_id}_{ligand_id}`` → (protein_config_id,
        ligand_id) via the authoritative combination map, with a best-effort
        ``split("_", 1)`` fallback for ids not present in the batch (shouldn't
        normally happen)."""
        if isinstance(cid, str) and cid in combo_to_keys:
            return combo_to_keys[cid]
        if not isinstance(cid, str) or "_" not in cid:
            return ("unknown_config", "unknown_ligand")
        protein_config_id, ligand_id = cid.split("_", 1)
        return (protein_config_id, ligand_id)

    try:
        df[PROTEIN_CONF_ID], df[LIGAND_ID] = zip(
            *df[COMBINATION_ID].apply(split_combination), strict=True
        )
        # Drop the legacy ``protein_id`` column that ``process_karmadock_files``
        # may have left behind (renamed from pdb_id) — bulk's rank engine only
        # needs ``protein_config_id``.
        return df[[COMBINATION_ID, PROTEIN_CONF_ID, LIGAND_ID, KARMADOCK_SCORE]].copy()
    except Exception:
        return pd.DataFrame(
            columns=[COMBINATION_ID, PROTEIN_CONF_ID, LIGAND_ID, KARMADOCK_SCORE]
        )
