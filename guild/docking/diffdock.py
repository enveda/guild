"""
Diffdock support functions
"""

import glob
import json
import logging
import os
import subprocess

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from guild.constants.bulk import (
    BATCH_FOLDER,
    COMBINATION_ID,
    COMBINATIONS_TO_RUN_KEY,
)
from guild.constants.diffdock import (
    CLEAN_RECEPTOR_SUFFIX,
    COMPLEX_NAME,
    DEFAULT_DIFFDOCK_POCKET,
    DIFFDOCK_ARGS_FILE,
    DIFFDOCK_COMBINATIONS_FILE,
    DIFFDOCK_DIRECTORY,
    DIFFDOCK_POCKET_AUTO,
    DIFFDOCK_POCKET_BLIND,
    DIFFDOCK_POCKET_BOX,
    DIFFDOCK_POCKET_KEY,
    DIFFDOCK_POCKET_MODES,
    DIFFDOCK_POSE_SELECTION,
    DIFFDOCK_RESULTS_FOLDER,
    LIGAND_DESCRIPTION,
    POSE_NO_BOX,
    POSE_NO_SAMPLE_IN_BOX,
    POSE_NO_SAMPLES,
    POSE_SELECTED_BLIND,
    POSE_SELECTED_BOX,
    PROTEIN_PATH,
    PROTEIN_SEQUENCE,
    RECEPTOR_CONTACT_CUTOFF,
    SELECTED_POSE_FILE,
)
from guild.constants.general import RANDOM_SEED
from guild.constants.guild import (
    BOXES_FOLDER,
    DIFFDOCK_FOLDER,
    DIFFDOCK_SCORE,
    GNINA_RESCORE_DIFFDOCK_CNN_SCORE,
    GNINA_RESCORE_DIFFDOCK_FOLDER,
    GNINA_RESCORE_DIFFDOCK_SCORE,
    LIGAND_ID,
    PROTEIN_CONF_ID,
    PROTEINS_FOLDER,
    SMILES,
    VINA_RESCORE_DIFFDOCK_FOLDER,
    VINA_RESCORE_DIFFDOCK_SCORE,
)
from guild.constants.system import (
    PYTHON_EXECUTABLE,
    SUPPORT_FOLDER,
)
from guild.constants.vina import VINA_BOXES_FOLDER
from guild.docking.gnina import gnina_score_pose
from guild.docking.vina import (
    compute_box_from_sdf,
    get_center_and_size_from_box_file,
    vina_score_pose,
)
from guild.transformers.converters import protein_pdb_to_pdbqt, sdf_to_pdbqt

logger = logging.getLogger(__name__)


def generate_diffdock_table(
    complex_name, protein_path, ligand_description="", protein_sequence="", output_csv=None
):
    """
    Generate the table for Diffdock.
    :param complex_name: Name of the complex.
    :param protein_path: Path to the protein.
    :param ligand_description: Description of the ligand.
    :param protein_sequence: Sequence of the protein.
    :param output_csv: Path to the output csv file.
    """
    with open(output_csv, "w") as f:
        f.write("complex_name,protein_path,ligand_description,protein_sequence\n")
        f.write(f"{complex_name},{protein_path},{ligand_description},{protein_sequence}\n")
    return output_csv


def clean_receptor_path(batch_folder: str, protein_conf_id: str, extension: str = "pdb") -> str:
    """Prepared receptor Guild writes during prep, in the Vina/pocket-box frame."""
    return os.path.join(
        batch_folder, PROTEINS_FOLDER, f"{protein_conf_id}{CLEAN_RECEPTOR_SUFFIX}.{extension}"
    )


def _diffdock_samples(combo_dir: str) -> list:
    """``(confidence, sdf_path)`` for every DiffDock sample, highest confidence first.

    Confidence comes from the ``rank<N>_confidence<score>.sdf`` filename, which
    avoids the lexicographic rank trap (``rank10`` sorts before ``rank2``).
    """
    if not os.path.isdir(combo_dir):
        return []
    samples = []
    for fname in os.listdir(combo_dir):
        if "_confidence" not in fname or not fname.endswith(".sdf"):
            continue
        try:
            confidence = float(fname.split("_confidence")[1].replace(".sdf", ""))
        except ValueError:
            continue
        samples.append((confidence, os.path.join(combo_dir, fname)))
    return sorted(samples, key=lambda item: item[0], reverse=True)


def _heavy_atom_coordinates(sdf_path: str):
    """Heavy-atom coordinates of the first molecule in an SDF, or ``None`` if unreadable."""
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    positions = mol.GetConformer().GetPositions()
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    return positions[heavy] if heavy else None


def _inside_box(point, center, size) -> bool:
    return all(abs(p - c) <= s / 2.0 for p, c, s in zip(point, center, size, strict=True))


def find_pocket_box(batch_folder: str, protein_conf_id: str, ligand_id: str):
    """Pocket box for a combination, or ``None`` when no pocket was defined.

    Every pocket source (user box, P2Rank, co-crystal ligand) ends up in
    ``boxes/vina_boxes/<combination>.txt``. That file is written per ligand
    during Vina prep, so when it is missing another ligand's box for the same
    protein is used: the pocket centre is per protein.
    """
    boxes_dir = os.path.join(batch_folder, BOXES_FOLDER, VINA_BOXES_FOLDER)
    exact = os.path.join(boxes_dir, f"{protein_conf_id}_{ligand_id}.txt")
    if os.path.isfile(exact):
        return exact
    siblings = sorted(
        glob.glob(os.path.join(glob.escape(boxes_dir), f"{glob.escape(protein_conf_id)}_*.txt"))
    )
    return siblings[0] if siblings else None


def select_diffdock_pose(combo_dir: str, box_file, mode: str = DEFAULT_DIFFDOCK_POCKET) -> dict:
    """Choose which DiffDock sample represents a combination.

    - ``box`` (or ``auto`` with a box): the highest-confidence sample whose
      heavy-atom centroid lies inside the pocket box.
    - ``blind`` (or ``auto`` without a box): the highest-confidence sample.

    :return: dict with ``sdf`` (path or ``None``), ``confidence`` (NaN when no
        pose), ``reason`` (one of the ``POSE_*`` outcomes), ``mode``,
        ``box_file`` and ``n_samples``.
    """
    if mode not in DIFFDOCK_POCKET_MODES:
        raise ValueError(
            f"Invalid DiffDock pocket mode {mode!r}; expected one of {DIFFDOCK_POCKET_MODES}."
        )

    samples = _diffdock_samples(combo_dir)
    selection = {
        "mode": mode,
        "box_file": box_file,
        "n_samples": len(samples),
        "sdf": None,
        "confidence": np.nan,
        "reason": POSE_NO_SAMPLES,
    }
    if not samples:
        return selection

    use_box = mode == DIFFDOCK_POCKET_BOX or (mode == DIFFDOCK_POCKET_AUTO and box_file is not None)
    if not use_box:
        confidence, sdf_path = samples[0]
        return {
            **selection,
            "sdf": sdf_path,
            "confidence": confidence,
            "reason": POSE_SELECTED_BLIND,
        }

    if box_file is None:
        return {**selection, "reason": POSE_NO_BOX}

    center, size = get_center_and_size_from_box_file(box_file)
    for confidence, sdf_path in samples:
        coordinates = _heavy_atom_coordinates(sdf_path)
        if coordinates is not None and _inside_box(coordinates.mean(axis=0), center, size):
            return {
                **selection,
                "sdf": sdf_path,
                "confidence": confidence,
                "reason": POSE_SELECTED_BOX,
            }
    return {**selection, "reason": POSE_NO_SAMPLE_IN_BOX}


def diffdock_combo_dir(batch_folder: str, protein_conf_id: str, ligand_id: str) -> str:
    return os.path.join(
        batch_folder, DIFFDOCK_FOLDER, DIFFDOCK_RESULTS_FOLDER, f"{protein_conf_id}_{ligand_id}"
    )


def resolve_diffdock_pose(
    batch_folder: str, protein_conf_id: str, ligand_id: str, mode: str = DEFAULT_DIFFDOCK_POCKET
) -> dict:
    """Select the pose for a combination and record the choice next to its samples.

    The record (``selected_pose.json``) is what PoseBusters reads, so every
    consumer of "the DiffDock pose" agrees without re-deriving the rule.
    """
    combo_dir = diffdock_combo_dir(batch_folder, protein_conf_id, ligand_id)
    box_file = (
        None
        if mode == DIFFDOCK_POCKET_BLIND
        else find_pocket_box(batch_folder, protein_conf_id, ligand_id)
    )
    selection = select_diffdock_pose(combo_dir, box_file, mode)

    if os.path.isdir(combo_dir):
        record = {
            **selection,
            "sdf": os.path.basename(selection["sdf"]) if selection["sdf"] else None,
            "confidence": None if pd.isna(selection["confidence"]) else selection["confidence"],
        }
        with open(os.path.join(combo_dir, SELECTED_POSE_FILE), "w") as handle:
            json.dump(record, handle, indent=2)
    return selection


def read_selected_pose(combo_dir: str):
    """The recorded selection for a combination, with ``sdf`` as a full path, or ``None``."""
    path = os.path.join(combo_dir, SELECTED_POSE_FILE)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        record = json.load(handle)
    if record.get("sdf"):
        record["sdf"] = os.path.join(combo_dir, record["sdf"])
    return record


def pocket_mode(batch_dictionary: dict) -> str:
    return batch_dictionary.get(DIFFDOCK_POCKET_KEY, DEFAULT_DIFFDOCK_POCKET)


def deploy_diffdock_single(
    home_path: str,
    combination_id: str,
    cleaned_protein: str,
    original_ligand_smile: str,
    project_dir: str,
    diffdock_results_dir: str,
    input_csv: str = None,
    use_gpu: bool = True,
):
    """
    Run DiffDock for docking the ligand to the protein. Output is saved in the diffdock directory, inside the project folder.
    :param home_path: Path to the home directory.
    :param combination_id: ID of the combination.
    :param cleaned_protein: Path to the cleaned protein.
    :param original_ligand_smile: Description of the ligand.
    :param project_dir: Path to the project directory.
    :param diffdock_results_dir: Path to the diffdock results directory.
    :param input_csv: Path to the input csv file.
    :return: Failed steps.
    """
    if input_csv is None:
        input_csv_path = generate_diffdock_table(
            complex_name=combination_id,
            protein_path=cleaned_protein,
            ligand_description=original_ligand_smile,
            protein_sequence="",
            output_csv=f"{project_dir}/{DIFFDOCK_COMBINATIONS_FILE}",
        )
    else:
        input_csv_path = input_csv
    status = deploy_diffdock(home_path, diffdock_results_dir, input_csv_path)
    return status


def deploy_diffdock(
    home_path: str,
    diffdock_results_dir: str,
    input_csv: str = None,
    subprocess_log_path: str = None,
):
    """
    Deploy DiffDock for docking the ligand to the protein. Output is saved in the diffdock directory, inside the project folder.
    :param home_path: Path to the home directory.
    :param diffdock_results_dir: Path to the diffdock results directory.
    :param input_csv: Path to the input csv file.
    :param subprocess_log_path: Optional path to write the full DiffDock
        stdout/stderr transcript. DiffDock runs once per batch (not per
        combination), so this is typically
        ``batches/<batch>/diffdock/_batch.subprocess.log``.
    :return: Status of the deployment.
    """
    os.makedirs(diffdock_results_dir, exist_ok=True)

    # Resolve the DiffDock directory: prefer /app/DiffDock (Docker) over
    # home_path/DiffDock (local dev).
    diffdock_dir = os.path.join(home_path, DIFFDOCK_DIRECTORY)
    if not os.path.isdir(diffdock_dir):
        diffdock_dir = os.path.join("/app", DIFFDOCK_DIRECTORY)
    if not os.path.isdir(diffdock_dir):
        logger.error(f"DiffDock directory not found at {diffdock_dir}")
        return 1

    arg_file = f"{SUPPORT_FOLDER}/{DIFFDOCK_ARGS_FILE}"

    # DiffDock's so3.py writes pre-computed numpy caches to CWD and
    # inference.py downloads model weights into model_dir.  When the
    # DiffDock repo is read-only (Docker image owned by another user)
    # both operations would fail with PermissionError.
    # Solution: run the subprocess from a writable scratch directory and
    # add diffdock_dir to PYTHONPATH so ``python -m inference`` resolves.
    diffdock_writable = not os.access(diffdock_dir, os.W_OK)
    if diffdock_writable:
        run_cwd = os.environ.get("DIFFDOCK_RUN_DIR", "/tmp/diffdock_run")
        os.makedirs(run_cwd, exist_ok=True)
        model_cache = os.environ.get("DIFFDOCK_MODEL_CACHE", "/tmp/diffdock_models/workdir")
    else:
        run_cwd = diffdock_dir
        model_cache = os.path.join(diffdock_dir, "workdir")

    score_model_dir = os.path.join(model_cache, "v1.1", "score_model")
    confidence_model_dir = os.path.join(model_cache, "v1.1", "confidence_model")

    cmd = [
        PYTHON_EXECUTABLE,
        "-m",
        "inference",
        "--config",
        arg_file,
        "--protein_ligand_csv",
        input_csv,
        "--out_dir",
        diffdock_results_dir,
        "--model_dir",
        score_model_dir,
        "--confidence_model_dir",
        confidence_model_dir,
    ]
    env = os.environ.copy()
    # Ensure DiffDock is importable when running from a different cwd.
    python_path = env.get("PYTHONPATH", "")
    if diffdock_dir not in python_path:
        env["PYTHONPATH"] = f"{diffdock_dir}:{python_path}" if python_path else diffdock_dir

    logger.info(f"DiffDock cmd: {' '.join(cmd)}")
    logger.info(f"DiffDock cwd: {run_cwd}  (repo: {diffdock_dir})")
    result = subprocess.run(cmd, text=True, env=env, capture_output=True, cwd=run_cwd)

    if result.stdout:
        logger.info(f"DiffDock STDOUT: {result.stdout[-2000:]}")
    if result.stderr:
        logger.warning(f"DiffDock STDERR: {result.stderr[-2000:]}")

    if subprocess_log_path is not None:
        from guild.tools.subprocess_log import write_subprocess_log

        write_subprocess_log(
            subprocess_log_path,
            argv=cmd,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    if result.returncode != 0:
        logger.error(f"DiffDock FAILED (exit code {result.returncode})")
        return 1

    return 0


def write_diffdock_combinations_table(input_table, output_dir):
    """
    Write the combinations table to the project directory for DiffDock.

    DiffDock docks into the same prepared receptor as Vina and gnina
    (``proteins/<id>_single_chain_clean.pdb``: the receptor chain(s) only, in
    the pocket-box frame), not the deposited file, which can carry G proteins,
    antibodies or fusion partners that a blind docker would search too.

    :param input_table: Table to write the combinations table for.
    :param output_dir: Path to the batch directory.
    """

    diffdock_combinations_table = input_table.copy()

    diffdock_combinations_table[PROTEIN_PATH] = diffdock_combinations_table[PROTEIN_CONF_ID].apply(
        lambda x: clean_receptor_path(output_dir, x)
    )

    diffdock_combinations_table[COMPLEX_NAME] = (
        diffdock_combinations_table[PROTEIN_CONF_ID] + "_" + diffdock_combinations_table[LIGAND_ID]
    )

    diffdock_combinations_table[LIGAND_DESCRIPTION] = diffdock_combinations_table[SMILES]
    diffdock_combinations_table[PROTEIN_SEQUENCE] = ""
    diffdock_combinations_table[
        [COMPLEX_NAME, PROTEIN_PATH, LIGAND_DESCRIPTION, PROTEIN_SEQUENCE]
    ].to_csv(
        f"{output_dir}/{DIFFDOCK_COMBINATIONS_FILE}",
        index=False,
    )


def diffdock_guild_scoring(batch_dictionary):
    """
    DiffDock confidence of the selected pose for each combination in a batch.

    A combination with no selectable pose scores NaN (not 0.0, which is a
    mid-range confidence). ``diffdock_pose_selection`` records which rule
    chose the pose, or why none was chosen.

    :param batch_dictionary: Dictionary containing the batch information.
    :return: DataFrame containing the docking scores.
    """
    batch_folder = batch_dictionary[BATCH_FOLDER]
    mode = pocket_mode(batch_dictionary)
    rows = []
    for protein_conf_id, ligand_id in tqdm(
        batch_dictionary[COMBINATIONS_TO_RUN_KEY],
        desc="DiffDock scoring",
    ):
        selection = resolve_diffdock_pose(batch_folder, protein_conf_id, ligand_id, mode)
        rows.append(
            {
                COMBINATION_ID: f"{protein_conf_id}_{ligand_id}",
                PROTEIN_CONF_ID: protein_conf_id,
                LIGAND_ID: ligand_id,
                DIFFDOCK_SCORE: selection["confidence"],
                DIFFDOCK_POSE_SELECTION: selection["reason"],
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            COMBINATION_ID,
            PROTEIN_CONF_ID,
            LIGAND_ID,
            DIFFDOCK_SCORE,
            DIFFDOCK_POSE_SELECTION,
        ],
    )


# ────────────────────────────────────────────────────────────────────────────
# Rescoring of DiffDock poses (Vina and gnina)
#
# DiffDock's confidence is not a binding energy, so the selected pose is also
# scored with Vina and gnina. Both use the receptor DiffDock docked into
# (Guild's prepared receptor, so the frames already agree), and the pose is
# locally minimised first: raw DiffDock poses routinely carry heavy-atom
# clashes that would otherwise dominate the score. A pose that touches no
# receptor atom is reported as NaN rather than the ~0 an empty grid returns.
# ────────────────────────────────────────────────────────────────────────────


def _has_receptor_contact(
    sdf_path: str, receptor_pdb: str, cutoff: float = RECEPTOR_CONTACT_CUTOFF
) -> bool:
    ligand = _heavy_atom_coordinates(sdf_path)
    if ligand is None:
        return False
    receptor = []
    with open(receptor_pdb) as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and line[76:78].strip().upper() != "H":
                receptor.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not receptor:
        return False
    distances = np.linalg.norm(ligand[:, None, :] - np.asarray(receptor)[None, :, :], axis=-1)
    return bool((distances < cutoff).any())


def _rescorable_pose(
    batch_folder: str, protein_conf_id: str, ligand_id: str, mode: str, track: str
):
    """The selected pose's SDF when it can be rescored, else ``None`` (reason logged)."""
    combination_id = f"{protein_conf_id}_{ligand_id}"
    selection = resolve_diffdock_pose(batch_folder, protein_conf_id, ligand_id, mode)
    if selection["sdf"] is None:
        logger.info(f"{track}: no DiffDock pose for {combination_id} ({selection['reason']})")
        return None

    receptor_pdb = clean_receptor_path(batch_folder, protein_conf_id)
    if not os.path.isfile(receptor_pdb):
        logger.warning(f"{track}: prepared receptor {receptor_pdb} not found for {combination_id}")
        return None
    if not _has_receptor_contact(selection["sdf"], receptor_pdb):
        logger.warning(
            f"{track}: DiffDock pose for {combination_id} has no receptor heavy atom within "
            f"{RECEPTOR_CONTACT_CUTOFF} Å; reporting NaN"
        )
        return None
    return selection["sdf"]


def rescore_diffdock_pose(
    receptor_pdbqt: str,
    sdf_path: str,
    output_dir: str,
    combination_id: str,
    box_padding: float = 6.0,
    seed: int = RANDOM_SEED,
) -> float:
    """
    Vina energy of a DiffDock pose after local minimisation (no re-docking).

    The grid is a box around the pose itself; the pocket restriction has
    already been applied when the pose was selected.

    :return: Vina energy in kcal/mol (lower = better).
    """
    ligand_pdbqt = os.path.join(output_dir, f"{combination_id}_rescore.pdbqt")
    # DiffDock writes heavy atoms only; Vina needs polar hydrogens to type donors.
    sdf_to_pdbqt(sdf_path, pdbqt=ligand_pdbqt, add_hydrogens=True)

    center, size = compute_box_from_sdf(sdf_path, padding=box_padding)
    score = vina_score_pose(
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        center=center,
        size=size,
        seed=seed,
        minimize=True,
        output_pdbqt=os.path.join(output_dir, f"{combination_id}_rescore_minimized.pdbqt"),
    )
    logger.info(f"Vina rescore (DiffDock pose): {combination_id} → {score:.3f} kcal/mol")
    return score


def _prepared_receptor_pdbqt(batch_folder: str, protein_conf_id: str) -> str:
    """Vina's receptor PDBQT, prepared the way Vina docking prepares it when Vina did not run."""
    receptor_pdbqt = clean_receptor_path(batch_folder, protein_conf_id, "pdbqt")
    if not os.path.isfile(receptor_pdbqt):
        protein_pdb_to_pdbqt(
            input_pdb=clean_receptor_path(batch_folder, protein_conf_id),
            output_pdbqt=receptor_pdbqt,
            allow_bad_res=True,
        )
    return receptor_pdbqt


def vina_rescore_diffdock_batch(
    batch_folder: str,
    combinations: list,
    mode: str = DEFAULT_DIFFDOCK_POCKET,
    box_padding: float = 6.0,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Vina-rescore the selected DiffDock pose of every combination in a batch."""
    output_dir = os.path.join(batch_folder, VINA_RESCORE_DIFFDOCK_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for protein_conf_id, ligand_id in combinations:
        combination_id = f"{protein_conf_id}_{ligand_id}"
        row = {
            COMBINATION_ID: combination_id,
            PROTEIN_CONF_ID: protein_conf_id,
            LIGAND_ID: ligand_id,
            VINA_RESCORE_DIFFDOCK_SCORE: np.nan,
        }
        try:
            sdf_path = _rescorable_pose(
                batch_folder, protein_conf_id, ligand_id, mode, "Vina rescore"
            )
            if sdf_path is not None:
                row[VINA_RESCORE_DIFFDOCK_SCORE] = rescore_diffdock_pose(
                    receptor_pdbqt=_prepared_receptor_pdbqt(batch_folder, protein_conf_id),
                    sdf_path=sdf_path,
                    output_dir=output_dir,
                    combination_id=combination_id,
                    box_padding=box_padding,
                    seed=seed,
                )
        except Exception as e:
            logger.warning(f"Vina rescore failed for {combination_id}: {e}")
        rows.append(row)

    return pd.DataFrame(
        rows, columns=[COMBINATION_ID, PROTEIN_CONF_ID, LIGAND_ID, VINA_RESCORE_DIFFDOCK_SCORE]
    )


def _diffdock_outputs_present(batch_folder: str) -> bool:
    diffdock_root = os.path.join(batch_folder, DIFFDOCK_FOLDER, DIFFDOCK_RESULTS_FOLDER)
    return os.path.isdir(diffdock_root) and len(os.listdir(diffdock_root)) > 0


def vina_rescore_diffdock_guild_scoring(batch_dictionary):
    """
    Vina re-scoring of DiffDock poses for a batch.

    :return: DataFrame with ``COMBINATION_ID``, ``PROTEIN_CONF_ID``,
        ``LIGAND_ID``, and ``VINA_RESCORE_DIFFDOCK_SCORE`` columns. Returns an
        empty frame (with those columns) if no DiffDock outputs are found.
    """
    batch_folder = batch_dictionary[BATCH_FOLDER]
    columns = [COMBINATION_ID, PROTEIN_CONF_ID, LIGAND_ID, VINA_RESCORE_DIFFDOCK_SCORE]
    if not _diffdock_outputs_present(batch_folder):
        logger.warning(
            "vina_rescore_diffdock: no DiffDock outputs found in %s — returning empty score table",
            batch_folder,
        )
        return pd.DataFrame(columns=columns)

    return vina_rescore_diffdock_batch(
        batch_folder=batch_folder,
        combinations=batch_dictionary[COMBINATIONS_TO_RUN_KEY],
        mode=pocket_mode(batch_dictionary),
    )


def rescore_diffdock_pose_gnina(
    receptor_pdb: str,
    sdf_path: str,
    output_dir: str,
    combination_id: str,
    box_padding: float = 6.0,
    seed: int = RANDOM_SEED,
    use_gpu: bool = False,
    subprocess_log_path: str = None,
) -> tuple[float, float]:
    """
    gnina energy and CNN score of a DiffDock pose after local minimisation.

    gnina reads the SDF and the plain-PDB receptor directly, so no PDBQT
    conversion is involved.

    :return: ``(affinity, cnn_score)``.
    """
    center, size = compute_box_from_sdf(sdf_path, padding=box_padding)
    score, cnn_score = gnina_score_pose(
        receptor=receptor_pdb,
        ligand=sdf_path,
        center=center,
        size=size,
        output_dir=output_dir,
        run_id=combination_id,
        seed=seed,
        use_gpu=use_gpu,
        subprocess_log_path=subprocess_log_path,
        minimize=True,
    )
    logger.info(f"gnina rescore (DiffDock pose): {combination_id} → {score:.3f} kcal/mol")
    return score, cnn_score


def gnina_rescore_diffdock_batch(
    batch_folder: str,
    combinations: list,
    mode: str = DEFAULT_DIFFDOCK_POCKET,
    box_padding: float = 6.0,
    seed: int = RANDOM_SEED,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """gnina-rescore the selected DiffDock pose of every combination in a batch."""
    output_dir = os.path.join(batch_folder, GNINA_RESCORE_DIFFDOCK_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for protein_conf_id, ligand_id in combinations:
        combination_id = f"{protein_conf_id}_{ligand_id}"
        row = {
            COMBINATION_ID: combination_id,
            PROTEIN_CONF_ID: protein_conf_id,
            LIGAND_ID: ligand_id,
            GNINA_RESCORE_DIFFDOCK_SCORE: np.nan,
            GNINA_RESCORE_DIFFDOCK_CNN_SCORE: np.nan,
        }
        try:
            sdf_path = _rescorable_pose(
                batch_folder, protein_conf_id, ligand_id, mode, "gnina rescore"
            )
            if sdf_path is not None:
                score, cnn_score = rescore_diffdock_pose_gnina(
                    receptor_pdb=clean_receptor_path(batch_folder, protein_conf_id),
                    sdf_path=sdf_path,
                    output_dir=output_dir,
                    combination_id=combination_id,
                    box_padding=box_padding,
                    seed=seed,
                    use_gpu=use_gpu,
                    subprocess_log_path=os.path.join(
                        output_dir, f"{combination_id}.subprocess.log"
                    ),
                )
                row[GNINA_RESCORE_DIFFDOCK_SCORE] = score
                row[GNINA_RESCORE_DIFFDOCK_CNN_SCORE] = cnn_score
        except Exception as e:
            logger.warning(f"gnina rescore failed for {combination_id}: {e}")
        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=[
            COMBINATION_ID,
            PROTEIN_CONF_ID,
            LIGAND_ID,
            GNINA_RESCORE_DIFFDOCK_SCORE,
            GNINA_RESCORE_DIFFDOCK_CNN_SCORE,
        ],
    )


def gnina_rescore_diffdock_guild_scoring(batch_dictionary, use_gpu: bool = False):
    """
    gnina re-scoring of DiffDock poses for a batch.

    :return: DataFrame with ``COMBINATION_ID``, ``PROTEIN_CONF_ID``,
        ``LIGAND_ID``, ``GNINA_RESCORE_DIFFDOCK_SCORE``, and the
        ``GNINA_RESCORE_DIFFDOCK_CNN_SCORE`` side-channel. Returns an empty
        frame (with those columns) if no DiffDock outputs are found.
    """
    batch_folder = batch_dictionary[BATCH_FOLDER]
    columns = [
        COMBINATION_ID,
        PROTEIN_CONF_ID,
        LIGAND_ID,
        GNINA_RESCORE_DIFFDOCK_SCORE,
        GNINA_RESCORE_DIFFDOCK_CNN_SCORE,
    ]
    if not _diffdock_outputs_present(batch_folder):
        logger.warning(
            "gnina_rescore_diffdock: no DiffDock outputs found in %s — returning empty score table",
            batch_folder,
        )
        return pd.DataFrame(columns=columns)

    df = gnina_rescore_diffdock_batch(
        batch_folder=batch_folder,
        combinations=batch_dictionary[COMBINATIONS_TO_RUN_KEY],
        mode=pocket_mode(batch_dictionary),
        use_gpu=use_gpu,
    )
    return df.reindex(columns=columns)
