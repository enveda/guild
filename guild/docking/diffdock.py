"""
Diffdock support functions
"""

import logging
import os
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm

from guild.constants.bulk import (
    BATCH_FOLDER,
    COMBINATION_ID,
    COMBINATIONS_TO_RUN_KEY,
)
from guild.constants.diffdock import (
    COMPLEX_NAME,
    DIFFDOCK_ARGS_FILE,
    DIFFDOCK_COMBINATIONS_FILE,
    DIFFDOCK_DIRECTORY,
    DIFFDOCK_RESULTS_FOLDER,
    LIGAND_DESCRIPTION,
    PROTEIN_PATH,
    PROTEIN_SEQUENCE,
)
from guild.constants.general import RANDOM_SEED
from guild.constants.guild import (
    DIFFDOCK_FOLDER,
    DIFFDOCK_SCORE,
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
from guild.docking.vina import (
    compute_box_from_sdf,
    vina_score_pose,
)
from guild.tools.preparation import _normalize_chain_list
from guild.transformers.converters import sdf_to_pdbqt

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


def process_diffdock_files(combination, diffdock_folder):
    """
    Process the diffdock csv files to retrieve the highest scores for each pdb
    :param input_combination: combination to process
    :param diffdock_folder: folder containing the diffdock results
    :param mode: single or bulk, depending on the input. If single, the ligand and protein names are the same for all rows. If bulk, the ligand and protein names are extracted from the pdb_id
    """
    folder_name = combination[0] + "_" + combination[1]
    full_folder = os.path.join(diffdock_folder, folder_name)

    # Extract IDs
    protein_conf, ligand_id = combination

    scores = [
        float(i.split("_confidence")[1].replace(".sdf", ""))
        for i in os.listdir(full_folder)
        if "_confidence" in i
    ]

    best_score = max(scores) if scores else 0.0

    return {
        COMBINATION_ID: folder_name,
        PROTEIN_CONF_ID: protein_conf,
        LIGAND_ID: ligand_id,
        DIFFDOCK_SCORE: best_score,
    }


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
        model_cache = os.environ.get(
            "DIFFDOCK_MODEL_CACHE", "/tmp/diffdock_models/workdir"
        )
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

    :param input_table: Table to write the combinations table for.
    :param output_dir: Path to the project directory.
    """

    diffdock_combinations_table = input_table.copy()

    diffdock_combinations_table[PROTEIN_PATH] = diffdock_combinations_table[PROTEIN_CONF_ID].apply(
        lambda x: f"{output_dir}/{PROTEINS_FOLDER}/{x}_raw.pdb"
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
    Perform the scoring of the dockinf results for DiffDock

    :param batch_dictionary: Dictionary containing the batch information.
    :return: DataFrame containing the docking scores.
    """
    diffdock_scores_data = []
    diffdock_folder = (
        f"{batch_dictionary[BATCH_FOLDER]}/{DIFFDOCK_FOLDER}/{DIFFDOCK_RESULTS_FOLDER}/"
    )
    for current_combination in tqdm(
        batch_dictionary[COMBINATIONS_TO_RUN_KEY],
        desc="DiffDock scoring",
    ):
        diffdock_scores_data.append(process_diffdock_files(current_combination, diffdock_folder))

    return pd.DataFrame(diffdock_scores_data)


# ────────────────────────────────────────────────────────────────────────────
# Vina score-only re-scoring of DiffDock poses
#
# DiffDock returns a relative confidence score, not a physics-based ΔG. For a
# comparable energy estimate we apply Vina's scoring function to the highest-
# confidence pose (no re-docking). DiffDock writes poses in the coordinate
# frame of the *raw* input PDB, so the rescore receptor must be derived from
# that same raw PDB — the cleaned/centred receptor used by Vina docking is in
# a different frame.
# ────────────────────────────────────────────────────────────────────────────


def _find_best_diffdock_sdf(diffdock_results_dir: str, protein_conf_id: str, ligand_id: str) -> str:
    """
    Locate the highest-confidence SDF file for a combination in a DiffDock
    results directory. DiffDock names output files like
    ``rank1_confidence-1.23.sdf``.

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


def _prepare_receptor_pdbqt_from_raw(
    raw_pdb: str,
    chain_id,
    output_pdbqt: str,
) -> str:
    """
    Extract one or more chains (ATOM records only) from a raw PDB and convert
    them to PDBQT via OpenBabel, preserving the original crystal coordinates so
    the receptor is in the same frame as DiffDock's output SDF files.

    ``chain_id`` may be a single chain ID, a list of IDs, or a comma-separated
    string (e.g. ``"A,B"``) so a multi-chain receptor is kept intact.
    """
    if not os.path.isfile(raw_pdb):
        raise FileNotFoundError(f"Raw PDB not found: {raw_pdb}")

    chain_ids = _normalize_chain_list(chain_id)
    chain_pdb = output_pdbqt.replace(".pdbqt", "_chain.pdb")
    kept = 0
    with open(raw_pdb) as fin, open(chain_pdb, "w") as fout:
        for line in fin:
            if line.startswith("ATOM") and len(line) > 21 and line[21] in chain_ids:
                fout.write(line)
                kept += 1
        fout.write("END\n")

    if kept == 0:
        raise ValueError(f"No ATOM records found for chain(s) {chain_ids} in {raw_pdb}")

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
        f"Prepared receptor PDBQT from raw PDB chain(s) {chain_ids}: "
        f"{kept} atoms → {output_pdbqt}"
    )
    return output_pdbqt


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
    Re-score a single DiffDock pose with Vina's physics-based scoring function.

    Pipeline:
    1. Find the highest-confidence SDF in ``diffdock_results_dir/{protein}_{ligand}/``.
    2. Convert SDF → PDBQT (preserving 3D coordinates).
    3. Compute a Vina box centred on the SDF pose.
    4. Call :func:`guild.docking.vina.vina_score_pose` (score-only).

    :return: Dict with ``combination``, ``protein_config_id``, ``ligand_id``,
        ``vina_rescore_diffdock_score``, ``diffdock_sdf``.
    """
    sdf_path = _find_best_diffdock_sdf(diffdock_results_dir, protein_conf_id, ligand_id)

    if output_dir is None:
        output_dir = os.path.dirname(sdf_path)

    ligand_pdbqt = os.path.join(output_dir, f"{protein_conf_id}_{ligand_id}_rescore.pdbqt")
    sdf_to_pdbqt(sdf_path, pdbqt=ligand_pdbqt)

    center, size = compute_box_from_sdf(sdf_path, padding=box_padding)

    score = vina_score_pose(
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        center=center,
        size=size,
        seed=seed,
    )

    combination_id = f"{protein_conf_id}_{ligand_id}"
    logger.info(f"Vina rescore (DiffDock pose): {combination_id} → {score:.3f} kcal/mol")

    return {
        COMBINATION_ID: combination_id,
        PROTEIN_CONF_ID: protein_conf_id,
        LIGAND_ID: ligand_id,
        VINA_RESCORE_DIFFDOCK_SCORE: score,
        "diffdock_sdf": sdf_path,
    }


def vina_rescore_diffdock_batch(
    batch_folder: str,
    combinations: list,
    receptor_pdbqt_dir: str = None,
    box_padding: float = 4.0,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Batch re-score DiffDock poses for all combinations in a batch.

    Receptor PDBQT resolution order:

    1. ``{receptor_pdbqt_dir}/{protein_conf_id}_raw.pdbqt`` — single-chain
       PDBQT already in the raw (crystal) coordinate frame.
    2. Auto-generated from ``{protein_conf_id}_raw.pdb`` by extracting the
       chain letter encoded in *protein_conf_id* and converting via OpenBabel.
    """
    diffdock_results_dir = os.path.join(batch_folder, DIFFDOCK_FOLDER, DIFFDOCK_RESULTS_FOLDER)
    if receptor_pdbqt_dir is None:
        receptor_pdbqt_dir = os.path.join(batch_folder, PROTEINS_FOLDER)

    rescore_output_dir = os.path.join(batch_folder, VINA_RESCORE_DIFFDOCK_FOLDER)
    os.makedirs(rescore_output_dir, exist_ok=True)

    results = []
    for protein_conf_id, ligand_id in combinations:
        receptor_pdbqt = os.path.join(
            receptor_pdbqt_dir, f"{protein_conf_id}_raw.pdbqt"
        )
        if not os.path.exists(receptor_pdbqt):
            raw_pdb = os.path.join(receptor_pdbqt_dir, f"{protein_conf_id}_raw.pdb")
            if os.path.isfile(raw_pdb):
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
                VINA_RESCORE_DIFFDOCK_SCORE: np.nan,
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
                VINA_RESCORE_DIFFDOCK_SCORE: np.nan,
                "diffdock_sdf": None,
            })

    return pd.DataFrame(results)


def vina_rescore_diffdock_guild_scoring(batch_dictionary):
    """
    Vina re-scoring of DiffDock poses for a batch.

    :return: DataFrame with ``COMBINATION_ID``, ``PROTEIN_CONF_ID``,
        ``LIGAND_ID``, and ``VINA_RESCORE_DIFFDOCK_SCORE`` columns. Returns an
        empty frame (with those columns) if no DiffDock outputs are found.
    """
    batch_folder = batch_dictionary[BATCH_FOLDER]
    combinations = batch_dictionary[COMBINATIONS_TO_RUN_KEY]

    diffdock_root = os.path.join(batch_folder, DIFFDOCK_FOLDER, DIFFDOCK_RESULTS_FOLDER)
    diffdock_present = os.path.isdir(diffdock_root) and len(os.listdir(diffdock_root)) > 0
    if not diffdock_present:
        logger.warning(
            "vina_rescore_diffdock: no DiffDock outputs found in %s — returning empty score table",
            batch_folder,
        )
        return pd.DataFrame(
            columns=[COMBINATION_ID, PROTEIN_CONF_ID, LIGAND_ID, VINA_RESCORE_DIFFDOCK_SCORE]
        )

    df = vina_rescore_diffdock_batch(batch_folder=batch_folder, combinations=combinations)
    keep = [COMBINATION_ID, PROTEIN_CONF_ID, LIGAND_ID, VINA_RESCORE_DIFFDOCK_SCORE]
    return df[[c for c in keep if c in df.columns]]
