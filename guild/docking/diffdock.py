"""
Diffdock support functions
"""

import logging
import os
import subprocess

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
from guild.constants.guild import (
    DIFFDOCK_FOLDER,
    DIFFDOCK_SCORE,
    LIGAND_ID,
    PROTEIN_CONF_ID,
    PROTEINS_FOLDER,
    SMILES,
)
from guild.constants.system import (
    PYTHON_EXECUTABLE,
    SUPPORT_FOLDER,
)

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


def deploy_diffdock(home_path: str, diffdock_results_dir: str, input_csv: str = None):
    """
    Deploy DiffDock for docking the ligand to the protein. Output is saved in the diffdock directory, inside the project folder.
    :param home_path: Path to the home directory.
    :param diffdock_results_dir: Path to the diffdock results directory.
    :param input_csv: Path to the input csv file.
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
