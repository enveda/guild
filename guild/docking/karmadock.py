"""
KarmaDock tools
"""

import logging
import os
import subprocess
import time

import pandas as pd

from guild.constants.bulk import BATCH_FOLDER, COMBINATION_ID
from guild.constants.guild import (
    KARMADOCK_FOLDER,
    KARMADOCK_SCORE,
    LIGAND_ID,
    PROTEIN_ID,
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
    pre_processing_command = f"""
    {PYTHON_EXECUTABLE} -u {home_path}/KarmaDock/utils/pre_processing.py --complex_file_dir {karmadock_data_dir} {SHELL_SILENCER}
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
    {PYTHON_EXECUTABLE} -u {home_path}/KarmaDock/utils/generate_graph.py --complex_file_dir {karmadock_data_dir} --graph_file_dir {karmadock_graphs_dir} {SHELL_SILENCER}
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
    {PYTHON_EXECUTABLE} -u {home_path}/KarmaDock/utils/ligand_docking.py  \
        --graph_file_dir {karmadock_graphs_dir} \
        --model_file {home_path}/KarmaDock/trained_models/karmadock_screening.pkl \
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
    Ensures that output contains:
        combination_id, protein_id, ligand_id, karmadock_score
    """

    df = (
        process_karmadock_files(
            f"{batch_dictionary[BATCH_FOLDER]}/{KARMADOCK_FOLDER}/{KARMADOCK_RESULTS_FOLDER}",
            mode="bulk",
        )
        .rename(columns={KARMADOCK_RAW_SCORE: KARMADOCK_SCORE, KARMADOCK_PDB_ID: COMBINATION_ID})
        .drop(columns=KARMADOCK_COLUMNS_TO_DROP, errors="ignore")
    )

    def split_combination(cid: str):
        if not isinstance(cid, str) or "_" not in cid:
            # fallback case → avoid crash
            return ("unknown_protein", "unknown_ligand")

        parts = cid.split("_")
        if len(parts) < 2:
            return ("unknown_protein", "unknown_ligand")

        protein = parts[0]
        ligand = parts[-1]
        return (protein, ligand)

    try:
        df[PROTEIN_ID], df[LIGAND_ID] = zip(
            *df[COMBINATION_ID].apply(split_combination), strict=True
        )
        df[[COMBINATION_ID, PROTEIN_ID, LIGAND_ID, KARMADOCK_SCORE]].copy()
    except Exception:
        return pd.DataFrame(columns=[COMBINATION_ID, PROTEIN_ID, LIGAND_ID, KARMADOCK_SCORE])
