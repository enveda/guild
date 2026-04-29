"""
Boltz support tools
"""

import json
import logging
import os
import subprocess

import pandas as pd
import yaml
from tqdm import tqdm

from guild.constants.boltz import (
    BOLTZ_PAIR_CHAINS_IPTM,
)
from guild.constants.bulk import (
    BATCH_FOLDER,
    COMBINATION_ID,
    COMBINATIONS_TABLE_KEY,
)
from guild.constants.guild import (
    BOLTZ_FOLDER,
    BOLTZ_SCORE,
    LIGAND_ID,
    PROTEIN_CONF_ID,
)

logger = logging.getLogger(__name__)


def generate_boltz_yaml(
    protein_sequence,
    protein_chain,
    ligand_sequences,
    ligand_ids,
    output_file,
    template_file=None,
    template_force=True,
    template_threshold=1.0,
    pocket_contacts=None,
    pocket_max_distance=4.0,
    pocket_force=True,
    msa_file=None,
):
    """
    Generate the Boltz yaml file.

    :param protein_sequence: Protein sequence.
    :param protein_chain: Protein chain.
    :param ligand_sequences: Ligand sequences.
    :param ligand_ids: Ligand ids.
    :param output_file: Output file.
    :param template_file: Template file.
    :param template_force: If True, enforce the template as a hard constraint (default True).
    :param template_threshold: Max backbone deviation (Å) from template when force=True (default 1.0).
    :param pocket_contacts: Optional list of ``[chain_id, res_seq]`` protein residue contacts
                            (from ``get_pocket_contacts_from_ligand``) for a Boltz pocket
                            constraint.  When provided, a ``constraints`` block is added to
                            the YAML.
    :param pocket_max_distance: Maximum heavy-atom distance (Å) for the pocket constraint (default 4.0).
    :param pocket_force: If True, use a potential to *enforce* the pocket constraint (default True).
    :param msa_file: Path to a pre-computed a3m MSA file for the protein.  When provided,
                     used instead of ``msa: empty``.
    """
    ligands_dictionary_list = []
    for ligand_sequence, ligand_id in zip(ligand_sequences, ligand_ids, strict=True):
        ligands_dictionary_list.append(
            {
                "ligand": {
                    "id": ligand_id,
                    "smiles": ligand_sequence,
                }
            }
        )
    affinities_dictionary_list = []
    for ligand_id in ligand_ids:
        affinities_dictionary_list.append(
            {
                "affinity": {
                    "binder": ligand_id,
                }
            }
        )

    boltz_yaml = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": protein_chain,
                    "sequence": protein_sequence,
                    "msa": msa_file if msa_file else "empty",
                }
            },
            *ligands_dictionary_list,
        ],
        "properties": affinities_dictionary_list,
    }

    if template_file:
        template_entry = {
            "pdb": template_file,
            "chain_id": [protein_chain],
            "template_id": [f"{protein_chain}1"],
        }
        if template_force:
            template_entry["force"] = True
            template_entry["threshold"] = template_threshold
        boltz_yaml["templates"] = [template_entry]

    if pocket_contacts:
        boltz_yaml["constraints"] = [
            {
                "pocket": {
                    "binder": ligand_ids[0],
                    "contacts": pocket_contacts,
                    "max_distance": pocket_max_distance,
                    "force": pocket_force,
                }
            }
        ]
        logger.debug(
            f"Boltz pocket constraint: {len(pocket_contacts)} contact residues, "
            f"binder={ligand_ids[0]}, max_distance={pocket_max_distance} Å"
        )

    with open(output_file, "w") as f:
        yaml.safe_dump(boltz_yaml, f, sort_keys=False, default_flow_style=None)


def deploy_boltz(
    yaml_file,
    out_dir="boltz_output",
    use_msa_server=False,
    use_gpu=True,
    cpu_threads=8,
    timeout=3600,
    recycling_steps=None,
    sampling_steps=None,
    sampling_steps_affinity=None,
    diffusion_samples_affinity=None,
    affinity_mw_correction=True,
):
    """
    Deploy Boltz for docking the ligand to the protein.
    :param yaml_file: Yaml file.
    :param out_dir: Output directory.
    :param use_msa_server: Use MSA server.
    :param use_gpu: Use GPU.
    :param cpu_threads: Number of CPU threads for Boltz (only applied when use_gpu=False).
    :param timeout: Timeout in seconds for the Boltz subprocess (default 1 hour).
    :param recycling_steps: Number of recycling steps (boltz default 3; use 1 on CPU).
    :param sampling_steps: Number of diffusion sampling steps (boltz default 200; use 10-50 on CPU).
    :param sampling_steps_affinity: Sampling steps for affinity head (boltz default 200; use 15 on CPU).
    :param diffusion_samples_affinity: Diffusion samples for affinity head (boltz default 5; use 1 on CPU).
    :param affinity_mw_correction: Apply molecular weight correction to affinity prediction.
    """
    gpu_flag = "gpu" if use_gpu else "cpu"

    # GPU defaults: halved sampling steps and one fewer recycling pass vs boltz
    # defaults (200 / 3).  Empirically this cuts wall-time ~50 % with minimal
    # accuracy loss; still well above the quality cliff (~50 steps / 1 recycle).
    if use_gpu:
        if recycling_steps is None:
            recycling_steps = 2
        if sampling_steps is None:
            sampling_steps = 100
        if sampling_steps_affinity is None:
            sampling_steps_affinity = 100
        if diffusion_samples_affinity is None:
            diffusion_samples_affinity = 3

    # Apply CPU-safe defaults when no GPU is available
    if not use_gpu:
        if recycling_steps is None:
            recycling_steps = 1
        if sampling_steps is None:
            sampling_steps = 15
        if sampling_steps_affinity is None:
            sampling_steps_affinity = 15
        if diffusion_samples_affinity is None:
            diffusion_samples_affinity = 1

    boltz_command = [
        "boltz",
        "predict",
        yaml_file,
        "--out_dir",
        out_dir,
        "--accelerator",
        gpu_flag,
    ]
    if use_msa_server:
        boltz_command.append("--use_msa_server")

    if recycling_steps is not None:
        boltz_command += ["--recycling_steps", str(recycling_steps)]

    if sampling_steps is not None:
        boltz_command += ["--sampling_steps", str(sampling_steps)]

    if sampling_steps_affinity is not None:
        boltz_command += ["--sampling_steps_affinity", str(sampling_steps_affinity)]

    if diffusion_samples_affinity is not None:
        boltz_command += ["--diffusion_samples_affinity", str(diffusion_samples_affinity)]

    if affinity_mw_correction:
        boltz_command.append("--affinity_mw_correction")

    if not use_gpu:
        boltz_command += ["--preprocessing-threads", str(cpu_threads)]

    env = os.environ.copy()
    if not use_gpu:
        # Prevent thread oversubscription on CPU-only machines
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[var] = str(cpu_threads)

    try:
        result = subprocess.run(
            boltz_command,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        logger.error(f"Boltz timed out after {timeout} seconds")
        if e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Error running Boltz command: {e}")
        raise

    if result.returncode != 0:
        logger.error(f"Boltz failed with exit code {result.returncode}")
        logger.error(f"STDOUT:\n{result.stdout}")
        logger.error(f"STDERR:\n{result.stderr}")
        raise subprocess.CalledProcessError(
            result.returncode, boltz_command, result.stdout, result.stderr
        )
    logger.info(f"Boltz finished successfully:\n{result.stdout}")


def process_boltz_output(json_file):
    """
    Process the Boltz output file.
    :param json_file: Input JSON file.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    return data[BOLTZ_PAIR_CHAINS_IPTM]


def boltz_guild_scoring(batch_dictionary):
    """
    Perform the scoring of the docking results for Boltz.
    Each protein-ligand pair was run as a separate 2-chain Boltz job, so we
    iterate over unique (protein_conf_id, ligand_id) pairs and read one JSON
    per pair.  In a 2-chain output the pair_chains_iptm matrix has:
        row/col 0 = protein,  row/col 1 = ligand
    and the off-diagonal element (iloc[1, 0]) is the protein-ligand ipTM score.

    :param batch_dictionary: Batch dictionary.
    :return: Boltz scores data.
    """
    boltz_scores_data = []
    missing_or_invalid = 0
    combinations_table = batch_dictionary[COMBINATIONS_TABLE_KEY]
    unique_pairs = combinations_table[[PROTEIN_CONF_ID, LIGAND_ID]].drop_duplicates()

    for _, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="Boltz scoring"):
        current_protein_configuration_id = row[PROTEIN_CONF_ID]
        current_ligand_id = row[LIGAND_ID]
        run_id = f"{current_protein_configuration_id}_{current_ligand_id}"

        boltz_folder = (
            f"{batch_dictionary[BATCH_FOLDER]}/{BOLTZ_FOLDER}/boltz_results_{run_id}_boltz"
        )
        current_boltz_json_file = (
            f"{boltz_folder}/predictions/{run_id}_boltz/confidence_{run_id}_boltz_model_0.json"
        )

        try:
            pair_chains_iptm = process_boltz_output(current_boltz_json_file)
            df = pd.DataFrame(pair_chains_iptm)
            # 2-chain layout: chain 0 = protein, chain 1 = ligand → off-diagonal score
            score = float(df.iloc[1, 0])
        except (FileNotFoundError, KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError) as e:
            missing_or_invalid += 1
            logger.warning(
                f"Skipping Boltz score for {run_id}: {e}"
            )
            continue

        boltz_scores_data.append(
            {
                LIGAND_ID: current_ligand_id,
                BOLTZ_SCORE: score,
                PROTEIN_CONF_ID: current_protein_configuration_id,
                COMBINATION_ID: f"{current_protein_configuration_id}_{current_ligand_id}",
            }
        )

    if missing_or_invalid > 0:
        logger.warning(
            f"Skipped {missing_or_invalid} Boltz result(s) due to missing/invalid output files"
        )

    return pd.DataFrame(
        boltz_scores_data,
        columns=[
            LIGAND_ID,
            BOLTZ_SCORE,
            PROTEIN_CONF_ID,
            COMBINATION_ID,
        ],
    )
