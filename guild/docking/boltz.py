"""
Boltz support tools
"""

import json
import logging
import os
import subprocess
import tempfile

import numpy as np
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
    COMBINATIONS_TO_RUN_KEY,
)
from guild.constants.general import RANDOM_SEED
from guild.constants.guild import (
    BOLTZ_FOLDER,
    BOLTZ_SCORE,
    LIGAND_ID,
    PROTEIN_CONF_ID,
    VINA_RESCORE_BOLTZ_FOLDER,
    VINA_RESCORE_BOLTZ_SCORE,
)
from guild.docking.vina import (
    _compute_box_from_pdb_atoms,
    _extract_ligand_records,
    _extract_protein_from_complex,
    vina_score_pose,
)
from guild.transformers.converters import (
    cif_to_pdb,
    ligand_pdb_to_pdbqt,
    protein_pdb_to_pdbqt,
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

    :param protein_sequence: Protein sequence, or a list of sequences for a
                             multi-chain receptor (one per chain in ``protein_chain``).
    :param protein_chain: Protein chain ID, or a list of chain IDs for a
                          multi-chain receptor (e.g. ``["A", "B"]``). When more
                          than one chain is given, one ``protein`` block is
                          emitted per chain and the template/MSA are applied
                          per chain.
    :param ligand_sequences: Ligand sequences.
    :param ligand_ids: Ligand ids.
    :param output_file: Output file.
    :param template_file: Template file.
    :param template_force: If True, enforce the template as a hard constraint (default True).
    :param template_threshold: Max backbone deviation (Å) from template when force=True (default 1.0).
    :param pocket_contacts: Optional list of ``[chain_id, res_seq]`` protein residue contacts
                            (from ``get_pocket_contacts_from_ligand``) for a Boltz pocket
                            constraint.  When provided, a ``constraints`` block is added to
                            the YAML. May span multiple chains.
    :param pocket_max_distance: Maximum heavy-atom distance (Å) for the pocket constraint (default 4.0).
    :param pocket_force: If True, use a potential to *enforce* the pocket constraint (default True).
    :param msa_file: Path to a pre-computed a3m MSA file for the protein, or a
                     list of paths (one per chain, same order as
                     ``protein_chain``). When provided, used instead of
                     ``msa: empty``.
    """
    # Normalize protein chains / sequences / MSAs to per-chain lists so single-
    # and multi-chain callers share one code path.
    protein_chains = [protein_chain] if isinstance(protein_chain, str) else list(protein_chain)
    protein_sequences = (
        [protein_sequence] if isinstance(protein_sequence, str) else list(protein_sequence)
    )
    if len(protein_sequences) != len(protein_chains):
        raise ValueError(
            f"generate_boltz_yaml: {len(protein_chains)} chains but "
            f"{len(protein_sequences)} sequences — must match."
        )
    if msa_file is None or isinstance(msa_file, str):
        msa_files = [msa_file] * len(protein_chains)
    else:
        msa_files = list(msa_file)
        if len(msa_files) != len(protein_chains):
            raise ValueError(
                f"generate_boltz_yaml: {len(protein_chains)} chains but "
                f"{len(msa_files)} MSA files — must match."
            )

    protein_dictionary_list = [
        {
            "protein": {
                "id": chain,
                "sequence": sequence,
                "msa": msa if msa else "empty",
            }
        }
        for chain, sequence, msa in zip(
            protein_chains, protein_sequences, msa_files, strict=True
        )
    ]

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
            *protein_dictionary_list,
            *ligands_dictionary_list,
        ],
        "properties": affinities_dictionary_list,
    }

    if template_file:
        template_entry = {
            "pdb": template_file,
            "chain_id": list(protein_chains),
            "template_id": [f"{chain}1" for chain in protein_chains],
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
    subprocess_log_path=None,
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

    from guild.tools.subprocess_log import write_subprocess_log

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
        if subprocess_log_path is not None:
            write_subprocess_log(
                subprocess_log_path,
                argv=boltz_command,
                returncode=-1,
                stdout=e.stdout,
                stderr=e.stderr,
                extra_header=f"timed out after {timeout}s",
            )
        raise
    except Exception as e:
        logger.error(f"Error running Boltz command: {e}")
        raise

    # Always persist the transcript when a log path was supplied — Boltz can
    # exit 0 and still produce no records (template parsing failure), so the
    # success branch needs the same trace as the failure branch for the
    # caller's manifest-emptiness check downstream.
    if subprocess_log_path is not None:
        write_subprocess_log(
            subprocess_log_path,
            argv=boltz_command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

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


# ────────────────────────────────────────────────────────────────────────────
# Vina score-only re-scoring of Boltz-predicted complexes
#
# Boltz returns a confidence-style score (ipTM) rather than a physics-based
# binding ΔG, so for a comparable energy estimate we rescore the predicted
# complex with Vina's scoring function (score-only, no re-docking).
#
# Critical: both the receptor and the ligand are extracted *from Boltz's
# complex PDB* (per pose) because Boltz typically re-centres the predicted
# complex into its own coordinate frame — a template-frame receptor would not
# be physically near the ligand and the resulting ΔG would be meaningless.
# ────────────────────────────────────────────────────────────────────────────


def _ensure_boltz_complex_pdb(boltz_folder: str, run_id: str) -> str:
    """
    Return the path to the relabelled Boltz complex PDB for ``run_id``,
    generating it on demand from the source CIF when missing.
    """
    complex_pdb = f"{boltz_folder}/{run_id}_complex.pdb"
    if os.path.exists(complex_pdb):
        return complex_pdb

    cif_file = (
        f"{boltz_folder}/boltz_results_{run_id}_boltz"
        f"/predictions/{run_id}_boltz/{run_id}_boltz_model_0.cif"
    )
    if not os.path.exists(cif_file):
        raise FileNotFoundError(
            f"Neither Boltz complex PDB nor source CIF found for {run_id}"
        )

    # Lazy import — guild.transformers.pdb imports from guild.docking.vina so
    # routing this through a top-level import would risk a circular load order.
    from guild.transformers.pdb import relabel_ligand_chain_in_pdb

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_pdb = tmp.name
    try:
        cif_to_pdb(cif_file, tmp_pdb)
        relabel_ligand_chain_in_pdb(
            input_pdb=tmp_pdb,
            output_pdb=complex_pdb,
            ligand_chain_id="L",  # Boltz YAML always uses ligand chain "L"
        )
    finally:
        if os.path.exists(tmp_pdb):
            os.unlink(tmp_pdb)
    return complex_pdb


def rescore_boltz_pose(
    boltz_folder: str,
    protein_conf_id: str,
    ligand_id: str,
    output_dir: str = None,
    box_padding: float = 4.0,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Re-score a single Boltz-predicted protein-ligand complex with Vina's
    physics-based scoring function (score-only — pose is not re-docked).

    Pipeline (all in Boltz's predicted coordinate frame):

    1. Ensure the relabelled complex PDB exists (regenerate from CIF if needed).
    2. Extract the receptor (everything that is not the ligand resname) → PDBQT.
    3. Extract the ligand (resname ``LIG``) → PDBQT.
    4. Compute the Vina box from the ligand's bounding box.
    5. Call :func:`guild.docking.vina.vina_score_pose` (score-only).

    :param boltz_folder: Directory containing Boltz outputs for this batch
        (``{batch_folder}/boltz``).
    :param protein_conf_id: Protein configuration ID.
    :param ligand_id: Ligand identifier.
    :param output_dir: Where to write intermediate PDB/PDBQT files.
        Defaults to ``boltz_folder``.
    :param box_padding: Padding around the ligand bounding box (Å).
    :param seed: Random seed for Vina.
    :return: Dict with ``combination``, ``protein_config_id``, ``ligand_id``,
        ``vina_rescore_boltz_score``.
    """
    run_id = f"{protein_conf_id}_{ligand_id}"
    complex_pdb = _ensure_boltz_complex_pdb(boltz_folder, run_id)

    if output_dir is None:
        output_dir = boltz_folder
    os.makedirs(output_dir, exist_ok=True)

    # Per-pose receptor in Boltz's frame
    receptor_pdb = os.path.join(output_dir, f"{run_id}_receptor_rescore.pdb")
    _extract_protein_from_complex(complex_pdb, receptor_pdb, ligand_resname="LIG")
    receptor_pdbqt = receptor_pdb.replace(".pdb", ".pdbqt")
    protein_pdb_to_pdbqt(
        input_pdb=receptor_pdb,
        output_pdbqt=receptor_pdbqt,
        allow_bad_res=True,
    )

    # Ligand in the same frame
    ligand_pdb = os.path.join(output_dir, f"{run_id}_ligand_rescore.pdb")
    _extract_ligand_records(complex_pdb, ligand_pdb, resname="LIG")
    ligand_pdb_to_pdbqt(pdb=ligand_pdb)
    ligand_pdbqt = ligand_pdb.replace(".pdb", ".pdbqt")

    center, size = _compute_box_from_pdb_atoms(ligand_pdb, padding=box_padding)

    score = vina_score_pose(
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        center=center,
        size=size,
        seed=seed,
    )

    combination_id = f"{protein_conf_id}_{ligand_id}"
    logger.info(f"Vina rescore (Boltz pose): {combination_id} → {score:.3f} kcal/mol")

    return {
        COMBINATION_ID: combination_id,
        PROTEIN_CONF_ID: protein_conf_id,
        LIGAND_ID: ligand_id,
        VINA_RESCORE_BOLTZ_SCORE: score,
    }


def vina_rescore_boltz_batch(
    batch_folder: str,
    combinations: list,
    box_padding: float = 4.0,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Batch-rescore every Boltz complex in a batch.

    :param batch_folder: Path to the batch folder.
    :param combinations: List of ``(protein_conf_id, ligand_id)`` tuples.
    :param box_padding: Padding around the ligand bounding box (Å).
    :param seed: Random seed for Vina.
    :return: DataFrame with ``combination``, ``protein_config_id``,
        ``ligand_id``, ``vina_rescore_boltz_score``.
    """
    boltz_folder = os.path.join(batch_folder, BOLTZ_FOLDER)
    rescore_output_dir = os.path.join(batch_folder, VINA_RESCORE_BOLTZ_FOLDER)
    os.makedirs(rescore_output_dir, exist_ok=True)

    results = []
    for protein_conf_id, ligand_id in combinations:
        combination_id = f"{protein_conf_id}_{ligand_id}"
        try:
            result = rescore_boltz_pose(
                boltz_folder=boltz_folder,
                protein_conf_id=protein_conf_id,
                ligand_id=ligand_id,
                output_dir=rescore_output_dir,
                box_padding=box_padding,
                seed=seed,
            )
        except Exception as e:
            logger.warning(
                f"Vina Boltz rescore failed for {combination_id}: {e}"
            )
            result = {
                COMBINATION_ID: combination_id,
                PROTEIN_CONF_ID: protein_conf_id,
                LIGAND_ID: ligand_id,
                VINA_RESCORE_BOLTZ_SCORE: np.nan,
            }
        results.append(result)

    return pd.DataFrame(results)


def vina_rescore_boltz_guild_scoring(batch_dictionary):
    """
    Vina re-scoring of Boltz-predicted complexes for a batch.

    :param batch_dictionary: Batch information dict (the standard structure
        every ``*_guild_scoring`` function receives).
    :return: DataFrame with ``COMBINATION_ID``, ``PROTEIN_CONF_ID``,
        ``LIGAND_ID``, and ``VINA_RESCORE_BOLTZ_SCORE`` columns. Returns an
        empty frame (with those columns) if no Boltz outputs are found.
    """
    batch_folder = batch_dictionary[BATCH_FOLDER]
    combinations = batch_dictionary[COMBINATIONS_TO_RUN_KEY]

    boltz_root = os.path.join(batch_folder, BOLTZ_FOLDER)
    boltz_present = os.path.isdir(boltz_root) and any(
        name.startswith("boltz_results_") for name in os.listdir(boltz_root)
    )
    if not boltz_present:
        logger.warning(
            "vina_rescore_boltz: no Boltz outputs found in %s — returning empty score table",
            batch_folder,
        )
        return pd.DataFrame(
            columns=[COMBINATION_ID, PROTEIN_CONF_ID, LIGAND_ID, VINA_RESCORE_BOLTZ_SCORE]
        )

    df = vina_rescore_boltz_batch(batch_folder=batch_folder, combinations=combinations)
    keep = [COMBINATION_ID, PROTEIN_CONF_ID, LIGAND_ID, VINA_RESCORE_BOLTZ_SCORE]
    return df[[c for c in keep if c in df.columns]]
