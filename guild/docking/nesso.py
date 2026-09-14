"""
Nesso-1 support tools.

Nesso-1 (Valence Labs / Recursion, Apache-2.0, github.com/recursionpharma/nesso)
is a coarse-grained cofolding affinity model in the same lineage as Boltz-2 —
same affinity training data and loss structure — but with the atomistic
diffusion module removed and MSAs replaced by frozen ESM-2 embeddings. Unlike
Boltz-2, Nesso takes no protein structure, MSA, template, or pocket — only a
protein sequence and a ligand SMILES — and writes a single scalar
``affinity.json`` per complex (no 3D output).

Nesso's package requires numpy>=2 and transformers>=4.40, which conflict with
guild's own numpy<2 pin, so it is installed into an isolated venv at
/opt/nesso inside the guild image (see the Dockerfile) and invoked purely by
subprocess — no Python import of ``nesso`` happens anywhere in this repo.

Directory-mode batching (one ``nesso predict <dir>`` call per batch, not one
per ligand) is where Nesso's >10x speed-up over Boltz-2 actually comes from,
since it amortizes model load across every complex in the directory. Always
batch; never invoke Nesso per-ligand.
"""

import json
import logging
import os
import subprocess

import pandas as pd
import yaml
from tqdm import tqdm

from guild.constants.bulk import (
    BATCH_FOLDER,
    COMBINATION_ID,
    COMBINATIONS_TABLE_KEY,
)
from guild.constants.general import RANDOM_SEED
from guild.constants.guild import (
    LIGAND_ID,
    NESSO_BINDER_PROBABILITY,
    NESSO_ENTROPY_PL,
    NESSO_FOLDER,
    NESSO_SCORE,
    PROTEIN_CONF_ID,
)
from guild.constants.nesso import (
    NESSO_AFFINITY_FILE,
    NESSO_AFFINITY_PRED_VALUE_FIELD,
    NESSO_AFFINITY_PROBABILITY_BINARY_FIELD,
    NESSO_ENTROPY_CROP_PL_FIELD,
)

logger = logging.getLogger(__name__)

# The isolated venv installed at build time (see Dockerfile). Invoking this
# binary directly — rather than "nesso" on PATH — avoids ever needing
# nesso's numpy>=2 / transformers>=4.40 requirements inside guild's own venv.
NESSO_BIN = os.environ.get("NESSO_BIN", "/opt/nesso/bin/nesso")


def generate_nesso_yaml(
    protein_sequence,
    protein_chain,
    ligand_smiles,
    ligand_id,
    output_file,
):
    """
    Generate a Nesso-1 input YAML for one protein-ligand complex.

    Nesso has no structure/MSA/template/pocket concept — only a protein
    sequence and a ligand SMILES, so this is deliberately simpler than
    ``generate_boltz_yaml``.

    :param protein_sequence: Protein amino-acid sequence.
    :param protein_chain: Protein entity id (any label; Nesso does not
        interpret it as a real chain — it's just the YAML entity id).
    :param ligand_smiles: Ligand SMILES string.
    :param ligand_id: Ligand entity id; also selected as the affinity binder.
    :param output_file: Path to write the YAML to.
    """
    nesso_yaml = {
        "sequences": [
            {"protein": {"id": protein_chain, "sequence": protein_sequence}},
            {"ligand": {"id": ligand_id, "smiles": ligand_smiles}},
        ],
        "properties": [{"affinity": {"binder": ligand_id}}],
    }
    with open(output_file, "w") as f:
        yaml.safe_dump(nesso_yaml, f, sort_keys=False, default_flow_style=None)


def deploy_nesso(
    input_dir,
    out_dir="nesso_output",
    use_gpu=True,
    recycling_steps=5,
    num_workers=2,
    require_affinity=True,
    no_kernels=False,
    seed=RANDOM_SEED,
    timeout=3600,
    subprocess_log_path=None,
):
    """
    Run Nesso-1 over every YAML file in ``input_dir`` in a single invocation.

    :param input_dir: Directory containing one ``.yaml`` per complex.
    :param out_dir: Output directory (predictions land at
        ``{out_dir}/predictions/{yaml_stem}/affinity.json``).
    :param use_gpu: Use GPU acceleration.
    :param recycling_steps: Trunk recycling iterations (Nesso default 5).
    :param num_workers: Dataloader worker threads.
    :param require_affinity: Fail loudly if the checkpoint lacks an affinity head.
    :param no_kernels: Disable cuEquivariance GPU kernels (required on CPU).
    :param seed: Random seed for reproducibility.
    :param timeout: Timeout in seconds for the Nesso subprocess.
    :param subprocess_log_path: Optional path to write a full subprocess transcript.
    """
    nesso_command = [
        NESSO_BIN,
        "predict",
        str(input_dir),
        "--out_dir",
        str(out_dir),
        "--accelerator",
        "gpu" if use_gpu else "cpu",
        "--recycling_steps",
        str(recycling_steps),
        "--num_workers",
        str(num_workers),
        "--seed",
        str(seed),
    ]
    if require_affinity:
        nesso_command.append("--require_affinity")
    if no_kernels or not use_gpu:
        nesso_command.append("--no_kernels")

    from guild.tools.subprocess_log import write_subprocess_log

    try:
        result = subprocess.run(
            nesso_command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        logger.error(f"Nesso binary not found at {NESSO_BIN}: {e}")
        if subprocess_log_path is not None:
            write_subprocess_log(
                subprocess_log_path,
                argv=nesso_command,
                returncode=127,
                stdout="",
                stderr=str(e),
                extra_header="Nesso binary not found",
            )
        return None
    except subprocess.TimeoutExpired as e:
        logger.error(f"Nesso timed out after {timeout} seconds")
        if e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
        if subprocess_log_path is not None:
            write_subprocess_log(
                subprocess_log_path,
                argv=nesso_command,
                returncode=-1,
                stdout=e.stdout,
                stderr=e.stderr,
                extra_header=f"Timed out after {timeout}s",
            )
        return None

    if subprocess_log_path is not None:
        write_subprocess_log(
            subprocess_log_path,
            argv=nesso_command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    if result.returncode != 0:
        logger.error(f"Nesso failed (exit {result.returncode}):\n{result.stderr}")

    return result


def process_nesso_output(affinity_json_file):
    """
    Read one Nesso ``affinity.json`` file.

    :param affinity_json_file: Path to the affinity JSON file.
    :return: Dict with ``NESSO_SCORE`` (log10(IC50/uM)), ``NESSO_BINDER_PROBABILITY``,
        and ``NESSO_ENTROPY_PL``.
    :raises FileNotFoundError: If the file does not exist.
    :raises KeyError: If ``affinity_pred_value`` is missing.
    """
    with open(affinity_json_file, "r") as f:
        data = json.load(f)
    return {
        NESSO_SCORE: float(data[NESSO_AFFINITY_PRED_VALUE_FIELD]),
        NESSO_BINDER_PROBABILITY: float(
            data.get(NESSO_AFFINITY_PROBABILITY_BINARY_FIELD, float("nan"))
        ),
        NESSO_ENTROPY_PL: float(data.get(NESSO_ENTROPY_CROP_PL_FIELD, float("nan"))),
    }


def nesso_guild_scoring(batch_dictionary):
    """
    Perform the scoring of Nesso-1 results for a batch.

    Every protein-ligand pair in the batch was run in one directory-batched
    Nesso invocation; predictions land at
    ``{NESSO_FOLDER}/predictions/{run_id}/affinity.json`` where
    ``run_id = f"{protein_conf_id}_{ligand_id}"`` (Nesso derives the record
    id from the input YAML's filename stem).

    :param batch_dictionary: Batch dictionary.
    :return: DataFrame with ``LIGAND_ID``, ``NESSO_SCORE``,
        ``NESSO_BINDER_PROBABILITY``, ``NESSO_ENTROPY_PL``,
        ``PROTEIN_CONF_ID``, ``COMBINATION_ID``.
    """
    nesso_scores_data = []
    missing_or_invalid = 0
    combinations_table = batch_dictionary[COMBINATIONS_TABLE_KEY]
    unique_pairs = combinations_table[[PROTEIN_CONF_ID, LIGAND_ID]].drop_duplicates()

    for _, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="Nesso scoring"):
        current_protein_configuration_id = row[PROTEIN_CONF_ID]
        current_ligand_id = row[LIGAND_ID]
        run_id = f"{current_protein_configuration_id}_{current_ligand_id}"

        affinity_json_file = (
            f"{batch_dictionary[BATCH_FOLDER]}/{NESSO_FOLDER}/predictions/"
            f"{run_id}/{NESSO_AFFINITY_FILE}"
        )

        try:
            fields = process_nesso_output(affinity_json_file)
        except (FileNotFoundError, KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            missing_or_invalid += 1
            logger.warning(f"Skipping Nesso score for {run_id}: {e}")
            continue

        nesso_scores_data.append(
            {
                LIGAND_ID: current_ligand_id,
                PROTEIN_CONF_ID: current_protein_configuration_id,
                COMBINATION_ID: run_id,
                **fields,
            }
        )

    if missing_or_invalid > 0:
        logger.warning(
            f"Skipped {missing_or_invalid} Nesso result(s) due to missing/invalid output files"
        )

    return pd.DataFrame(
        nesso_scores_data,
        columns=[
            LIGAND_ID,
            NESSO_SCORE,
            NESSO_BINDER_PROBABILITY,
            NESSO_ENTROPY_PL,
            PROTEIN_CONF_ID,
            COMBINATION_ID,
        ],
    )
