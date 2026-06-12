"""
Utilities for fetching protein Multiple Sequence Alignments (MSA) using a
local ColabFold installation.

The MSA is computed once per unique protein sequence and cached on disk.
All ligand Boltz YAML files for the same protein reuse the cached a3m,
replacing the ``msa: empty`` fallback and providing richer evolutionary
input features to the diffusion model.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path


LOCALCOLABFOLD_DIR_ENV_VAR = "GUILD_LOCALCOLABFOLD_DIR"
COLABFOLD_RUN_SCRIPT = "run_colabfoldbatch_sample.sh"

logger = logging.getLogger(__name__)


def _resolve_localcolabfold_dirs() -> list[Path]:
    """Resolve candidate localcolabfold directories across env, module path, and cwd."""
    candidates: list[Path] = []

    env_localcolabfold_dir = os.environ.get(LOCALCOLABFOLD_DIR_ENV_VAR)
    if env_localcolabfold_dir:
        candidates.append(Path(env_localcolabfold_dir).expanduser())

    current_file_path = Path(__file__).resolve()
    for parent in [current_file_path.parent, *current_file_path.parents]:
        if parent.name == "localcolabfold":
            candidates.append(parent)
        candidates.append(parent / "localcolabfold")

    current_working_dir = Path.cwd().resolve()
    for parent in [current_working_dir, *current_working_dir.parents]:
        if parent.name == "localcolabfold":
            candidates.append(parent)
        candidates.append(parent / "localcolabfold")

    unique_existing_candidates: list[Path] = []
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate.is_dir() and resolved_candidate not in unique_existing_candidates:
            unique_existing_candidates.append(resolved_candidate)

    return unique_existing_candidates


def _resolve_colabfold_run_script() -> Path | None:
    """Resolve the local shell script wrapper for MSA generation."""
    module_script = Path(__file__).resolve().parent / COLABFOLD_RUN_SCRIPT
    if module_script.is_file():
        return module_script

    for localcolabfold_dir in _resolve_localcolabfold_dirs():
        script_path = localcolabfold_dir / COLABFOLD_RUN_SCRIPT
        if script_path.is_file():
            return script_path
    return None


def _cleanup_temp_msa_artifacts(files_to_remove: list) -> None:
    """Best-effort cleanup of temporary files created for local MSA generation."""
    for file_path in files_to_remove:
        try:
            if file_path.exists():
                shutil.rmtree(file_path) if file_path.is_dir() else file_path.unlink()
        except OSError as exc:
            logger.debug(
                "fetch_protein_msa: could not remove temporary file %s: %s", file_path, exc
            )


def fetch_protein_msa(
    sequence: str,
    protein_id: str,
    protein_chain_id: str,
    output_a3m_dir: str,
    timeout: int = 300,
) -> str | None:
    """
    Fetch a protein MSA from a local ColabFold installation and save it as an a3m file.

    The function checks for a cached a3m at *output_a3m_dir* before invoking
    ``colabfold_batch --msa-only``, so repeated runs for the same protein are free.
    All ligand Boltz YAML
    files for the same protein should point to the same *output_a3m_dir* path.

    :param sequence: Protein amino-acid sequence (single-letter codes).
    :param output_a3m_dir: Directory where the a3m file will be written.
    :param timeout: Maximum seconds to wait for local MSA generation (default 300).
    :return: Path to the saved a3m file on success, ``None`` on any failure
             (the caller should fall back to ``msa: empty``).
    """

    output_a3m_path = Path(f"{output_a3m_dir}/{protein_id}_{protein_chain_id}.a3m")

    # --- cache hit ---
    if output_a3m_path.exists() and output_a3m_path.stat().st_size > 0:
        logger.info(
            f"fetch_protein_msa: cached MSA found at {output_a3m_path}, skipping local ColabFold call."
        )
        return str(output_a3m_path)

    output_dir_path = Path(output_a3m_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    run_script = _resolve_colabfold_run_script()
    if not run_script:
        searched_dirs = ", ".join(str(path) for path in _resolve_localcolabfold_dirs())
        logger.warning(
            f"fetch_protein_msa: could not find `{COLABFOLD_RUN_SCRIPT}` near transformers or localcolabfold. "
            f"Set `{LOCALCOLABFOLD_DIR_ENV_VAR}`. "
            f"Searched: {searched_dirs if searched_dirs else 'none'}. "
            "Falling back to empty MSA."
        )
        return None

    # Write the input sequence to a temporary FASTA file for ColabFold
    input_fasta_path = output_dir_path / f"{protein_id}_{protein_chain_id}.fasta"
    with open(input_fasta_path, "w", encoding="utf-8") as fasta_file:
        fasta_file.write(f">{protein_id}_{protein_chain_id}\n{sequence.strip()}\n")

    cmd = [
        "bash",
        str(run_script),
        str(input_fasta_path),
        str(output_dir_path),
        "42",
    ]
    run_env = os.environ.copy()
    run_env["MPLBACKEND"] = "Agg"
    logger.info(
        "fetch_protein_msa: running local ColabFold MSA script with command: %s",
        " ".join(cmd),
    )

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=run_env,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "fetch_protein_msa: local ColabFold MSA generation timed out after %ss. "
            "Falling back to empty MSA.",
            timeout,
        )
        return None
    except OSError as exc:
        logger.warning(f"fetch_protein_msa: failed to execute local colabfold_batch: {exc}")
        return None

    if completed.returncode != 0:
        logger.warning(
            "fetch_protein_msa: local ColabFold script failed (code=%s). stderr=%s",
            completed.returncode,
            (completed.stderr or "").strip()[-1000:],
        )
        return None

    # ColabFold writes the a3m using the FASTA header as the filename stem, which
    # matches output_a3m_path exactly.  Clean up the other artifacts it produces
    # (pickle, png, _env dir, log, config, bibtex) without touching any pre-existing
    # a3m files from earlier proteins that share the same output directory.
    if not output_a3m_path.exists() or output_a3m_path.stat().st_size == 0:
        logger.warning(
            "fetch_protein_msa: local script completed but %s was not created. "
            "Falling back to empty MSA.",
            output_a3m_path,
        )
        return None

    colabfold_artifacts = [
        output_dir_path / f"{protein_id}_{protein_chain_id}.pickle",
        output_dir_path / f"{protein_id}_{protein_chain_id}_coverage.png",
        output_dir_path / f"{protein_id}_{protein_chain_id}_env",
        output_dir_path / f"{protein_id}_{protein_chain_id}.fasta",
        output_dir_path / "cite.bibtex",
        output_dir_path / "config.json",
        output_dir_path / "log.txt",
    ]
    _cleanup_temp_msa_artifacts(colabfold_artifacts)
    logger.info("fetch_protein_msa: MSA saved to %s", output_a3m_path)
    return str(output_a3m_path)
