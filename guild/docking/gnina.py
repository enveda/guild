"""
GNINA docking tools.

GNINA (https://github.com/gnina/gnina) is a fork of smina/AutoDock Vina that
rescores Vina-style poses with a CNN. It is invoked here as a standalone
docker — it produces a Vina-style binding affinity (kcal/mol, lower is better)
that drives guild's rank-percentile aggregation, plus a CNN confidence score
(`CNNscore`, [0,1], higher is better) that is carried alongside as a
side-channel column.
"""

import logging
import os
import re
import subprocess
import tempfile

import numpy as np
import pandas as pd

from guild.constants.bulk import (
    BATCH_FOLDER,
    COMBINATION_ID,
    COMBINATIONS_TO_RUN_KEY,
)
from guild.constants.general import RANDOM_SEED
from guild.constants.gnina import (
    GNINA_BINARY,
    GNINA_DEFAULT_CNN_SCORING,
    GNINA_DEFAULT_EXHAUSTIVENESS,
    GNINA_DEFAULT_NUMBER_OF_POSES,
    GNINA_LIB_PATH,
    GNINA_OB_DATA_DIR,
    GNINA_OB_PLUGIN_DIR,
    GNINA_OB_SYSTEM_LIB,
)
from guild.constants.guild import (
    GNINA_CNN_SCORE,
    GNINA_FOLDER,
    GNINA_SCORE,
    LIGAND_ID,
    PROTEIN_CONF_ID,
)
from guild.docking.vina import _validate_pdbqt
from guild.tools.subprocess_log import write_subprocess_log

logger = logging.getLogger(__name__)

# Bound to avoid hangs from a runaway gnina worker. Matches DOCKING_TIMEOUT in
# guild/constants/bulk.py (kept as a module-local constant to avoid pulling a
# bulk-orchestration dep into the docking module).
GNINA_SUBPROCESS_TIMEOUT = 600  # seconds

# gnina prints a table to stdout that looks like:
#
#     mode |  affinity  |  CNN     | CNN
#          | (kcal/mol) | pose-score| affinity
#     -----+------------+----------+----------
#         1     -8.345      0.7891      6.234
#         2     -7.910      0.6512      5.987
#         ...
#
# Each data row starts with the integer mode number; the rest are floats.
_GNINA_POSE_ROW = re.compile(
    r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)


def _ensure_openbabel_plugin_shim() -> tuple[str, str, str] | None:
    """
    Make gnina's Open Babel able to write pose files.

    gnina's bundled ``libopenbabel.so.7`` has no format-plugin tree in the
    image, so its OB can't write any ``--out`` format and every pose file ends
    up empty. The system Open Babel 3.2 (``.so.8``) *does* ship a complete
    plugin set. We create a tiny shim dir holding ``libopenbabel.so.7`` ->
    system ``.so.8`` symlink; prepending it to ``LD_LIBRARY_PATH`` makes gnina
    resolve its OB to the .so.8 (ABI-compatible for gnina's calls), which then
    finds the plugins via ``BABEL_LIBDIR``/``BABEL_DATADIR``.

    :return: ``(shim_dir, plugin_dir, data_dir)`` to feed into the gnina
        subprocess env, or ``None`` if the system OpenBabel isn't present (in
        which case the caller leaves the env untouched — gnina still scores via
        its native pdbqt parser, just with empty pose files as before).
    """
    if not (
        os.path.exists(GNINA_OB_SYSTEM_LIB)
        and os.path.isdir(GNINA_OB_PLUGIN_DIR)
        and os.path.isdir(GNINA_OB_DATA_DIR)
    ):
        return None

    shim_dir = os.path.join(tempfile.gettempdir(), "guild_gnina_ob_shim")
    os.makedirs(shim_dir, exist_ok=True)
    link = os.path.join(shim_dir, "libopenbabel.so.7")
    # Idempotent + race-safe across parallel gnina workers: the target is a
    # fixed path, so a pre-existing link is already correct.
    try:
        if os.path.islink(link):
            if os.readlink(link) != GNINA_OB_SYSTEM_LIB:
                os.unlink(link)
        elif os.path.exists(link):
            os.remove(link)

        if not os.path.lexists(link):
            os.symlink(GNINA_OB_SYSTEM_LIB, link)
    except OSError as e:
        logger.warning("gnina: failed to set up OpenBabel shim (%s)", e)
        return None
    return shim_dir, GNINA_OB_PLUGIN_DIR, GNINA_OB_DATA_DIR


def parse_gnina_stdout(stdout: str) -> list[tuple[int, float, float, float]]:
    """
    Parse gnina's stdout pose table.

    :param stdout: Captured stdout from a ``gnina`` subprocess call.
    :return: List of ``(mode, affinity, cnn_score, cnn_affinity)`` tuples,
        one per docked pose, in the order gnina emitted them.
    :raises ValueError: when no pose rows can be parsed (treated as a docking
        failure by callers).
    """
    rows: list[tuple[int, float, float, float]] = []
    for line in stdout.splitlines():
        match = _GNINA_POSE_ROW.match(line)
        if not match:
            continue
        mode = int(match.group(1))
        affinity = float(match.group(2))
        cnn_score = float(match.group(3))
        cnn_affinity = float(match.group(4))
        rows.append((mode, affinity, cnn_score, cnn_affinity))
    if not rows:
        raise ValueError("Could not parse any pose rows from gnina stdout")
    return rows


def deploy_gnina(
    receptor: str,
    ligand: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    output_pdbqt: str,
    output_scores: str,
    exhaustiveness: int = GNINA_DEFAULT_EXHAUSTIVENESS,
    n_poses: int = GNINA_DEFAULT_NUMBER_OF_POSES,
    seed: int = RANDOM_SEED,
    cnn_scoring: str = GNINA_DEFAULT_CNN_SCORING,
    use_gpu: bool = True,
    subprocess_log_path: str | None = None,
) -> dict:
    """
    Run docking with the gnina CLI.

    :param receptor: Path to the receptor file. ``.pdbqt`` and ``.pdb`` both
        work — gnina sniffs the format from the extension. PDBQT inputs get
        an extra structural validation pass; PDB inputs are handed to gnina
        verbatim.
    :param ligand: Path to the ligand file. ``.pdbqt`` and ``.sdf`` both
        work. Same validation rule as above.
    :param center: Box center (x, y, z).
    :param size: Box size (x, y, z).
    :param output_pdbqt: Where gnina should write the multi-pose output PDBQT.
    :param output_scores: Path to write the per-pose score file. Format mirrors
        Vina's (``mode: affinity``) extended with a tab-separated CNN score:
        ``mode: affinity\\tcnn_score``.
    :param exhaustiveness: Search exhaustiveness (gnina default 8).
    :param n_poses: Number of poses to keep (``--num_modes``).
    :param seed: RNG seed.
    :param cnn_scoring: One of gnina's CNN modes (``none``/``rescore``/
        ``refinement``/``metrorescore``/``metrorefine``/``all``). Default
        ``rescore`` does Vina search then CNN-rescores the top poses.
    :param use_gpu: When False, pass ``--no_gpu`` to gnina (CPU-only inference).
    :return: ``{"scores": [...affinities], "cnn_scores": [...], "out_pdbqt":
        output_pdbqt, "output_scores": output_scores}``.
    """
    if receptor.endswith(".pdbqt"):
        _validate_pdbqt(receptor, "receptor")
    if ligand.endswith(".pdbqt"):
        _validate_pdbqt(ligand, "ligand")

    cx, cy, cz = center
    sx, sy, sz = size

    argv = [
        GNINA_BINARY,
        "--receptor", receptor,
        "--ligand", ligand,
        "--center_x", str(cx),
        "--center_y", str(cy),
        "--center_z", str(cz),
        "--size_x", str(sx),
        "--size_y", str(sy),
        "--size_z", str(sz),
        "--out", output_pdbqt,
        "--num_modes", str(n_poses),
        "--exhaustiveness", str(exhaustiveness),
        "--seed", str(seed),
        "--cnn_scoring", cnn_scoring,
    ]
    if not use_gpu:
        argv.append("--no_gpu")

    # gnina's torch/openbabel/boost live under /opt/gnina/lib, isolated from
    # the rest of the image. Prepend that to LD_LIBRARY_PATH for this call
    # only — gnina ABI demands its own libtorch + libcudart 12, which would
    # clash with our venv-managed CUDA 13 torch if we set it globally.
    env = os.environ.copy()
    ld_parts = [GNINA_LIB_PATH]

    # Repoint gnina's broken OpenBabel at the system .so.8 + its plugins so the
    # ``--out`` pose file isn't written empty. The shim dir goes *ahead* of
    # GNINA_LIB_PATH so its libopenbabel.so.7 symlink wins over the bundled
    # (plugin-less) one, while gnina's own libtorch/boost still resolve from
    # /opt/gnina/lib. No-ops to today's behaviour if the system OB is absent.
    ob_shim = _ensure_openbabel_plugin_shim()
    if ob_shim is not None:
        shim_dir, plugin_dir, data_dir = ob_shim
        ld_parts.insert(0, shim_dir)
        env["BABEL_LIBDIR"] = plugin_dir
        env["BABEL_DATADIR"] = data_dir

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        ld_parts.append(existing_ld)
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    try:
        completed = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=GNINA_SUBPROCESS_TIMEOUT,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        # Persist the failure transcript before re-raising so the caller can
        # point users at it.
        if subprocess_log_path is not None:
            write_subprocess_log(
                subprocess_log_path,
                argv=argv,
                returncode=e.returncode,
                stdout=e.stdout,
                stderr=e.stderr,
            )
        raise RuntimeError(
            f"gnina failed (exit {e.returncode}) for receptor={receptor} "
            f"ligand={ligand}: {e.stderr.strip()[:500]}"
        ) from e

    if subprocess_log_path is not None:
        write_subprocess_log(
            subprocess_log_path,
            argv=argv,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    poses = parse_gnina_stdout(completed.stdout)

    scores = [affinity for (_mode, affinity, _cnn, _cnn_aff) in poses]
    cnn_scores = [cnn for (_mode, _affinity, cnn, _cnn_aff) in poses]

    with open(output_scores, "w") as f:
        for (mode, affinity, cnn, _cnn_aff) in poses:
            # mode is 1-based in gnina's table; keep that to make the file
            # round-trippable to the CLI output.
            f.write(f"{mode}: {affinity}\t{cnn}\n")

    logger.info(f"gnina: docking completed. Scores saved to {output_scores}")

    return {
        "scores": scores,
        "cnn_scores": cnn_scores,
        "out_pdbqt": output_pdbqt,
        "output_scores": output_scores,
    }


def process_gnina_output(input_file: str) -> tuple[float, float]:
    """
    Read a gnina score file and return the best ``(affinity, cnn_score)``.

    "Best" follows the Vina convention: the row with the **minimum** affinity
    (lower kcal/mol = stronger binder). The CNN score returned is the one
    reported for that same pose — not the max CNN across all poses.

    :param input_file: Path to the file written by :func:`deploy_gnina`.
    :return: ``(affinity, cnn_score)``. If the file is empty/unreadable,
        returns ``(nan, nan)`` — matching :func:`process_vina_output`.
    """
    df = pd.read_csv(input_file, sep=":", header=None)
    if df.empty:
        return float("nan"), float("nan")

    # The second column holds "affinity\tcnn_score"; split it back out.
    affinity_cnn = df[1].astype(str).str.split("\t", n=1, expand=True)
    affinities = affinity_cnn[0].astype(float)
    # Older score files might be Vina-format (no CNN column) — guard for that
    # so a partial migration doesn't break callers.
    if affinity_cnn.shape[1] > 1:
        cnn_scores = pd.to_numeric(affinity_cnn[1], errors="coerce")
    else:
        cnn_scores = pd.Series([np.nan] * len(affinities))

    best_idx = affinities.idxmin()
    return float(affinities.loc[best_idx]), float(cnn_scores.loc[best_idx])


def gnina_guild_scoring(batch_dictionary) -> pd.DataFrame:
    """
    Collect per-combination gnina scores into a DataFrame.

    Mirrors :func:`guild.docking.vina.vina_guild_scoring` but emits both the
    primary ``gnina_score`` (Vina-style affinity, used in rank-percentile
    aggregation) and the side-channel ``gnina_cnn_score`` (CNN confidence,
    not registered in ``RANKS_DICTIONARY``/``RP_SCORES_DICTIONARY``).

    :param batch_dictionary: Standard bulk batch dictionary.
    :return: DataFrame with columns
        ``[COMBINATION_ID, GNINA_SCORE, GNINA_CNN_SCORE, PROTEIN_CONF_ID, LIGAND_ID]``.
    """
    combinations_df = pd.DataFrame(
        batch_dictionary[COMBINATIONS_TO_RUN_KEY],
        columns=[PROTEIN_CONF_ID, LIGAND_ID],
    )

    def score_combination(row):
        try:
            return process_gnina_output(
                f"{batch_dictionary[BATCH_FOLDER]}/{GNINA_FOLDER}/"
                f"{row[PROTEIN_CONF_ID]}_{row[LIGAND_ID]}.txt"
            )
        except Exception as e:
            logger.info(
                f"Failed gnina scoring for {(row[PROTEIN_CONF_ID], row[LIGAND_ID])} with error {e}"
            )
            return np.nan, np.nan

    score_pairs = combinations_df.apply(score_combination, axis=1)
    combinations_df[GNINA_SCORE] = [s[0] for s in score_pairs]
    combinations_df[GNINA_CNN_SCORE] = [s[1] for s in score_pairs]
    combinations_df[COMBINATION_ID] = (
        combinations_df[PROTEIN_CONF_ID] + "_" + combinations_df[LIGAND_ID]
    )

    return combinations_df[
        [COMBINATION_ID, GNINA_SCORE, GNINA_CNN_SCORE, PROTEIN_CONF_ID, LIGAND_ID]
    ]
