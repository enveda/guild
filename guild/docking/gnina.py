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
    GNINA_COVALENT_BOND_ORDER,
    GNINA_COVALENT_OPTIMIZE_LIG,
    GNINA_DEFAULT_CNN_SCORING,
    GNINA_DEFAULT_CNN_SCORING_COVALENT_ONLY,
    GNINA_DEFAULT_EXHAUSTIVENESS,
    GNINA_DEFAULT_NUMBER_OF_POSES,
    GNINA_LIB_PATH,
    GNINA_OB_DATA_DIR,
)
from guild.constants.guild import (
    GNINA_CNN_SCORE,
    GNINA_FOLDER,
    GNINA_SCORE,
    LIGAND_ID,
    PROTEIN_CONF_ID,
)
from guild.constants.poses import (
    DEFAULT_POSE_MODE,
    POSE_MODE_LOCAL,
    POSE_MODE_SCORE,
    POSE_MODES,
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
_GNINA_POSE_ROW = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def _ensure_openbabel_plugin_shim() -> None:
    """Deprecated no-op.

    Kept as a stub for one release cycle in case external callers imported
    it.  gnina v1.3.3+ statically links OpenBabel with plugins baked in, so
    the shim is no longer needed — remove this stub in the following
    release.
    """
    return None


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
    cnn_scoring: str | None = None,
    use_gpu: bool = True,
    subprocess_log_path: str | None = None,
    pose_mode: str = DEFAULT_POSE_MODE,
    flex_pdbqt: str | None = None,
    flexres: str | None = None,
    flexdist_ligand: str | None = None,
    flexdist: float | None = None,
    out_flex_pdbqt: str | None = None,
    covalent_rec_atom: str | None = None,
    covalent_lig_atom_pattern: str | None = None,
    covalent_optimize_lig: bool = GNINA_COVALENT_OPTIMIZE_LIG,
    covalent_bond_order: int = GNINA_COVALENT_BOND_ORDER,
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
    :param exhaustiveness: Search exhaustiveness (gnina default 8). Ignored
        when ``pose_mode`` is ``local`` or ``score`` (gnina drops the
        global-search step in those modes).
    :param n_poses: Number of poses to keep (``--num_modes``). Effectively
        forced to 1 in ``local`` / ``score`` modes — those emit a single
        pose row.
    :param seed: RNG seed.
    :param cnn_scoring: One of gnina's CNN modes (``none``/``rescore``/
        ``refinement``/``metrorescore``/``metrorefine``/``all``). Default
        ``rescore`` does Vina search then CNN-rescores the top poses.
    :param use_gpu: When False, pass ``--no_gpu`` to gnina (CPU-only inference).
    :param pose_mode: One of ``"dock"`` (default; normal global search),
        ``"local"`` (appends ``--local_only`` — single local refinement of
        the supplied pose), or ``"score"`` (appends ``--score_only`` —
        evaluate the supplied pose, no movement). ``local``/``score`` are
        the modes that honour an experimentally-informed starting pose;
        the global search in ``dock`` mode ignores the supplied
        coordinates.
    :param flex_pdbqt: Pre-prepared flexible-residue PDBQT (from obabel split).
        Mutually exclusive with ``flexres``/``flexdist_ligand``/``flexdist``.
    :param flexres: Explicit flexible-residue spec in gnina's ``chain:resnum[,resnum]``
        format (e.g. ``"A:88,91"``).  Passed directly as ``--flexres``.  Takes
        priority over ``flexdist_ligand``/``flexdist``; ``flex_pdbqt`` takes
        priority over this.
    :param flexdist_ligand: Path to the ligand file used as the flexdist anchor.
        gnina selects flexible residues within ``flexdist`` Å of this ligand.
    :param flexdist: Distance threshold (Å) for automatic flexible-residue
        selection via ``--flexdist_ligand``.
    :param out_flex_pdbqt: Where gnina should write the flexible-residue poses.
    :param covalent_rec_atom: Receptor atom the ligand covalently bonds to, in
        gnina's ``chain:resnum:atomname`` form (e.g. ``A:145:SG``). Triggers
        covalent docking when given together with ``covalent_lig_atom_pattern``.
        The residue number must match the *receptor file passed here* — for
        guild that is the cleaned/renumbered protein.
    :param covalent_lig_atom_pattern: SMARTS matching the ligand warhead atom
        that forms the covalent bond (e.g. ``[CX4]Cl`` for a chloroacetamide,
        ``C#N`` for a nitrile).
    :param covalent_optimize_lig: Pass ``--covalent_optimize_lig`` to UFF-optimise
        the ligand+residue adduct (recommended for sensible covalent geometry).
    :param covalent_bond_order: Bond order of the covalent bond (default 1).
    :return: ``{"scores": [...affinities], "cnn_scores": [...], "out_pdbqt":
        output_pdbqt, "output_scores": output_scores}``.
    """
    if pose_mode not in POSE_MODES:
        raise ValueError(f"Invalid pose_mode={pose_mode!r}; expected one of {POSE_MODES}.")

    if receptor.endswith(".pdbqt"):
        _validate_pdbqt(receptor, "receptor")
    if ligand.endswith(".pdbqt"):
        _validate_pdbqt(ligand, "ligand")
    if flex_pdbqt:
        _validate_pdbqt(flex_pdbqt, "flex receptor")

    cx, cy, cz = center
    sx, sy, sz = size

    if cnn_scoring is None:
        if covalent_rec_atom and covalent_lig_atom_pattern:
            cnn_scoring = GNINA_DEFAULT_CNN_SCORING_COVALENT_ONLY
        else:
            cnn_scoring = GNINA_DEFAULT_CNN_SCORING

    argv = [
        GNINA_BINARY,
        "--receptor",
        receptor,
        "--ligand",
        ligand,
        "--center_x",
        str(cx),
        "--center_y",
        str(cy),
        "--center_z",
        str(cz),
        "--size_x",
        str(sx),
        "--size_y",
        str(sy),
        "--size_z",
        str(sz),
        "--out",
        output_pdbqt,
        "--num_modes",
        str(n_poses),
        "--exhaustiveness",
        str(exhaustiveness),
        "--seed",
        str(seed),
        "--cnn_scoring",
        cnn_scoring,
    ]
    if not use_gpu:
        argv.append("--no_gpu")
    if pose_mode == POSE_MODE_LOCAL:
        argv.append("--local_only")
    elif pose_mode == POSE_MODE_SCORE:
        argv.append("--score_only")
    if flex_pdbqt:
        argv += ["--flex", flex_pdbqt]
    elif flexres:
        argv += ["--flexres", flexres]
    elif flexdist_ligand and flexdist is not None:
        argv += ["--flexdist_ligand", flexdist_ligand, "--flexdist", str(flexdist)]
    if out_flex_pdbqt and (flex_pdbqt or flexres or (flexdist_ligand and flexdist is not None)):
        argv += ["--out_flex", out_flex_pdbqt]
    if covalent_rec_atom and covalent_lig_atom_pattern:
        argv += [
            "--covalent_rec_atom",
            covalent_rec_atom,
            "--covalent_lig_atom_pattern",
            covalent_lig_atom_pattern,
            "--covalent_bond_order",
            str(covalent_bond_order),
        ]
        if covalent_optimize_lig:
            argv.append("--covalent_optimize_lig")

    # gnina's torch/CUDA runtime live under /opt/gnina/lib, isolated from
    # the rest of the image. Prepend that to LD_LIBRARY_PATH for this call
    # only — gnina's CUDA 12 runtime would otherwise clash with the main
    # image's venv-managed torch if we set it globally.
    env = os.environ.copy()
    ld_parts = [GNINA_LIB_PATH]

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        ld_parts.append(existing_ld)
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    # OpenBabel data files (UFF.prm etc.) live under /opt/gnina/share/openbabel.
    # gnina's static OB needs BABEL_DATADIR set to find them — without this,
    # --covalent_optimize_lig prints "Cannot open UFF.prm" and skips the
    # post-bond UFF minimisation.
    env["BABEL_DATADIR"] = GNINA_OB_DATA_DIR

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
        for mode, affinity, cnn, _cnn_aff in poses:
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
