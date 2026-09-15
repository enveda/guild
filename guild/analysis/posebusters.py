"""PoseBusters pose-validity analysis.

Runs PoseBusters' ``dock`` suite over guild's docked poses. A filter
*input*, not a filter: nothing is dropped or blanked, every pose just
gets a verdict.

Two invariants are load-bearing. ``mol_pred`` is an in-memory Mol whose
bond orders come from SMILES, never a file path -- inferring them from
geometry would make the intramolecular checks judge the coordinates under
suspicion. And the receptor is extracted from the complex PDB, not the
template, because Boltz re-centres its prediction so the frames differ.

``pb_valid`` fails closed, so ``df[df.pb_valid]`` cannot admit an
unverified pose; ``posebusters_status`` separates "invalid" from "could
not check".
"""

import glob
import logging
import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from guild.constants.guild import (
    DIFFDOCK_PREFIX,
    GNINA_PREFIX,
    PROTEIN_CONF_ID,
    SMILES,
    VINA_PREFIX,
)
from guild.constants.interactions import LIGAND_RESNAME
from guild.constants.posebusters import (
    CHECK_INTERNAL_ENERGY,
    DEFAULT_POSE_SCOPE,
    DEFAULT_POSEBUSTERS_CONFIG,
    PB_CHECK_COLUMNS,
    PB_COMBINATION_ID,
    PB_COMPLEX_PDB,
    PB_DOCKING_METHOD,
    PB_ERROR,
    PB_FAILED_CHECKS,
    PB_INTERMOLECULAR_CHECK_COLUMNS,
    PB_INTERMOLECULAR_VALID,
    PB_INTRAMOLECULAR_CHECK_COLUMNS,
    PB_INTRAMOLECULAR_VALID,
    PB_N_CHECKS_FAILED,
    PB_POSE,
    PB_STATUS,
    PB_STATUS_BUST_FAILED,
    PB_STATUS_LIGAND_BUILD_FAILED,
    PB_STATUS_MISSING_COMPLEX,
    PB_STATUS_NO_POSES,
    PB_STATUS_OK,
    PB_STATUS_PROTEIN_BUILD_FAILED,
    PB_STATUS_TEMPLATE_FALLBACK,
    PB_STATUS_UNAVAILABLE,
    PB_VALID,
    POSE_SCOPE_ALL,
    POSE_SCOPE_BEST,
    POSEBUSTERS_COLUMNS,
    POSEBUSTERS_MAX_POSES,
    POSEBUSTERS_REPORT_ID_COLUMNS,
)
from guild.tools.pose_molecules import (
    mol_from_pdb_block,
    mols_from_sdf,
    split_complex_records,
    split_pdbqt_models,
)

logger = logging.getLogger(__name__)


def _read_complex(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _normalise_check_name(name: str) -> str:
    # PoseBusters emits hyphens where guild's constants use underscores;
    # spaces are folded too, in case of an upstream relabelling.
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _pose_mols(
    complex_pdb_path: str,
    batch_folder: str,
    docking_method: str,
    combination_id: str,
    smiles: str,
    ligand_resname: str,
    max_poses: int,
) -> List[Tuple[object, Optional[str], bool]]:
    """Collect this combination's poses as RDKit mols, best-first.

    Only the top pose lives in the complex PDB, so going past it means
    reading the engine's own output. Falls back to the complex PDB's ligand
    when that is absent; Boltz always lands there, predicting one complex.
    """
    method_folder = f"{batch_folder}/{docking_method}"
    poses: List[Tuple[object, Optional[str], bool]] = []

    if docking_method in (VINA_PREFIX, GNINA_PREFIX):
        # gnina in sdf input mode writes an SDF instead of a PDBQT; prefer it,
        # since native bond orders need no template.
        sdf_path = f"{method_folder}/{combination_id}.sdf"
        pdbqt_path = f"{method_folder}/{combination_id}.pdbqt"
        if os.path.exists(sdf_path):
            poses = mols_from_sdf(sdf_path, smiles)
        elif os.path.exists(pdbqt_path):
            poses = [mol_from_pdb_block(b, smiles) for b in split_pdbqt_models(pdbqt_path)]

    elif docking_method == DIFFDOCK_PREFIX:
        # Rank by the confidence in the filename, as generate_diffdock_complex_pdbs
        # does, so pose 1 is the pose that reached the complex PDB. Parsing
        # confidence also sidesteps the rank<N> lexicographic trap (rank10 <
        # rank2).
        results_dir = f"{method_folder}/results/{combination_id}"
        scored = []
        for path in glob.glob(f"{results_dir}/*_confidence*.sdf"):
            try:
                confidence = float(
                    os.path.basename(path).split("_confidence")[1].replace(".sdf", "")
                )
            except (IndexError, ValueError):
                continue
            scored.append((confidence, path))
        for _, path in sorted(scored, key=lambda item: item[0], reverse=True):
            poses.extend(mols_from_sdf(path, smiles))

    if not poses:
        try:
            ligand_lines, _ = split_complex_records(_read_complex(complex_pdb_path), ligand_resname)
        except OSError:
            return []
        if ligand_lines:
            poses = [mol_from_pdb_block("".join(ligand_lines) + "END\n", smiles)]

    return poses[:max_poses]


def _base_record(
    combination_id: str,
    protein_conf_id: str,
    smiles: str,
    docking_method: str,
    complex_pdb_path: str,
    pose: Optional[int] = None,
) -> Dict:
    """Identity columns shared by every summary row this module emits."""
    return {
        PB_COMBINATION_ID: combination_id,
        PROTEIN_CONF_ID: protein_conf_id,
        SMILES: smiles,
        PB_DOCKING_METHOD: docking_method,
        PB_POSE: pose,
        PB_COMPLEX_PDB: complex_pdb_path,
    }


def _failed_record(base: Dict, status: str, error: Optional[str]) -> Dict:
    """A row for a pose that could not be evaluated.

    Check columns are left absent: "not checked" must not read as "passed".
    """
    return {
        **base,
        PB_STATUS: status,
        PB_ERROR: error,
        PB_VALID: False,
        PB_INTRAMOLECULAR_VALID: False,
        PB_INTERMOLECULAR_VALID: False,
        PB_N_CHECKS_FAILED: None,
        PB_FAILED_CHECKS: None,
    }


def _passed(value) -> bool:
    """A check counts as passed only when it is present and truthy.

    NaN is explicitly excluded: ``bool(float('nan'))`` is True, so a check
    PoseBusters could not evaluate would otherwise count as a pass and make
    ``pb_valid`` fail open.
    """
    return value is not None and not pd.isna(value) and bool(value) is True


def _summarise_checks(base: Dict, results: Dict, status: str, error: Optional[str]) -> Dict:
    """Turn one normalised PoseBusters result row into a guild summary record.

    Only checks present are judged -- ``dock_fast`` omits
    ``internal_energy``. A check present but unevaluated counts as failed.
    """
    record = {**base, PB_STATUS: status, PB_ERROR: error}

    failed = []
    for column in PB_CHECK_COLUMNS:
        if column not in results:
            continue
        record[column] = results[column]
        if not _passed(results[column]):
            failed.append(column)

    def _group_valid(columns):
        present = [c for c in columns if c in results]
        return bool(present) and all(_passed(results[c]) for c in present)

    record[PB_INTRAMOLECULAR_VALID] = _group_valid(PB_INTRAMOLECULAR_CHECK_COLUMNS)
    record[PB_INTERMOLECULAR_VALID] = _group_valid(PB_INTERMOLECULAR_CHECK_COLUMNS)
    record[PB_VALID] = record[PB_INTRAMOLECULAR_VALID] and record[PB_INTERMOLECULAR_VALID]
    record[PB_N_CHECKS_FAILED] = len(failed)
    record[PB_FAILED_CHECKS] = ";".join(sorted(failed))
    return record


def _warn_missing_checks(results: Dict, config: str) -> None:
    """Warn when a declared check is absent from PoseBusters' output.

    Guards the dangerous direction: a renamed check drops out of the verdict
    and makes ``pb_valid`` more permissive. Added checks are simply unused.
    """
    missing = [c for c in PB_CHECK_COLUMNS if c not in results]
    if config.endswith("_fast"):
        missing = [c for c in missing if c != CHECK_INTERNAL_ENERGY]
    if missing:
        logger.warning(
            f"PoseBusters config {config!r} did not emit these declared checks: "
            f"{sorted(missing)}. They are excluded from the verdict, which makes "
            f"pb_valid more permissive — update PB_CHECK_COLUMNS in "
            f"guild/constants/posebusters.py if posebusters renamed them."
        )


def validate_pose(
    complex_pdb_path: str,
    combination_id: str,
    protein_conf_id: str,
    smiles: str,
    docking_method: str,
    batch_folder: str,
    ligand_resname: str = LIGAND_RESNAME,
    config: str = DEFAULT_POSEBUSTERS_CONFIG,
    pose_scope: str = DEFAULT_POSE_SCOPE,
    max_poses: int = POSEBUSTERS_MAX_POSES,
    buster=None,
) -> Tuple[List[Dict], List[Dict]]:
    """Validate one combination's pose(s) with PoseBusters.

    Never raises, and always returns at least one summary row: a combination
    that could not be evaluated must still appear, with a status saying why.

    :param pose_scope: ``best`` | ``escalate`` | ``all``.
    :param buster: Pre-built ``PoseBusters``, so a batch parses the config YAML
        once rather than per pose.
    :return: ``(summary_rows, full_report_rows)``.
    """
    base = _base_record(combination_id, protein_conf_id, smiles, docking_method, complex_pdb_path)

    try:
        from posebusters import PoseBusters
    except ImportError as error:
        # Report the real exception: posebusters pulls in rdkit, so this often
        # means a broken dependency rather than a missing posebusters.
        logger.warning(
            f"PoseBusters analysis unavailable — {type(error).__name__}: {error}. "
            f"If posebusters itself is missing, check it is in pyproject.toml, that "
            f"uv.lock is up to date, and that the image was rebuilt; otherwise this "
            f"is a broken or incompatible dependency."
        )
        return [_failed_record(base, PB_STATUS_UNAVAILABLE, f"{type(error).__name__}: {error}")], []

    if not os.path.exists(complex_pdb_path):
        return [
            _failed_record(base, PB_STATUS_MISSING_COMPLEX, f"not found: {complex_pdb_path}")
        ], []

    n_wanted = 1 if pose_scope == POSE_SCOPE_BEST else max_poses
    poses = _pose_mols(
        complex_pdb_path,
        batch_folder,
        docking_method,
        combination_id,
        smiles,
        ligand_resname,
        n_wanted,
    )
    if not poses:
        return [
            _failed_record(base, PB_STATUS_NO_POSES, f"no pose records found for {combination_id}")
        ], []

    if buster is None:
        try:
            buster = PoseBusters(config=config)
        except Exception as error:
            return [
                _failed_record(base, PB_STATUS_BUST_FAILED, f"{type(error).__name__}: {error}")
            ], []

    summary_rows: List[Dict] = []
    report_rows: List[Dict] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        receptor_path = os.path.join(tmp_dir, "receptor.pdb")
        try:
            _, protein_lines = split_complex_records(
                _read_complex(complex_pdb_path), ligand_resname
            )
            if not protein_lines:
                raise ValueError(
                    f"no protein atoms left after excluding resname {ligand_resname!r}"
                )
            with open(receptor_path, "w") as handle:
                handle.writelines(protein_lines)
                handle.write("END\n")
        except Exception as error:
            return [
                _failed_record(
                    base, PB_STATUS_PROTEIN_BUILD_FAILED, f"{type(error).__name__}: {error}"
                )
            ], []

        for index, (mol, reason, used_fallback) in enumerate(poses, start=1):
            pose_base = dict(base, **{PB_POSE: index})

            if mol is None:
                summary_rows.append(
                    _failed_record(pose_base, PB_STATUS_LIGAND_BUILD_FAILED, reason)
                )
                continue
            if used_fallback:
                logger.warning(
                    f"PoseBusters: SMILES template did not match pose {index} of "
                    f"{combination_id} ({docking_method}) — bond orders inferred from "
                    f"geometry, so intramolecular checks are weakened. {reason}"
                )

            try:
                # full_report also returns the measurements behind the booleans.
                frame = buster.bust(
                    mol_pred=mol, mol_true=None, mol_cond=receptor_path, full_report=True
                )
            except Exception as error:
                summary_rows.append(
                    _failed_record(
                        pose_base, PB_STATUS_BUST_FAILED, f"{type(error).__name__}: {error}"
                    )
                )
                continue

            if frame is None or frame.empty:
                summary_rows.append(
                    _failed_record(pose_base, PB_STATUS_BUST_FAILED, "PoseBusters returned no rows")
                )
                continue

            raw = frame.reset_index(drop=True).iloc[0].to_dict()
            results = {_normalise_check_name(key): value for key, value in raw.items()}
            _warn_missing_checks(results, config)

            status = PB_STATUS_TEMPLATE_FALLBACK if used_fallback else PB_STATUS_OK
            summary_rows.append(_summarise_checks(pose_base, results, status, reason))
            report_rows.append({**pose_base, **results})

            if pose_scope != POSE_SCOPE_ALL and summary_rows[-1][PB_VALID]:
                # best/escalate ask only whether a valid pose exists.
                break

    return summary_rows, report_rows


# Per-process cache keyed by config; building a PoseBusters parses YAML.
_WORKER_BUSTERS: Dict[str, object] = {}


def _validate_pose_worker(task: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Picklable entry point for the multiprocessing path."""

    class _UnavailableBuster:
        def __init__(self, error: Exception):
            self._error = error

        def bust(self, *args, **kwargs):
            raise self._error

    config = task.get("config", DEFAULT_POSEBUSTERS_CONFIG)
    if config not in _WORKER_BUSTERS:
        try:
            from posebusters import PoseBusters

            _WORKER_BUSTERS[config] = PoseBusters(config=config)
        except Exception as error:
            # validate_pose surfaces the reason per row; cache a stub so
            # construction is not retried for every task.
            _WORKER_BUSTERS[config] = _UnavailableBuster(error)
    return validate_pose(**task, buster=_WORKER_BUSTERS[config])


def validate_batch_poses(
    complex_metadata: List[Tuple[str, str, str, str, str]],
    batch_folder: str,
    ligand_resname: str = LIGAND_RESNAME,
    config: str = DEFAULT_POSEBUSTERS_CONFIG,
    pose_scope: str = DEFAULT_POSE_SCOPE,
    max_poses: int = POSEBUSTERS_MAX_POSES,
    n_processes: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate every complex in one batch.

    ``complex_metadata`` is the 5-tuple shape
    :func:`guild.analysis.prolif.get_batch_prolif_interactions` also takes,
    so one discovery pass feeds PLIP, ProLIF and PoseBusters alike.

    PoseBusters' own ``max_workers`` stays at its in-process default: its
    pool raises ``BrokenProcessPool`` when a worker dies, and one pool
    boundary we control beats two we do not.

    :return: ``(summary_df, report_df)``, header-only when there is nothing
        to validate. Every input combination appears in the summary.
    """
    if not complex_metadata:
        return (
            pd.DataFrame(columns=POSEBUSTERS_COLUMNS),
            pd.DataFrame(columns=POSEBUSTERS_REPORT_ID_COLUMNS),
        )

    tasks = [
        {
            "complex_pdb_path": complex_pdb,
            "combination_id": combination_id,
            "protein_conf_id": protein_conf_id,
            "smiles": smiles,
            "docking_method": docking_method,
            "batch_folder": batch_folder,
            "ligand_resname": ligand_resname,
            "config": config,
            "pose_scope": pose_scope,
            "max_poses": max_poses,
        }
        for complex_pdb, combination_id, protein_conf_id, smiles, docking_method in (
            complex_metadata
        )
    ]

    summary_records: List[Dict] = []
    report_records: List[Dict] = []

    if n_processes and n_processes > 1 and len(tasks) > 1:
        # Never more workers than work: n_processes defaults to cpu_count().
        with mp.Pool(processes=min(n_processes, len(tasks))) as pool:
            for summary_rows, report_rows in pool.imap(_validate_pose_worker, tasks):
                summary_records.extend(summary_rows)
                report_records.extend(report_rows)
    else:
        buster = None
        try:
            from posebusters import PoseBusters

            buster = PoseBusters(config=config)
        except Exception as error:
            # validate_pose surfaces this per row; note it once here.
            logger.debug(f"Deferring PoseBusters construction to per-pose handling: {error}")
        for task in tasks:
            summary_rows, report_rows = validate_pose(**task, buster=buster)
            summary_records.extend(summary_rows)
            report_records.extend(report_rows)

    summary_df = _frame_with_schema(summary_records, POSEBUSTERS_COLUMNS, exact=True)
    report_df = _frame_with_schema(report_records, POSEBUSTERS_REPORT_ID_COLUMNS, exact=False)
    return summary_df, report_df


def _frame_with_schema(records: List[Dict], columns: List[str], exact: bool) -> pd.DataFrame:
    """Build a DataFrame with ``columns`` guaranteed present and leading.

    :param exact: False keeps extra keys after them, so a posebusters
        upgrade that adds a measurement widens the report.
    """
    if not records:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(records)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    if exact:
        return frame[columns]
    extras = [c for c in frame.columns if c not in columns]
    return frame[columns + extras]
