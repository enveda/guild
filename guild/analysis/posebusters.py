"""
PoseBusters pose-validity analysis.

Runs PoseBusters' ``dock`` check suite over guild's docked poses and reports
which are physically implausible. This is a *filter input*, not a filter: no
score is ever blanked and no row is ever dropped — every pose gets a verdict
and downstream consumers decide what to do with it.

Three things about this module are load-bearing and easy to break:

1. **Bond orders must be right.** Complex PDBs carry no ligand ``CONECT``
   records, so RDKit would infer bonds from 3D distance — and 3D geometry is
   exactly what is under suspicion. The intramolecular checks (bond lengths,
   bond angles, aromatic-ring flatness, internal energy) are meaningless
   against inferred bonds. ``mol_pred`` is therefore an in-memory
   :class:`rdkit.Chem.Mol`: taken straight from the engine's SDF where one
   exists, and otherwise rebuilt from the combination's SMILES via
   ``AssignBondOrdersFromTemplate``, mirroring
   :func:`guild.analysis.prolif.analyze_prolif_interactions`. "Simplifying"
   this to a file path silently guts half the checks.

2. **The receptor comes from the complex PDB, not from the template.** Boltz
   re-centres its predicted complex, so a template-frame receptor and a
   Boltz-frame ligand are not in the same physical space (see the Boltz
   coordinate-frame note in CLAUDE.md). Extracting both sides from one complex
   PDB makes the frames agree by construction, for every engine.

3. **PoseBusters raises on a bad molecule; it does not record a failure row.**
   An unusable pose surfaces as ``ValueError: Bad Conformer Id`` out of
   ``bust()``. Every call is wrapped, and every failure becomes a row with a
   ``posebusters_status`` explaining itself.

``pb_valid`` fails **closed** — it is never ``None``, so ``df[df.pb_valid]``
can never admit a pose that was not actually verified. Use
``posebusters_status`` to tell an invalid pose (``ok`` + ``pb_valid`` False)
from one that could not be checked.
"""

import glob
import logging
import multiprocessing as mp
import os
import tempfile
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

logger = logging.getLogger(__name__)

# PDBQT lines carry extra columns past 66 (partial charge, autodock type) that
# a PDB parser chokes on. Same truncation _convert_pdbqt_to_pdb falls back to
# in guild/transformers/pdb.py.
_PDB_LINE_KEEP = 66


def _normalise_check_name(name: str) -> str:
    """
    Normalise a PoseBusters output column to a guild column name.

    PoseBusters emits lowercase snake_case with embedded hyphens —
    ``non-aromatic_ring_non-flatness``, ``protein-ligand_maximum_distance``.
    Guild's constants use underscores throughout. Spaces are folded too so a
    future upstream relabelling to human-readable headers still maps cleanly.
    """
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _split_complex_pdb(pdb_text: str, ligand_resname: str) -> Tuple[List[str], List[str]]:
    """
    Split complex-PDB text into (ligand_lines, protein_lines) on residue name.

    Resname is the reliable ligand marker, not chain ID — ``cif_to_pdb`` may
    rename Boltz's ligand chain, which is why the chain rewrite in
    :func:`guild.docking.boltz.relabel_ligand_chain_in_pdb` cannot be trusted
    here.
    """
    ligand_lines, protein_lines = [], []
    for line in pdb_text.splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")) and len(line) > 20:
            if line[17:20].strip() == ligand_resname:
                ligand_lines.append(line)
            else:
                protein_lines.append(line)
    return ligand_lines, protein_lines


def _split_pdbqt_models(pdbqt_path: str) -> List[str]:
    """
    Split a multi-model Vina/gnina PDBQT into per-pose PDB text blocks.

    Vina and gnina write their poses score-sorted, so the returned list is
    already best-first. Extra PDBQT columns are truncated so RDKit's PDB parser
    accepts each block.

    :return: One PDB text block per MODEL. A PDBQT with no MODEL records (a
        single-pose file) yields one block.
    """
    blocks, current = [], []
    with open(pdbqt_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("MODEL"):
                current = []
            elif line.startswith("ENDMDL"):
                if current:
                    blocks.append("".join(current) + "END\n")
                current = []
            elif line.startswith(("ATOM", "HETATM")):
                current.append(line[:_PDB_LINE_KEEP].rstrip("\n") + "\n")

    # No MODEL/ENDMDL framing (single-pose PDBQT) — emit what we collected.
    if not blocks and current:
        blocks.append("".join(current) + "END\n")
    return blocks


def _mol_from_pdb_block(pdb_block: str, smiles: str):
    """
    Build an RDKit ligand from a PDB block, taking bond orders from ``smiles``.

    Ported from :func:`guild.analysis.prolif.analyze_prolif_interactions` so
    both analyses perceive the same molecule. See this module's docstring for
    why the SMILES template is not optional.

    :return: ``(mol, reason, used_template_fallback)``. ``mol`` is None only
        when the block itself is unusable, in which case ``reason`` says so.
        ``used_template_fallback`` is True when the SMILES did not match and
        bonds were inferred from geometry instead.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if not pdb_block.strip():
        return None, "empty ligand PDB block", False

    raw = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    if raw is None:
        return None, "RDKit could not parse the ligand PDB block", False

    template = Chem.MolFromSmiles(smiles) if smiles else None
    if template is None:
        reason = f"unusable SMILES template: {smiles!r}"
    elif template.GetNumHeavyAtoms() != raw.GetNumHeavyAtoms():
        # Guard against a partial substructure match. AssignBondOrdersFromTemplate
        # does NOT require the template to describe the whole molecule — given a
        # template that merely matches part of the pose it succeeds silently and
        # returns a molecule whose unmatched bonds keep their geometry-inferred
        # orders. That is indistinguishable from success at the call site, so the
        # atom counts have to be compared explicitly. Heavy atoms only:
        # protonation differences between the SMILES and the pose are expected
        # and harmless, a different heavy-atom skeleton is not.
        reason = (
            f"SMILES template has {template.GetNumHeavyAtoms()} heavy atoms but the "
            f"pose has {raw.GetNumHeavyAtoms()} — template does not describe this ligand"
        )
    else:
        try:
            mol = AllChem.AssignBondOrdersFromTemplate(template, raw)
            Chem.SanitizeMol(mol)
            return mol, None, False
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"

    # Fall back to the geometry-perceived molecule. Intermolecular checks stay
    # fully valid; the intramolecular ones are judged against inferred bonds,
    # which is why the caller promotes this to its own status value.
    try:
        Chem.SanitizeMol(
            raw, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        )
    except Exception as error:
        return None, f"sanitization failed after template fallback: {error}", False
    return raw, reason, True


def _mols_from_sdf(sdf_path: str, smiles: str) -> List[Tuple[object, Optional[str], bool]]:
    """
    Read every record of an SDF as a ligand, preserving file order.

    Used for DiffDock's ranked pose SDFs and gnina's ``--gnina-input-mode sdf``
    output. An SDF already carries bond orders, so these poses skip the SMILES
    template entirely — which also means they are immune to the
    template-mismatch weakening that PDB-sourced poses can hit. A record that
    will not sanitize falls back to the template path via its PDB block.
    """
    from rdkit import Chem

    mols = []
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)

    fallback_records = None
    for index, mol in enumerate(supplier):
        if mol is not None:
            mols.append((mol, None, False))
            continue

        # Unusual valences (e.g. gnina covalent output) defeat sanitization;
        # re-read unsanitized and rebuild through the SMILES template.
        if fallback_records is None:
            fallback = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
            fallback_records = [m for m in fallback if m is not None]

        if index < len(fallback_records):
            try:
                block = Chem.MolToPDBBlock(fallback_records[index])
            except Exception as error:
                mols.append((None, f"unreadable SDF record {index}: {error}", False))
                continue
            mols.append(_mol_from_pdb_block(block, smiles))
        else:
            mols.append((None, f"unreadable SDF record {index}", False))
    return mols


def _pose_mols(
    complex_pdb_path: str,
    batch_folder: str,
    docking_method: str,
    combination_id: str,
    smiles: str,
    ligand_resname: str,
    max_poses: int,
) -> List[Tuple[object, Optional[str], bool]]:
    """
    Collect this combination's poses as RDKit mols, best-first.

    Only the top pose lives in ``<combination>_complex.pdb`` (``build_complex_pdb``
    merges MODEL 1 only), so escalating past it means reading each engine's own
    output. Every pose of a given engine shares the coordinate frame of the
    receptor that engine's complex PDB was built from, so one extracted
    receptor serves them all.

    Falls back to the complex PDB's own ligand whenever the native pose file is
    absent — a pruned or partially-synced project tree still yields the top pose
    rather than nothing. Boltz always lands on that path: it predicts one
    complex, so its best pose is its only pose.

    :return: ``(mol, reason, used_template_fallback)`` triples, capped at
        ``max_poses``.
    """
    method_folder = f"{batch_folder}/{docking_method}"
    poses: List[Tuple[object, Optional[str], bool]] = []

    if docking_method in (VINA_PREFIX, GNINA_PREFIX):
        # gnina in --gnina-input-mode sdf writes an SDF instead of a PDBQT.
        # Prefer it when present: native SDF bond orders need no template.
        sdf_path = f"{method_folder}/{combination_id}.sdf"
        pdbqt_path = f"{method_folder}/{combination_id}.pdbqt"
        if os.path.exists(sdf_path):
            poses = _mols_from_sdf(sdf_path, smiles)
        elif os.path.exists(pdbqt_path):
            poses = [_mol_from_pdb_block(b, smiles) for b in _split_pdbqt_models(pdbqt_path)]

    elif docking_method == DIFFDOCK_PREFIX:
        # Rank by the confidence encoded in the filename, matching how
        # generate_diffdock_complex_pdbs picks its best pose, so pose 1 here is
        # the same pose that ended up in the complex PDB. Parsing confidence
        # also sidesteps the rank<N> lexicographic trap (rank10 sorts before
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
            poses.extend(_mols_from_sdf(path, smiles))

    if not poses:
        try:
            ligand_lines, _ = _split_complex_pdb(_read_text(complex_pdb_path), ligand_resname)
        except OSError:
            return []
        if ligand_lines:
            poses = [_mol_from_pdb_block("".join(ligand_lines) + "END\n", smiles)]

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
    """
    A row for a pose that could not be evaluated.

    Check columns are left absent (they become NaN on DataFrame construction):
    "not checked" must not read as "checked and passed". The verdicts fail
    closed so the row can never survive a ``df[df.pb_valid]`` filter.
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
    """A check counts as passed only when it is present and truthy."""
    return value is not None and bool(value) is True


def _summarise_checks(base: Dict, results: Dict, status: str, error: Optional[str]) -> Dict:
    """
    Turn one normalised PoseBusters result row into a guild summary record.

    Only checks actually present in the result are judged — the ``dock_fast``
    config legitimately omits ``internal_energy``, and an absent check must not
    count against the pose. A check present but None (PoseBusters could not
    evaluate it — e.g. every intermolecular check when the receptor fails to
    load) does count as not-passed: nothing about it was verified.
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
    """
    Warn when a declared check is absent from PoseBusters' output.

    This is the upgrade-safety valve, and it guards the dangerous direction: a
    renamed check silently drops out of the verdict, which makes ``pb_valid``
    *more* permissive rather than less. Missing columns therefore warn; newly
    added ones do not (they are simply not used yet).

    ``internal_energy`` is expected to be absent under ``dock_fast``, which
    omits it by design.
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
    """
    Validate one combination's pose(s) with PoseBusters.

    Never raises, and always returns at least one summary row — a combination
    that could not be evaluated must still appear in the output table, with a
    ``posebusters_status`` saying why.

    :param complex_pdb_path: Path to ``<combination>_complex.pdb``.
    :param combination_id: ``f"{protein_conf_id}_{ligand_id}"``.
    :param smiles: Ligand SMILES — the bond-order template. See module docstring.
    :param docking_method: Method prefix, e.g. ``"vina"``.
    :param batch_folder: Batch root, used to locate the engine's own pose files
        when escalating past the top pose.
    :param pose_scope: ``best`` | ``escalate`` | ``all``.
    :param max_poses: Hard cap on poses validated, for escalate and all.
    :param buster: Optional pre-built ``PoseBusters`` instance, so a batch pays
        the config-YAML parse once rather than per pose.
    :return: ``(summary_rows, full_report_rows)``.
    """
    base = _base_record(combination_id, protein_conf_id, smiles, docking_method, complex_pdb_path)

    try:
        from posebusters import PoseBusters
    except ImportError as error:
        # Report the real exception. posebusters pulls in rdkit, so an
        # ImportError here often means an installed-but-broken dependency
        # rather than a missing posebusters, and reporting only the latter sends
        # you looking in the wrong place. The step is on by default, so an image
        # built before the dependency was added must degrade to a row plus a
        # warning rather than breaking every run.
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
            _, protein_lines = _split_complex_pdb(_read_text(complex_pdb_path), ligand_resname)
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
                # full_report=True returns the boolean checks AND the numeric
                # measurements behind them in one pass, so the report file costs
                # nothing beyond the summary.
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
                # best/escalate: the question is whether a valid pose exists, so
                # stop as soon as one does.
                break

    return summary_rows, report_rows


# Per-process PoseBusters cache for the multiprocessing path, keyed by config.
# Building one parses a YAML config, so without this every pose in a worker
# would pay that cost again.
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
            # validate_pose will surface the reason as a per-row status; cache a
            # stub so we don't retry construction for every task.
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
    """
    Validate every complex in one batch.

    :param complex_metadata: List of
        ``(complex_pdb, combination_id, protein_conf_id, smiles, docking_method)``
        — the same 5-tuple
        :func:`guild.analysis.prolif.get_batch_prolif_interactions` takes, so one
        discovery pass feeds PLIP, ProLIF and PoseBusters alike.
    :param n_processes: Guild-level worker count. PoseBusters' own
        ``max_workers`` is deliberately left at its in-process default: its pool
        raises ``BrokenProcessPool`` when a worker dies, and this codebase has
        already been burned by exactly that (the OpenBabel-abort recovery loops
        in guild/bulk.py). One pool boundary we control beats two we do not.
    :return: ``(summary_df, report_df)``. The summary always has exactly
        :data:`POSEBUSTERS_COLUMNS`; both are header-only when there is nothing
        to validate. Every input combination is represented in the summary, so
        row identity never has to be recovered positionally.
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
        # Never start more workers than there is work: n_processes defaults to
        # BulkRun.n_workers, which is mp.cpu_count() unless set, and spawning 64
        # interpreters to validate 3 poses costs far more than it saves.
        with mp.Pool(processes=min(n_processes, len(tasks))) as pool:
            for summary_rows, report_rows in pool.imap(_validate_pose_worker, tasks):
                summary_records.extend(summary_rows)
                report_records.extend(report_rows)
    else:
        # Serial path builds the PoseBusters instance once for the whole batch.
        buster = None
        try:
            from posebusters import PoseBusters

            buster = PoseBusters(config=config)
        except Exception as error:
            # validate_pose surfaces this as a per-row status; note it once here
            # instead of once per pose.
            logger.debug(f"Deferring PoseBusters construction to per-pose handling: {error}")
        for task in tasks:
            summary_rows, report_rows = validate_pose(**task, buster=buster)
            summary_records.extend(summary_rows)
            report_records.extend(report_rows)

    summary_df = _frame_with_schema(summary_records, POSEBUSTERS_COLUMNS, exact=True)
    report_df = _frame_with_schema(report_records, POSEBUSTERS_REPORT_ID_COLUMNS, exact=False)
    return summary_df, report_df


def _frame_with_schema(records: List[Dict], columns: List[str], exact: bool) -> pd.DataFrame:
    """
    Build a DataFrame with ``columns`` guaranteed present and leading.

    :param exact: When True the result holds exactly ``columns`` (the summary
        table's fixed schema). When False any extra keys are kept after them —
        the full report is deliberately open-ended so a posebusters upgrade that
        adds a measurement widens the file instead of tripping an assert.
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
