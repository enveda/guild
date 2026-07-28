"""
Master Guild runner — accepts all key parameters as CLI arguments.

Usage (directly or via Makefile):
    python scripts/run_guild.py \
        --project my_project \
        --combinations /workspace/notebooks/data_prep/full_combinations_table.csv \
        --methods boltz vina \
        [--decoys /workspace/guild/support/decoys/chembl_36_decoys_2.tsv] \
        [--batch-size 2] \
        [--head 100] \
        [--clean]

Via Makefile:
    make run-boltz       PROJECT=my_project COMBINATIONS=/workspace/path/to/combos.tsv
    make run-vina        PROJECT=my_project COMBINATIONS=/workspace/path/to/combos.tsv
    make run-guild  PROJECT=my_project METHODS="vina boltz diffdock" HEAD=100 BATCH_SIZE=5
"""

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment hardening — must happen BEFORE importing torch / guild.
# Inside Docker the container UID may not have an /etc/passwd entry, so any
# library that calls pwd.getpwuid() or getpass.getuser() would crash.
# Setting UV_CACHE_DIR avoids uv writing to an unwritable default location.
# TORCHINDUCTOR_CACHE_DIR short-circuits PyTorch's default_cache_dir() which
# internally calls getpass.getuser().
# ---------------------------------------------------------------------------
os.environ.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Guild bulk docking pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project",
        "-p",
        required=True,
        help="Project name (used as output folder under data/).",
    )
    parser.add_argument(
        "--combinations",
        "-c",
        required=True,
        help="Path to the combinations TSV/CSV (protein–ligand pairs table).",
    )
    parser.add_argument(
        "--decoys",
        "-d",
        default=None,
        help=(
            "Path to the decoys TSV. "
            "Defaults to guild/support/decoys/chembl_36_decoys_2.tsv "
            "relative to the project root."
        ),
    )
    parser.add_argument(
        "--methods",
        "-m",
        nargs="+",
        default=["boltz"],
        choices=["boltz", "vina", "karmadock", "diffdock", "gnina"],
        help="Docking methods to run (space-separated).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for BulkRun.",
    )
    parser.add_argument(
        "--min-mol-wt",
        type=float,
        default=250,
        help="Minimum molecule weight filter.",
    )
    parser.add_argument(
        "--max-mol-wt",
        type=float,
        default=450,
        help="Maximum molecule weight filter.",
    )
    parser.add_argument(
        "--chembl-version",
        default="chembl_36",
        help="ChEMBL version string.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Delete the project output folder before running (fresh start).",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="Take only the first N rows from the combinations table (0 = all rows).",
    )
    parser.add_argument(
        "--use-known-binders",
        action="store_true",
        default=False,
        help="Enable known-binders expansion in BulkRun.",
    )
    parser.add_argument(
        "--no-decoys",
        action="store_true",
        default=False,
        help="Disable decoy generation (useful for single runs).",
    )
    parser.add_argument(
        "--box",
        default=None,
        help=(
            "Optional path to a Vina box file (center_{x,y,z} + size_{x,y,z}) used as a "
            "global fallback for combinations whose 'box_location' column is empty. "
            "Per-row values in the CSV always take precedence."
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes for Vina docking. Vina's internal "
            "threading auto-scales, so values >1 oversubscribe slightly but typically "
            "still help. Default 1 (safe but serial)."
        ),
    )
    parser.add_argument(
        "--vina-exhaustiveness",
        type=int,
        default=None,
        help=(
            "Vina exhaustiveness parameter (controls search thoroughness). "
            "Higher values improve pose quality at the cost of runtime. "
            "Defaults to VINA_DEFAULT_EXHAUSTIVENESS (16) when omitted."
        ),
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        default=False,
        help=(
            "Force CPU-only execution. Passed through to BulkRun(use_gpu=False); "
            "gnina respects it via its own --no_gpu flag. Vina and DiffDock are "
            "already CPU-only and unaffected; Boltz is genuinely GPU-bound and will "
            "still attempt to use a GPU (so don't combine --no-gpu with --methods boltz)."
        ),
    )
    parser.add_argument(
        "--no-plip",
        action="store_true",
        default=False,
        help=(
            "Skip the PLIP interactions analysis step. By default, after docking "
            "and scoring complete, guild runs PLIP over every method's complex "
            "PDBs and writes data/<project>/plip_interactions.tsv (always — "
            "header-only when nothing was produced). External consumers should "
            "read that file rather than installing plip locally; pass this flag "
            "only if you don't need the interactions output."
        ),
    )
    parser.add_argument(
        "--plip-only",
        action="store_true",
        default=False,
        help=(
            "Skip docking + scoring and run only the PLIP interactions step over "
            "an existing data/<project>/ tree. Useful for re-generating "
            "plip_interactions.tsv when only the PLIP code changed."
        ),
    )
    parser.add_argument(
        "--gnina-input-mode",
        choices=["pdbqt", "sdf"],
        default="pdbqt",
        help=(
            "Ligand/receptor input format for gnina. 'sdf' skips OpenBabel "
            "PDBQT prep entirely (gnina reads RDKit-generated SDF + cleaned "
            "PDB natively), but only takes effect when gnina is the sole "
            "PDBQT-relevant method requested — co-running with Vina or any "
            "Vina-rescore downgrades to PDBQT with a warning."
        ),
    )
    parser.add_argument(
        "--poses-dir",
        default=None,
        help=(
            "Optional path to a directory of user-supplied ligand starting "
            "poses, one '<ligand_id>.sdf' per ligand. When set, every "
            "ligand_id in the combinations CSV must have a matching SDF in "
            "the directory; otherwise BulkRun aborts before docking with a "
            "single error listing the missing IDs. Requires --pose-mode "
            "local or score (the default 'dock' mode is stochastic and "
            "ignores the supplied coordinates, so combining POSES_DIR with "
            "pose-mode=dock is rejected). Incompatible with "
            "--use-known-binders and decoy expansion (use --no-decoys)."
        ),
    )
    parser.add_argument(
        "--pose-mode",
        choices=["dock", "local", "score"],
        default="dock",
        help=(
            "How Vina/gnina consume a user-supplied starting pose. 'dock' "
            "(default) runs a normal global search — the supplied pose has "
            "*no effect* on Vina's starting coordinates. 'local' refines "
            "the pose with Vina.optimize() / gnina --local_only. 'score' "
            "evaluates the pose with Vina.score() / gnina --score_only. "
            "'local' and 'score' require --poses-dir to be set."
        ),
    )
    parser.add_argument(
        "--flexible-docking",
        action="store_true",
        default=False,
        help=(
            "Enable flexible receptor docking for Vina and gnina. Side-chain "
            "atoms of residues whose Cα lies inside the docking box are allowed "
            "to move during the search. In pdbqt mode the receptor is split into "
            "rigid+flex PDBQT files via mk_prepare_receptor.py; in gnina sdf mode "
            "gnina's --flexdist_ligand / --flexdist flags are used instead. Falls "
            "back to rigid docking if flex prep fails. Boltz, DiffDock, and "
            "KarmaDock are unaffected."
        ),
    )
    parser.add_argument(
        "--flexres-gnina",
        default=None,
        metavar="FLEXRES",
        help=(
            "gnina-only. Explicit flexible-residue spec passed directly as gnina's "
            "--flexres flag (e.g. 'A:88,91' or 'A:88,91_B:7'). Applied to every "
            "combination in the run. When set, takes priority over the automatic "
            "box-based / flexdist residue selection that --flexible-docking uses. "
            "Ignored by Vina, Boltz, DiffDock, and KarmaDock."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    np.random.seed(42)

    # PROJECT_ROOT is the host directory the Makefile bind-mounts at
    # ``/workspace`` — the place where ``data/<project>/`` outputs land and
    # where ``COMBINATIONS``/``BOX`` paths resolve. We read it from the
    # ``WORKSPACE_ROOT`` env var (set in the Makefile's DOCKER_COMMON) so the
    # script can live anywhere on disk: it does NOT have to sit under
    # ``WORKSPACE/scripts/``. The fallback to ``parent.parent`` covers
    # invocations that don't go through the Makefile (e.g. someone running
    # ``python scripts/run_guild.py`` locally).
    workspace_env = os.environ.get("WORKSPACE_ROOT")
    if workspace_env:
        PROJECT_ROOT = Path(workspace_env)
    else:
        PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Override guild's hardcoded paths so output lands in
    # PROJECT_ROOT/data/ (the volume-mounted workspace) rather than /app.
    import guild as _guild_pkg
    import guild.bulk as _bulk_mod
    import guild.constants.system as _sys_const
    import guild.run as _run_mod

    # ``guild/support/`` ships with the guild package itself. Resolving it
    # from the imported package's location works both in-repo
    # (``./guild/support``) and from an external caller where ``WORKSPACE``
    # is a different repo entirely and the guild source is bind-mounted at
    # ``/app/guild`` — there is no ``guild/`` directory at the caller's
    # workspace root.
    GUILD_PKG_DIR = Path(_guild_pkg.__file__).resolve().parent
    GUILD_SUPPORT_DIR = GUILD_PKG_DIR / "support"

    _sys_const.WORKING_DIR_PATH = PROJECT_ROOT
    _sys_const.PROJECTS_FOLDER = str(PROJECT_ROOT / "data")
    _sys_const.SUPPORT_FOLDER = str(GUILD_SUPPORT_DIR)
    _sys_const.PYTHON_EXECUTABLE = sys.executable  # works both locally & in Docker
    _bulk_mod.WORKING_DIR_PATH = PROJECT_ROOT
    _bulk_mod.PROJECTS_FOLDER = str(PROJECT_ROOT / "data")
    _run_mod.WORKING_DIR_PATH = PROJECT_ROOT

    # Patch diffdock module's imported constants so they see the overrides.
    import guild.docking.diffdock as _dd_mod

    _dd_mod.PYTHON_EXECUTABLE = sys.executable
    _dd_mod.SUPPORT_FOLDER = str(GUILD_SUPPORT_DIR)

    from guild.bulk import BulkRun

    project_name: str = args.project
    combinations_path = Path(args.combinations)
    decoys_path = (
        args.decoys
        if args.decoys is not None
        else str(GUILD_SUPPORT_DIR / "decoys" / "chembl_36_decoys_2.tsv")
    )

    if args.clean:
        shutil.rmtree(PROJECT_ROOT / "data" / project_name, ignore_errors=True)
        print(f"Cleaned output folder: data/{project_name}")

    # Auto-detect separator (handles .tsv named as .csv, etc.)
    runs_table = pd.read_csv(combinations_path, sep=None, engine="python")

    # Remap protein_path / box_location: strip any absolute host prefix up to
    # /notebooks/ or /temp_data/ so paths work both locally and inside Docker
    # (where the repo is mounted at /workspace).
    _PATH_PREFIX = re.compile(r"^.*?/(notebooks|temp_data)/")

    def _normalize_workspace_path(p):
        if pd.isna(p):
            return p
        s = str(p)
        m = _PATH_PREFIX.search(s)
        if not m:
            return p
        return str(PROJECT_ROOT / _PATH_PREFIX.sub(r"\1/", s, count=1))

    for col in ("protein_path", "box_location"):
        if col in runs_table.columns:
            runs_table[col] = runs_table[col].apply(_normalize_workspace_path)

    # Optionally take only the first N rows
    if args.head > 0:
        runs_table = runs_table.head(args.head).reset_index(drop=True)

    # Global box fallback: fill empty 'box_location' rows with --box, when supplied.
    # Per-row CSV values win.
    if args.box:
        if "box_location" not in runs_table.columns:
            runs_table["box_location"] = args.box
        else:
            mask = runs_table["box_location"].isna() | (
                runs_table["box_location"].astype(str).str.strip() == ""
            )
            runs_table.loc[mask, "box_location"] = args.box
        print(f"Box (global): {args.box}")

    print(f"Project:      {project_name}")
    print(f"Combinations: {combinations_path}  ({len(runs_table)} rows)")
    print(f"Decoys:       {decoys_path}")
    print(f"Methods:      {args.methods}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Known bndrs:  {args.use_known_binders}")
    print(f"GPU:          {not args.no_gpu}")

    poses_dir = _normalize_workspace_path(args.poses_dir) if args.poses_dir else None
    if poses_dir is not None:
        print(f"Poses dir:    {poses_dir}")
        print(f"Pose mode:    {args.pose_mode}")

    bulk = BulkRun(
        runs_table,
        project_name,
        methods_to_run=args.methods,
        batch_size=args.batch_size,
        min_mol_wt=args.min_mol_wt,
        max_mol_wt=args.max_mol_wt,
        chembl_version=args.chembl_version,
        decoys=decoys_path,
        use_decoys=not args.no_decoys,
        use_known_binders=args.use_known_binders,
        n_workers=args.n_workers,
        vina_exhaustiveness=args.vina_exhaustiveness,
        use_gpu=not args.no_gpu,
        gnina_input_mode=args.gnina_input_mode,
        poses_dir=poses_dir,
        pose_mode=args.pose_mode,
        flexible_docking=args.flexible_docking,
        flexres_gnina=args.flexres_gnina,
    )

    if not args.plip_only:
        t0 = time.time()
        bulk.run_docking()
        print(f"Docking time: {time.time() - t0:.1f}s")

        t0 = time.time()
        bulk.run_guild_scoring()
        print(f"Scoring time: {time.time() - t0:.1f}s")
    else:
        print("Skipping docking + scoring (--plip-only).")

    # PLIP interaction analysis: always run by default. The contract is that
    # data/<project>/plip_interactions.tsv exists after every run (header-only
    # if no method produced complex PDBs), so external notebooks can read it
    # without installing plip locally. --no-plip skips this step; --plip-only
    # runs ONLY this step.
    if args.plip_only or not args.no_plip:
        t0 = time.time()
        bulk.run_interactions_analysis()
        print(f"PLIP time:    {time.time() - t0:.1f}s")

    # ── Print final scores summary ──────────────────────────────────────
    if (
        hasattr(bulk, "rp_scores_df")
        and bulk.rp_scores_df is not None
        and not bulk.rp_scores_df.empty
    ):
        df = bulk.rp_scores_df
        # Pick the columns that actually exist: id, raw scores, rp scores
        show_cols = ["combination", "protein_config_id", "ligand_id", "ligand_category"]
        score_cols = [c for c in df.columns if c.endswith("_score") and not c.startswith("rank_")]
        show_cols = [c for c in show_cols if c in df.columns] + sorted(score_cols)
        print(f"\n{'='*80}")
        print(f"Final scores  ({len(df)} combinations)  →  {bulk.rp_scores_path}")
        print(f"{'='*80}")
        print(df[show_cols].to_string(index=False, max_rows=60))
        print()

    # Silence adlfs/fsspec weakref finalizer noise on shutdown.
    sys.stderr = open(os.devnull, "w")


if __name__ == "__main__":
    main()
