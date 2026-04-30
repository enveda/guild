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
        "--project", "-p",
        required=True,
        help="Project name (used as output folder under data/).",
    )
    parser.add_argument(
        "--combinations", "-c",
        required=True,
        help="Path to the combinations TSV/CSV (protein–ligand pairs table).",
    )
    parser.add_argument(
        "--decoys", "-d",
        default=None,
        help=(
            "Path to the decoys TSV. "
            "Defaults to guild/support/decoys/chembl_36_decoys_2.tsv "
            "relative to the project root."
        ),
    )
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        default=["boltz"],
        choices=["boltz", "vina", "karmadock", "diffdock"],
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
        "--no-decoys",
        action="store_true",
        default=False,
        help="Disable decoy generation (useful when decoy file is not available).",
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(42)

    # scripts/ lives directly under the project root, so one level up is enough.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Override guild's hardcoded paths so output lands in
    # PROJECT_ROOT/data/ (the volume-mounted workspace) rather than /app.
    import guild.bulk as _bulk_mod
    import guild.constants.system as _sys_const
    import guild.run as _run_mod

    _sys_const.WORKING_DIR_PATH = PROJECT_ROOT
    _sys_const.PROJECTS_FOLDER = str(PROJECT_ROOT / "data")
    _sys_const.SUPPORT_FOLDER = str(PROJECT_ROOT / "guild" / "support")
    _sys_const.PYTHON_EXECUTABLE = sys.executable  # works both locally & in Docker
    _bulk_mod.WORKING_DIR_PATH = PROJECT_ROOT
    _bulk_mod.PROJECTS_FOLDER = str(PROJECT_ROOT / "data")
    _run_mod.WORKING_DIR_PATH = PROJECT_ROOT

    # Patch diffdock module's imported constants so they see the overrides.
    import guild.docking.diffdock as _dd_mod
    _dd_mod.PYTHON_EXECUTABLE = sys.executable
    _dd_mod.SUPPORT_FOLDER = str(PROJECT_ROOT / "guild" / "support")

    from guild.bulk import BulkRun

    project_name: str = args.project
    combinations_path = Path(args.combinations)
    decoys_path = (
        args.decoys
        if args.decoys is not None
        else str(PROJECT_ROOT / "guild" / "support" / "decoys" / "chembl_36_decoys_2.tsv")
    )

    if args.clean:
        shutil.rmtree(PROJECT_ROOT / "data" / project_name, ignore_errors=True)
        print(f"Cleaned output folder: data/{project_name}")

    # Auto-detect separator (handles .tsv named as .csv, etc.)
    runs_table = pd.read_csv(combinations_path, sep=None, engine="python")

    # Remap protein_path: strip any absolute prefix up to /notebooks/ so paths
    # work both locally and inside Docker.
    if "protein_path" in runs_table.columns:
        runs_table["protein_path"] = runs_table["protein_path"].apply(
            lambda p: str(PROJECT_ROOT / re.sub(r"^.*?/notebooks/", "notebooks/", p))
            if pd.notna(p) and "/notebooks/" in str(p)
            else p
        )

    # Optionally take only the first N rows
    if args.head > 0:
        runs_table = runs_table.head(args.head).reset_index(drop=True)

    print(f"Project:      {project_name}")
    print(f"Combinations: {combinations_path}  ({len(runs_table)} rows)")
    print(f"Decoys:       {decoys_path}")
    print(f"Methods:      {args.methods}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Known bndrs:  {args.use_known_binders}")
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
        n_workers=1,
    )

    t0 = time.time()
    bulk.run_docking()
    print(f"Docking time: {time.time() - t0:.1f}s")

    t0 = time.time()
    bulk.run_guild_scoring()
    print(f"Scoring time: {time.time() - t0:.1f}s")

    # ── Print final scores summary ──────────────────────────────────────
    if hasattr(bulk, "rp_scores_df") and bulk.rp_scores_df is not None and not bulk.rp_scores_df.empty:
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
