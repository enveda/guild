import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor")

PROJECT_ROOT = Path("/workspace")

import guild.constants.system as _sys_const
import guild.bulk as _bulk_mod

_sys_const.WORKING_DIR_PATH = PROJECT_ROOT
_sys_const.PROJECTS_FOLDER = str(PROJECT_ROOT / "data")
_bulk_mod.WORKING_DIR_PATH = PROJECT_ROOT
_bulk_mod.PROJECTS_FOLDER = str(PROJECT_ROOT / "data")

from guild.bulk import BulkRun

runs_table = pd.read_csv("/workspace/combinations.csv")

# Must match the run-guild invocation exactly: project name, methods, batch_size, and
# the decoy/known-binder flags. BulkRun reconstructs the batch folder layout from them,
# and a mismatch silently finds no poses and writes empty TSVs rather than erroring.
# This run used NO_DECOYS=1 (decoys were explicit rows) and the Makefile's default
# BATCH_SIZE=2, confirmed against batch_progress.log (83 batches for 165 combos).
BATCH_SIZE = 2

bulk = BulkRun(
    runs_table,
    "small-example-all-methods",
    methods_to_run=["vina", "gnina", "karmadock", "diffdock", "boltz"],
    batch_size=BATCH_SIZE,
    use_decoys=False,
    use_known_binders=False,
)
bulk.run_pose_validity_analysis()
print(f"Wrote {bulk.posebusters_path} and {bulk.posebusters_report_path}")
