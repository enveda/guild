#!/usr/bin/env python3
"""Join PoseBusters flags into guild_scores.txt.

Guild's `BulkRun._merge_posebusters_into_scores` normally does this at the end of
`run_pose_validity_analysis`, but it did not run for the
`small-example-all-methods` rerun -- `guild_scores.txt` arrived with no `pb_`
columns even though `posebusters_validity.tsv` was written. Reproducing that
merge here rather than re-running the pipeline, since it needs only the two
tables and not the 1.2 GB pose tree.

Logic is copied from guild/bulk.py so the result is identical:

* per (combination, method), `<method>_pb_valid` is True if ANY validated pose
  passed;
* `<method>_pb_pose` is the lowest-numbered pose that passed;
* a combination with no PoseBusters row at all gets NA, not False -- absence of
  evidence is not invalidity;
* the row count must not change. Flags are added, rows are never dropped.

Usage:
    python merge_posebusters_flags.py --rerun ../rerun
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

# Column names, matching guild/constants
SCORES_ID = "combination"          # COMBINATION_ID in guild_scores.txt
PB_ID = "combination_id"           # PB_COMBINATION_ID in the validity table
PB_METHOD = "docking_method"
PB_VALID = "pb_valid"
PB_POSE = "pose"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rerun", type=Path, default=Path("../rerun"),
                        help="directory holding guild_scores.txt and posebusters_validity.tsv")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    args = parser.parse_args(argv)

    scores_path = args.rerun / "guild_scores.txt"
    validity_path = args.rerun / "posebusters_validity.tsv"
    for path in (scores_path, validity_path):
        if not path.exists():
            parser.error(f"missing {path}")

    scores = pd.read_csv(scores_path, sep="\t", low_memory=False)
    validity = pd.read_csv(validity_path, sep="\t", low_memory=False)
    n_before, cols_before = len(scores), len(scores.columns)
    print(f"scores   {n_before} rows x {cols_before} cols")
    print(f"validity {len(validity)} rows, methods: "
          f"{sorted(validity[PB_METHOD].unique())}")

    existing = [c for c in scores.columns if "_pb_" in c]
    if existing:
        print(f"already has pb columns ({existing}); nothing to do")
        return 0

    valid_any = validity.groupby([PB_ID, PB_METHOD])[PB_VALID].any().unstack(PB_METHOD)
    first_valid = (
        validity[validity[PB_VALID]]
        .groupby([PB_ID, PB_METHOD])[PB_POSE]
        .min()
        .unstack(PB_METHOD)
    )

    for method in valid_any.columns:
        scores[f"{method}_pb_valid"] = scores[SCORES_ID].map(valid_any[method])
        scores[f"{method}_pb_pose"] = (
            scores[SCORES_ID].map(first_valid[method])
            if method in first_valid.columns else None
        )

    if len(scores) != n_before:
        raise RuntimeError(
            f"row count changed ({n_before} -> {len(scores)}); the merge must only add columns"
        )

    print(f"\nadded {len(scores.columns) - cols_before} columns:")
    for method in valid_any.columns:
        col = f"{method}_pb_valid"
        flags = scores[col]
        print(f"  {col:26s} True {int(flags.eq(True).sum()):3d}  "
              f"False {int(flags.eq(False).sum()):3d}  NA {int(flags.isna().sum()):3d}")

    # KarmaDock writes no complex PDB, so it has no validity rows and therefore no
    # columns here at all. That is a coverage gap and must not read as a pass.
    absent = sorted({"vina", "gnina", "karmadock", "diffdock", "boltz"} - set(valid_any.columns))
    if absent:
        print(f"\nno PoseBusters rows, so no columns added for: {', '.join(absent)}")

    if args.dry_run:
        print("\ndry run; nothing written")
        return 0

    backup = scores_path.with_suffix(".txt.pre_pb_merge")
    if not backup.exists():
        shutil.copy2(scores_path, backup)
        print(f"\nbackup -> {backup.name}")
    scores.to_csv(scores_path, sep="\t", index=False)
    print(f"wrote {scores_path.name}  ({len(scores)} rows x {len(scores.columns)} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
