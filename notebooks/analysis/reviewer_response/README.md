# Reviewer-response analyses — reproduction

Every number quoted in the Guild reviewer response that is not read directly off a
`guild` output table is produced by one of two scripts here. One command each
regenerates all of them.

```shell
python reproduce_response_analyses.py --data ./data --out ./output
python reproduce_response_analyses.py --data ./data --only a2 a3                    # subset
python reproduce_response_analyses.py --data ./data --only r2_5_training_overlap --verify-dates
python build_supplementary_tables.py --data ./data --out ./output                   # Tables S1-S6
```

`reproduce_response_analyses.py` requires only `pandas` and `numpy` and does not import
`guild` (rank percentiles are recomputed locally, and `test_matches_guild` asserts the
reimplementation reproduces guild's own `rp_vina_score` exactly, to < 1e-9) -- with one
exception: `r2_5_training_overlap` imports `guild.constants.bulk.SCORES_DIRECTION_DICTIONARY`
to get each track's minimum/maximum convention right rather than hardcoding it. That module
is pure Python constants with no heavy dependencies, so this still needs no Docker image,
just `guild` importable (e.g. an editable install).

`build_supplementary_tables.py` does import `guild` (for
`compute_rank_percentile_scores`, to recompute rank percentiles through the shipped
code rather than trust stale ones on disk) and needs it importable the same way.

Nothing is downloaded and nothing outside `--out` is written, except
`r2_5_training_overlap --verify-dates`, which makes one read-only network call per
target to `data.rcsb.org` and is off by default.

## Where this code belongs

Lives here, under `notebooks/analysis/reviewer_response/`, alongside the existing figure
notebooks but in its own subdirectory since these are response-specific scripts rather than
notebooks. `--data` and `--out` are not committed — provide them locally (see below).

## Inputs — `--data`

| file | provenance | needs to go to Zenodo |
| --- | --- | --- |
| `guild_scores.txt` | `small-example-all-methods` rerun, 168 combinations, 3 targets. Used by `a2`, `reserve`, `r2_5_training_overlap` and `build_supplementary_tables.py`. | **yes** |
| `posebusters_validity.tsv` | same rerun — pose validity checks. `build_supplementary_tables.py` only (Tables S2, S3). | **yes** |
| `native_ligand_rmsd.tsv` | same rerun — native-ligand redocking RMSD. `build_supplementary_tables.py` only (Table S5). | **yes** |
| `vinarun_scores.txt` | `guild/vina_data/data/vinarun/dockwizard_scores.txt` on Azure — 403,000 rows, 133 targets, decoy/NP/synthetic | **yes** (56 MB) |
| `knownbinders_scores.txt` | `guild/vina_data_2/data/knownbindersvinarun/drrp_scores.txt` on Azure — 655 rows, 5 strong binders × 131 targets | **yes** |
| `vinarun_batch_progress.log` | `guild/vina_data/data/vinarun/batch_progress.log` | yes, small |
| `npsvinarun_batch.log` | `guild/vina_data/data/npsvinarun/batch_progress.log` | yes, small |
| `knownbindersvinarun_batch.log` | `guild/vina_data_2/data/knownbindersvinarun/batch_progress.log` | yes, small |

Azure root:
`abfss://cheminformatics@platformncus01.dfs.core.windows.net/`, subscription
`sub-enveda-data-dev-01`. The full 1.2 GB rerun tree (poses, ligand prep, per-batch logs)
lives at `guild/data/small-example-all-methods/`; only the summary tables above are
needed to reproduce the numbers.

Note the column-name drift: the Azure files are named `dockwizard_scores.txt` and
`drrp_scores.txt` and use the legacy `dockwizard_*` / `drrp_*` / `guild_*` prefixes for
what current guild calls `rp_*`. The script reads the raw `vina_score` column only, which
is unaffected, so the rename is cosmetic here — but do not read a `guild_*` percentile
column from those files expecting the current 0 = best orientation. See
`guild/support/results/README.md`.

## Outputs — `--out`

### `reproduce_response_analyses.py`

| file | backs |
| --- | --- |
| `R3-4_runtime.tsv` | R3-4 — 282.4 h over 538,931 pairs, 1.89 s/pair |
| `a2_aggregation_rules.tsv` | a2, R2-2 — AUC under eight aggregation rules plus each method standalone |
| `a3_normalisations.tsv` | a3 — pooled AUC for rank percentile, min-max, z-score, raw |
| `a3_mechanism.tsv` | a3 — per-target mean drift and its correlation with skewness |
| `a3_per_target_skew.tsv` | a3 — per-target skewness and normalised means, 47 targets |
| `a3_failure_tail_summary.tsv` | a3 — the poorly-ranked-binder characterisation |
| `a3_failure_tail_binders.tsv` | a3 — per-binder detail behind that summary |
| `reserve_size_matched_auc.tsv` | not quoted; see `../analysis_size_matched_control.md` |
| `r2_5_training_overlap.tsv` | R2-5 — within-target AUC per track, ordered by PDB release date |

### `build_supplementary_tables.py`

| file | backs |
| --- | --- |
| `Table_S1.tsv` | S1, quoted throughout R2-3/R2-4 — per-method screening AUC, EF and non-physical rate |
| `Table_S2.tsv` | S2, R2-4 — pose validity rate by method. Four of five methods: KarmaDock emits no complex PDB, so it has no PoseBusters rows at all (a coverage gap, not a pass) — see `merge_posebusters_flags.py`'s own note on this. |
| `Table_S3.tsv` | S3, R2-4 — pose validity, individual checks. Same four-of-five coverage as S2. |
| `Table_S4.tsv` | R2-4 — per-target AUC, pooled column is the one to read (5 binders per target is imprecise alone) |
| `Table_S5.tsv` | R3-4 — native-ligand redocking RMSD, illustrative (n = 1 per method per target). KarmaDock is null here too, for the same underlying reason as S2/S3 -- `native_ligand_rmsd.tsv` does carry a KarmaDock row per target, but its `rmsd` value itself is null throughout. |
| `Table_S6.tsv` | R3-4 — runtime by stage |
| `supplementary_tables.html` | all of the above, one page, for the supplementary PDF |

## Caveats that belong in any write-up

1. **The a3 binders and decoys come from different guild runs.** Binders from
   `knownbindersvinarun`, decoys from `vinarun`, matched on `protein_config_id`. Confirm
   the docking box and protein preparation were identical between those runs before
   treating the comparison as within-run. This affects every a3 number.
2. **z-score is not reproduced as inferior to rank percentile.** This analysis gives
   0.675 for z-score against 0.674 for rank percentile — statistically indistinguishable
   — while min-max is clearly worse at 0.577. The mechanism table explains why: both
   rank percentile and z-score fix the per-target location (SD of the per-target mean
   0.0003 and 0.0000 respectively) whereas min-max does not (0.179). The manuscript's
   Figure 3 reports rank percentile ahead of both, so the discrepancy must be reconciled
   against however that figure was constructed before the a3 reply is finalised. The
   corresponding sentence in the response draft is deliberately still marked red.
3. **a2 rests on 15 binders and 150 decoys**, so the confidence intervals overlap
   heavily. The ordering of the rules is informative; the individual values are not
   precise. Do not quote a difference between two rules as significant.
4. **Descriptors are parsed from SMILES without RDKit** — see `heavy_atoms`. Adequate for
   the size comparison, but recompute properly before quoting any descriptor externally.
5. **Non-physical scores are dropped**, not imputed: any Vina-family energy outside
   −20 to 0 kcal/mol is excluded (`PHYSICAL_VINA`). About 6% of the large pool is
   non-negative and 0.8% exceeds 1,000, with a maximum of 43,851,078.
6. **`r2_5_training_overlap` is deliberately within-target and raw-score; `Table_S1` is
   deliberately pooled and rank-percentile.** They can legitimately disagree (Boltz-2
   affinity: 0.970 pooled-raw here vs. 0.985 pooled-rank-percentile in S1; Vina: 0.736 vs.
   0.788) — that is not an error in either one, and is not fixed by switching one to the
   other's method.
7. **Two of `r2_5_training_overlap`'s five tracks needed a second look at their sign
   convention.** `boltz_affinity_score`, `diffdock_score` and `vina_score` verify exactly
   against an earlier hand-computed draft of this table. `boltz_score` (ipTM) and
   `karmadock_score` did not — that draft had them as the exact complement (1 − auc) of
   what `SCORES_DIRECTION_DICTIONARY`-correct scoring gives, cross-checked independently
   against `sklearn.roc_auc_score`. Treated as an error in that earlier draft, not in this
   analysis; corrected here. R2-5's actual argument rests only on `boltz_affinity_score`,
   which was never in question either way.

## Determinism

No random sampling anywhere in this script, so repeated runs are byte-identical. The one
place a seed matters is the decoy selection for the rerun and for the exhaustiveness
sweep, both of which sort by `ligand_id` before sampling with `random_state=42` and are
specified in their respective run prompts.
