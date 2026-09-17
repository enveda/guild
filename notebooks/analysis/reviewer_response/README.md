# Reviewer-response analyses — reproduction

Every number quoted in the Guild reviewer response that is not read directly off a
`guild` output table is produced by `reproduce_response_analyses.py`. One command
regenerates all of them.

```shell
python reproduce_response_analyses.py --data ./data --out ./output
python reproduce_response_analyses.py --data ./data --only a2 a3   # subset
```

Requires only `pandas` and `numpy`. It does not import `guild`, so it runs outside the
Docker image; rank percentiles are recomputed locally and `test_matches_guild` asserts
the reimplementation reproduces guild's own `rp_vina_score` exactly (it does, to
< 1e-9). Nothing is downloaded and nothing outside `--out` is written.

## Where this code belongs

Lives here, under `notebooks/analysis/reviewer_response/`, alongside the existing figure
notebooks but in its own subdirectory since these are response-specific scripts rather than
notebooks. `--data` and `--out` are not committed — provide them locally (see below).

## Inputs — `--data`

| file | provenance | needs to go to Zenodo |
| --- | --- | --- |
| `guild_scores.txt` | `small-example-all-methods` rerun, 168 combinations, 3 targets | **yes** |
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

Tables S1–S6 in `../supplementary/` are generated separately and read the rerun outputs
directly rather than recomputing anything.

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

## Determinism

No random sampling anywhere in this script, so repeated runs are byte-identical. The one
place a seed matters is the decoy selection for the rerun and for the exhaustiveness
sweep, both of which sort by `ligand_id` before sampling with `random_state=42` and are
specified in their respective run prompts.
