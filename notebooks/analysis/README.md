# Manuscript figures — inputs and regeneration

`score_distribution.ipynb` (Figure 2) and `score_comparison.ipynb` (Figure 3) each read a
results table from a `--data`-style directory rather than a path baked into the notebook.
Set it with the `GUILD_FIGURES_DATA_DIR` environment variable (default: `data`, relative to
the notebook's own directory), mirroring `--data` in
`reviewer_response/reproduce_response_analyses.py`.

**None of these inputs are committed to the repo.** They are large results tables (up to
56 MB), already covered by the blanket `*.txt` rule in `.gitignore`, and belong on the
Zenodo deposit instead. `revised_metrics.ipynb` (Figure 4) is unaffected by any of this —
its own inputs are unchanged and it is out of scope for this note.

## Figure 2 — `score_distribution.ipynb`

| | |
| --- | --- |
| File | `$GUILD_FIGURES_DATA_DIR/guild_scores.txt` |
| Columns needed | `protein_config_id`, `ligand_category`, and the raw `*_score` columns for `vina`, `gnina`, `karmadock`, `diffdock`, `boltz`, `boltz_affinity`, `vina_rescore_diffdock`, `vina_rescore_boltz` |
| Provenance | The 3-target rerun (`7v3z`, `6ot0`, `8gdc`; 15 strong binders, 150 decoys, 3 natives) — all five methods plus both rescore tracks |

This **replaces** the original many-target `figure_2_guild_scores.txt`, which no longer
exists anywhere reachable (not in the repo, not in this laptop's files, and not on the
Zenodo deposit — see below). The rerun is a deliberate replacement, not a stand-in: the
published figure plotted DiffDock's and Boltz-2's *pose confidences*, which commit
`20b3c50` stopped letting vote, so rebuilding on the Vina-rescore tracks
(`vina_rescore_diffdock_score`, `vina_rescore_boltz_score`) that actually enter the score is
the intended change. The notebook recomputes every rank-percentile column itself via
`guild.tools.scores.compute_rank_percentile_scores` rather than trusting any rank-percentile
column already in the file, since the rerun predates the `boltz_affinity_score` ranking
(`0d36671`) and the median aggregation default (`0e7c959`).

This is a real reduction in scope — three targets where the published figure showed many —
and the figure's caption says so explicitly rather than leaving it implicit.

## Figure 3 — `score_comparison.ipynb`

| | |
| --- | --- |
| Decoys | `$GUILD_FIGURES_DATA_DIR/vinarun_scores.txt`, filtered to `ligand_category == "decoy"` |
| Known binders | `$GUILD_FIGURES_DATA_DIR/knownbinders_scores.txt` |
| Columns needed | `protein_config_id`, `vina_score`, `ligand_category` (decoys file only, for the filter) |
| Provenance | Large Vina case study — 133 targets, decoy/NP/synthetic pool, plus a matching known-binders-only run; staged for the Zenodo upload at `review/analysis/data/` |

Both inputs exist today, just under older column-naming generations (see below);
`legacy_columns.LEGACY_RENAME` maps them onto the current `rp_*` / `global_rp_score` names.
The rank-percentile columns themselves are not actually load-bearing here:
`compute_rank_percentile_scores` is re-run on the raw `vina_score` column, so the rename
mainly avoids stale column names lingering in `unified`.

## Supplementary Text 3 schematic — `build_rank_percentile_schematic.py`

| | |
| --- | --- |
| File | same rerun as Figure 2: `.../review/rerun/guild_scores.txt` |
| Provenance | 3-target rerun, all five methods |

Generates the previously hand-built, uncaptioned schematic diagram from real data instead
of invented distributions, so it can no longer drift out of step with the scoring code.

## The Zenodo deposit does not cover any of this

The deposit (DOI `10.5281/zenodo.20024339`, `guild_data.zip`, 4.2 GB, 4.7 M entries)
archives exactly two run trees:

```
guild_data/vina_runs/{nps,synthetics}/dockwizard_scores.txt
guild_data/diffdock_runs/diffdock-shard####of0100/guild_scores.txt
```

Sampling 0.42% of its zip central directory in 120 evenly spread windows found zero
`boltz`, `karmadock`, `gnina`, `knownbinder` or `decoy` entries. So the deposit cannot
reproduce the *published* Figure 2 or Figure 3, and (per
`review/zenodo_correction/README.md`) its DiffDock scores don't even reproduce Figure 4's
DiffDock curve — a superseded run. The staged upload under `review/analysis/data/`
(`vinarun_scores.txt`, `knownbinders_scores.txt`) is what closes the gap for Figure 3; the
rerun closes it for Figure 2 and the schematic. Do not substitute the deposit's raw shards
for either.

## Column-naming generations

The results files behind these figures span three naming generations for the same
quantities:

| Generation | Vina raw | Vina rank percentile | Combined |
| --- | --- | --- | --- |
| published | `autodock_vina_score` | `guild_autodock_vina_score` | `global_guild_score` |
| case study | `vina_score` | `dockwizard_vina_score` | `global_dockwizard_score` |
| current | `vina_score` | `rp_vina_score` | `global_rp_score` |

`legacy_columns.py` holds one `LEGACY_RENAME` dict covering all of the above (plus the
`guild_diffdock_score`/`guild_boltz_score`/`guild_karmadock_score` analogues from earlier
drafts of Figure 2, and the earlier `drrp_*` names), imported by both notebooks rather than
duplicated. `rename(columns=...)` only touches columns that are present, so entries for a
generation a given file doesn't use are harmless no-ops.

**Orientation is unaffected by any of this.** The stored rank percentile is 0 = best in
every one of these generations (checked directly against `knownbinders_scores.txt`:
raw Vina −12.465 → `guild_vina_score` 0.2; raw +24.668 → 1.0). Regenerating against current
code does not mirror any figure; do not "fix" an orientation here. `score_comparison.ipynb`
does plot `1 - rp_vina_score` in its Figure 3 panel — that is a deliberate, commented
inversion for the plotted axis only, not a fix to the stored data.
