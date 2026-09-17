=======
History
=======

Unreleased
----------
* Added ``notebooks/analysis/reviewer_response/build_supplementary_tables.py``, ported
  from a session scratchpad so Supplementary Tables S1-S6 (every per-method AUC quoted in
  R2-3/R2-4, the PoseBusters pose-validity rates in R2-4) have a committed, regenerable
  path instead of existing nowhere but a prior session. Follows its committed sibling
  ``reproduce_response_analyses.py``'s conventions (``--data``/``--out``, module
  docstring, ``write()`` helper). Ported changes only, output unchanged: hardcoded
  ``C:/Users/...`` paths replaced with ``--data``/``--out``; the ``sys.path`` insert
  dropped (unnecessary once the file lives in the repo); the pandas-3 ``_pcid_keep``
  save/restore workaround deleted as dead code now that ``e9a6fe7`` fixed
  ``compute_rank_percentile_scores`` to keep its grouping column at source (the
  recompute itself is kept, and why: the file on disk predates both the median
  aggregation, ``0e7c959``, and ``boltz_affinity_score``'s ranking, ``0d36671``); and a
  stale docstring line rewritten (Boltz-2's affinity head has had its own
  ``rp_boltz_affinity_score`` column, ranked like every other track, since ``0d36671`` --
  the code already read it correctly, only the comment was wrong). Verified against the
  real rerun data that the four numbers the response letter quotes verbatim from Table S1
  are unchanged: Guild combined 0.838, AutoDock Vina 0.788, Boltz-2 affinity 0.985, GNINA
  0.739.
* Added ``r2_5_training_overlap`` to ``reproduce_response_analyses.py``: the within-target,
  per-track, raw-score AUC ordered by each benchmark target's PDB release date that backs
  R2-5's answer on training-set leakage (any cutoff admitting the most recently released
  structure, 8GDC, necessarily admits the other two, so their ordering is informative
  without knowing any method's actual cutoff). Boltz-2's affinity head verifies exactly:
  0.956 (6OT0, 2019), 1.000 (7V3Z, 2021), 1.000 (8GDC, 2024) -- the newest structure ranked
  best, the oldest worst, as the letter claims. Release dates are a hardcoded, cited
  constant (RCSB entry endpoint, retrieved 2026-09-17), not a live call; an opt-in
  ``--verify-dates`` re-fetches and asserts them instead. Deliberately does not attempt a
  ligand-level training-set audit (no PDBBind overlap, no ECFP4 similarity to a training
  set) -- the argument rests on the release-date contrast and the scoping logic, not on
  enumerating what Boltz-2 was trained on, and building one would answer a question this
  reply does not ask. Direction is read from ``guild.constants.bulk.SCORES_DIRECTION_DICTIONARY``
  rather than hardcoded -- the only guild import anywhere in this otherwise guild-free
  script, and still no heavy dependency, since that module is pure constants. The
  direction-aware AUC itself is factored into a small, tested helper,
  ``direction_aware_auc`` (``tests/reviewer_response/``, including a direction-flip case).
  Verifying this against real data also surfaced that an earlier hand-computed draft of
  this table had two of its five rows (``boltz_score``, ``karmadock_score``) backwards --
  their values were the exact complement (1 - auc) of what direction-correct scoring
  gives, independently cross-checked against ``sklearn.roc_auc_score``. Corrected here;
  flagged in the module docstring and the response-analyses README. Does not affect
  ``boltz_affinity_score``, the only track R2-5's argument depends on.
* Relabelled two stale comments in ``reproduce_response_analyses.py``'s ``a2`` analysis
  that went stale when the median default (``0e7c959``) landed after it was written: the
  rule labelled ``"unweighted mean (current)"`` is now just ``"unweighted mean"``, and
  ``"median"`` is now ``"median (current default, 0e7c959)"``; a comment above
  ``POSE_SOURCE_VOTES`` no longer claims the cross-engine combination is a mean. Labels
  only -- ``a2`` computes every aggregation rule itself rather than calling
  ``compute_rank_percentile_scores``, so no number moves.
* Added ``notebooks/analysis/build_rank_percentile_schematic.py``, which generates the
  Supplementary Text 3 rank-percentile schematic from the real 3-target rerun instead of
  the hand-built, uncaptioned original (``guild_rank_percentile_figure.png`` /
  ``Revision 1.docx``'s ``image4.png``) -- nothing in the repository produced that image,
  so it could drift from the scoring code silently, and it already had. Two concrete
  errors are fixed by construction: it showed four pose sources (Vina, DiffDock,
  KarmaDock, Boltz-2), GNINA was missing; and its DiffDock panel was labelled "confidence
  (higher = better)", exactly the input R2-3 objects to and what the rescore tracks and
  ``20b3c50`` stopped letting vote. The regenerated schematic shows all five pose sources,
  labels DiffDock's and Boltz-2's panels with the Vina-rescore ΔG that actually enters the
  score (native confidences are still noted as reported, but not ranked or voted, mirroring
  ``gnina_cnn_score``), shows Boltz-2 as one Step-4 vote (the mean of its two shown tracks,
  including ``boltz_affinity_score`` per ``0d36671``) rather than one panel per track, and
  states the cross-source combination is a median (``0e7c959``), not a mean.
* R2-7: added ``guild.tools.decoy_matching``, the property-matched decoy panel and
  descriptor-only-baseline control the reply to R2-7 promises for Figure 3, as ordinary
  tested library code rather than a final render. ``match_decoys_to_binders`` keeps a
  decoy only if it is within tolerance of at least one of its target's known binders on
  molecular weight, cLogP, HBA, HBD, rotatable bonds and net charge *and* its ECFP4
  Tanimoto similarity to every known binder for that target is below 0.35, so a "matched"
  decoy is not simply a close structural analogue; MW/logP/HBA/HBD are reused from
  ``guild.tools.ligand_properties.assign_properties`` rather than reimplemented, only
  rotatable-bond count and net formal charge are new. ``descriptor_only_auc`` reports
  binder-vs-decoy AUC per descriptor with no docking score at all, and
  ``combined_descriptor_auc`` a cross-validated logistic-regression AUC across all six --
  the control that separates binding signal from property bias (on the 3-target rerun,
  an unmatched heavy-atom-count baseline alone reaches AUC 0.865, higher than any docking
  method; size-matching collapses it to 0.645). ``score_comparison.ipynb``'s decoy-loading
  cell now takes a ``DECOY_SUBSET`` (an explicit ligand_id list, or a predicate),
  defaulting to ``None`` (every decoy, current behaviour); verified end-to-end that the
  default reproduces Figure 3's three AUCs exactly (0.684 / 0.547 / 0.597, unchanged) and
  added a demonstration cell that runs the real matching + baseline against a 5-target
  slice of the actual Figure 3 inputs (460/5,000 decoys kept; combined AUC 0.778). Full
  scale (133,000 decoys x 655 binders) is not run in the notebook -- descriptor computation
  is roughly linear in decoy count and the check above took several seconds per 5,000, so
  rendering the matched panel itself is left as follow-up work. 18 new tests in
  ``tests/ligand_properties/test_decoy_matching.py``.
* Figure 3 (``score_comparison.ipynb``): the grey/orange rug ticks (R4 b3), the "Density"
  y-axis (R4 b4), and the score-orientation inversion are now all explicit instead of
  implicit. The rug ticks are per-molecule values -- the decoy rug (grey) is a random
  subsample (``RUG_SUBSAMPLE_N = 400``, ``RUG_SEED = 42``, both now named and printed),
  the known-binder rug (orange) is the complete set -- not "per-target values underlying
  the pooled distributions" as the current draft reply to b3 says; that sentence needs
  correcting in the letter. The y-axis is relabelled "Probability density" and now uses
  the same ``kde_curve_bounded`` as Figure 2's b2 fix, so each panel's curve genuinely
  integrates to 1 on its own bounded support; the shared y-scale is kept (comparability
  across the three panels is the point of the figure) and is now stated in the printed
  output rather than left to be inferred from the hidden y-ticks. The x-axis label is now
  "Normalized score (higher = better)": panel A plots ``1 - rp_vina_score`` even though
  the stored column is 0 = best, which is the actual source of the Supp. Text 6 vs. code
  discrepancy ("1 indicates the top-ranked molecule" is about the plotted axis, not the
  stored value) -- both were right about different things, and nobody had flagged that
  the notebook flips it. Neither the stored value nor the inversion itself is changed.
  Also now prints the three panel AUCs, which were computed for the in-panel annotation
  but never emitted: on this dataset, rank percentile (0.684) beats z-score (0.597) by a
  real margin, unlike the reviewer-response reproduction over 47 targets (0.675 vs.
  0.674, effectively tied) -- these are different analyses (this notebook's own
  binder/decoy join vs. that script's stricter "both classes present per target" filter),
  so the discrepancy is reported rather than resolved here; whoever finalises the a3 reply
  needs to reconcile it. ``kde_curve``/``kde_curve_bounded`` now live in a shared
  ``kde_helpers.py``, imported by both figure notebooks, so the b2 fix is one function,
  not two copies that could drift.
* ``score_comparison.ipynb``'s ``_decoy_pct``, ``_zscore`` and ``_decoy_fit_unbounded``
  no longer use ``groupby(protein_col, group_keys=False).apply(...)`` -- the same pattern
  ``e9a6fe7`` removed from ``guild.tools.scores.compute_rank_percentile_scores``, and
  confirmed here to emit the same ``FutureWarning`` on pandas 2.2+ that fix described.
  ``_zscore`` is now a straightforward ``groupby(...).transform("mean"/"std")``; the other
  two reference a different set (decoys only) than the group being scored, so they cannot
  be a single `.transform()` call, but no longer touch ``DataFrameGroupBy.apply()``
  either -- rewritten as a per-protein decoy-array lookup (built by iterating the groupby
  object, not ``.apply()``) plus the same per-value arithmetic as before. Verified
  numerically identical to the old implementation on the real Figure 3 input data (max
  abs diff 8.8e-16, floating-point noise only) and that row count and row order are
  unchanged (asserted in the notebook itself, not just checked once here).
* Figure 2 (``score_distribution.ipynb``) is rebuilt on the 3-target rerun and no longer
  plots DiffDock's or Boltz-2's *pose confidence* as the ranked quantity -- it now shows
  the Vina-rescore ΔG that actually enters the score (``vina_rescore_diffdock_score``,
  ``vina_rescore_boltz_score``), matching what ``20b3c50`` stopped letting the raw
  confidences do, and adds GNINA and KarmaDock so the figure covers all five pose sources
  its caption already claimed. The native confidences (``diffdock_score``, ``boltz_score``)
  are still reported per panel but are not ranked or plotted as the voted quantity, the
  same treatment ``gnina_cnn_score`` already gets; the panel notes that
  ``boltz_affinity_score`` also joins Boltz's vote (``0d36671``). ``vina_rescore_boltz_score``
  is populated for 134/165 rows (receptor-preparation failures) and
  ``vina_rescore_diffdock_score`` for only 34/165 once non-negative values -- a
  rescoring-failure sentinel, 79.4% of the column -- are nulled alongside the strictly
  non-physical ones; both counts are printed rather than left implicit, and the figure
  states on its own face that three targets replace what the published version covered
  many more of.
* Figure 2's right column (rank percentile) no longer plots a density that overshoots
  [0, 1] (R4 b2) -- the KDE grid used to run from -0.06 to 1.06 with the axis merely
  clipped to (-0.04, 1.04), which still let the drawn curve cross the valid boundary.
  Replaced with ``kde_curve_bounded``, a reflected boundary-corrected KDE evaluated only
  on ``[0, 1]``, so the curve cannot leave the valid range and still integrates to 1
  (checked with ``np.trapezoid`` against every panel with enough points to make that
  check meaningful; verified to ~1e-3 in every one). The left column (raw docking score)
  is genuinely unbounded and keeps the original, unreflected KDE.
* ``score_distribution.ipynb`` (Figure 2) and ``score_comparison.ipynb`` (Figure 3) now
  read their input tables from a ``--data``-style directory
  (``GUILD_FIGURES_DATA_DIR`` env var, default ``data``) instead of a hardcoded relative
  path, mirroring ``reviewer_response/reproduce_response_analyses.py``. Neither
  notebook's input was reachable before this: both paths are covered by the blanket
  ``*.txt`` rule in ``.gitignore`` and were never committed, so no figure could be
  regenerated from a clean checkout. Figure 2 now points at the 3-target rerun
  (replacing a many-target case-study file that no longer exists anywhere reachable,
  including the Zenodo deposit -- see Task 1 below); Figure 3's two inputs are unchanged
  and load today. Added ``notebooks/analysis/README.md`` documenting, per figure, which
  file is needed, which columns it must carry, and its provenance, plus a
  ``legacy_columns.LEGACY_RENAME`` module shared by both notebooks (previously
  duplicated in ``score_comparison.ipynb`` and absent from ``score_distribution.ipynb``
  entirely) that now also covers ``global_dockwizard_score``, the one column-naming
  generation it was missing. Orientation was checked directly against all three
  generations and is unaffected -- 0 = best throughout; nothing here changes any stored
  score.
* Fixed the ``test`` CI job, which failed to even collect: the ``test``
  Docker stage copied in ``guild/`` and ``tests/`` but never ``scripts/``,
  and ``.dockerignore`` allowlisted only ``scripts/apply_karmadock_patches.py``,
  so ``tests/scripts/test_run_guild.py``'s ``import run_guild`` had nothing to
  import inside the container even though it works locally. Added
  ``scripts/run_guild.py`` to both. Pre-existing since that test file was
  added; unrelated to the other changes in this branch.
* ``compute_rank_percentile_scores`` no longer drops ``protein_config_id``
  (or scrambles row order against any other column) on pandas 3 — it grouped
  with ``groupby(protein_col, group_keys=False).apply(...)``, relying on the
  grouping column being passed through to the callable and back out, which
  stopped being the default on pandas 2.2+ and is gone on 3.x. Not reachable
  today (``pyproject.toml`` pins ``pandas<3``), but every downstream
  consumer of ``guild_scores.txt`` groups by that column, so it was a
  landmine for whenever that pin lifts. Rewritten as groupby transforms over
  the whole frame instead of a per-group ``.apply()``, which keeps every
  column and every row's original position by construction and needs no
  ``.reset_index()``. Numbers are unchanged — verified against the existing
  orientation and denominator tests, and manually against pandas 3.0.5.
* ``boltz_affinity_score`` (Boltz-2's own affinity head, log10(IC50/uM), read
  from the same output tree ``boltz_guild_scoring`` already parses) is now
  ranked and votes in ``global_rp_score``, via the new
  ``BOLTZ_AFFINITY_PREFIX`` prefix. Previously deliberately excluded as a
  side channel for validating Nesso-1 against; that rationale doesn't
  survive the criterion adopted for the pose-confidence exclusion above —
  it's a genuine affinity estimate, not a confidence, so it qualifies on the
  same grounds ``vina_rescore_boltz``/``gnina_rescore_boltz`` do. It joins
  Boltz's existing pose-source group as a third estimate
  (``POSE_SOURCE_DICTIONARY[BOLTZ_AFFINITY_PREFIX] == BOLTZ_PREFIX``) rather
  than voting independently, so requesting ``boltz`` still contributes one
  pose-source vote, not two. Whether this is a fair test is unresolved —
  Boltz-2's affinity head is trained on binding-affinity data and the
  benchmark's 15 known binders are ChEMBL compounds at pChEMBL 9.15–10.7, so
  some of its benchmark strength may be training-set recall; that overlap
  audit is still outstanding, and no performance claim is made here on the
  strength of the benchmark number.
* ``global_rp_score`` now combines pose sources with their **median**, not
  their mean (``compute_rank_percentile_scores(..., aggregation=
  "pose_source_median")``, the new default). Measured on the three-target
  benchmark (165 pairs, 15 known binders, 150 decoys): the unweighted mean
  scored 0.781 AUC, below AutoDock Vina alone (0.790), because one track
  (DiffDock, 0.281 standalone) dragged it down with no defence against a
  single aberrant vote; the median scored 0.824 with all five methods still
  included. Chosen over performance-weighting (0.851) or dropping DiffDock
  (0.852) because it needs no fitted parameter and makes no engine-specific
  judgement. This is a direction, not a significant result — the confidence
  intervals overlap heavily at 15 binders. ``aggregation="pose_source"``
  (the previous default, a mean) and ``aggregation="flat"`` are both still
  available for reproducing older scores.
* Added ``notebooks/analysis/reviewer_response/`` — the scripts and
  provenance notes behind every number quoted in the reviewer response that
  isn't read directly off a guild output table (runtime, aggregation-rule
  comparison, normalisation comparison, decoy size-matched control), plus
  the recovery script for a merge that didn't run (see the PoseBusters merge
  fix below). ``pandas``/``numpy`` only, no guild import needed.
* Fixed a stale docstring in ``score_distribution.ipynb`` — the markdown
  header cited ``figure_2_dockwizard_scores.txt`` while the code cell
  correctly loads ``figure_2_guild_scores.txt``.
* ``guild.tools.scores.is_physical_score(value, method)`` flags Vina-family raw
  scores (``vina_score``, ``gnina_score``, and the four rescore tracks)
  outside a plausible −20 to 0 kcal/mol range — measured on the large Vina
  case study, 6.06% of scored rows were non-negative and 0.79% exceeded
  1,000,000 in magnitude (observed maximum 43,851,078). ``run_guild_scoring``
  now logs a per-method count of these as a warning; stored values are
  unchanged by default. ``run_guild_scoring(exclude_non_physical=True)``
  (``--exclude-non-physical`` / ``EXCLUDE_NON_PHYSICAL=1``) nulls them for a
  new run instead, opt-in only, since the published case-study numbers were
  generated with these values left in the table.
* PoseBusters and PLIP/ProLIF now log which requested methods produce no
  complex PDB by design (karmadock, nesso) instead of letting them simply
  not appear in ``posebusters_validity.tsv`` / ``plip_interactions.tsv`` —
  a rows-only table reads that silence as "checked, nothing to report"
  rather than "structurally not applicable". KarmaDock's own docking script
  writes a scores CSV only (see ``karmadock_guild_scoring``); nothing in
  this repo defines a predicted-pose file for it, so generating a KarmaDock
  complex PDB would mean guessing that external tool's output convention
  rather than reading it off a verified path — left for a follow-up once
  that's confirmed.
* ``_merge_posebusters_into_scores`` now raises instead of only logging a
  warning when ``guild_scores.txt`` is missing at merge time, unless the
  caller explicitly expected that (``run_pose_validity_analysis(...,
  expect_existing_scores=False)``, which ``--posebusters-only`` now passes).
  It also raises if PoseBusters validated poses but the merge added zero
  columns — previously possible, silently, whenever a combination_id or
  docking_method value came through null. Both close the same gap: a real
  run once produced a correct, non-empty ``posebusters_validity.tsv`` while
  ``guild_scores.txt`` quietly kept zero ``pb_`` columns, with nothing louder
  than a warning to notice by.
* PoseBusters pose-validity analysis is now wired into ``scripts/run_guild.py``
  and the Makefile (``--posebusters`` / ``--no-posebusters`` /
  ``--posebusters-only`` / ``--posebusters-config``, ``NO_POSEBUSTERS`` /
  ``POSEBUSTERS_CONFIG`` / ``make run-posebusters``). It runs by default,
  mirroring PLIP; previously ``run_pose_validity_analysis`` was reachable only
  from the test suite, and every real invocation went through a hand-rolled
  script that re-instantiated ``BulkRun`` and risked silently validating the
  wrong batch layout.
* ``global_rp_score`` no longer averages ``diffdock_score`` and ``boltz_score``
  in with the affinity tracks — they're pose confidences, not affinity
  estimates, so (like ``gnina_cnn_score``) they keep their own ``rp_*`` column
  but stop voting. The remaining tracks are grouped by which engine generated
  the pose before averaging, so DiffDock's and Boltz's two auto-added rescores
  count as one vote for that pose source instead of three. The previous flat
  mean is still available via ``compute_rank_percentile_scores(...,
  aggregation="flat")`` for reproducing older scores.

1.4.0 (2026-09-15)
------------------
* ``compute_rank_percentile_scores`` gained a ``denominator`` option
  (``"valid"``, the default and previous behaviour, or ``"attempted"``). The
  published case-study results were generated with ``"attempted"``.
* The rank-percentile orientation (0 = best) is now documented in the README
  and pinned by a regression test. Behaviour is unchanged.

1.3.0 (2026-09-14)
------------------
* PoseBusters pose validity as a post-analysis step: ``pb_valid``,
  ``posebusters_status`` and a full per-check report.
* Nesso-1 affinity prediction — sequence plus SMILES, no pose.
* gnina rescoring of DiffDock and Boltz poses, mirroring the Vina rescore
  tracks.
* Per-pose score tables (``vina_scores.txt``, ``gnina_scores.txt``).

1.2.0 (2026-07-28)
------------------
* Flexible, covalent and pose-guided docking modes.
* ProLIF interaction fingerprints alongside PLIP.
* Covalent atom lookups scoped to the intended records.

1.1.5 (2026-06-12)
------------------
* GNINA docking support.
* Multi-chain protein handling.
* KarmaDock edge-feature dimension fix.

1.0.0 (2026-04-29)
------------------
* First public release.
