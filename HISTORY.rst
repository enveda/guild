=======
History
=======

Unreleased
----------
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
