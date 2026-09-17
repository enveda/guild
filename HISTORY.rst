=======
History
=======

Unreleased
----------
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
