=======
History
=======

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
