=======
History
=======

1.3.1 (2026-09-15)
------------------
**Breaking:** rank-percentile scores now run 1 = best (previously 0 = best).
``rp_*_score`` and ``global_rp_score`` produced by earlier versions are inverted
relative to this release. Raw ``*_score`` columns are unchanged, and the
``rank_*`` columns still hold rank 1 = best binder.

The orientation is now pinned by a regression test so it cannot silently flip
again.

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
