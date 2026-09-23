================================
Adding a new prediction method
================================

Adding a new protein-ligand binding prediction (PLBP) method to Guild is a **six-step code
contract**, not a plugin system or a configuration file. This page documents that contract as
it exists today, using the GNINA integration as a worked example.

There is one genuinely modular part, and it is worth stating up front: once a method is
registered in the six dictionaries in ``guild/constants/bulk.py``, the rank-percentile
computation (``compute_rank_percentile_scores``) picks it up automatically and folds it into
``global_rp_score`` with no changes to that function itself. Everything upstream of
registration — running the tool, parsing its output — is bespoke, tool-specific code.

What counts as a "method"
==========================

A method is anything that produces one raw numeric score per protein-ligand combination, in a
consistent direction (either "lower is better" or "higher is better" for every row). Guild
ranks that raw score into a rank percentile, per protein, and optionally folds it into the
combined score. A method does not have to dock — Nesso predicts affinity from a sequence and a
SMILES alone, with no 3D pose — but it does have to produce that one column.

The six-step contract
======================

1. Constants module
--------------------

``guild/constants/<method>.py`` — the tool's own configuration: binary path, default
parameters, timeouts. See ``guild/constants/gnina.py`` (CLI defaults, the CNN-scoring mode,
the covalent-docking flags) or ``guild/constants/nesso.py`` (a much shorter one — batch size
and output field names, since Nesso needs no docking-search parameters at all).

2. Runner module
-----------------

``guild/docking/<method>.py`` — invokes the tool and turns its output into a score. Follow
``guild/docking/gnina.py``:

* A ``deploy_<method>`` function that builds an argv, runs it via ``subprocess.run``, and
  writes a subprocess transcript with :func:`guild.tools.subprocess_log.write_subprocess_log`
  — **on both the success and failure paths**, not just on failure. That transcript is the
  contract behind the per-combination log paths in the README's
  `troubleshooting table <../README.md#troubleshooting-a-failed-combination>`_; a runner that
  only logs failures breaks that contract even though nothing raises.
* A results parser (``parse_<method>_stdout`` or a file reader, depending on whether the tool
  prints to stdout or writes a results file — gnina does the former, Nesso the latter with its
  ``affinity.json``).
* A ``<method>_guild_scoring(batch_dictionary)`` function returning a DataFrame keyed by
  ``COMBINATION_ID``, with the raw score column plus ``PROTEIN_CONF_ID`` / ``LIGAND_ID``. This
  is what ``guild/bulk.py`` collects per batch and hands to
  ``compute_rank_percentile_scores``.

3. Registration in ``guild/constants/bulk.py``
-----------------------------------------------

Six registries, each keyed by the method's prefix string. ``guild/constants/bulk.py`` carries
unusually good inline comments explaining *why* each one exists — read them before copying the
table below blind.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Registry
     - What it decides
   * - ``SCORES_DIRECTION_DICTIONARY``
     - ``"minimum"`` or ``"maximum"``. The single most dangerous entry: getting it backwards
       doesn't crash anything, it silently inverts the ranking, producing a plausible-looking
       column whose AUC is the complement of the truth. Sanity-check it against a handful of
       known binders before trusting a new method's numbers — a real signal should put them
       toward the "good" end, not the "bad" one.
   * - ``RANKS_DICTIONARY``
     - The rank column name (``rank_<method>_score``).
   * - ``RP_SCORES_DICTIONARY``
     - The rank-percentile column name (``rp_<method>_score``) — this is the column
       ``compute_rank_percentile_scores`` actually writes and folds into ``global_rp_score``.
   * - ``SCORES_TO_USE_DICTIONARY``
     - The ``[raw, rank, rp]`` column triple carried into the results table.
   * - ``POSE_SOURCE_DICTIONARY``
     - Which engine's *pose* the track judges. A rescore belongs to the engine whose pose it
       scores, not to Vina or gnina — e.g. ``vina_rescore_boltz`` is grouped under
       ``BOLTZ_PREFIX``, not ``VINA_PREFIX``. This is what keeps DiffDock/Boltz from getting
       three votes apiece for one pose (see "How a method earns its vote" below). A method
       absent from this dictionary is treated as its own pose source, which is the sensible
       default for a genuinely new engine.
   * - ``CONFIDENCE_ONLY_METHODS``
     - Opts a method out of voting in ``global_rp_score`` while it still gets ranked and gets
       its own ``rp_`` column. For a pose-confidence output (DiffDock's diffusion confidence,
       Boltz's ipTM), not an affinity estimate.

4. Orchestration
-----------------

Registration alone does not make a method runnable — it has to be reachable from a real
invocation:

* ``guild/bulk.py`` (``BulkRun``): add the method to the ``method_runners`` dict in
  ``run_docking()`` with your own ``_run_<method>_for_batch``, and add a
  ``if <PREFIX> in self.methods_to_run: docked_methods.append(<method>_guild_scoring(...))``
  line in ``_process_batch_scoring``. If the method should auto-enable a rescore the way
  ``boltz``/``diffdock`` do (see ``BulkRun.__init__``), that's an explicit design decision, not
  a default.
* ``scripts/run_guild.py``: add the prefix to the ``--methods`` ``choices`` list, or the CLI
  rejects it before ``BulkRun`` ever sees it.
* ``Makefile``: a ``run-<method>`` target, mirroring ``run-gnina``, if the method should be
  runnable as its own shortcut rather than only via the generic ``run-guild METHODS=...``.

5. Complex PDB output — or explicitly none
--------------------------------------------

PoseBusters, PLIP and ProLIF all read complex PDBs from
``_COMPLEX_PDB_FOLDER_BY_METHOD`` in ``guild/bulk.py``. If your method writes one, add it
there and it is picked up by every one of those analyses. If it does not — like KarmaDock
(external docking script, no predicted-pose file this repo controls) or Nesso (no 3D output at
all) — **say so explicitly** rather than leaving the method out of the dictionary silently.
``_complex_pdb_coverage()`` exists for exactly this: it partitions the requested methods into
supported and excluded, and both ``run_pose_validity_analysis`` and
``run_interactions_analysis`` log the excluded set by name. A silent omission here reads
exactly like a clean bill of health in a results table that only lists methods as rows — see
the commit that made this explicit (``467e778``) for the failure mode this avoids.

6. A test
----------

``tests/`` is organised by area — ``bulk_run/`` for orchestration wiring, ``scores/`` for the
rank-percentile machinery, ``single_run/`` for the single-combination path. Put your test
where it fits; ``tests/bulk_run/test_gnina_orchestration.py`` is a reasonable template for a
new docking method (registration, standalone-vs-auto-enabled-rescore behaviour, scoring-column
shape).

The genuinely modular part, made checkable
============================================

Once a method clears steps 1-4 above, nothing in ``guild/tools/scores.py`` needs to change.
``compute_rank_percentile_scores`` iterates whatever is in its ``methods`` argument (which
comes from ``self.methods_to_run``) and reads its direction/rank/rp column names out of the
three dictionaries above — a new method's ``rp_<method>_score`` column appears the same way
every existing one does. You can check this yourself without touching the ranking code at
all: register a throwaway method (even with a stub runner that returns constant scores) in the
three scoring dictionaries, pass its prefix into ``methods=``, and confirm the new ``rp_``
column shows up and (if it isn't in ``CONFIDENCE_ONLY_METHODS``) that ``global_rp_score``
changes. This is exactly what
``tests/scores/test_scores.py::TestNewMethodRegistrationIsModular`` does.

How a method earns its vote
==============================

``global_rp_score`` does not average every voting track flat. It first averages the tracks
that share a pose source (so ``boltz``, ``vina_rescore_boltz``, ``gnina_rescore_boltz`` and
``boltz_affinity`` collapse into one Boltz-source estimate), then combines *across* sources.
The default, ``aggregation="pose_source_median"``, takes the **median** across sources rather
than the mean — on the three-target benchmark this scored 0.824 AUC against 0.781 for the
flat mean, because the flat mean let one weak track (DiffDock) drag the whole score down.
A newly-added engine therefore contributes **one vote**, regardless of how many rescore tracks
attach to it — a contributor who assumes a flat mean over every registered column will
mis-predict what their method actually does to ``global_rp_score``.

Direction is per-track, not per-tool
=======================================

``SCORES_DIRECTION_DICTIONARY`` is keyed by track, not by tool, and Boltz-2 is the clearest
illustration of why: it contributes two tracks with **opposite** directions.
``boltz_score`` is an ipTM structural-confidence value — ``"maximum"``. ``boltz_affinity_score``
is Boltz-2's own affinity head, a log10(IC50/µM) potency — ``"minimum"``, the same convention
as Nesso's ``nesso_score``. A new method that emits more than one column needs a direction
entry *per column*, not one assumption applied to the tool as a whole.

Worked example: GNINA
========================

GNINA was added in ``2502502`` (aggregation has since changed under it — ``20b3c50``,
``0e7c959``, ``0d36671`` — the description below is current behaviour, not that diff verbatim).

* Constants: ``guild/constants/gnina.py`` — binary path, CNN-scoring defaults, covalent-docking
  flags, subprocess timeout.
* Runner: ``guild/docking/gnina.py`` — ``deploy_gnina`` invokes the ``gnina`` CLI, parses
  either its pose table or its score-only key/value block depending on mode, and writes a
  subprocess transcript on every call. ``gnina_guild_scoring`` returns ``gnina_score`` (the
  Vina-style affinity that gets ranked) alongside ``gnina_cnn_score`` — a side-channel
  pose-confidence value that rides along in the results table but is **not** registered in any
  of the six dictionaries, so it is never ranked or voted.
* Registration: ``GNINA_PREFIX`` appears in all six dictionaries in
  ``guild/constants/bulk.py`` — ``"minimum"`` direction (kcal/mol, lower is better), its own
  pose source (gnina judges its own poses, not Vina's).
* Orchestration: ``method_runners[GNINA_PREFIX] = self._run_gnina_for_batch`` in
  ``guild/bulk.py``, a scoring-dispatch line in ``_process_batch_scoring``, ``"gnina"`` in
  ``scripts/run_guild.py``'s ``--methods`` choices, and the ``run-gnina`` Makefile target.
* Complex PDB: gnina shares Vina's PDBQT-based complex generation, so it has an entry in
  ``_COMPLEX_PDB_FOLDER_BY_METHOD`` and is covered by PoseBusters/PLIP like Vina is.
* Test: ``tests/bulk_run/test_gnina_orchestration.py``.

A second reference: Nesso
============================

``guild/docking/nesso.py`` / ``guild/constants/nesso.py`` are a second, more recent worked
example, useful for a method that skips docking entirely — no receptor structure, no pose, a
single ``nesso predict <directory>`` call scoring every complex in a batch at once (the
directory-batched call is not an optimisation; it is where Nesso's speed advantage over Boltz-2
actually comes from, so a per-ligand runner would defeat the point of adding it).

Nesso is a deliberately GitHub-only addition, not one of the five methods described in the
associated publication — a useful second reference for this contract, not evidence that the
published method count has grown.

One gap worth flagging rather than fixing here: Nesso is registered in
``SCORES_DIRECTION_DICTIONARY``, ``RANKS_DICTIONARY``, ``RP_SCORES_DICTIONARY`` and
``POSE_SOURCE_DICTIONARY``, and appears in ``ALL_AVAILABLE_METHODS``, but its prefix is
missing from ``scripts/run_guild.py``'s ``--methods`` ``choices`` list — so it cannot be
selected from the CLI or a ``make run-guild METHODS=...`` invocation today, only via the
Python API (``BulkRun(methods_to_run=["nesso"])``). That may be intentional or an oversight;
it is not addressed by this page.

What this does not give you
==============================

This is a code contract, not a plugin system. There is no configuration file that adds a
method, no dynamic discovery, no way to register a new prediction source without writing
Python. Adding a method means writing a constants module, a runner module, six registry
entries, orchestration wiring in three files, and a test — real engineering effort, the same
amount whether you are one of the authors or an external contributor. If your use case needs
something more turnkey than that, this repository does not currently offer it.
