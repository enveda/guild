<p align="center">
  <img src="files/logo.svg" alt="Guild Logo" width="400">
</p>

# Guild

> **Version 1.1.4** — Python ≥3.10, <3.11

Guild is an open-source Protein-Ligand Binding Tools orchestrator that covers the end-to-end pipeline while leveraging multiple docking methods in each step.

## Table of Contents

* [Docker (recommended)](#docker)
* [Running locally](#how-to-run)
* [Installations](#installations)
  * [Consuming PLIP interactions output](#consuming-plip-interactions-output)
  * [Troubleshooting a failed combination](#troubleshooting-a-failed-combination)
* [Usage](#usage)
  * [Single run](#single-run)
  * [BulkRun](#bulkrun)
* [Methods](#methods)
  * [Docking](#docking)
  * [Vina rescore (automatic with DiffDock and Boltz)](#vina-rescore-automatic-with-diffdock-and-boltz)
  * [Custom binding pocket](#custom-binding-pocket)
  * [Multi-chain binding pocket](#multi-chain-binding-pocket)
  * [Starting from a user-supplied pose](#starting-from-a-user-supplied-pose)
  * [Flexible receptor docking](#flexible-receptor-docking)
  * [Covalent docking (gnina)](#covalent-docking-gnina)
  * [Post-analysis](#post-analysis)

## Docker

The recommended way to run Guild is via Docker, which bundles all dependencies (Vina, OpenBabel, LocalColabFold, KarmaDock, DiffDock, Boltz).

### Build the image

```shell
make docker-local
```

### Run docking

All run targets accept the same set of parameters, passed as Make variables.
The local repository is volume-mounted into the container at `/workspace`, so
changes to `guild/` are reflected immediately without rebuilding.

```shell
# All three methods, first 100 rows, batch size 2, with known binders, clean start
make run-guild \
  COMBINATIONS=/workspace/notebooks/data_prep/full_combinations_table.csv \
  METHODS="vina boltz diffdock" \
  HEAD=100 \
  BATCH_SIZE=2 \
  KNOWN_BINDERS=1 \
  CLEAN=1

# Boltz only (GPU)
make run-boltz \
  COMBINATIONS=/workspace/path/to/combos.csv \
  PROJECT=myproject

# Vina only (CPU, no GPU required)
make run-vina \
  COMBINATIONS=/workspace/path/to/combos.csv \
  PROJECT=myproject \
  BATCH_SIZE=5
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `COMBINATIONS` | *(required)* | Path to the protein–ligand pairs CSV/TSV (use `/workspace/…` paths) |
| `PROJECT` | `imagerun` | Output folder name under `data/` (no underscores allowed) |
| `METHODS` | `boltz` | Space-separated list: `boltz`, `vina`, `karmadock`, `diffdock`, `gnina` |
| `BATCH_SIZE` | `2` | Number of combinations per batch |
| `HEAD` | `0` | Take only the first N rows from the combinations table (0 = all) |
| `DECOYS` | *(script default)* | Path to the decoys file; omit to use built-in default (`chembl_36_decoys_2.tsv`) |
| `NO_DECOYS` | *(empty)* | Set to `1` to skip decoy expansion entirely (useful for single-protein runs where you only want to score the supplied ligands) |
| `CLEAN` | *(empty)* | Set to `1` to delete the project output folder before running |
| `KNOWN_BINDERS` | *(empty)* | Set to `1` to enable known-binders expansion |
| `N_WORKERS` | `1` | Vina parallel-worker processes. Vina internally also threads — values >1 may oversubscribe on high-core hosts but are typically fine. |
| `BOX` | *(empty)* | Global fallback Vina box file (`center_{x,y,z}` + `size_{x,y,z}`). Used for combinations whose CSV `box_location` cell is empty; per-row values always take precedence. See [Custom binding pocket](#custom-binding-pocket). |
| `USE_GPU` | `1` | Set empty (`USE_GPU=`) to drop `--gpus all` from `docker run` and forward `--no-gpu` to the python script. Use on no-GPU hosts. gnina falls back to CPU; vina and diffdock are unaffected. **Do not combine with `METHODS=boltz`** — Boltz is genuinely GPU-bound. |
| `GNINA_INPUT_MODE` | *(empty)* | Set to `sdf` to skip OpenBabel PDBQT prep entirely when gnina is the only docking method requested — gnina then reads the RDKit-generated SDF + cleaned PDB directly. Co-requesting Vina or any Vina-rescore (boltz/diffdock auto-add a Vina-rescore) silently falls back to PDBQT with a warning, since OpenBabel still has to run for those methods. |
| `POSES_DIR` | *(empty)* | Directory of user-supplied ligand starting poses, one `<ligand_id>.sdf` per ligand. Every `ligand_id` in the combinations CSV must have a match or the run aborts before docking. Requires `POSE_MODE=local` or `score`, and `NO_DECOYS=1`. See [Starting from a user-supplied pose](#starting-from-a-user-supplied-pose). |
| `POSE_MODE` | *(empty → `dock`)* | `dock` \| `local` \| `score`. How Vina/gnina consume a supplied pose. `dock` ignores the supplied coordinates, so it is rejected together with `POSES_DIR`. |
| `FLEXIBLE_DOCKING` | *(empty)* | Set to `1` to let side chains of residues inside the docking box move during the search (Vina and gnina only). See [Flexible receptor docking](#flexible-receptor-docking). |
| `FLEXRES_GNINA` | *(empty)* | gnina-only explicit flexible-residue spec, e.g. `"A:88,91"`. Takes priority over `FLEXIBLE_DOCKING`'s automatic selection for gnina. |
| `VINA_EXHAUSTIVENESS` | *(empty)* | Vina search exhaustiveness. Higher improves pose quality at the cost of runtime. Defaults to `16` when omitted. |
| `MIN_MOL_WT` | `250` | Minimum molecular weight filter for known-binder expansion |
| `MAX_MOL_WT` | `450` | Maximum molecular weight filter for known-binder expansion |
| `CHEMBL_VERSION` | `chembl_36` | ChEMBL version string used for known-binder lookup |

### Targets

| Target | GPU | Description |
|---|---|---|
| `run-guild` | Yes | Generic target — pass any combination of `METHODS` |
| `run-boltz` | Yes | Shortcut for boltz docking |
| `run-vina` | No | Shortcut for vina docking (CPU only) |
| `run-diffdock` | No | Shortcut for diffdock docking |
| `run-gnina` | Yes* | Shortcut for gnina docking (*GPU used for CNN rescoring; pass `USE_GPU=` for CPU-only) |
| `run-plip` | No | Re-run only the PLIP interactions step over an existing `data/<project>/` tree |

### Direct script invocation

You can also call the master script directly inside a container:

```shell
python scripts/run_guild.py \
    --project my_project \
    --combinations /workspace/path/to/combos.csv \
    --methods boltz vina diffdock \
    --batch-size 5 \
    --head 100 \
    --decoys /workspace/path/to/decoys.tsv \
    --min-mol-wt 250 \
    --max-mol-wt 450 \
    --chembl-version chembl_36 \
    --use-known-binders \
    --clean
```

### Requirements

* NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) (for GPU methods)

### Consuming PLIP interactions output

Every `make run-*` invocation runs PLIP after scoring and writes
`data/<project>/plip_interactions.tsv` — a tab-separated file with one row
per docked complex that PLIP could analyze. The file is **always written**,
even header-only if no method produced complex PDBs, so downstream code
paths are deterministic. External notebooks should read this file directly
instead of installing `plip` locally (its sdist tries to build openbabel
from source, which fails on most CI / lab hosts):

```python
import pandas as pd

plip = pd.read_csv(f"data/{project}/plip_interactions.tsv", sep="\t")
```

Schema (see [guild/constants/plip.py](guild/constants/plip.py) for the
canonical list):

| column | meaning |
|---|---|
| `protein_config_id` | matches the row in `combinations.csv` |
| `smiles` | the docked ligand |
| `n_hbonds` | hydrogen bonds |
| `n_hydrophobic` | hydrophobic contacts |
| `n_pistacking`, `n_pication` | π-stacking and π-cation |
| `n_saltbridges`, `n_halogen`, `n_waterbridges`, `n_metal` | other interaction types |
| `total_interactions` | sum across all categories |
| `n_unique_residues` | unique residue contacts |

To skip PLIP for a docking run, pass `--no-plip` to `run_guild.py`. To
**re-run only PLIP** over an existing project (no re-docking), use:

```shell
make run-plip PROJECT=myproject COMBINATIONS=/workspace/path/to/combos.csv METHODS="vina diffdock"
```

That iterates the existing `data/<project>/batches/*/` tree and regenerates
`plip_interactions.tsv` from whatever complex PDBs are present.

### Troubleshooting a failed combination

When a docking method fails for a given combination, the project's
`batch_progress.log` and each batch's `output.log` carry a `FAILED ...`
line that points at a dedicated subprocess transcript:

| Method | Per-combination log path |
|---|---|
| Boltz | `batches/<batch>/boltz/<run_id>.subprocess.log` |
| gnina | `batches/<batch>/gnina/<protein>_<ligand>.subprocess.log` |
| DiffDock | `batches/<batch>/diffdock/_batch.subprocess.log` *(batch-level — DiffDock runs once per batch)* |
| Vina | *(uses the Python API — failure trace lands in `output.log`)* |

The file contains the full argv, exit code, stdout and stderr — written
on every invocation, success or failure. Example FAILED line you'd grep
for:

```
FAILED Boltz 6CTA-A-protein_1 (empty manifest after retry) — see /workspace/data/myproject/batches/batch_1/boltz/6CTA-A-protein_1.subprocess.log
```

Open that file to see exactly what Boltz / gnina / DiffDock printed before
exiting — no `grep` archaeology in the batch-wide log needed.

---

## How to run

`uv run python guild/run.py`

If using a notebook to run the code, make sure you pass the home_path as well.

## Installations

This set of installations aims to allow the full usage of Guild, even if the user does not leverage all its capacities.
If you have a CPU-only machine, delete the `pyproject.toml`, rename the `pyproject_cpu.toml` as `pyproject.toml` and only then run `uv sync`.

Pre-requisites:

* [Karmadock](https://github.com/schrojunzhang/KarmaDock)
* [Diffdock](https://github.com/gcorso/DiffDock.git)
* [Openbabel]

```shell
git clone https://github.com/openbabel/openbabel.git
mkdir openbabel/build
sudo apt install -y cmake
cmake -DBUILD_GUI=OFF -S openbabel -B openbabel/build
make -C openbabel/build
sudo make install
sudo ldconfig /usr/local/lib64/
obabel -V
```

PLIP dependencies (beyond openbabel):

```shell
sudo apt-get update
sudo apt-get install -y swig
sudo apt-get install -y libopenbabel-dev
```

P2Rank (binding site prediction):

```shell
sudo apt update
sudo apt install openjdk-17-jre

wget https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz
tar -xvzf p2rank_2.4.2.tar.gz
```

## Usage

### Single run

The Guild object is the focal point of this tool. It takes the input protein and ligand and generates the appropriate folder structure to run all tools. Furthermore, it generates replicates or versions of the input files appropriate for all the tools.

**Basic Example:**

```python
from guild.run import Guild

# Initialize Guild with protein and ligand information
dock_wizard_object = Guild(
    ligand_smile="CC(=COC=O)CCC1=C(C)CCCC1(C)C",  # SMILES string of the ligand
    ligand_idx="ligand1",                          # Unique identifier for the ligand
    protein_idx="3pbl",                            # Unique identifier for the protein
    protein_file="/path/to/protein.pdb",           # Path to the protein PDB file
    project_name="my_project",                     # Name of the project
    protein_chain="A",                             # Optional: specific chain to use
    original_ligand="3C0",                          # Optional: original ligand ID in PDB
    original_ligand_chain="A",                      # Optional: chain of original ligand
)

# Run docking with all available methods
# Note: box_location is required for AutoDock Vina
dock_wizard_object.dock(
    box_location="/path/to/autodock_box.txt",      # Required for AutoDock Vina
    methods=["vina", "karmadock", "diffdock", "boltz"]  # Optional: specify methods
)

# Run individual docking methods
dock_wizard_object.run_autodock_vina()  # Requires box_location to be set
dock_wizard_object.run_karmadock()
dock_wizard_object.run_diffdock()
dock_wizard_object.run_boltz()

# Analyze docking results (PLIP interaction profiling)
dock_wizard_object.analyze()
```

**Complete Example with Box File:**

The box file is necessary to run AutoDock Vina. It defines the search space for docking. An example box file format can be found in the `files` folder. The box file should contain center coordinates (x, y, z) and size dimensions.

```python
from guild.run import Guild

# Example with all parameters
dock_wizard_object = Guild(
    ligand_smile="CCCC",
    ligand_idx="test1",
    protein_idx="5c1m",
    protein_file="/home/user/Guild/5c1m.pdb",
    project_name="debug_project",
    protein_chain="A",
    original_ligand="LIG",
    original_ligand_chain="A",
)

# Run docking with box file
dock_wizard_object.dock(box_location="/home/user/Guild/autodock_box.txt")
```

### BulkRun

Running in bulk is necessary to leverage the rank percentile score, as it is empirically derived by comparing a ligand of interest against a panel of proteins. The bulk run automatically handles multiple protein-ligand combinations, generates decoys, and computes rank percentile scores.

**Input Table Format:**

Your input table should be a pandas DataFrame with the following columns:

|protein_config_id|protein_id|protein_path|protein_chain|original_ligand|original_ligand_chain|ligand_id|smiles|ligand_category|is_pdb|box_location|
|-----------------|----------|------------|-------------|---------------|---------------------|---------|------|---------------|------|------------|
|5zk8-A-3C0-A|5zk8|path/to/file.pdb|A|3C0|A|drug_1|CCCC|LOI|1| *(optional)* |

**Column Descriptions:**
- `protein_config_id`: Unique identifier for the protein configuration (e.g., `{protein_id}-{chain}-{ligand}-{ligand_chain}`)
- `protein_id`: PDB ID or identifier for the protein
- `protein_path`: Full path to the protein PDB file
- `protein_chain`: Chain identifier to use for docking. To dock into a pocket
  that spans **multiple chains** (e.g. a dimer interface), give a comma-separated
  list such as `A,B` (any number of chains). See
  [Multi-chain binding pocket](#multi-chain-binding-pocket).
- `original_ligand`: Ligand identifier from the PDB file
- `original_ligand_chain`: Chain of the original ligand
- `ligand_id`: Unique identifier for the ligand
- `smiles`: SMILES string of the ligand
- `ligand_category`: Category of ligand (e.g., "LOI" for ligand of interest, "known_binder", etc.) - required for plotting
- `is_pdb`: Binary indicator (1 if PDB file, 0 otherwise)
- `box_location` *(optional)*: Path to a Vina box file (`center_{x,y,z}` + `size_{x,y,z}`) that
  defines the binding pocket for both Vina and Boltz. Supplied per row but conceptually per
  protein — see [Custom binding pocket](#custom-binding-pocket).

**Basic Example:**

```python
import pandas as pd
from guild.bulk import BulkRun

# Create or load your input table
input_table = pd.DataFrame({
    'protein_config_id': ['5zk8-A-3C0-A'],
    'protein_id': ['5zk8'],
    'protein_path': ['/path/to/5zk8.pdb'],
    'protein_chain': ['A'],
    'original_ligand': ['3C0'],
    'original_ligand_chain': ['A'],
    'ligand_id': ['drug_1'],
    'smiles': ['CCCC'],
    'ligand_category': ['LOI'],
    'is_pdb': [1]
})

# Initialize BulkRun
bulk_analysis_object = BulkRun(
    input_table=input_table,
    project_name="my_bulk_project",              # Project name (cannot contain underscores)
    methods_to_run=["vina", "karmadock", "diffdock", "boltz"],  # Optional: specify methods
    batch_size=1000,                             # Number of combinations per batch
    decoys=None,                                 # Optional: path to custom decoy file
    min_mol_wt=250,                              # Minimum molecular weight for known binders
    max_mol_wt=450,                              # Maximum molecular weight for known binders
    chembl_version="chembl_36",                  # ChEMBL version for known binders
)

# Run docking for all combinations
bulk_analysis_object.run_docking()

# Compute guild scores (normalizes scores across methods)
bulk_analysis_object.run_guild_scoring(n_processes=None)  # None = use all CPUs

# Generate plots
bulk_analysis_object.plot_guild_scoring()
bulk_analysis_object.plot_unique_proteins_scorings(top_n_hits=5)

# Run PLIP interaction profiling for a specific batch
bulk_analysis_object.run_plip(current_batch="batch_1", verbose=True)

# Plot PLIP interaction comparison
bulk_analysis_object.plot_plip_comparison()
```

**Advanced Example with Custom Settings:**

```python
import pandas as pd
from guild.bulk import BulkRun

# Load input table from CSV
input_table = pd.read_csv("input_combinations.csv")

# Initialize with custom settings
bulk_analysis_object = BulkRun(
    input_table=input_table,
    project_name="large_scale_screening",
    methods_to_run=["vina", "karmadock"],  # Only run specific methods
    batch_size=500,                                  # Smaller batches for memory management
    decoys="/path/to/custom_decoys.tsv",            # Custom decoy dataset
    min_mol_wt=200,
    max_mol_wt=500,
    chembl_version="chembl_36",
)

# Run docking (processes all batches)
bulk_analysis_object.run_docking()

# Run scoring with multiprocessing
bulk_analysis_object.run_guild_scoring(n_processes=8)  # Use 8 CPU cores

# Access results
print(bulk_analysis_object.guild_scores_df)  # DataFrame with all scores
```

## Methods

### Docking

The docking methods available via Guild are, to date:
* [Autodock Vina](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3041641/)
*Trott O, Olson AJ*. **AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading**. J Comput Chem. 2010 Jan 30;31(2):455-61. doi: 10.1002/jcc.21334. PMID: 19499576; PMCID: PMC3041641.
* [Karmadock](https://www.nature.com/articles/s43588-023-00511-5)
*Zhang, X., Zhang, O., Shen, C., Qu, W., Chen, S., Cao, H., Kang, Y., Wang, Z., Wang, E., Zhang, J., Deng, Y., Liu, F., Wang, T., Du, H., Wang, L., Pan, P., Chen, G., Hsieh, C. Y., & Hou, T.*. **Efficient and accurate large library ligand docking with KarmaDock**. Nature computational science, 2023, 3(9), 789–804. <https://doi.org/10.1038/s43588-023-00511-5>
* [DiffDock](https://arxiv.org/abs/2210.01776)
*Gabriele Corso, Hannes Stärk, Bowen Jing, Regina Barzilay, Tommi Jaakkola*, **DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking**. arxiv: <https://arxiv.org/abs/2210.01776>.
* [Boltz2](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1)
*Saro Passaro,  Gabriele Corso,  Jeremy Wohlwend,  Mateo Reveiz,  Stephan Thaler, Vignesh Ram Somnath, Noah Getz,  Tally Portnoi, Julien Roy,  Hannes Stark, David Kwabi-Addo,  Dominique Beaini,  Tommi Jaakkola, Regina Barzilay*, **Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction**.
biorxiv: <https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1>

If you use results from any of these tools, please make sure to cite the authors as indicated in the hyperlinks.

### Vina rescore (automatic with DiffDock and Boltz)

When `diffdock` or `boltz` is included in the methods list, Guild **automatically adds** a
matching Vina rescore step. The rescore applies Vina's physics-based scoring function to the
predicted pose (score-only, no re-docking), giving a kcal/mol ΔG estimate that's comparable
across methods.

The two rescore tracks are **independent** — each produces its own column, so a run that uses
both DiffDock and Boltz gets two distinct rescore scores:

| Upstream method | Auto-enabled rescore   | Score column                  |
|-----------------|------------------------|-------------------------------|
| `boltz`         | `vina_rescore_boltz`   | `vina_rescore_boltz_score`    |
| `diffdock`      | `vina_rescore_diffdock`| `vina_rescore_diffdock_score` |

Both score columns are in kcal/mol (lower = stronger predicted binding). Each is independently
ranked per protein and folded into the `global_rp_score`.

> **Note:** `boltz_score` itself is the protein-ligand ipTM confidence (range [0, 1], higher =
> more confident structure) — not a binding score. For a binding-strength signal from Boltz,
> use `vina_rescore_boltz_score`. Likewise `gnina`'s `gnina_score` is the Vina-style affinity
> (kcal/mol, lower = better) while `gnina_cnn_score` is a pose-confidence side channel that does
> not participate in guild's rank-percentile aggregation.

#### Coordinate-frame caveat

Boltz often recentres its predicted complex into its own internal frame, so a receptor PDBQT
prepared from the *template* PDB will not be in the same physical space as the Boltz-output
ligand. Guild handles this by extracting both the receptor and the ligand from Boltz's complex
PDB on every rescore call. If you call `rescore_boltz_pose` directly outside the bulk pipeline,
do the same — do not reuse the template-frame receptor.

### Custom binding pocket

By default, Guild derives the binding pocket from the co-crystal ligand declared in
`original_ligand` / `original_ligand_chain` (for Boltz, residues within 4 Å of that ligand become
`pocket_contacts`; for Vina, a box is built from its centre). When that information isn't
available — apo structures, recombinant assemblies, or pockets predicted by external tools like
fpocket / P2Rank — supply a **Vina box file** instead:

```
center_x = -7.470
center_y = -15.230
center_z =   5.970
size_x = 13.830
size_y = 15.070
size_z = 15.230
```

There are two ways to wire it in:

1. **Per-row** via the optional `box_location` column in the combinations CSV — best for
   multi-protein runs where each protein has its own pocket file.
2. **Global** via the `BOX=` Makefile flag (or `--box` on the script) — fills the column on rows
   where it is empty. Per-row values always win.

Precedence per combination: `box_location` (explicit) > P2Rank prediction (when
`predict_binding_pocket=True`) > derived from `original_ligand`.

For Boltz, the box is converted to a residue list at runtime: every residue whose Cα is inside
the axis-aligned box becomes a `contact` constraint in the YAML.

### Multi-chain binding pocket

Some pockets sit at the **interface of two (or more) protein chains** — a dimer
interface, a recombinant assembly, an allosteric site between subunits. To dock a
small-molecule ligand into such a pocket, list every chain in the `protein_chain`
column, comma-separated:

| protein_config_id | protein_id | protein_path | protein_chain | ... |
|-------------------|------------|--------------|---------------|-----|
| 6CTA-A,B--lig1    | 6CTA       | /path/6CTA.pdb | `A,B`       | ... |

This works for an arbitrary number of chains (`A`, `A,B`, `A,B,C,D`, …). A single
chain (`A`) behaves exactly as before, so existing tables are unaffected.

What changes under the hood when more than one chain is listed:

- **Receptor** (Vina, gnina, KarmaDock, DiffDock): all listed chains are kept in
  the prepared receptor, so docking and scoring see the full interface. Residues
  are renumbered 1-based *per chain*.
- **Boltz**: emits one `protein` block per chain (each with its own MSA) and a
  template/pocket constraint spanning every chain. Per-chain MSA generation and
  the larger complex make Boltz runs proportionally more expensive with more
  chains.
- **Pocket contacts**: collected from every listed chain, each indexed
  independently from 1, so an interface pocket is fully constrained.

**Recommended:** supply a `box_location` (see [Custom binding pocket](#custom-binding-pocket))
that covers the interface. The co-crystal-ligand pocket derivation
(`original_ligand`) still uses the primary (first) chain only; a box covers the
whole interface cleanly. If you encode the chain segment of `protein_config_id`,
use the same comma form (`6CTA-A,B-...`) so DiffDock/complex receptor extraction
keeps both chains.

### Starting from a user-supplied pose

By default Guild generates a 3D conformer of each ligand from its SMILES with
OpenBabel `--gen3d`, then hands that to Vina / gnina / KarmaDock for docking.
For experiments where you already have an **experimentally-informed starting
pose** (a crystal-ligand placement, a known cognate-binder pose, a manually-built
hypothesis) you can ask Guild to start from that pose instead.

Drop one `<ligand_id>.sdf` file per ligand into a directory and pass it as
`POSES_DIR`. The file name *must* match the `ligand_id` column in the
combinations CSV.

```shell
make run-vina \
    PROJECT=myproject \
    COMBINATIONS=/workspace/combos.csv \
    POSES_DIR=/workspace/poses/ \
    POSE_MODE=local \
    NO_DECOYS=1
```

**Fail-fast contract.** When `POSES_DIR` is set, *every* `ligand_id` in the
combinations CSV must have a matching `<ligand_id>.sdf` in the directory. Any
missing file aborts the run before docking starts, with a single error
message listing all missing IDs (capped at 20 with `(+N more)`). Partial
coverage would silently mix pose-anchored and SMILES-random ligands in the
same project, invalidating cross-ligand comparisons, so we reject it.

`POSES_DIR` is also rejected together with `KNOWN_BINDERS=1` or decoy
expansion (`NO_DECOYS=1` is required) — those expansions add ChEMBL ligands
the caller couldn't have prepared poses for.

#### Pose modes

The `POSE_MODE` flag controls how Vina / gnina consume the supplied pose:

| `POSE_MODE` | Vina behaviour | When to use |
|---|---|---|
| `dock` *(default)* | Normal global stochastic search. **The supplied pose has *no effect* on Vina's starting coordinates** — Vina samples random positions/orientations/torsions inside the box per run. The pose is used only for ligand topology / connectivity. | Backwards-compatible default. Don't combine with `POSES_DIR`. |
| `local` | `Vina.optimize()` — single local minimisation starting from the supplied coords. No global search. | **Recommended for biasing toward an experimental pose.** The mode that actually honours the supplied coordinates. |
| `score` | `Vina.score()` — evaluate the supplied pose as-is, no movement. | Pure energy evaluation of a known pose. |

Setting `POSES_DIR` with `POSE_MODE=dock` is rejected — the combination would
do no useful work (Vina/gnina's global search ignores the supplied coords)
and would burn ligand-prep cycles for nothing. Pass `POSE_MODE=local` or
`POSE_MODE=score` explicitly to opt in.

For gnina specifically, `POSE_MODE=local` pairs naturally with the default
`--cnn_scoring=rescore`/`refinement` knobs for CNN-aided local refinement.

#### Supported engines

| Engine | Honours `POSES_DIR`? | Notes |
|---|---|---|
| **Vina** | yes | `dock` / `local` / `score` all wired. |
| **gnina** | yes | `dock` / `local` / `score` all wired (`--local_only` / `--score_only`). |
| **KarmaDock** | no | KarmaDock isn't currently supported in this codepath (deferred). |
| **Boltz** | no — logged on `BulkRun.__init__` | End-to-end sequence/SMILES → complex predictor; no starting-pose API. |
| **DiffDock** | no — logged on `BulkRun.__init__` | Score-based diffusion sampler; no starting-pose API in our pinned version. |

#### What we don't support

Hard distance / harmonic positional restraints (e.g. "stay within 2 Å of these
anchor atoms"). Stock AutoDock Vina has no API for them; you'd need a
Smina-style fork. If you need stronger biasing than `local` provides, that's
the future direction.

### Flexible receptor docking

By default, Guild treats the receptor as fully rigid during docking. Enabling flexible receptor docking allows the **side chains** of residues inside the docking box to adjust their conformation to accommodate the ligand, which can improve pose quality for tightly packed binding sites.

Flexible docking is supported for **Vina and gnina only**. Boltz, DiffDock, and KarmaDock are unaffected.

#### Enabling it

Via Make:

```shell
make run-vina \
  COMBINATIONS=/workspace/path/to/combos.csv \
  PROJECT=myproject \
  FLEXIBLE_DOCKING=1
```

Via Python:

```python
from guild.bulk import BulkRun

bulk = BulkRun(
    input_table=input_table,
    project_name="myproject",
    methods_to_run=["vina", "gnina"],
    flexible_docking=True,
)
bulk.run_docking()
```

Via the script directly:

```shell
python scripts/run_guild.py \
    --project myproject \
    --combinations /workspace/combos.csv \
    --methods vina gnina \
    --flexible-docking
```

#### How residues are selected

**Vina and gnina (pdbqt mode):** residues whose Cα atom falls inside the docking box are automatically identified and passed to `mk_prepare_receptor.py --flexres`. The receptor is split into a rigid PDBQT (backbone + non-flexible residues) and a flex PDBQT (mobile side chains). Vina receives both via `set_receptor(rigid, flex)`; gnina receives the flex PDBQT via `--flex`.

**gnina (sdf mode):** when gnina is running in `GNINA_INPUT_MODE=sdf`, no PDBQT splitting is needed. Instead, gnina's `--flexdist_ligand` / `--flexdist` flags are used: gnina selects flexible residues automatically at search time by finding all residues within **4 Å** of the docked ligand.

> **⚠️ Known limitation — not safe for SMILES-only ligands.** "Within 4 Å of the docked
> ligand" means 4 Å of the ligand *input file's own coordinates*, not the docking box. A
> ligand built from a bare SMILES (via `smiles_to_sdf`, i.e. anything that isn't a
> `--poses-dir`-supplied or otherwise pre-aligned pose) is embedded in an arbitrary local
> frame with no reference to the receptor at all — its coordinates can end up nowhere
> near the real binding site. When that happens, `--flexdist_ligand` locks onto whatever
> happens to be nearby (observed in practice: residues on an unrelated chain), and the
> search can fail to find *any* valid pose ("ligand outside box" on effectively every
> sample) — which looks like a docking failure but is actually this setup bug. If your
> ligands aren't pose-supplied, use the explicit `gnina_flexres`/`FLEXRES_GNINA`
> override below instead of automatic sdf-mode `FLEXIBLE_DOCKING`.

#### Fallback behaviour

If flex prep fails for any combination (e.g. `mk_prepare_receptor.py` rejects a residue, or no residues fall inside the box), that combination falls back silently to **rigid docking** with a warning in the log. No combination is dropped.

#### Explicit residue selection (gnina)

Instead of the automatic box-based or flexdist-based selection, you can pin specific residues as flexible for gnina by naming them explicitly. The spec is passed directly as gnina's `--flexres` flag and takes priority over `FLEXIBLE_DOCKING`'s automatic selection for gnina. Vina is unaffected.

There are two ways to supply the spec, which can be combined:

| Scope | How to set | Precedence |
|---|---|---|
| Per-combination | `gnina_flexres` column in the combinations CSV | Wins over the project-level flag |
| Project-wide | `FLEXRES_GNINA` (Make) / `--flexres-gnina` (CLI) / `flexres_gnina=` (Python) | Used when the CSV column is absent or empty |

**Residue spec format:** comma-separated residue numbers within a chain block, with chain-blocks joined by underscores:

```
chain:resnum[,resnum,...]_chain:resnum[,resnum,...]
```

| Example spec | Meaning |
|---|---|
| `A:88` | Residue 88 on chain A |
| `A:88,91` | Residues 88 and 91 on chain A |
| `A:88,91_B:7` | Residues 88 and 91 on chain A, and residue 7 on chain B |

> **Residue numbering.** Use residue numbers from the **cleaned receptor** (per-chain 1-based after Guild's `renumber_pdb_residues`), not the original PDB. This is the same frame used by `covalent_rec_atom`. The easiest way to confirm the right numbers is to look at the cleaned PDB under `data/<project>/batches/<batch>/gnina/<protein>_cleaned.pdb`.

##### Per-combination (CSV column)

Add a `gnina_flexres` column to the combinations CSV. Rows with a value use it; rows with an empty/missing cell fall back to the project-wide flag (or no explicit flexres if that is also unset).

```csv
protein_config_id,ligand_id,smiles,...,gnina_flexres
3pbl-A-ETQ-A,lig1,CC(=O)O,...,"A:88,91"
3pbl-A-ETQ-A,lig2,c1ccccc1,...,"A:88,91_B:7"
3pbl-A-ETQ-A,lig3,CCN,...,
```

##### Project-wide

Via Make (gnina only):

```shell
make run-gnina \
  COMBINATIONS=/workspace/path/to/combos.csv \
  PROJECT=myproject \
  FLEXRES_GNINA="A:88,91"
```

To combine with `FLEXIBLE_DOCKING` for Vina (gnina uses the explicit list, Vina uses box-based automatic selection):

```shell
make run-guild \
  COMBINATIONS=/workspace/path/to/combos.csv \
  PROJECT=myproject \
  METHODS="vina gnina" \
  FLEXIBLE_DOCKING=1 \
  FLEXRES_GNINA="A:88,91"
```

Via Python:

```python
bulk = BulkRun(
    input_table=input_table,
    project_name="myproject",
    methods_to_run=["gnina"],
    flexres_gnina="A:88,91",
)
bulk.run_docking()
```

Via the script directly:

```shell
python scripts/run_guild.py \
    --project myproject \
    --combinations /workspace/combos.csv \
    --methods gnina \
    --flexres-gnina "A:88,91"
```

#### Caveats

- Flexible docking increases runtime — side-chain sampling adds conformational degrees of freedom to the search.
- The residue selection is box-driven (Cα inside the AABB), not contact-driven, so very large boxes may include more residues than needed.
- For pdbqt mode, the box file must exist before flex prep runs. Per-row `box_location` values and the global `BOX=` fallback both work.
- To vary flexible residues per combination, use the `gnina_flexres` CSV column. The project-wide `FLEXRES_GNINA` / `flexres_gnina=` flag is a convenient shorthand when every combination should use the same residues.

### Covalent docking (gnina)

Covalent docking models inhibitors that form a covalent bond to a specific receptor atom (e.g. a catalytic cysteine). It is supported for **gnina only** — gnina exposes covalent docking natively, whereas Vina has no native covalent mode. Boltz, DiffDock, KarmaDock, and Vina are unaffected.

Unlike flexible docking (a global on/off), covalent docking is configured **per combination** through two optional combinations-CSV columns:

| Column | Example | Meaning |
|---|---|---|
| `covalent_rec_atom` | `A:145:SG` | The receptor atom the ligand bonds to, as `chain:resnum:atomname`. |
| `covalent_lig_smarts` | `C#N` | SMARTS matching the ligand warhead atom that forms the bond. |

gnina runs covalent docking for a row only when **both** columns are present and non-empty; otherwise that row docks normally. No Make flag or CLI option is required — the data flows in through the CSV like `box_location`.

> **Residue numbering — important.** `covalent_rec_atom` must reference the residue numbers of the **cleaned receptor** that Guild docks against. Guild runs `isolate_protein_chain` + `renumber_pdb_residues`, which resets residue numbers to **1-based per chain**. A number copied from the original PDB will usually be wrong. Guild validates the spec against the cleaned protein up-front; if `chain:resnum:atomname` does not resolve, it logs an error and that row falls back to normal (non-covalent) docking.

> **gnina build.** Covalent docking requires a gnina build that accepts the `--covalent_*` flags. Verify with `docker run --rm guild:latest gnina --help | grep -i covalent` after `make docker-local`.

Example combinations CSV row (covalent inhibitor against a catalytic Cys):

```csv
protein_config_id,protein_id,protein_chain,protein_path,smiles,ligand_id,ligand_category,is_pdb,original_ligand,original_ligand_chain,box_location,covalent_rec_atom,covalent_lig_smarts
6CTA-A,6CTA,A,/workspace/.../6cta.pdb,O=C(CCl)Nc1ccccc1,cov_1,ex,False,,,/workspace/.../box.txt,A:145:SG,[CH2]Cl
```

### Post-analysis

Post-analysis allows guild to leverage the results from the multiple docking approaches.

#### PLIP

PLIP (Protein-Ligand Interaction Profiler) allows evaluating structural interactions between proteins and ligands, including hydrogen bonds, hydrophobic contacts, salt bridges, π-stacking, and more. To cite PLIP use:
* [PLIP](https://doi.org/10.1093/nar/gkv315)
*Sebastian Salentin, Sven Schreiber, V. Joachim Haupt, Melissa F. Adasme, Michael Schroeder*, **PLIP: fully automated protein-ligand interaction profiler**. Nucleic Acids Res. 2015 Jul 1;43(W1):W443-7. doi: 10.1093/nar/gkv315. PMID: 25873628.

#### Guild score

Guild score is derived by:

1. Comparing a ligand of interest against a panel of random molecules, selected from ChEMBL.
2. When available, compare the results with known binders.
3. Rank the ligand of interest according to the random molecules, by the the specific docking method score.
This provides an empirical way to uniformize the different scoring systems.

## Karmadock fix

There is a mismatch with rdkit version that creates different input files and causes a downstream dimension failure between mol2 and sdf.
In `KarmaDock/dataset/ligand_feature.py`, find these two blocks (there are four places where edge_feature_new is defined in `get_ligand_feature()`):

```
edge_feature_new = torch.zeros((edge_index_new.size(1), 20))
edge_feature_new[:, [4, 5, 18]] = 1
```

Replace their occurrences with:

```
feat_dim = edge_feature.size(1)
edge_feature_new = torch.zeros((edge_index_new.size(1), feat_dim),
                               dtype=edge_feature.dtype,
                               device=edge_feature.device)
```

and find this line in the `forward()` method of the GraphTransformer Block (around line 436) in `KarmaDock/architecture/GraphTransformer_Block.py`:

```
edge_feats = self.edge_encoder(edge_s)
```

Insert the following block immediately before it:

```
if edge_s.size(1) > self.edge_encoder.in_features:
    edge_s = edge_s[:, :self.edge_encoder.in_features]
elif edge_s.size(1) < self.edge_encoder.in_features:
    pad = th.zeros(edge_s.size(0),
                      self.edge_encoder.in_features - edge_s.size(1),
                      device=edge_s.device,
                      dtype=edge_s.dtype)
    edge_s = th.cat([edge_s, pad], dim=1)
```
