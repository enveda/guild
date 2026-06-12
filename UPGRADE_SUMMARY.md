# guild_upgrade — sync from guild-internal

Branch: `guild_upgrade`. **Nothing committed** — all changes are uncommitted in the working tree for your review.

This brings the public `guild` repo up to date with the latest engineering work in the
internal `guild-internal` repo, **without exposing any internal secrets, infrastructure,
database details, or personal paths**. The two repos have separate git histories (public
was a sanitized import), so this was done by comparing working trees file-by-file, not by
merging git history.

## Guiding rule

Internal-only material was **excluded entirely**, not just commented out. Where a shared
file mixed good engineering with internal coupling (e.g. `bulk.py`, the `Dockerfile`,
the `Makefile`), the improvements were ported and the internal parts removed — including
indirect tells like internal function names, the DB API shape, internal CI workflows,
private registry URLs, and example compound names.

A full token sweep over the result found **zero new internal tokens** introduced by this
upgrade (see "Verification" below).

## What was brought in (sanitized)

### New features
- **GNINA docking** — `guild/docking/gnina.py`, `guild/constants/gnina.py`, plus tests
  (`tests/single_run/test_gnina_*.py`, `tests/bulk_run/test_gnina_orchestration.py`).
  The Python code only references container-internal paths (`/opt/gnina/...`), no secrets.
- **GNINA bundle build** — `Dockerfile.gnina-bundle` builds a curated runtime bundle from
  the **public** `gnina/gnina` image. The private `envedancusacr01.azurecr.io` registry
  reference was removed; the bundle image is now an overridable build arg
  `GNINA_BUNDLE_IMAGE` (default `gnina-bundle:latest`).
- **Multi-chain binding pocket** — comma-separated `protein_chain` (`A,B`) support, threaded
  through `run.py`, `bulk.py`, `tools/preparation.py`, `transformers/pdb.py`, and
  `scripts/vina_rescore_all.py` (multi-chain port applied onto the public, relative-path
  version of that script).
- **Per-subprocess troubleshooting logs** — `guild/tools/subprocess_log.py` + integration.
- **KarmaDock compatibility patches** — `scripts/apply_karmadock_patches.py`, applied at
  Docker build time (rdkit/torch ABI fixes).
- **`run_guild.py` CLI** — gained `--methods gnina`, `--box`, `--n-workers`, `--no-gpu`,
  `--plip-only`, `--no-plip`, `--gnina-input-mode`. The `--no-database-update` flag and the
  `database_update=` wiring were **stripped**.
- **`SHELL_SILENCER` bugfix** — `&>/dev/null` → `> /dev/null 2>&1` (dash-safe).

### Updated wholesale (verified clean of internal refs)
`guild/run.py`, `guild/docking/{boltz,diffdock,karmadock,vina}.py`,
`guild/tools/{preparation,bulk,utils,scores}.py`,
`guild/transformers/{pdb,msa,converters,chembl}.py`,
`guild/constants/{bulk,guild,p2rank,visualization}.py`, `docs/adfrsuite_class.py`.

### Config / build (sanitized merges)
- **Dockerfile** — added GNINA bundle stage (genericized) + KarmaDock patches. **Kept** the
  public repo's clean `uv sync` (no `GH_TOKEN` secret) and **omitted** the Azure CLI block —
  both existed only to serve the excluded `enveda-toolkit`/keyvault/database layer.
- **Makefile** — added `run-gnina`, `run-plip`, a `USE_GPU` toggle, gnina/box/workers flags,
  and gnina-bundle auto-build in `docker-local`. **Excluded** all `azurecr.io` push targets,
  the `dev`/`run` targets that pulled internal images, the `~/.azure` credential mounts, and
  the `github-init`/`init` targets (`gh repo create enveda/guild`, internal remote URL).
- **pyproject.toml** — version bump 1.0.0 → 1.1.4, added one author. **Did not** add
  `azure-identity`, `azure-keyvault-secrets`, `enveda-toolkit`, or the `[tool.uv.sources]`
  git source.
- **uv.lock** — public lock was already free of internal packages; only the `guild` version
  pin was updated to 1.1.4 (no dependency changes were made, so the lock stays consistent).
- **.gitignore** — added safety rules: `/notebooks/database`, `/guild/OLD`, `/temp_data`, and
  a `scripts/*` allowlist (prevents committing local experiment scripts that carry
  machine-specific paths). Kept the public allowlist for shipped support data + logo.
- **.dockerignore** — un-ignore `scripts/apply_karmadock_patches.py` (now COPY'd by Docker).
- **.pre-commit-config.yaml** — ruff/uv hook version bumps only.

## What was deliberately EXCLUDED (internal-only)

- **Database / Azure layer**: `guild/azure/database.py`, `guild/azure/get_secret.py`,
  `guild/constants/database.py`, `DATABASE_SCHEMA.md`, `notebooks/database/*`,
  `scripts/test_db_real_submission.py`, `scripts/test_db_upload_dummy.py`. The public repo
  keeps its existing "no database mode". In `bulk.py` the `database_update` param, the
  `retrieve_existing_rp_scores`/`upload_rp_scores` import and all call sites, and the
  DB-merge/upload code paths were fully removed (not just disabled).
- **Internal CI**: `.github/workflows/deploy-dev.yml`, `.github/workflows/build-gnina-bundle.yml`
  (both use `enveda/reusable-workflows` + ACR + Prefect). The public `test.yml`/`lint.yml`
  were kept as-is.
- **Internal tooling/meta**: `CLAUDE.md` (internal dev notes — references ACR, Databricks,
  internal compound names), `.cookiecutter.json` (internal template + `it@envedabio.com`),
  `.claude/settings.json`, `.gitmodules` (empty), `pyptoject_cpu.toml` (typo-named CPU
  variant carrying azure/prefect deps), `environments/installations.sh` (hardcoded
  `/home/ec2-user/...` paths).
- **Azure/Prefect constants** in `constants/system.py` (`AZURE_KEYVAULT_URI`,
  `AZURE_CLIENT_ID`, `AZURE_ACR_IMAGE_URI`, etc.) — used only by excluded code.
- **New analysis notebooks / plot scripts / result figures** under `notebooks/` — these are
  analysis artifacts whose output cells and hardcoded paths (several contained
  `/home/antonio.gomes/...`) are a high indirect-leak risk for low value. Existing public
  notebooks were left untouched. **Worth a selective, manual revisit if you want any of
  these published.**

## Pre-existing references (NOT introduced by this upgrade — flagged for your call)

- `.gitignore:55` → `/.databrickscfg` — a protective ignore rule (prevents committing a
  Databricks credential file). Left in place; removing it would *reduce* safety.
- `renovate.json:4` → `local>enveda/renovate-config` — was already in the public repo,
  unchanged. References an internal renovate config repo.

## Verification

- All 15 ported modules import cleanly in the repo venv.
- Full internal-token sweep (`enveda`, `azurecr`, `databricks`, `keyvault`, `enveda_toolkit`,
  `enveda_prod`, `cdd-benchling`, the Azure client GUID, `/home/antonio`, `/home/ec2-user`,
  `reusable-workflows`, `prefect`, …) over all source/config files → only the two
  pre-existing references above; zero new tokens.
- Docker image rebuild + empirical docking results: see the "Empirical results" section below.

## Empirical results

Image rebuilt from scratch with `--build-arg GNINA_BUNDLE_IMAGE=gnina-bundle:local`
(`guild:latest`, 38.2 GB). `gnina v1.3.1` runs from the genericized local bundle.

**Test suite** (`docker run --rm guild-test python -m pytest tests/ -q`): **100 passed**.

**Empirical docking** — single combination `3pbl-A-ETQ-A / lig1` (D3 receptor + a small
ligand), box-driven pocket, `--no-decoys`, run on an A100 80GB. Every engine produced a
score and completed scoring + PLIP (exit 0):

| Engine    | Mode | Result |
|-----------|------|--------|
| Vina      | CPU  | `vina_score = -4.892` |
| GNINA     | GPU  | `gnina_score = -5.09`, `gnina_cnn_score = -0.28` |
| Karmadock | GPU  | `karmadock_score = 13.86` (build-time rdkit/torch patches applied OK) |
| DiffDock  | GPU  | `diffdock_score = -1.7`; auto-rescore `vina_rescore_diffdock_score = -0.029` |
| Boltz     | GPU  | `boltz_score = 0.327` (ipTM); auto-rescore `vina_rescore_boltz_score = -4.336` |

Notes:
- The DiffDock→Vina and Boltz→Vina auto-rescore tracks both fired correctly, confirming the
  multi-method scoring wiring survived the database-strip of `bulk.py`.
- One benign Boltz PLIP warning (`Binding site LIG:Z:1 not found`) — a known ligand-resname
  quirk, not a failure (run exited 0); unrelated to this upgrade.
- A `pandas` `DataFrameGroupBy.apply` FutureWarning appears across runs — pre-existing in
  `tools/scores.py`, not introduced here.

## How to reproduce the build + runs

```sh
# Build the gnina bundle once (from the public gnina/gnina image), then the app image:
make docker-local              # auto-builds gnina-bundle:local if missing, then guild:latest

# Empirical docking (workspace holds data + combinations.csv at /workspace):
make run-vina  PROJECT=testvina  COMBINATIONS=/workspace/combinations.csv NO_DECOYS=1
make run-gnina PROJECT=testgnina COMBINATIONS=/workspace/combinations.csv NO_DECOYS=1
make run-guild PROJECT=testall   COMBINATIONS=/workspace/combinations.csv METHODS="diffdock karmadock boltz" NO_DECOYS=1
```
