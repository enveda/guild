.PHONY : docker-local docker-test dev test run-boltz run-vina run-diffdock run-gnina run-plip run-guild project-init project-setup

# Run the tests locally
test:
	uv run pytest -v

# Build a local guild image.
# Uses gnina-bundle:local if already present; otherwise builds it first from
# Dockerfile.gnina-bundle (pulls gnina/gnina:latest once, ~9 GB) so the gnina
# docking method works out of the box without needing an external registry.
docker-local:
	@if ! docker image inspect gnina-bundle:local >/dev/null 2>&1; then \
		echo "gnina-bundle:local not found — building from Dockerfile.gnina-bundle..."; \
		docker build -t gnina-bundle:local -f Dockerfile.gnina-bundle .; \
	fi
	DOCKER_BUILDKIT=1 \
	docker build \
		--build-arg APP_NAME=guild \
		--build-arg GNINA_BUNDLE_IMAGE=gnina-bundle:local \
		-t guild:latest \
		-f Dockerfile \
		--target=docker \
		.

# ---------------------------------------------------------------------------
# Shared parameters for run-boltz / run-vina / run-guild targets.
#
# Usage examples:
#   make run-guild COMBINATIONS=/workspace/path/to/combos.csv METHODS="vina boltz diffdock" HEAD=100 BATCH_SIZE=5 CLEAN=1
#   make run-boltz   PROJECT=my_project COMBINATIONS=/workspace/path/to/combos.tsv
#   make run-vina    PROJECT=my_project COMBINATIONS=/workspace/path/to/combos.tsv KNOWN_BINDERS=1
#
# COMBINATIONS is required. All other parameters are optional.
# DECOYS defaults to guild/support/decoys/chembl_36_decoys_2.tsv if omitted.
# ---------------------------------------------------------------------------

MASTER_SCRIPT   ?= /workspace/scripts/run_guild.py
PROJECT         ?= imagerun
COMBINATIONS    ?=
DECOYS          ?=
BATCH_SIZE      ?= 2
HEAD            ?= 0
CLEAN           ?=
KNOWN_BINDERS   ?=
NO_DECOYS       ?=
BOX             ?=
N_WORKERS       ?=
PASSWD_FILE     ?= /tmp/guild_passwd
VINA_EXHAUSTIVENESS ?=

# GPU toggle. Default 1 (enabled). Set USE_GPU= (empty) to drop
# `--gpus all --shm-size=8g` from docker run AND forward `--no-gpu` to
# run_guild.py — required on hosts without a usable GPU (gnina then falls back
# to CPU; Boltz is genuinely GPU-bound and shouldn't be combined with USE_GPU=).
USE_GPU         ?= 1
_GPU_FLAGS       = $(if $(USE_GPU),--gpus all --shm-size=8g,)
_NO_GPU_FLAG     = $(if $(USE_GPU),,--no-gpu)

# Gnina input mode. Empty (default) → omit the flag (pdbqt default applies).
# Set GNINA_INPUT_MODE=sdf to skip OpenBabel PDBQT prep when gnina is the only
# docking method requested (otherwise BulkRun downgrades to pdbqt with a warning).
GNINA_INPUT_MODE ?=
_GNINA_INPUT_MODE_FLAG = $(if $(GNINA_INPUT_MODE),--gnina-input-mode $(GNINA_INPUT_MODE),)

# User-supplied starting pose directory. Empty (default) → omit the flag and the
# SMILES→3D ligand-prep path is used. When set, every ligand_id in the
# combinations CSV must have a matching <ligand_id>.sdf in the directory;
# BulkRun fails fast otherwise. Use with POSE_MODE=local (refine the supplied
# pose) or POSE_MODE=score (evaluate without movement); the default
# POSE_MODE=dock is rejected when POSES_DIR is set, because Vina/gnina's global
# search ignores the supplied coordinates.
POSES_DIR ?=
_POSES_DIR_FLAG = $(if $(POSES_DIR),--poses-dir $(POSES_DIR),)

# Pose mode for runs that supply POSES_DIR. Empty (default) → omit the flag and
# run_guild.py defaults to 'dock' (which only makes sense without POSES_DIR).
# Valid values: dock | local | score.
POSE_MODE ?=
_POSE_MODE_FLAG = $(if $(POSE_MODE),--pose-mode $(POSE_MODE),)

# Flexible receptor docking (Vina + gnina only). Empty string (default) → rigid.
# Set FLEXIBLE_DOCKING=1 to allow side chains inside the docking box to move
# during the search.
FLEXIBLE_DOCKING ?=
_FLEXIBLE_DOCKING_FLAG = $(if $(FLEXIBLE_DOCKING),--flexible-docking,)

# Explicit flexible residues for gnina (gnina-only). Empty (default) → omit.
# Set to a gnina --flexres spec, e.g. FLEXRES_GNINA="A:88,91" to pin specific
# residues as flexible regardless of box geometry. Takes priority over
# FLEXIBLE_DOCKING's automatic selection for gnina; Vina is unaffected.
FLEXRES_GNINA ?=
_FLEXRES_GNINA_FLAG = $(if $(FLEXRES_GNINA),--flexres-gnina $(FLEXRES_GNINA),)

# Internal docker run flags reused across targets.
# Mounts a generated /etc/passwd so pwd.getpwuid() works for the host UID
# (required by PyTorch / boltz inside the container).
define DOCKER_COMMON
	--rm \
	--network host \
	--user $(shell id -u):$(shell id -g) \
	-e USER=$(shell echo $$USER) \
	-e LOGNAME=$(shell echo $$USER) \
	-e UV_CACHE_DIR=/tmp/uv-cache \
	-e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
	-e WORKSPACE_ROOT=/workspace \
	-v $(PASSWD_FILE):/etc/passwd:ro \
	-v $(shell pwd):/workspace \
	-v $(shell pwd)/guild:/app/guild
endef

_CLEAN_FLAG          = $(if $(CLEAN),--clean,)
_KNOWN_BINDERS_FLAG  = $(if $(KNOWN_BINDERS),--use-known-binders,)
# Build --head flag only when HEAD > 0
_HEAD_FLAG = $(if $(filter-out 0,$(HEAD)),--head $(HEAD),)

# Collect all optional flags into one variable for DRY target definitions
_DECOYS_FLAG         = $(if $(DECOYS),--decoys $(DECOYS),)
_NO_DECOYS_FLAG      = $(if $(NO_DECOYS),--no-decoys,)
_BOX_FLAG            = $(if $(BOX),--box $(BOX),)
_N_WORKERS_FLAG      = $(if $(N_WORKERS),--n-workers $(N_WORKERS),)
_VINA_EXHAUSTIVENESS_FLAG = $(if $(VINA_EXHAUSTIVENESS),--vina-exhaustiveness $(VINA_EXHAUSTIVENESS),)
_OPTIONAL_FLAGS = $(_CLEAN_FLAG) $(_KNOWN_BINDERS_FLAG) $(_HEAD_FLAG) $(_DECOYS_FLAG) $(_NO_DECOYS_FLAG) $(_BOX_FLAG) $(_N_WORKERS_FLAG) $(_NO_GPU_FLAG) $(_GNINA_INPUT_MODE_FLAG) $(_FLEXIBLE_DOCKING_FLAG) $(_FLEXRES_GNINA_FLAG) $(_VINA_EXHAUSTIVENESS_FLAG) $(_POSES_DIR_FLAG) $(_POSE_MODE_FLAG)

# Generate an /etc/passwd that includes the container's original entries plus
# the host user.  This fixes pwd.getpwuid() failures for LDAP/SSSD users
# whose UID does not appear in the container image.
.PHONY: _prepare-passwd
_prepare-passwd:
	@docker run --rm --entrypoint cat guild:latest /etc/passwd > $(PASSWD_FILE) 2>/dev/null || true
	@if ! grep -q ":x:$$(id -u):" $(PASSWD_FILE) 2>/dev/null; then \
		printf '%s:x:%d:%d::%s:/bin/sh\n' "$$(id -un)" "$$(id -u)" "$$(id -g)" "/workspace" >> $(PASSWD_FILE); \
	fi

# Run boltz docking inside the local Docker image (GPU required).
# Requires: make docker-local first.
METHODS ?= boltz
run-boltz: _prepare-passwd
	docker run \
		$(DOCKER_COMMON) \
		$(_GPU_FLAGS) \
		-e LD_LIBRARY_PATH=/opt/localcolabfold/.pixi/envs/default/lib:/usr/local/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cu13/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cublas/lib \
		guild:latest \
		python $(MASTER_SCRIPT) \
			--project $(PROJECT) \
			--combinations $(COMBINATIONS) \
			--methods $(METHODS) \
			--batch-size $(BATCH_SIZE) \
			$(_OPTIONAL_FLAGS)

# Run vina docking inside the local Docker image (CPU only, no GPU required).
# Requires: make docker-local first.
VINA_METHODS ?= vina
run-vina: _prepare-passwd
	docker run \
		$(DOCKER_COMMON) \
		guild:latest \
		python $(MASTER_SCRIPT) \
			--project $(PROJECT) \
			--combinations $(COMBINATIONS) \
			--methods $(VINA_METHODS) \
			--batch-size $(BATCH_SIZE) \
			$(_OPTIONAL_FLAGS)

# Run diffdock docking inside the local Docker image (CPU only — GPU disabled in deploy_diffdock).
# Requires: make docker-local first.
DIFFDOCK_METHODS ?= diffdock
run-diffdock: _prepare-passwd
	docker run \
		$(DOCKER_COMMON) \
		guild:latest \
		python $(MASTER_SCRIPT) \
			--project $(PROJECT) \
			--combinations $(COMBINATIONS) \
			--methods $(DIFFDOCK_METHODS) \
			--batch-size $(BATCH_SIZE) \
			$(_OPTIONAL_FLAGS)

# Run gnina docking inside the local Docker image. Uses the GPU by default for
# CNN rescoring; pass USE_GPU= (empty) on no-GPU hosts to drop --gpus and run
# gnina CPU-only. Requires: make docker-local first.
GNINA_METHODS ?= gnina
run-gnina: _prepare-passwd
	docker run \
		$(DOCKER_COMMON) \
		$(_GPU_FLAGS) \
		guild:latest \
		python $(MASTER_SCRIPT) \
			--project $(PROJECT) \
			--combinations $(COMBINATIONS) \
			--methods $(GNINA_METHODS) \
			--batch-size $(BATCH_SIZE) \
			$(_OPTIONAL_FLAGS)

# Re-run only the PLIP interactions step over an existing data/<project>/ tree.
# CPU-safe and skips docking + scoring entirely — useful for regenerating
# plip_interactions.tsv when only the PLIP code changed. Requires the same
# COMBINATIONS / PROJECT used by the original run.
run-plip: _prepare-passwd
	docker run \
		$(DOCKER_COMMON) \
		guild:latest \
		python $(MASTER_SCRIPT) \
			--project $(PROJECT) \
			--combinations $(COMBINATIONS) \
			--methods $(METHODS) \
			--batch-size $(BATCH_SIZE) \
			--plip-only \
			$(_OPTIONAL_FLAGS)

# Generic target — pass METHODS="boltz vina karmadock diffdock gnina" as needed.
# GPU on by default; pass USE_GPU= (empty) for CPU-only hosts (drops --gpus all
# and forwards --no-gpu). Boltz requires a GPU regardless.
run-guild: _prepare-passwd
	docker run \
		$(DOCKER_COMMON) \
		$(_GPU_FLAGS) \
		-e LD_LIBRARY_PATH=/opt/localcolabfold/.pixi/envs/default/lib:/usr/local/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cu13/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/app/.venv/lib/python3.10/site-packages/nvidia/cublas/lib \
		guild:latest \
		python $(MASTER_SCRIPT) \
			--project $(PROJECT) \
			--combinations $(COMBINATIONS) \
			--methods $(METHODS) \
			--batch-size $(BATCH_SIZE) \
			$(_OPTIONAL_FLAGS)

# Run the first time initialization of the project
project-init:
	uv lock && \
	uv sync --all-groups && \
	git init && \
	uv run pre-commit install

# setup project as a developer
project-setup:
	uv sync --all-groups
	uv run pre-commit install
