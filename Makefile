.PHONY : docker-local docker-test dev test run-boltz run-vina run-diffdock run-guild project-init project-setup

# Run the tests locally
test:
	uv run pytest -v

# Build a local guild image
docker-local:
	DOCKER_BUILDKIT=1 \
	docker build \
		--build-arg APP_NAME=guild \
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
PASSWD_FILE     ?= /tmp/guild_passwd

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
_OPTIONAL_FLAGS = $(_CLEAN_FLAG) $(_KNOWN_BINDERS_FLAG) $(_HEAD_FLAG) $(_DECOYS_FLAG)

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
		--gpus all \
		--shm-size=8g \
		-e LD_LIBRARY_PATH=/opt/localcolabfold/.pixi/envs/default/lib:/usr/local/lib \
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

# Generic target — pass METHODS="boltz vina karmadock diffdock" as needed.
# Picks up GPU flags automatically when any GPU method is present.
run-guild: _prepare-passwd
	docker run \
		$(DOCKER_COMMON) \
		--gpus all \
		--shm-size=8g \
		-e LD_LIBRARY_PATH=/opt/localcolabfold/.pixi/envs/default/lib:/usr/local/lib \
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
