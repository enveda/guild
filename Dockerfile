# syntax=docker/dockerfile:1

####################################################################################################
# GNINA — pulled as a curated ~1.4 GB bundle (binary + the .so deps it loads
# at runtime). Build the bundle separately with Dockerfile.gnina-bundle and
# tag it (locally or in your own registry), then point GNINA_BUNDLE_IMAGE at
# that tag:
#
#     docker build -f Dockerfile.gnina-bundle -t gnina-bundle:local .
#     docker build --build-arg GNINA_BUNDLE_IMAGE=gnina-bundle:local -t guild .
#
# We do NOT consume gnina/gnina:latest directly here: pulling the full ~9 GB
# upstream image into the build graph caused disk exhaustion in CI. The bundle
# is a `FROM scratch` image whose only contents are /export/{bin,lib}, so it
# stays off the main build's dependency graph until the final runtime stage.
ARG GNINA_BUNDLE_IMAGE=gnina-bundle:latest
FROM ${GNINA_BUNDLE_IMAGE} AS gnina-source

####################################################################################################
FROM python:3.10-slim-bookworm AS base

ARG APP_NAME="guild"
ARG ENV

ENV APP_NAME="${APP_NAME}" \
    APP_ROOT="/app" \
    APP_TESTS="tests" \
    APP_USER="appuser" \
    ENV=${ENV} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

ENV APP_USER_HOME="/home/${APP_USER}"
ENV PROJECT_ROOT="${APP_ROOT}/${APP_NAME}"
ENV PROJECT_TEST_ROOT="${APP_ROOT}/${APP_TESTS}"

# RDKit's drawing module links libXrender/libXext, and importing ProLIF pulls it
# in, so `import prolif` fails with "libXrender.so.1: cannot open shared object
# file" without these. They live in `base` rather than in the final `docker`
# stage because `test` branches off `base-build`, not off `docker` -- putting
# them downstream leaves the test image unable to import ProLIF at all.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends libxrender1 libxext6; \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 ${APP_USER} && \
    useradd --uid 1000 --gid 1000 -m ${APP_USER} && \
    mkdir -p ${APP_ROOT} ${PROJECT_ROOT} ${PROJECT_TEST_ROOT} && \
    chown -R ${APP_USER}:${APP_USER} ${APP_ROOT}

WORKDIR ${APP_ROOT}

####################################################################################################
FROM base AS base-build
USER root

# System build deps
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      cmake \
      swig \
      wget \
      git \
      ca-certificates \
      libc-bin \
    ; \
    rm -rf /var/lib/apt/lists/*

# AutoDock Vina
RUN wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64 \
      -O /usr/local/bin/vina && \
    chmod a+x /usr/local/bin/vina

# NOTE: GNINA is intentionally not copied into base-build. Every target that
# transitively depends on base-build (including `test`) would otherwise have to
# materialize the bundle on disk. The gnina bundle is only needed at runtime,
# so the COPY lives in the final `docker` stage instead — buildkit then skips
# gnina-source entirely for the `test` target.

# Python build tools
RUN python -m pip install --upgrade pip setuptools wheel

# OpenBabel build
# Pinned to the openbabel-3-1-1 release tag rather than tracking master, so an
# upstream commit cannot change the library this image builds against.
#
# The `sed` cherry-picks a one-line upstream fix that never made it into any
# tagged 3.1.x release: on GCC 12+, `obutil.h` uses `clock()` / `CLOCKS_PER_SEC`
# without including `<ctime>`, so the asciiformat build fails with
# `error: 'clock' was not declared in this scope`. Injecting the include is
# equivalent to https://github.com/openbabel/openbabel/pull/2313 . Remove the
# sed if we ever bump past the 3.1.x line.
ARG OPENBABEL_VERSION=openbabel-3-1-1
RUN git clone --depth 1 --branch ${OPENBABEL_VERSION} \
        https://github.com/openbabel/openbabel.git && \
    sed -i '/^#include <math.h>/a #include <ctime>' \
        openbabel/include/openbabel/obutil.h && \
    mkdir -p openbabel/build && \
    cmake -DBUILD_GUI=OFF -S openbabel -B openbabel/build && \
    make -C openbabel/build && \
    make install -C openbabel/build && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/openbabel.conf && \
    ldconfig

ENV LD_LIBRARY_PATH=/usr/local/lib

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- LocalColabFold ----
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

RUN git clone --depth 1 https://github.com/YoshitakaMo/localcolabfold.git /opt/localcolabfold && \
    cd /opt/localcolabfold && \
    pixi install && \
    pixi run setup

ENV COLABFOLD_BIN="/opt/localcolabfold/.pixi/envs/default/bin"

####################################################################################################
# Nesso-1 — installed into its OWN venv, entirely independent of the main
# uv-managed .venv (built below in project-prod-deps).
#
# Why isolated: nesso requires numpy>=2 and transformers>=4.40, which
# conflict outright with this repo's numpy<2 pin and fair-esm's pinned
# transformers range — they cannot share one virtualenv. This stage is not
# in project-prod-deps' or base-build's dependency graph, so it builds in
# parallel with them and never touches the main venv.
#
# guild only ever shells out to /opt/nesso/bin/nesso (see NESSO_BIN in
# guild/docking/nesso.py) — there is no Python-level `import nesso` anywhere
# in this repo, so the two venvs never need to interoperate at import time.
#
# Torch is pre-installed from the CUDA 12.1 wheel index before nesso itself:
# nesso's own dependency resolution only requires torch>=2.2 (loose) and
# would otherwise happily settle for the CPU-only PyPI wheel. The pin here
# matters because nesso's optional "kernels" extra (cuequivariance-ops-torch)
# is documented as CUDA 12 only — verify this still matches the GPU/driver on
# the target hosts before bumping either the torch or cuequivariance version.
# This runs entirely inside its own venv's site-packages (manylinux wheels
# ship their own libcudart/libcudnn via RPATH), so — unlike gnina — no
# LD_LIBRARY_PATH juggling is needed for it to coexist with the main venv's
# CUDA 13 torch.
FROM python:3.10-slim-bookworm AS nesso-build

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      git \
      ca-certificates \
    ; \
    rm -rf /var/lib/apt/lists/*

ARG NESSO_TORCH_VERSION=2.5.1
RUN python -m venv /opt/nesso && \
    /opt/nesso/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/nesso/bin/pip install --no-cache-dir \
        "torch==${NESSO_TORCH_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu121 && \
    /opt/nesso/bin/pip install --no-cache-dir \
        "nesso[kernels] @ git+https://github.com/recursionpharma/nesso.git"
####################################################################################################
FROM base-build AS project-prod-deps

USER appuser
WORKDIR /app

COPY --chown=appuser:appuser pyproject.toml /app
COPY --chown=appuser:appuser uv.lock* /app
COPY --chown=appuser:appuser README.md /app

# --locked, not --frozen. Both install straight from uv.lock without
# re-resolving, but --frozen does not check that the lock still agrees with
# pyproject.toml: a dependency added to pyproject and never locked is silently
# omitted from the image, and the first sign of it is a puzzling test failure
# much later. --locked fails here instead, telling you to run `uv lock`.
RUN uv sync --locked --no-install-project --no-group dev && \
    rm -rf /home/appuser/.cache /home/appuser/.config

# Replace pure-Python PyG extension stubs with pre-built CUDA wheels from the
# official PyG wheel index (exact match for torch 2.11.0+cu130).
# uv venvs don't include pip, so use "uv pip install" which targets the active venv.
RUN uv pip install --reinstall --no-deps \
      "https://data.pyg.org/whl/torch-2.11.0%2Bcu130/torch_cluster-1.6.3%2Bpt211cu130-cp310-cp310-linux_x86_64.whl" \
      "https://data.pyg.org/whl/torch-2.11.0%2Bcu130/torch_scatter-2.1.2%2Bpt211cu130-cp310-cp310-linux_x86_64.whl" \
      "https://data.pyg.org/whl/torch-2.11.0%2Bcu130/torch_sparse-0.6.18%2Bpt211cu130-cp310-cp310-linux_x86_64.whl"

####################################################################################################
FROM project-prod-deps AS project-all-deps
# Dev deps (black, ruff, pytest, ipykernel, pre-commit) are all public
RUN uv sync --locked --no-install-project

####################################################################################################
FROM project-all-deps AS test
USER appuser
WORKDIR /app

COPY --chown=appuser:appuser guild /app/guild
COPY --chown=appuser:appuser tests /app/tests
COPY --chown=appuser:appuser scripts/run_guild.py /app/scripts/run_guild.py
COPY --chown=appuser:appuser notebooks/analysis/reviewer_response/reproduce_response_analyses.py /app/notebooks/analysis/reviewer_response/reproduce_response_analyses.py

ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:/app:${PATH}"
ENV PYTHONPATH="/app"

CMD ["python", "-m", "pytest", "-q"]

####################################################################################################
FROM base AS docker
USER root

# Install runtime libs (libinchi1 for bookworm)
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      git \
      curl \
      gcc \
      libc6-dev \
      libxml2 \
      libinchi1 \
      libc-bin \
    ; \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/lib/x86_64-linux-gnu/libinchi.so.1 /usr/lib/x86_64-linux-gnu/libinchi.so.0 && \
    ldconfig

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy OpenBabel runtime
COPY --from=base-build /usr/local/lib/libopenbabel* /usr/local/lib/
COPY --from=base-build /usr/local/lib/openbabel/ /usr/local/lib/openbabel/
COPY --from=base-build /usr/local/share/openbabel/ /usr/local/share/openbabel/
COPY --from=base-build /usr/local/bin/obabel /usr/local/bin/obabel
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/openbabel.conf && ldconfig
ENV LD_LIBRARY_PATH=/app/.venv/lib/python3.10/site-packages/nvidia/cu13/lib:/opt/localcolabfold/.pixi/envs/default/lib:/usr/local/lib

# Vina
COPY --from=base-build /usr/local/bin/vina /usr/local/bin/vina

# GNINA — curated bundle (statically linked binary + the CUDA 12 runtime it
# still loads dynamically). Pulled directly from the gnina-source stage so
# base-build (and therefore the `test` target) does not have to materialize it.
# Invoked with LD_LIBRARY_PATH=/opt/gnina/lib so the bundle's CUDA runtime
# doesn't conflict with the venv-managed torch.
COPY --from=gnina-source /export /opt/gnina
RUN chmod a+x /opt/gnina/bin/gnina

# LocalColabFold
COPY --from=base-build /opt/localcolabfold /opt/localcolabfold
ENV COLABFOLD_BIN="/opt/localcolabfold/.pixi/envs/default/bin"

# Nesso-1 — its own self-contained venv (see the nesso-build stage above for
# why it can't share the main .venv). guild invokes it only by subprocess
# (guild/docking/nesso.py::NESSO_BIN), so no PATH/PYTHONPATH entry is needed
# — just world-readable+executable permissions, since the container runs as
# an arbitrary host uid:gid (see Makefile DOCKER_COMMON's --user), not root
# or appuser.
COPY --from=nesso-build /opt/nesso /opt/nesso
RUN chmod -R a+rX /opt/nesso

USER appuser
WORKDIR /app

ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:/app:${COLABFOLD_BIN}:${PATH}"
ENV PYTHONPATH="/app"
# Runtime environment
ENV HOME="/workspace"
# Nesso downloads its checkpoint + ESM-2 weights from HuggingFace Hub on
# first use (see nesso/hub/config.json upstream); pointing its cache at the
# mounted workspace (like TRITON_CACHE_DIR below) means that download only
# happens once across container restarts instead of once per run. Baking the
# weights into the image at build time would remove even that one-time cost,
# but requires knowing their exact size/auth requirements up front — revisit
# if cold-start download time becomes a bottleneck.
ENV NESSO_CACHE="/workspace/.nesso_cache"
ENV MPLCONFIGDIR="/tmp/matplotlib"
# Triton JIT cache — persisted to the mounted workspace so compiled kernels
# survive container restarts (avoids recompiling cuequivariance/boltz kernels).
ENV TRITON_CACHE_DIR="/workspace/.triton"

# Copy dependencies + venv
COPY --from=project-prod-deps --chown=appuser:appuser /app /app

# Copy project source
COPY --chown=appuser:appuser guild /app/guild

# Optional clones
RUN git clone https://github.com/schrojunzhang/KarmaDock.git /app/KarmaDock && \
    git -C /app/KarmaDock checkout 9a35d0cb7caaa1a4d0a61f6ea96821dc1edefa81 && \
    git clone https://github.com/gcorso/DiffDock.git /app/DiffDock && \
    git -C /app/DiffDock checkout 85c49b60d3e0b0182a59ee43a34a6d7036981284

# KarmaDock at the pinned commit was written for an older rdkit and crashes on
# the rdkit/torch combo we use today (dimension mismatch in ligand_feature.py
# and the GraphTransformer block). Apply the compatibility patches documented
# in the project README. The script is idempotent.
COPY --chown=appuser:appuser scripts/apply_karmadock_patches.py /tmp/apply_karmadock_patches.py
RUN /app/.venv/bin/python /tmp/apply_karmadock_patches.py /app/KarmaDock && \
    rm -f /tmp/apply_karmadock_patches.py