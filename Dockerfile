# syntax=docker/dockerfile:1
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

# Python build tools
RUN python -m pip install --upgrade pip setuptools wheel

# OpenBabel build
RUN git clone https://github.com/openbabel/openbabel.git && \
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
FROM base-build AS project-prod-deps

USER appuser
WORKDIR /app

COPY --chown=appuser:appuser pyproject.toml /app
COPY --chown=appuser:appuser uv.lock* /app
COPY --chown=appuser:appuser README.md /app

RUN uv sync --frozen --no-install-project --no-group dev && \
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
# Dev deps (black, ruff, pytest, ipykernel, pre-commit) are all public — no GH auth needed
RUN uv sync --frozen --no-install-project

####################################################################################################
FROM project-all-deps AS test
USER appuser
WORKDIR /app

COPY --chown=appuser:appuser guild /app/guild
COPY --chown=appuser:appuser tests /app/tests

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

# LocalColabFold
COPY --from=base-build /opt/localcolabfold /opt/localcolabfold
ENV COLABFOLD_BIN="/opt/localcolabfold/.pixi/envs/default/bin"

USER appuser
WORKDIR /app

ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:/app:${COLABFOLD_BIN}:${PATH}"
ENV PYTHONPATH="/app"
# Runtime environment
ENV HOME="/workspace"
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