#!/bin/bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input_fasta> <output_dir> [random_seed]"
  exit 1
fi

INPUT_FASTA="$1"
OUTPUTDIR="$2"
RANDOMSEED="${3:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOCALCOLABFOLD_DIR_CANDIDATES=()
if [[ -n "${GUILD_LOCALCOLABFOLD_DIR:-}" ]]; then
  LOCALCOLABFOLD_DIR_CANDIDATES+=("${GUILD_LOCALCOLABFOLD_DIR}")
fi
LOCALCOLABFOLD_DIR_CANDIDATES+=(
  "${SCRIPT_DIR}/../../localcolabfold"
  "${SCRIPT_DIR}/../localcolabfold"
  "${SCRIPT_DIR}/localcolabfold"
)

COLABFOLD_BATCH_BIN=""
for candidate in "${LOCALCOLABFOLD_DIR_CANDIDATES[@]}"; do
  candidate_abs="$(cd "${candidate}" 2>/dev/null && pwd || true)"
  if [[ -n "${candidate_abs}" && -x "${candidate_abs}/.pixi/envs/default/bin/colabfold_batch" ]]; then
    COLABFOLD_BATCH_BIN="${candidate_abs}/.pixi/envs/default/bin"
    break
  fi
done

if [[ -n "${COLABFOLD_BATCH_BIN}" ]]; then
  export PATH="${COLABFOLD_BATCH_BIN}:${PATH}"
fi

if ! command -v colabfold_batch >/dev/null 2>&1; then
  echo "Error: colabfold_batch not found. Set GUILD_LOCALCOLABFOLD_DIR to your localcolabfold folder."
  exit 1
fi

# Notebook kernels often export MPLBACKEND=module://matplotlib_inline.backend_inline,
# which is invalid inside the localcolabfold runtime. Use a safe headless backend.
export MPLBACKEND="Agg"

mkdir -p "${OUTPUTDIR}"

colabfold_batch \
  --msa-only \
  --random-seed "${RANDOMSEED}" \
  "${INPUT_FASTA}" \
  "${OUTPUTDIR}"
