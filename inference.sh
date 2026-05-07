#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash inference.sh <input_json_path>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-fast_layernorm}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-${SCRIPT_DIR}/release_data}"

INPUT_JSON_PATH="$1"
MODEL_NAME="${MODEL_NAME:-protenix_base_default_v0.5.0}"
TYPE="${TYPE:-default}"
SEEDS="${SEEDS:-${seed:-42}}"
DUMP_DIR="${DUMP_DIR:-${SCRIPT_DIR}/outputs/inference/${TYPE}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${LOAD_CHECKPOINT_DIR:-${PROTENIX_DATA_ROOT_DIR}/checkpoint}}"
DTYPE="${DTYPE:-bf16}"
USE_MSA="${USE_MSA:-true}"
N_SAMPLE="${N_SAMPLE:-5}"
N_STEP="${N_STEP:-200}"
N_CYCLE="${N_CYCLE:-10}"
TRIANGLE_ATTENTION="${TRIANGLE_ATTENTION:-triattention}"
TRIANGLE_MULTIPLICATIVE="${TRIANGLE_MULTIPLICATIVE:-cuequivariance}"
LOAD_STRICT="${LOAD_STRICT:-true}"

mkdir -p "${DUMP_DIR}"

cmd=(
  python3 "${SCRIPT_DIR}/runner/inference.py"
  --model_name "${MODEL_NAME}"
  --seeds "${SEEDS}"
  --dump_dir "${DUMP_DIR}"
  --input_json_path "${INPUT_JSON_PATH}"
  --dtype "${DTYPE}"
  --use_msa "${USE_MSA}"
  --model.N_cycle "${N_CYCLE}"
  --sample_diffusion.N_sample "${N_SAMPLE}"
  --sample_diffusion.N_step "${N_STEP}"
  --load_checkpoint_dir "${CHECKPOINT_DIR}"
  --load_strict "${LOAD_STRICT}"
  --triangle_attention "${TRIANGLE_ATTENTION}"
  --triangle_multiplicative "${TRIANGLE_MULTIPLICATIVE}"
)

if [[ "${TYPE}" == *glycan* ]]; then
  cmd+=(--model.glycan_feature "${GLYCAN_FEATURE:-true}")
fi

"${cmd[@]}"
