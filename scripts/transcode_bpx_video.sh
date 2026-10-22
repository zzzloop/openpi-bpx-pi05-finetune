#!/usr/bin/env bash
# Uses only the installed server environment; never installs dependencies.
set -Eeuo pipefail

readonly BPX_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly BPX_PYTHON="${BPX_PROJECT_ROOT}/.venv/bin/python"
readonly BPX_STORAGE_ROOT="${BPX_DATA_ROOT:-/scratch/bpx_datasets}"

if [[ ! -x "${BPX_PYTHON}" ]]; then
  echo "[ERROR] Project Python not found: ${BPX_PYTHON}" >&2
  echo "Run this launcher in the server's existing OpenPI project." >&2
  exit 2
fi

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${BPX_STORAGE_ROOT}/lerobot}"
export HF_HOME="${HF_HOME:-${BPX_STORAGE_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${BPX_STORAGE_ROOT}/cache/huggingface/datasets}"
export TMPDIR="${TMPDIR:-${BPX_STORAGE_ROOT}/tmp}"
export PYTHONUNBUFFERED=1
# Episode workers do their own bounded decoding/encoding; avoid nested BLAS pools.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p -- "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}"
cd -- "${BPX_PROJECT_ROOT}"
echo "[BOOT] Offline streaming converter; Python: ${BPX_PYTHON}"
echo "[BOOT] LeRobot data root: ${HF_LEROBOT_HOME}; no Hugging Face dataset initialization"
exec "${BPX_PYTHON}" -u -m openpi.bpx_studio.video_transcoder "$@"
