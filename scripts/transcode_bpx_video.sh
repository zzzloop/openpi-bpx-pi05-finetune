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

mkdir -p -- "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}"
cd -- "${BPX_PROJECT_ROOT}"
exec "${BPX_PYTHON}" examples/bpx/transcode_bpx_lerobot_images_to_video.py \
  --image-writer-threads 32 --video-encode-workers 3 "$@"
