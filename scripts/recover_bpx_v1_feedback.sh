#!/usr/bin/env bash
# Offline, server-only recovery entry point. Never installs dependencies.
set -Eeuo pipefail

readonly BPX_RECOVERY_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly BPX_RECOVERY_PYTHON="${BPX_RECOVERY_PROJECT}/.venv/bin/python"

if [[ ! -x "${BPX_RECOVERY_PYTHON}" ]]; then
  echo "[ERROR] Run in the training server's existing OpenPI project (.venv/bin/python is missing)." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd -- "${BPX_RECOVERY_PROJECT}"
echo "[BOOT] BPX v1 experimental feedback estimation. No ROS, GPU model, or dependency downloads."
exec "${BPX_RECOVERY_PYTHON}" -u -m openpi.bpx_studio.recover_feedback "$@"
