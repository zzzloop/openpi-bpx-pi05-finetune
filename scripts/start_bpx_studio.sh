#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly PROJECT_PYTHON="${BPX_STUDIO_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
readonly WHEEL_DIR="${PROJECT_ROOT}/third_party/bpx_studio_wheels"

if [[ ! -x "${PROJECT_PYTHON}" ]]; then
  echo "[ERROR] Project Python not found: ${PROJECT_PYTHON}" >&2
  echo "Run the one-time installation while online: uv sync --frozen" >&2
  exit 2
fi

# Bind before even the web-runtime import check. The helper consumes this
# selector, re-execs this launcher, and children inherit its kernel affinity.
# No global CPU settings, existing jobs or training parameters change.
if [[ -n "${BPX_CPU_SOCKET:-}" ]]; then
  cd "${PROJECT_ROOT}"
  exec "${PROJECT_PYTHON}" -u -m openpi.bpx_studio.cpu_affinity \
    --socket "${BPX_CPU_SOCKET}" -- bash "${SCRIPT_DIR}/start_bpx_studio.sh" "$@"
fi

# BPX Studio's small, pure-Python web stack is committed as wheel files so the
# dashboard can start on an air-gapped server. Python can import directly from
# wheels (zip files); no installation, resolver, cache, or package index is used.
wheel_pythonpath=""
for wheel in "${WHEEL_DIR}"/*.whl; do
  [[ -f "${wheel}" ]] || continue
  if [[ -z "${wheel_pythonpath}" ]]; then
    wheel_pythonpath="${wheel}"
  else
    wheel_pythonpath="${wheel_pythonpath}:${wheel}"
  fi
done
if [[ -z "${wheel_pythonpath}" ]]; then
  echo "[ERROR] Vendored BPX Studio wheels not found: ${WHEEL_DIR}" >&2
  exit 3
fi
export PYTHONPATH="${wheel_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}"

if ! "${PROJECT_PYTHON}" -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
  echo "[ERROR] Vendored BPX Studio runtime failed its import check." >&2
  echo "Verify that Git LFS/content filters did not omit third_party/bpx_studio_wheels." >&2
  exit 4
fi

cd "${PROJECT_ROOT}"
exec "${PROJECT_PYTHON}" -m openpi.bpx_studio.api "$@"
