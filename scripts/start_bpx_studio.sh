#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly PROJECT_PYTHON="${BPX_STUDIO_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

if [[ ! -x "${PROJECT_PYTHON}" ]]; then
  echo "[ERROR] Project Python not found: ${PROJECT_PYTHON}" >&2
  echo "Run the one-time installation while online: uv sync --frozen" >&2
  exit 2
fi

if ! "${PROJECT_PYTHON}" -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
  echo "[ERROR] BPX Studio dependencies are not installed in the project .venv." >&2
  echo "Run the one-time installation while online: uv sync --frozen" >&2
  echo "This launcher will never download or modify dependencies automatically." >&2
  exit 3
fi

cd "${PROJECT_ROOT}"
exec "${PROJECT_PYTHON}" -m openpi.bpx_studio.api "$@"
