#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BPX_CPU_SOCKET=1
exec bash "${SCRIPT_DIR}/start_bpx_studio.sh" "$@"
