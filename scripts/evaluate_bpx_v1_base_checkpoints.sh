#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly PROJECT_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
readonly RUN_DIR="${BPX_EVAL_RUN:-/scratch/bpx_datasets/checkpoints/pi05_bpx_full_low_mem_v1_base_estimated/bpx_v2_humanoid_bs2_v001_new}"
readonly DATA_ROOT="${BPX_DATA_ROOT:-/scratch/bpx_datasets}"
readonly OUTPUT_ROOT="${BPX_EVAL_OUTPUT:-${DATA_ROOT}/eval_outputs/bpx_v1_base_20k/replay_$(date +%Y%m%d_%H%M%S)}"
episodes=("$@")
if [[ ${#episodes[@]} -eq 0 ]]; then episodes=(20 61); fi
for episode in "${episodes[@]}"; do
  if [[ ! "${episode}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Pass episode numbers only, e.g. bash scripts/evaluate_bpx_v1_base_checkpoints.sh 20 61" >&2
    exit 2
  fi
done

if [[ ! -x "${PROJECT_PYTHON}" ]]; then
  echo "[ERROR] Existing project .venv is required; this launcher never installs anything." >&2
  exit 2
fi
for step in 10000 18000 19999; do
  checkpoint="${RUN_DIR}/selected_checkpoints/${step}"
  [[ -d "${checkpoint}/params" ]] || checkpoint="${RUN_DIR}/${step}"
  if [[ ! -d "${checkpoint}/params" || ! -d "${checkpoint}/assets" ]]; then
    echo "[ERROR] Complete checkpoint missing at step ${step}: ${checkpoint}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "[ERROR] Output exists; use a new BPX_EVAL_OUTPUT (nothing will be overwritten): ${OUTPUT_ROOT}" >&2
  exit 2
fi
export HF_LEROBOT_HOME="${DATA_ROOT}/lerobot"
export HF_HOME="${DATA_ROOT}/cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TMPDIR="${DATA_ROOT}/tmp"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${BPX_EVAL_CUDA:-2,3}"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"
cd "${PROJECT_ROOT}"
echo "[BPX EVAL] Read-only dataset/checkpoints; outputs: ${OUTPUT_ROOT}"
for step in 10000 18000 19999; do
  checkpoint="${RUN_DIR}/selected_checkpoints/${step}"
  [[ -d "${checkpoint}/params" ]] || checkpoint="${RUN_DIR}/${step}"
  # Fresh process per model: no simultaneous residency of three full policies.
  "${PROJECT_PYTHON}" -u -m openpi.bpx_studio.cpu_affinity --socket 1 -- \
    "${PROJECT_PYTHON}" -u examples/bpx/evaluate_bpx_lerobot_checkpoint.py \
    --checkpoint-dir "${checkpoint}" --output-dir "${OUTPUT_ROOT}/${step}" \
    --dataset-root "${HF_LEROBOT_HOME}/bpx/bpx_v1_base_estimated_v001" \
    --validation-manifest "${DATA_ROOT}/reports/bpx_v1_base_recovery_v001/validation_split.json" \
    --episode-indices "${episodes[@]}"
done
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  "${PROJECT_PYTHON}" -u -m openpi.bpx_studio.cpu_affinity --socket 1 -- \
  "${PROJECT_PYTHON}" -u examples/bpx/compare_bpx_offline_evaluations.py --root "${OUTPUT_ROOT}"
echo "[BPX EVAL] DONE. Compare PNGs/CSV: ${OUTPUT_ROOT}/comparison"
