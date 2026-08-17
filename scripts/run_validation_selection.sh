#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${PROJECT_ROOT}/runs/t1_main_seed20260730"
OUTPUT_ROOT="${PROJECT_ROOT}/results/new_protocol/validation_trajectory"

mkdir -p "${OUTPUT_ROOT}"

for step in $(seq 50000 5000 140000); do
  checkpoint="${RUN_DIR}/checkpoint_step_$(printf '%06d' "${step}").pth"
  output_dir="${OUTPUT_ROOT}/step_$(printf '%06d' "${step}")"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite: ${output_dir}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate.py" \
    --config "${PROJECT_ROOT}/configs/train_t1_main.yaml" \
    --checkpoint "${checkpoint}" \
    --split validation \
    --output-dir "${output_dir}" \
    --batch-size 1 \
    --num-workers 4 \
    --expected-subjects 107 \
    --expected-pairs 2140
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/select_checkpoint.py" \
  --evaluation-root "${OUTPUT_ROOT}" \
  --primary-metric psnr \
  --plateau-db 0.01 \
  --output "${PROJECT_ROOT}/results/new_protocol/selected_checkpoint_global_psnr.json"
