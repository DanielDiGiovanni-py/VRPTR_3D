#!/usr/bin/env bash
# Shared configuration for the public command-line workflows.
set -euo pipefail

require_value() {
  local name="$1"
  [[ -n "${!name:-}" ]] || { echo "Set $name; see configs/environment.example.sh." >&2; exit 2; }
}

require_file() {
  [[ -s "$1" ]] || { echo "Missing or empty file: $1" >&2; exit 2; }
}

configure_workspace() {
  require_value OUT_BASE
  require_value DATA_DIR
  require_value TRAIN_CONTROLS
  require_value TEST_CONTROLS
  [[ -d "$DATA_DIR" ]] || { echo "Data directory does not exist: $DATA_DIR" >&2; exit 2; }
  require_file "$TRAIN_CONTROLS"
  require_file "$TEST_CONTROLS"
  PYTHON_BIN="${PYTHON_BIN:-python}"
  RESULTS_ROOT="${RESULTS_ROOT:-$OUT_BASE/thesis_vrptr}"
  COMPARISONS="${COMPARISONS:-$OUT_BASE/comparisons}"
  CACHE_DIR="${CACHE_DIR:-$OUT_BASE/image_cache}"
  TRAINING_TEMPLATE="${TRAINING_TEMPLATE:-$OUT_BASE/training_task_template.npz}"
  FINGERPRINT_PATTERN="${FINGERPRINT_PATTERN:-output/rest_connectivity_100/fingerprint_100.nii.gz}"
  BETA_PATTERN="${BETA_PATTERN:-output/task_betas/beta_story_vs_math.nii.gz}"
  DEVICE="${DEVICE:-cuda}"
  AMP="${AMP:-fp16}"
  SEED="${SEED:-42}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
}

checkpoint_for() {
  printf '%s/checkpoints/%s/final_thesis_vrptr_%s.pt\n' "$RESULTS_ROOT" "$1" "$1"
}
