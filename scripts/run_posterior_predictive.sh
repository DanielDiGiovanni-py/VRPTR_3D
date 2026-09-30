#!/usr/bin/env bash
# Posterior sampling, input corruption, and participant-fold calibration.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
After sourcing the environment, set OUT_BASE to the evaluation root.
POSTERIOR_MODE=core (default), clean, corruption, or compile.
Core runs clean sampling (128 draws), corruption (48 draws), then compilation.
EOF
  exit 0
fi
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
configure_workspace
if [[ "${POSTERIOR_MODE:-core}" == "core" || "${POSTERIOR_MODE:-core}" == "corruption" ]]; then
  # The manuscript's 44-test corruption family includes deterministic comparisons.
  require_file "$(checkpoint_for deterministic)"
fi

sample_posterior() {
  local analysis="$1" samples="$2" voxels="$3"
  require_file "$(checkpoint_for full)"
  local extra=()
  if [[ "$analysis" == "clean" ]]; then
    [[ "${WRITE_NIFTI:-0}" != "1" ]] || extra+=(--write-nifti --nifti-max-subjects "${NIFTI_MAX_SUBJECTS:-5}")
  else
    local noise dropout
    read -r -a noise <<< "${NOISE_LEVELS:-0.25 0.50 1.00}"
    read -r -a dropout <<< "${CHANNEL_DROPOUT_LEVELS:-0.10 0.25 0.50}"
    extra+=(--noise-levels "${noise[@]}" --channel-dropout-levels "${dropout[@]}")
  fi
  "$PYTHON_BIN" -m vrptr_clinical.posterior_predictive_analysis \
    --analysis "$analysis" --results-root "$RESULTS_ROOT" --subjects "$TEST_CONTROLS" \
    --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" --samples "$samples" \
    --sample-batch "${POSTERIOR_SAMPLE_BATCH:-2}" --max-voxels "$voxels" \
    --max-subjects "${MAX_SUBJECTS:-0}" --device "$DEVICE" --amp "$AMP" --seed "$SEED" "${extra[@]}"
}

case "${POSTERIOR_MODE:-core}" in
  core)
    sample_posterior clean "${POSTERIOR_SAMPLES:-128}" "${POSTERIOR_MAX_VOXELS:-25000}"
    sample_posterior corruption "${CORRUPTION_SAMPLES:-48}" "${CORRUPTION_MAX_VOXELS:-10000}"
    ;;
  clean) sample_posterior clean "${POSTERIOR_SAMPLES:-128}" "${POSTERIOR_MAX_VOXELS:-25000}" ;;
  corruption) sample_posterior corruption "${CORRUPTION_SAMPLES:-48}" "${CORRUPTION_MAX_VOXELS:-10000}" ;;
  compile) ;;
  *) echo "Unknown POSTERIOR_MODE: $POSTERIOR_MODE" >&2; exit 2 ;;
esac
"$PYTHON_BIN" -m vrptr_clinical.compile_posterior_predictive \
  --results-root "$RESULTS_ROOT" --folds "${CALIBRATION_FOLDS:-5}" \
  --fit-voxels "${CALIBRATION_FIT_VOXELS:-500000}" --bootstrap "${BOOTSTRAP:-20000}" \
  --signflip "${SIGNFLIP:-20000}" --seed "$SEED"
