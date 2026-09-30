#!/usr/bin/env bash
# Re-evaluate fixed checkpoints with a training-only template in a new directory.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Source configs/environment.local.sh, then set UPDATE_ROOT to a new directory.
Uses RESULTS_ROOT/checkpoints and COMPARISONS as checkpoint sources; never trains.
Creates the template, evaluates six models, compiles matching/factorial results,
and runs full-VRPTR component swaps. Run posterior analysis and figures separately.
Set LOO_SENSITIVITY=1 to additionally write test leave-one-out sensitivity results.
EOF
  exit 0
fi
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
configure_workspace
stats_args=(--seed "$SEED" --bootstrap "${BOOTSTRAP:-20000}" --signflip "${SIGNFLIP:-20000}")
require_value UPDATE_ROOT
[[ ! -e "$UPDATE_ROOT" ]] || { echo "UPDATE_ROOT must be a new directory: $UPDATE_ROOT" >&2; exit 2; }
for variant in full deterministic no_transformer compression_only; do require_file "$(checkpoint_for "$variant")"; done
for name in brainsurfcnn_lite swifun_lite; do require_file "$COMPARISONS/$name/final_model.pt"; done
source_checkpoints="$(cd "$RESULTS_ROOT/checkpoints" && pwd)"
source_comparisons="$(cd "$COMPARISONS" && pwd)"
mkdir -p "$UPDATE_ROOT/thesis_vrptr" "$UPDATE_ROOT/comparisons"
UPDATE_ROOT="$(cd "$UPDATE_ROOT" && pwd)"
RESULTS_ROOT="$UPDATE_ROOT/thesis_vrptr"
COMPARISONS="$UPDATE_ROOT/comparisons"
ln -s "$source_checkpoints" "$RESULTS_ROOT/checkpoints"
for name in brainsurfcnn_lite swifun_lite; do
  mkdir -p "$COMPARISONS/$name"
  ln -s "$source_comparisons/$name/final_model.pt" "$COMPARISONS/$name/final_model.pt"
done
TRAINING_TEMPLATE="$UPDATE_ROOT/training_task_template.npz"
"$PYTHON_BIN" -m vrptr_clinical.training_template \
  --train-subjects "$TRAIN_CONTROLS" --test-subjects "$TEST_CONTROLS" \
  --checkpoint "$(checkpoint_for full)" --data-dir "$DATA_DIR" \
  --fingerprint-pattern "$FINGERPRINT_PATTERN" --beta-pattern "$BETA_PATTERN" \
  --expected-train-count "${EXPECTED_TRAIN_COUNT:-360}" --out "$TRAINING_TEMPLATE"
extra=()
[[ "${LOO_SENSITIVITY:-0}" != "1" ]] || extra+=(--test-loo-sensitivity)
for variant in full deterministic no_transformer compression_only; do
  "$PYTHON_BIN" -m vrptr_clinical.predict_clinical \
    --checkpoint "$(checkpoint_for "$variant")" --subjects "$TEST_CONTROLS" --data-dir "$DATA_DIR" \
    --group controls --model-name "thesis_vrptr_${variant}" --out-dir "$RESULTS_ROOT/predictions" \
    --cache-dir "$CACHE_DIR" --batch-size 1 --device "$DEVICE" --save-latents \
    --training-template "$TRAINING_TEMPLATE" "${extra[@]}"
done
"$PYTHON_BIN" -m vrptr_clinical.comparison_models --eval-only --checkpoint-kind final \
  --models brainsurfcnn_lite swifun_lite --eval-groups controls --data-dir "$DATA_DIR" \
  --train-controls "$TRAIN_CONTROLS" --test-controls "$TEST_CONTROLS" \
  --fingerprint-pattern "$FINGERPRINT_PATTERN" --beta-pattern "$BETA_PATTERN" \
  --out-dir "$COMPARISONS" --cache-dir "$CACHE_DIR" --batch-size 1 --device "$DEVICE" \
  --training-template "$TRAINING_TEMPLATE" "${extra[@]}"
"$PYTHON_BIN" -m vrptr_clinical.compile_thesis_results \
  --predictions-root "$RESULTS_ROOT/predictions" \
  --comparator-per-subject "$COMPARISONS/comparison_per_subject_metrics.csv" \
  --out-dir "$RESULTS_ROOT/compiled" --seed "$SEED"
"$PYTHON_BIN" -m vrptr_clinical.compile_individualization --results-root "$RESULTS_ROOT" --comparisons-dir "$COMPARISONS" "${stats_args[@]}"
"$PYTHON_BIN" -m vrptr_clinical.final_factorial_analysis --results-root "$RESULTS_ROOT" "${stats_args[@]}"
"$PYTHON_BIN" -m vrptr_clinical.skip_bottleneck_swap_analysis \
  --checkpoint "$(checkpoint_for full)" --subjects "$TEST_CONTROLS" --data-dir "$DATA_DIR" \
  --out-dir "$RESULTS_ROOT/final_analysis/skip_bottleneck_swap/full" --cache-dir "$CACHE_DIR" \
  --device "$DEVICE" --amp "$AMP" --training-template "$TRAINING_TEMPLATE" "${stats_args[@]}"
"$PYTHON_BIN" -m vrptr_clinical.compile_skip_bottleneck_swap \
  --analysis-root "$RESULTS_ROOT/final_analysis/skip_bottleneck_swap" --variants full "${stats_args[@]}"
printf 'Evaluation written to %s\n' "$UPDATE_ROOT"
