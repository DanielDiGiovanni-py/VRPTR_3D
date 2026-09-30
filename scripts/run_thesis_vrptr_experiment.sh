#!/usr/bin/env bash
# Train the four matched variants using the settings preserved in the run manifests.
set -euo pipefail
MODE="${MODE:-help}"
if [[ "$MODE" == "help" || "$MODE" == "--help" ]]; then
  cat <<'EOF'
Usage after sourcing configs/environment.local.sh:
  MODE=train VARIANT=full bash scripts/run_thesis_vrptr_experiment.sh
  MODE=train-all bash scripts/run_thesis_vrptr_experiment.sh
  MODE=eval bash scripts/run_thesis_vrptr_experiment.sh
  MODE=compile bash scripts/run_thesis_vrptr_experiment.sh

Training defaults match the archived VRPTR manifests. Evaluation requires the
saved training-only template. Comparators have a separate training workflow.
Set NPROC explicitly for distributed training (default: one process).
EOF
  exit 0
fi
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
configure_workspace
read -r -a variants <<< "${VARIANTS:-full deterministic no_transformer compression_only}"
for variant in "${variants[@]}" "${VARIANT:-full}"; do
  case "$variant" in full|deterministic|no_transformer|compression_only) ;; *) echo "Unknown variant: $variant" >&2; exit 2;; esac
done

train_variant() {
  local variant="$1"
  local checkpoint
  checkpoint="$(checkpoint_for "$variant")"
  [[ ! -e "$checkpoint" ]] || { echo "Checkpoint already exists; choose a new RESULTS_ROOT: $checkpoint" >&2; exit 2; }
  local args=(
    --data-dir "$DATA_DIR" --train-subjects "$TRAIN_CONTROLS"
    --out-dir "$RESULTS_ROOT/checkpoints/$variant" --cache-dir "$CACHE_DIR"
    --fingerprint-pattern "$FINGERPRINT_PATTERN" --beta-pattern "$BETA_PATTERN"
    --n-rois "${N_ROIS:-100}" --variant "$variant" --seed "$SEED"
    --epochs "${EPOCHS:-100}" --batch-size "${BATCH_SIZE:-1}"
    --lr "${LR:-2e-5}" --weight-decay "${WEIGHT_DECAY:-1e-4}"
    --width "${WIDTH:-32}" --depth "${DEPTH:-4}" --dropout "${DROPOUT:-0.15}"
    --latent-spatial-dim "${LATENT_SPATIAL_DIM:-32}" --latent-dim "${LATENT_DIM:-32}"
    --transformer-layers "${TRANSFORMER_LAYERS:-6}" --transformer-heads "${TRANSFORMER_HEADS:-8}"
    --transformer-mlp-ratio "${TRANSFORMER_MLP_RATIO:-2.0}" --head-init-std "${HEAD_INIT_STD:-1e-4}"
    --deterministic-warmup-epochs "${DETERMINISTIC_WARMUP_EPOCHS:-20}"
    --kl-warmup-epochs "${KL_WARMUP_EPOCHS:-30}" --noise-warmup-epochs "${NOISE_WARMUP_EPOCHS:-10}"
    --initial-noise-scale "${INITIAL_NOISE_SCALE:-0.10}"
    --w-kl "${W_KL:-1e-4}" --w-reconstruct "${W_RECONSTRUCT:-1.0}" --w-corr "${W_CORR:-0.0}"
    --w-contrastive "${W_CONTRASTIVE:-0.08}" --contrastive-temperature "${CONTRASTIVE_TEMPERATURE:-0.20}"
    --contrastive-after-epoch "${CONTRASTIVE_AFTER_EPOCH:-51}" --mask-fc-prob "${MASK_FC_PROB:-0.0}"
    --grad-clip "${GRAD_CLIP:-1.0}" --num-workers "${NUM_WORKERS:-2}"
    --prefetch-factor "${PREFETCH_FACTOR:-2}" --device "$DEVICE" --amp "$AMP"
    --amp-init-scale "${AMP_INIT_SCALE:-1024}" --stem-input-chunk-size "${STEM_INPUT_CHUNK_SIZE:-20}"
  )
  [[ "${ACTIVATION_CHECKPOINTING:-1}" == "1" ]] || args+=(--disable-activation-checkpointing)
  [[ "${REFRESH_CACHE:-0}" != "1" ]] || args+=(--refresh-cache)
  [[ -z "${RESUME:-}" ]] || { require_file "$RESUME"; args+=(--resume "$RESUME"); }
  local nproc="${NPROC:-1}"
  [[ "$nproc" =~ ^[1-9][0-9]*$ ]] || { echo "NPROC must be a positive integer." >&2; exit 2; }
  if (( nproc > 1 )); then
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$nproc" -m vrptr_clinical.train_control "${args[@]}"
  else
    "$PYTHON_BIN" -m vrptr_clinical.train_control "${args[@]}"
  fi
}

evaluate_variant() {
  local variant="$1"
  local checkpoint
  checkpoint="$(checkpoint_for "$variant")"
  require_file "$checkpoint"
  require_file "$TRAINING_TEMPLATE"
  local extra=()
  [[ "${WRITE_NIFTI:-0}" != "1" ]] || extra+=(--write-nifti)
  "$PYTHON_BIN" -m vrptr_clinical.predict_clinical \
    --checkpoint "$checkpoint" --subjects "$TEST_CONTROLS" --data-dir "$DATA_DIR" \
    --group controls --model-name "thesis_vrptr_${variant}" --out-dir "$RESULTS_ROOT/predictions" \
    --cache-dir "$CACHE_DIR" --batch-size 1 --device "$DEVICE" --save-latents \
    --training-template "$TRAINING_TEMPLATE" "${extra[@]}"
}

case "$MODE" in
  train) train_variant "${VARIANT:-full}" ;;
  train-all|ablation) for variant in "${variants[@]}"; do train_variant "$variant"; done ;;
  eval) for variant in "${variants[@]}"; do evaluate_variant "$variant"; done ;;
  compile)
    comparator_csv="${COMPARATOR_PER_SUBJECT:-$COMPARISONS/comparison_per_subject_metrics.csv}"
    require_file "$comparator_csv"
    "$PYTHON_BIN" -m vrptr_clinical.compile_thesis_results \
      --predictions-root "$RESULTS_ROOT/predictions" --comparator-per-subject "$comparator_csv" \
      --out-dir "$RESULTS_ROOT/compiled" --variants "${variants[@]}" --seed "$SEED"
    ;;
  *) echo "Unknown MODE: $MODE; use MODE=help." >&2; exit 2 ;;
esac
