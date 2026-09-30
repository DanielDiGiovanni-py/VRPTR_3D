#!/usr/bin/env bash
# Compact volumetric comparators: settings from the archived original launcher.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Train the compact volumetric BrainSurfCNN-like and SWIFUN-like comparators.
Source configs/environment.local.sh first. The saved training template is needed
for final evaluation. Set NPROC explicitly (default one process). Override recipe
settings with COMPARATOR_EPOCHS, COMPARATOR_LR, COMPARATOR_BATCH_SIZE, or CLI arguments.
The recipe follows the archived launcher; no original comparator run manifest or
checkpoint is distributed to verify any historical environment-variable overrides.
EOF
  exit 0
fi
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
configure_workspace
require_file "$TRAINING_TEMPLATE"
for name in brainsurfcnn_lite swifun_lite; do
  [[ ! -e "$COMPARISONS/$name/final_model.pt" ]] || { echo "Checkpoint already exists; choose a new COMPARISONS directory." >&2; exit 2; }
done
args=(
  --models brainsurfcnn_lite swifun_lite --data-dir "$DATA_DIR"
  --train-controls "$TRAIN_CONTROLS" --test-controls "$TEST_CONTROLS"
  --out-dir "$COMPARISONS" --cache-dir "$CACHE_DIR"
  --fingerprint-pattern "$FINGERPRINT_PATTERN" --beta-pattern "$BETA_PATTERN"
  --epochs "${COMPARATOR_EPOCHS:-100}" --lr "${COMPARATOR_LR:-2e-5}"
  --batch-size "${COMPARATOR_BATCH_SIZE:-1}" --no-val-split
  --weight-decay 1e-4 --map-only-epochs 50 --eval-every 20
  --surf-width 32 --surf-depth 4 --swifun-dim 32 --swifun-depth 4 --swifun-heads 4 --swifun-window 4
  --dropout 0.15 --head-init-std 1e-4 --grad-clip 1.0 --w-mse 1.0
  --w-contrastive 0.08 --contrastive-temperature 0.20 --contrastive-mse-weight 0.15
  --num-workers "${NUM_WORKERS:-2}" --prefetch-factor "${PREFETCH_FACTOR:-2}"
  --seed "$SEED" --device "$DEVICE" --training-template "$TRAINING_TEMPLATE"
)
nproc="${NPROC:-1}"
[[ "$nproc" =~ ^[1-9][0-9]*$ ]] || { echo "NPROC must be a positive integer." >&2; exit 2; }
if (( nproc > 1 )); then
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$nproc" -m vrptr_clinical.comparison_models "${args[@]}" "$@"
else
  "$PYTHON_BIN" -m vrptr_clinical.comparison_models "${args[@]}" "$@"
fi
# Distributed training defers evaluation; this pass also fixes final-checkpoint use.
"$PYTHON_BIN" -m vrptr_clinical.comparison_models --eval-only --checkpoint-kind final \
  --models brainsurfcnn_lite swifun_lite --eval-groups controls --data-dir "$DATA_DIR" \
  --train-controls "$TRAIN_CONTROLS" --test-controls "$TEST_CONTROLS" \
  --out-dir "$COMPARISONS" --cache-dir "$CACHE_DIR" --batch-size 1 --device "$DEVICE" \
  --fingerprint-pattern "$FINGERPRINT_PATTERN" --beta-pattern "$BETA_PATTERN" \
  --training-template "$TRAINING_TEMPLATE"
