#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Generate manuscript figures after template evaluation and posterior compilation.
Source the environment and set OUT_BASE to the evaluation root. Additional figure
arguments are forwarded; --skip-uncertainty generates the first four figures.
EOF
  exit 0
fi
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
configure_workspace
require_file "$TRAINING_TEMPLATE"
"$PYTHON_BIN" -m vrptr_clinical.make_vrptr_current_paper_figures \
  --root "$OUT_BASE" --analysis-root "$RESULTS_ROOT/final_analysis" \
  --test-subjects "$TEST_CONTROLS" --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
  --device "$DEVICE" --seed "$SEED" --training-template "$TRAINING_TEMPLATE" "$@"
