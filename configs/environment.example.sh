# Copy to configs/environment.local.sh, edit the four required paths, then run:
#   source configs/environment.local.sh
# Paths may contain spaces. Do not commit your local copy or participant lists.
export DATA_DIR="/absolute/path/to/preprocessed_subjects"
export TRAIN_CONTROLS="/absolute/path/to/train_subjects.txt"
export TEST_CONTROLS="/absolute/path/to/test_subjects.txt"
export OUT_BASE="/absolute/path/to/vrptr_results"

# Derived defaults: OUT_BASE/thesis_vrptr, OUT_BASE/comparisons,
# OUT_BASE/image_cache, and OUT_BASE/training_task_template.npz.
# Uncomment only if a different location is needed:
# export RESULTS_ROOT="/absolute/path/to/vrptr_checkpoints_and_results"
# export COMPARISONS="/absolute/path/to/comparator_checkpoints_and_results"
# export CACHE_DIR="/absolute/path/to/image_cache"
# export TRAINING_TEMPLATE="/absolute/path/to/training_task_template.npz"

export DEVICE="cuda"
export AMP="fp16"
export SEED="42"

# Each process receives one subject per step. Set NPROC to the intended GPU count.
# The available training histories recorded 180 batches for 360 participants.
export NPROC="2"

# Paths relative to DATA_DIR/<subject>/; modify for your input layout.
export FINGERPRINT_PATTERN="output/rest_connectivity_100/fingerprint_100.nii.gz"
export BETA_PATTERN="output/task_betas/beta_story_vs_math.nii.gz"
