# VRPTR

Code accompanying **“VRPTR prediction of individual language activation and uncertainty from resting state fMRI”**, by Daniel A. Di Giovanni and D. Louis Collins.

VRPTR predicts an individual language activation map from resting-state connectivity fingerprints. This release contains the volumetric model, four matched variants, two compact volumetric comparators, training-template evaluation, component swaps, posterior analyses, and manuscript figures. It also includes the preprocessing scripts for constructing the inputs.

The manuscript evaluates 360 training participants and 40 held-out participants from the Human Connectome Project. Images, participant lists, trained checkpoints, and subject-level results are not included. You must supply authorized inputs and your own paths. This is a cleaned research-code release, not an independently reproduced set of manuscript results.

## Install

Use Python 3.10 or newer and a PyTorch build appropriate for your hardware. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For raw-data preprocessing, install `python -m pip install -e '.[preprocessing]'` and configure FSL; see [preprocessing/README.md](preprocessing/README.md). Installation requirements are unpinned. The available historical environment and its limits are recorded in [docs/reproducibility.md](docs/reproducibility.md).

## Configure inputs

```bash
cp configs/environment.example.sh configs/environment.local.sh
# Edit the required paths and GPU count in your local copy.
source configs/environment.local.sh
```

Set `DATA_DIR`, `TRAIN_CONTROLS`, `TEST_CONTROLS`, and `OUT_BASE`. The subject lists contain one identifier per line, matching subject directories below `DATA_DIR`. Each subject needs a 100-channel fingerprint NIfTI and a task-contrast beta NIfTI. The configurable file patterns are relative to each subject directory. Inputs must share anatomical space; grid resampling alone does not perform registration. Evaluation requires the observed task map to calculate accuracy and define the analysis mask.

`NPROC=2` and a per-process batch size of one give the global batch size of two used by the VRPTR training runs. A single-process run is supported, but changes the contrastive training batch. Choose `DEVICE=cpu` and `AMP=none` for CPU-only inspection; full-volume training is computationally demanding.

## Run the analysis

Run these commands from this directory. Each shell workflow provides `--help` (the VRPTR runner uses `MODE=help`). All paths are explicit; no original workstation paths are assumed.

**1. Train the four VRPTR variants.** The wrapper preserves the hyperparameters in the archived training manifests, including 100 epochs and a learning rate of `2e-5`.

```bash
MODE=train-all bash scripts/run_thesis_vrptr_experiment.sh
```

For one variant, use `MODE=train VARIANT=full`. `VARIANTS` accepts a space-separated subset. The variants are `full`, `deterministic`, `no_transformer`, and `compression_only`; see the [configuration reference](docs/reproducibility.md). Training refuses to overwrite a completed final checkpoint.

**2. Build the training-only task template.** The template averages the standardized training targets and verifies that the training and evaluation participant lists are disjoint.

```bash
python -m vrptr_clinical.training_template \
  --train-subjects "$TRAIN_CONTROLS" --test-subjects "$TEST_CONTROLS" \
  --checkpoint "$OUT_BASE/thesis_vrptr/checkpoints/full/final_thesis_vrptr_full.pt" \
  --data-dir "$DATA_DIR" --fingerprint-pattern "$FINGERPRINT_PATTERN" \
  --beta-pattern "$BETA_PATTERN" --expected-train-count 360 \
  --out "$OUT_BASE/training_task_template.npz"
```

Change `--expected-train-count` for a different cohort. This is an explicit count check, not a participant-selection procedure.

**3. Train the comparators, if you do not already have their checkpoints.** This separate recipe follows the original archived launcher: 100 epochs, learning rate `2e-5`, and no internal validation split. The provenance of these settings differs from the verified VRPTR run manifests; see the reproducibility notes.

```bash
bash scripts/run_comparator_training.sh
```

The implementations are **BrainSurfCNN-like** and **SWIFUN-like** volumetric adaptations. They are not the authors' original surface-model implementations. Internal names `brainsurfcnn_lite` and `swifun_lite` remain unchanged so saved configurations and checkpoint filenames stay compatible.

**4. Evaluate fixed checkpoints in a new directory.** This constructs a new training-only template, evaluates all six models, compiles individualized prediction and factorial results, and runs full-VRPTR component swaps. It does not retrain the models.

```bash
export UPDATE_ROOT="$OUT_BASE/evaluation_training_template"
bash scripts/run_training_template_revision.sh
```

The new directory contains symbolic links to the source checkpoints. Keep their original locations available. Existing output directories are rejected to prevent mixing result generations. `LOO_SENSITIVITY=1` additionally writes the test leave-one-out template sensitivity outputs; the primary reference remains the training mean.

**5. Sample the posterior and generate figures.** Point the remaining workflows to the evaluation directory. Clear optional derived-path overrides if you previously set them.

```bash
export OUT_BASE="$UPDATE_ROOT"
unset RESULTS_ROOT COMPARISONS TRAINING_TEMPLATE
POSTERIOR_MODE=core bash scripts/run_posterior_predictive.sh
bash scripts/make_vrptr_main_figures.sh
```

The posterior workflow uses 128 clean-input samples and 48 samples per corruption condition. Five participant folds calibrate uncertainty within the evaluation cohort; this is internal calibration. To generate figures without posterior results, pass `--skip-uncertainty` to the figure wrapper.

## Outputs and provenance

Training writes final checkpoints and run metadata under `thesis_vrptr/checkpoints/`. Evaluation writes per-subject metrics, individualized signal summaries, factorial and component-swap results under `thesis_vrptr/`. Posterior results are under `thesis_vrptr/final_analysis/posterior_predictive/`; figures are under `paper_figures_imaging_neuroscience/`.

The saved training template records a content hash and provenance. Evaluation and compilation check this metadata so that incompatible template references are not silently combined. Input scaling is learned on the training set and stored in checkpoints. Pass `--data-dir` to relocate checkpoint-based inference to another data root without changing the saved model or scaler.

[docs/reproducibility.md](docs/reproducibility.md) documents parameters, statistical families, source provenance, and remaining limits. [CITATION.cff](CITATION.cff) identifies the accompanying manuscript; no publication DOI is asserted.

Release checks and the regression-test command are documented in [VALIDATION.md](VALIDATION.md).
