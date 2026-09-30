# Reproducibility reference

## Source and scope

The primary source is the **18 September 2026 Supplementary Software 1** snapshot, including the training-template revision. Archived VRPTR manifests and training histories verify the four variant configurations. The earlier source bundle, prepared in July 2026, supplies the original comparator launcher and an environment snapshot. These sources serve different purposes: a launcher states a recipe; a run manifest records the arguments saved during a run.

This release removes unrelated experimental branches, unused helpers, machine-specific paths, and one-off packaging utilities while retaining the model and checkpoint interfaces used by the manuscript workflow. It preserves internal `thesis_vrptr_*` and `*_lite` identifiers for compatibility. The public descriptions use “BrainSurfCNN-like” and “SWIFUN-like.” No new model results are implied by this cleanup.

No imaging data, participant lists, checkpoints, or subject-level results are distributed. The historical participant-selection and manual quality-control records were not available for reconstruction. The included preprocessing code describes its operations; it does not establish which optional historical commands or manual decisions were used for every participant. See [../preprocessing/README.md](../preprocessing/README.md) for that workflow.

## Verified VRPTR configuration

The four run manifests agree on the settings below. The shell wrapper specifies them explicitly because bare Python defaults in a research module may differ from the settings used in the archived experiment.

| Setting | Value |
|---|---|
| Training epochs; seed | 100; 42 |
| Optimizer | AdamW |
| Learning rate; weight decay | `2e-5`; `1e-4` |
| Learning-rate schedule | Constant |
| Batch | One participant per process; histories record 180 steps per epoch for 360 participants, consistent with two processes |
| Convolutional width; depth | 32; 4 |
| Dropout | 0.15 |
| Spatial compression; latent dimension | 32; 32 |
| Transformer layers; heads | 6; 8 in each Transformer stage |
| Feedforward expansion | 2 |
| Reconstruction weight | 1.0, smooth L1 |
| Correlation-loss weight | 0.0 |
| Contrastive weight; temperature; start | 0.08; 0.20; epoch 51 |
| KL weight | `1e-4` |
| Deterministic warmup | 20 epochs |
| KL warmup; latent-noise warmup | 30; 10 epochs after the deterministic phase |
| Initial latent-noise scale | 0.10 |
| Gradient clipping | 1.0 |
| Output-head initialization | Normal weights with SD `1e-4`; zero bias |
| Numeric mode | FP16 automatic mixed precision, initial loss scale 1024 |
| Memory settings | Activation checkpointing; first convolution chunked in 20-channel blocks |
| Random fingerprint masking | Disabled |
| Checkpoint used for evaluation | Final epoch |

| Variant | Transformer | Variational sampling |
|---|---|---|
| `full` | Yes | Yes |
| `deterministic` | Yes | No |
| `no_transformer` | No | Yes |
| `compression_only` | No | No |

The contrastive objective uses up to 8,192 evenly spaced flattened positions per map, centers each sampled vector, normalizes its Euclidean norm, and compares participant-matched maps within the distributed batch. Reconstruction uses all valid voxels. A one-process batch of one changes the contrastive setting; the public wrappers therefore expose process count rather than silently choosing it.

The training-only template averages targets after each participant's within-mask standardization. The resulting mean is not standardized again. Primary template metrics must not be recomputed from a test-cohort mean.

## Comparator recipe

The original archived launcher specifies 100 epochs, 50 initial map-only epochs, AdamW learning rate `2e-5`, weight decay `1e-4`, no validation split, width 32, depth 4, and a SWIFUN-like window size of 4. Its distributed defaults yield global batch size two with one participant per process on two GPUs. The implementation supplies four SWIFUN-like attention heads, dropout 0.15, gradient clipping 1.0, contrastive weight 0.08, and temperature 0.20. Comparator training uses FP32.

The public recipe follows that launcher rather than the module's bare learning-rate default. No original comparator checkpoint or run manifest was supplied to establish whether environment variables overrode the launcher for the reported fits. The recipe is transparent about this remaining distinction; the available source is not a substitute for missing run metadata.

## Posterior analysis and inference

Clean-input posterior analysis uses 128 draws and at most 25,000 voxels per participant. Corruption uses 48 draws and at most 10,000 voxels. Gaussian-noise levels are 0.25, 0.50, and 1.00; channel-dropout proportions are 0.10, 0.25, and 0.50. Noise is applied to standardized fingerprint values, and channel dropout sets complete standardized channels to zero. Corrupted inputs are clipped to the interval −8 to 8. Each participant's noise field or channel ordering is reused across severities.

Calibration uses five participant folds, up to 500,000 fitting voxels, and seed 42. Bootstrap and sign-flip calculations use 20,000 draws. Calibration stays within the 40-participant evaluation cohort; it does not establish performance in an independent population.

Benjamini–Hochberg adjustment is applied to the following separate families. Counts include evaluated outcomes beyond those emphasized in the manuscript.

| Family | Scope | Tests |
|---|---|---:|
| Factorial: prediction | Transformer, variational, and interaction effects × Pearson r, Spearman r, R², Dice AUC, Dice at 25%, absolute LI error | 18 |
| Factorial: individualized signal | The same three effects × template-correlation gain, residual Pearson and Spearman correlations, residual Dice AUC and Dice at 25%, LI-error improvement | 18 |
| Factorial: residual matching | The same three effects × best-other margin, mean-other margin, rank, percentile | 12 |
| Factorial: whole-map matching | The same three effects × best-other margin, mean-other margin, rank, percentile | 12 |
| Direct full-VRPTR comparisons | Five other models × whole-map correlation, template gain, residual correlation, residual Dice AUC, two residual matching margins, whole-map best-other margin, LI error | 40 |
| Full-VRPTR component swaps | Four recipient/donor swap contrasts, plus mean-bottleneck and mean-skip replacements each tested for Pearson r, Dice AUC, Dice at 25%, and residual Pearson r | 12 |
| Clean posterior utility and coverage | Error/uncertainty associations, selective-error contrasts, and raw interval coverage | 10 |
| Cross-fitted calibration | Calibrated probabilistic-score and coverage contrasts | 7 |
| Input corruption | All degradation comparisons and severity associations together | 44 |

Unadjusted probabilities are labeled `p`; adjusted values are labeled `q`. The code and output CSVs retain each tested contrast. Factorial simple effects are reported separately from the adjusted main-effect/interaction families.

### Complete component-swap and posterior family membership

The **12 component-swap tests** comprise four skip-swap versus bottleneck-swap contrasts—whole-map identity shift toward the donor, residual identity shift toward the donor, recipient Pearson-correlation loss, and RMSE change from the recipient's baseline prediction—plus baseline minus mean-bottleneck replacement and baseline minus mean-skip replacement, each tested for Pearson correlation, Dice AUC, Dice at 25%, and residual Pearson correlation (4 + 2 × 4).

The **10 clean-posterior tests** use these exact output contrast identifiers:

| Contrast | Quantity |
|---|---|
| `posterior_mean_minus_mu_pearson` | Posterior-average minus latent-mean Pearson correlation |
| `posterior_mean_minus_mu_r2` | Posterior-average minus latent-mean R² |
| `posterior_mean_minus_mu_dice_auc` | Posterior-average minus latent-mean Dice AUC |
| `mu_minus_posterior_mean_li_abs_error` | Latent-mean minus posterior-average absolute LI error |
| `voxel_uncertainty_error_spearman` | Within-participant Spearman association between posterior SD and absolute voxel error |
| `selective_gain_fraction` | Fractional error reduction from withholding uncertain voxels |
| `raw_coverage_error_50` | Signed observed minus nominal 50% coverage |
| `raw_coverage_error_80` | Signed observed minus nominal 80% coverage |
| `raw_coverage_error_90` | Signed observed minus nominal 90% coverage |
| `raw_coverage_error_95` | Signed observed minus nominal 95% coverage |

The **7 cross-fitted calibration tests** are `constant_minus_heteroscedastic_nll`, `constant_minus_heteroscedastic_crps`, `raw_minus_heteroscedastic_nll`, `raw_minus_heteroscedastic_crps`, and `constant_minus_heteroscedastic_abs_coverage_error_80`, `_90`, and `_95`. NLL is negative log-likelihood; CRPS is the continuous ranked probability score. Constant-width and heteroscedastic intervals are fitted within the same participant-fold scheme.

The **44 input-corruption tests** combine 36 condition-specific tests and eight severity tests into one adjustment family:

| Repeated over | Contrasts in each repetition | Number |
|---|---|---:|
| Each Gaussian-noise level (0.25, 0.50, 1.00) and channel-dropout proportion (0.10, 0.25, 0.50) | `posterior_sd_increase`, `posterior_mean_pearson_loss`, `mu_pearson_loss`, `dominance_entropy_increase`, `li_posterior_sd_increase`, `deterministic_minus_vae_performance_loss` | 6 × 6 = 36 |
| Each corruption family (Gaussian noise; channel dropout), including its clean condition when fitting severity relationships | `uncertainty_slope`, `error_slope`, `dominance_entropy_slope`, `uncertainty_error_spearman_across_severity` | 2 × 4 = 8 |

The 44-test family requires both full-VRPTR and deterministic-VRPTR corruption outputs. The public corruption wrapper therefore requires the deterministic checkpoint before sampling. Changing severity levels or omitting deterministic comparisons changes family membership and cannot be described as the same 44-test analysis. `BOOTSTRAP` and `SIGNFLIP` override the public statistical wrappers' default 20,000 draws; `SEED` overrides their default 42.

## Available environment record

An earlier source bundle contains a July 2026 snapshot with Python 3.10.18, PyTorch 2.6.0+cu124, NumPy 2.2.5, SciPy 1.15.3, pandas 2.2.3, NiBabel 5.3.2, Nilearn 0.11.1, and two NVIDIA TITAN Xp GPUs with 12 GB each. This establishes that environment snapshot, not an immutable record of every later training or analysis execution. The exact environment of the September runs was not supplied.

The install metadata therefore lists dependencies without claiming that unverified version pins reproduce every original result. Record your own package versions, GPU model, driver, CUDA runtime, process count, command-line arguments, and source checksum alongside new results. Preserve the final checkpoints, training manifests, and participant split outside any public package.
