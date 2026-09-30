from __future__ import annotations

"""Evaluation-only skip-versus-bottleneck swap analysis for thesis VRPTR.

For held-out subjects i (recipient) and j (donor), this program constructs:

  * donor bottleneck + recipient skips
  * recipient bottleneck + donor skips

All ordered donor-recipient pairs are evaluated, but statistical inference is
performed after averaging over donors within each recipient (n = subjects), so
1560 pairwise hybrids are not incorrectly treated as independent observations.

The analysis never changes model weights and never retrains a checkpoint.
"""

import argparse
from .training_template import add_template_argument, load_for_evaluation, metric_provenance, require_complete_subjects
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import (
    ClinicalPaths,
    ImageSubjectDataset,
    build_subject_matrix_cached,
    create_image_space,
    load_subject_list,
    write_image_nifti,
)
from .final_stats_common import bh_adjust, paired_stats
from .metrics import pearsonr_np, subject_metrics
from .predict_clinical import load_model_checkpoint, prepare_matrix


PAIR_CONDITIONS = (
    "donor_bottleneck_recipient_skips",
    "recipient_bottleneck_donor_skips",
)
SUBJECT_CONDITIONS = (
    "correct_bottleneck_correct_skips",
    "cohort_mean_bottleneck_recipient_skips",
    "recipient_bottleneck_cohort_mean_skips",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Swap held-out VRPTR bottlenecks and U-Net skips without retraining."
    )
    add_template_argument(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default=None, help="Override the input data root stored in the checkpoint; leaves the checkpoint unchanged.")
    p.add_argument("--subjects", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-dir", default="")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--group", default="controls")
    p.add_argument("--model-name", default="thesis_vrptr_full")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--signflip", type=int, default=20000)
    p.add_argument(
        "--max-donors-per-recipient",
        type=int,
        default=0,
        help="0 uses every other held-out subject; positive values sample this many donors per recipient.",
    )
    p.add_argument(
        "--component-cache-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="CPU dtype for cached bottlenecks; skip maps are streamed and are not stored per subject.",
    )
    p.add_argument("--write-nifti", action="store_true")
    p.add_argument("--n-example-pairs", type=int, default=3)
    return p.parse_args(argv)


def _autocast(device: torch.device, mode: str):
    if device.type != "cuda" or mode == "none":
        return nullcontext()
    dtype = torch.float16 if mode == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _cpu_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.float32


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device)


def _encode_components(model, x: torch.Tensor, amp: str):
    with torch.no_grad(), _autocast(x.device, amp):
        prepared = model._prepare_input(x, 0.0)
        deepest, skips = model.encoder(prepared)
        latent = model.bottleneck.encode(deepest)
        mu = latent["mu_tokens"]
        transformed = model.bottleneck.decode(mu)
    return mu, transformed, skips


def _decode_components(model, bottleneck: torch.Tensor, skips: list[torch.Tensor], amp: str) -> torch.Tensor:
    with torch.no_grad(), _autocast(bottleneck.device, amp):
        decoded = model.decoder(bottleneck, skips)
        pred = torch.nan_to_num(model.head(decoded), nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)
    return pred


def _finite_rmse(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 1:
        return float("nan")
    return float(np.sqrt(np.mean((aa[mask] - bb[mask]) ** 2)))


def _donor_indices(n: int, recipient: int, maximum: int, rng: np.random.Generator) -> list[int]:
    candidates = np.asarray([j for j in range(n) if j != recipient], dtype=int)
    if maximum <= 0 or maximum >= len(candidates):
        return candidates.tolist()
    return np.sort(rng.choice(candidates, size=int(maximum), replace=False)).tolist()


def _pair_row(
    *,
    variant: str,
    condition: str,
    recipient_subject: str,
    donor_subject: str,
    true_recipient: np.ndarray,
    true_donor: np.ndarray,
    pred: np.ndarray,
    baseline_recipient: np.ndarray,
    baseline_donor: np.ndarray,
    common_template: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
) -> dict:
    recipient_metrics = subject_metrics(true_recipient, pred, left_idx, right_idx)
    whole_recipient = pearsonr_np(true_recipient, pred)
    whole_donor = pearsonr_np(true_donor, pred)
    baseline_whole_recipient = pearsonr_np(true_recipient, baseline_recipient)
    baseline_whole_donor = pearsonr_np(true_donor, baseline_recipient)

    pred_residual = np.asarray(pred, dtype=np.float32) - common_template
    recipient_residual = np.asarray(true_recipient, dtype=np.float32) - common_template
    donor_residual = np.asarray(true_donor, dtype=np.float32) - common_template
    baseline_recipient_residual = np.asarray(baseline_recipient, dtype=np.float32) - common_template

    residual_recipient = pearsonr_np(recipient_residual, pred_residual)
    residual_donor = pearsonr_np(donor_residual, pred_residual)
    baseline_residual_recipient = pearsonr_np(recipient_residual, baseline_recipient_residual)
    baseline_residual_donor = pearsonr_np(donor_residual, baseline_recipient_residual)

    return {
        "variant": variant,
        "condition": condition,
        "recipient_subject": recipient_subject,
        "donor_subject": donor_subject,
        **{f"recipient_{k}": v for k, v in recipient_metrics.items()},
        "whole_pearson_to_recipient": whole_recipient,
        "whole_pearson_to_donor": whole_donor,
        "whole_recipient_preference": whole_recipient - whole_donor,
        "baseline_whole_recipient_preference": baseline_whole_recipient - baseline_whole_donor,
        "whole_identity_shift_toward_donor": (
            (whole_donor - whole_recipient) - (baseline_whole_donor - baseline_whole_recipient)
        ),
        "residual_pearson_to_recipient": residual_recipient,
        "residual_pearson_to_donor": residual_donor,
        "residual_recipient_preference": residual_recipient - residual_donor,
        "baseline_residual_recipient_preference": baseline_residual_recipient - baseline_residual_donor,
        "residual_identity_shift_toward_donor": (
            (residual_donor - residual_recipient)
            - (baseline_residual_donor - baseline_residual_recipient)
        ),
        "prediction_pearson_to_recipient_baseline": pearsonr_np(pred, baseline_recipient),
        "prediction_rmse_from_recipient_baseline": _finite_rmse(pred, baseline_recipient),
        "prediction_pearson_to_donor_baseline": pearsonr_np(pred, baseline_donor),
        "donor_preferred_whole": float(whole_donor > whole_recipient),
        "donor_preferred_residual": float(residual_donor > residual_recipient),
    }


def _subject_condition_row(
    *,
    variant: str,
    condition: str,
    subject: str,
    true: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
    template: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
) -> dict:
    metrics = subject_metrics(true, pred, left_idx, right_idx)
    true_residual = np.asarray(true, dtype=np.float32) - template
    pred_residual = np.asarray(pred, dtype=np.float32) - template
    return {
        "variant": variant,
        "condition": condition,
        "subject": subject,
        **metrics,
        "residual_pearson_r": pearsonr_np(true_residual, pred_residual),
        "prediction_pearson_to_baseline": pearsonr_np(pred, baseline),
        "prediction_rmse_from_baseline": _finite_rmse(pred, baseline),
    }


def _aggregate_pair_rows(pair_frame: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["variant", "condition", "recipient_subject"]
    numeric = pair_frame.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in id_cols]
    out = pair_frame.groupby(id_cols, as_index=False)[numeric].mean()
    donor_counts = pair_frame.groupby(id_cols, as_index=False)["donor_subject"].nunique()
    donor_counts = donor_counts.rename(columns={"donor_subject": "n_donors"})
    return out.merge(donor_counts, on=id_cols, how="left")


def _condition_summary(recipient_frame: pd.DataFrame, subject_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for condition, frame in recipient_frame.groupby("condition", sort=False):
        row = {"condition": condition, "n_recipients": int(frame["recipient_subject"].nunique())}
        for metric in [
            "recipient_pearson_r",
            "recipient_dice_auc",
            "recipient_dice25",
            "whole_identity_shift_toward_donor",
            "residual_identity_shift_toward_donor",
            "whole_recipient_preference",
            "residual_recipient_preference",
            "prediction_rmse_from_recipient_baseline",
            "donor_preferred_whole",
            "donor_preferred_residual",
        ]:
            if metric in frame:
                values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                row[f"{metric}_mean"] = float(values.mean()) if values.size else np.nan
                row[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else np.nan
        rows.append(row)
    for condition, frame in subject_frame.groupby("condition", sort=False):
        row = {"condition": condition, "n_recipients": int(frame["subject"].nunique())}
        for metric in ["pearson_r", "dice_auc", "dice25", "residual_pearson_r", "prediction_rmse_from_baseline"]:
            if metric in frame:
                values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                row[f"{metric}_mean"] = float(values.mean()) if values.size else np.nan
                row[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_effects(
    recipient_frame: pd.DataFrame,
    subject_frame: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    wide = recipient_frame.set_index(["condition", "recipient_subject"])
    conditions = set(recipient_frame["condition"].astype(str))
    if set(PAIR_CONDITIONS).issubset(conditions):
        bottleneck = wide.loc[PAIR_CONDITIONS[0]]
        skips = wide.loc[PAIR_CONDITIONS[1]]
        common = sorted(set(bottleneck.index.astype(str)) & set(skips.index.astype(str)))
        comparisons = [
            (
                "skip_swap_minus_bottleneck_swap_identity_transfer",
                "whole_identity_shift_toward_donor",
                skips.loc[common, "whole_identity_shift_toward_donor"].to_numpy(dtype=float)
                - bottleneck.loc[common, "whole_identity_shift_toward_donor"].to_numpy(dtype=float),
                "positive means swapped skips transfer more whole-map identity",
            ),
            (
                "skip_swap_minus_bottleneck_swap_residual_identity_transfer",
                "residual_identity_shift_toward_donor",
                skips.loc[common, "residual_identity_shift_toward_donor"].to_numpy(dtype=float)
                - bottleneck.loc[common, "residual_identity_shift_toward_donor"].to_numpy(dtype=float),
                "positive means swapped skips transfer more residual identity",
            ),
            (
                "skip_swap_minus_bottleneck_swap_recipient_quality_loss",
                "recipient_pearson_r",
                bottleneck.loc[common, "recipient_pearson_r"].to_numpy(dtype=float)
                - skips.loc[common, "recipient_pearson_r"].to_numpy(dtype=float),
                "positive means swapping skips harms recipient Pearson more",
            ),
            (
                "skip_swap_minus_bottleneck_swap_map_change",
                "prediction_rmse_from_recipient_baseline",
                skips.loc[common, "prediction_rmse_from_recipient_baseline"].to_numpy(dtype=float)
                - bottleneck.loc[common, "prediction_rmse_from_recipient_baseline"].to_numpy(dtype=float),
                "positive means swapping skips changes the prediction more",
            ),
        ]
        for contrast, metric, values, interpretation in comparisons:
            rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "positive_effect_interpretation": interpretation,
                    **paired_stats(values, rng, n_boot=n_boot, n_perm=n_perm),
                }
            )

    subject_wide = subject_frame.set_index(["condition", "subject"])
    subject_conditions = set(subject_frame["condition"].astype(str))
    baseline_name = SUBJECT_CONDITIONS[0]
    if baseline_name in subject_conditions:
        baseline = subject_wide.loc[baseline_name]
        for condition in SUBJECT_CONDITIONS[1:]:
            if condition not in subject_conditions:
                continue
            other = subject_wide.loc[condition]
            common = sorted(set(baseline.index.astype(str)) & set(other.index.astype(str)))
            for metric in ["pearson_r", "dice_auc", "dice25", "residual_pearson_r"]:
                values = baseline.loc[common, metric].to_numpy(dtype=float) - other.loc[common, metric].to_numpy(dtype=float)
                rows.append(
                    {
                        "contrast": f"baseline_minus_{condition}",
                        "metric": metric,
                        "positive_effect_interpretation": "positive means replacing this component reduces performance",
                        **paired_stats(values, rng, n_boot=n_boot, n_perm=n_perm),
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["bh_q"] = bh_adjust(frame["signflip_p_two_sided"].to_numpy(dtype=float))
    return frame


def _write_summary_md(path: Path, variant: str, condition_summary: pd.DataFrame, effects: pd.DataFrame) -> None:
    lines = [
        f"# Skip–bottleneck swap analysis: {variant}",
        "",
        "All ordered held-out donor–recipient pairs were evaluated. Pairwise hybrids were averaged over donors within each recipient before inferential statistics, so the subject remains the unit of inference.",
        "",
        "## Interpretation guide",
        "",
        "- Positive `residual_identity_shift_toward_donor` means the hybrid residual map moved toward the donor subject.",
        "- A larger identity shift after swapping skips than after swapping bottlenecks indicates that shallower skips carry more subject identity.",
        "- A larger recipient-quality loss after replacing one component indicates that component contributes more to accurate reconstruction.",
        "",
        "## Condition means",
        "",
    ]
    if condition_summary.empty:
        lines.append("No condition summaries were produced.")
    else:
        lines.extend(["```text", condition_summary.to_string(index=False), "```"] )
    lines.extend(["", "## Paired recipient-level effects", ""])
    if effects.empty:
        lines.append("No paired effects were produced.")
    else:
        keep = [
            "contrast", "metric", "n", "mean_effect", "bootstrap_ci95_low", "bootstrap_ci95_high",
            "signflip_p_two_sided", "bh_q", "positive_effect_interpretation",
        ]
        lines.extend(["```text", effects[[c for c in keep if c in effects]].to_string(index=False), "```"] )
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, scaler, ckpt = load_model_checkpoint(checkpoint, device)
    model.eval()
    variant = str(ckpt.get("model_config", {}).get("variant", args.model_name))

    paths = ClinicalPaths(**ckpt["paths"])
    if args.data_dir is not None:
        paths.data_dir = args.data_dir
    atlas = create_image_space(paths)
    subjects_requested = load_subject_list(args.subjects)
    cache_dir = args.cache_dir or str(checkpoint.parent.parent.parent / "image_cache")
    raw = build_subject_matrix_cached(
        subjects_requested,
        paths,
        atlas,
        cache_dir=cache_dir,
        cache_tag=f"skip_bottleneck_swap_{Path(args.subjects).stem}",
        refresh_cache=args.refresh_cache,
    )
    require_complete_subjects(subjects_requested, raw["subjects"])
    common_template, template_metadata = load_for_evaluation(args.training_template, subjects_requested, paths, raw["y"].shape[1:])
    matrix = prepare_matrix(raw, scaler)
    dataset = ImageSubjectDataset(matrix)
    subjects = [str(s) for s in matrix["subjects"]]
    n = len(subjects)
    if n < 3:
        raise RuntimeError("At least three subjects are required for component swapping")

    cache_dtype = _cpu_dtype(args.component_cache_dtype)
    true_maps: list[np.ndarray] = []
    baseline_predictions: list[np.ndarray] = []
    bottlenecks: list[torch.Tensor] = []
    mu_tokens: list[torch.Tensor] = []
    mean_skip_accumulators: list[torch.Tensor] | None = None

    print(f"[swap] pass 1/3: baseline predictions and compact bottleneck cache for {n} subjects")
    for i in range(n):
        item = dataset[i]
        x = _to_tensor(item["x"], device)
        mu, transformed, skips = _encode_components(model, x, args.amp)
        pred = _decode_components(model, transformed, skips, args.amp)
        true_maps.append(np.asarray(item["y"], dtype=np.float32).copy())
        baseline_predictions.append(pred[0].float().cpu().numpy())
        bottlenecks.append(transformed.detach().to("cpu", dtype=cache_dtype).contiguous())
        mu_tokens.append(mu.detach().to("cpu", dtype=torch.float32).contiguous())
        if mean_skip_accumulators is None:
            mean_skip_accumulators = [torch.zeros_like(s.detach().cpu(), dtype=torch.float32) for s in skips]
        for level, skip in enumerate(skips):
            mean_skip_accumulators[level].add_(skip.detach().float().cpu())
        del x, mu, transformed, skips, pred
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert mean_skip_accumulators is not None
    mean_skips_cpu = [(s / float(n)).to(dtype=cache_dtype).contiguous() for s in mean_skip_accumulators]
    del mean_skip_accumulators
    mean_mu = torch.stack(mu_tokens, dim=0).mean(dim=0).to(device)
    with torch.no_grad(), _autocast(device, args.amp):
        mean_bottleneck = model.bottleneck.decode(mean_mu).detach().to("cpu", dtype=cache_dtype).contiguous()
    del mean_mu
    if any(np.any(np.isfinite(y) & ~np.isfinite(common_template)) for y in true_maps):
        raise ValueError("Evaluation voxels missing training-template support")

    subject_rows: list[dict] = []
    # Baseline rows are already available from pass 1.
    for i, subject in enumerate(subjects):
        subject_rows.append(
            _subject_condition_row(
                variant=variant,
                condition=SUBJECT_CONDITIONS[0],
                subject=subject,
                true=true_maps[i],
                pred=baseline_predictions[i],
                baseline=baseline_predictions[i],
                template=common_template,
                left_idx=matrix["left_idx"],
                right_idx=matrix["right_idx"],
            )
        )

    pair_rows: list[dict] = []
    rng = np.random.default_rng(args.seed)
    donors_by_recipient = {
        i: _donor_indices(n, i, args.max_donors_per_recipient, rng) for i in range(n)
    }

    print("[swap] pass 2/3: donor bottleneck + recipient skips")
    for i, recipient in enumerate(subjects):
        item = dataset[i]
        x = _to_tensor(item["x"], device)
        _, _, recipient_skips = _encode_components(model, x, args.amp)

        mean_b = mean_bottleneck.to(device)
        pred_mean_b = _decode_components(model, mean_b, recipient_skips, args.amp)[0].float().cpu().numpy()
        subject_rows.append(
            _subject_condition_row(
                variant=variant,
                condition=SUBJECT_CONDITIONS[1],
                subject=recipient,
                true=true_maps[i],
                pred=pred_mean_b,
                baseline=baseline_predictions[i],
                template=common_template,
                left_idx=matrix["left_idx"],
                right_idx=matrix["right_idx"],
            )
        )
        del mean_b, pred_mean_b

        for j in donors_by_recipient[i]:
            donor_b = bottlenecks[j].to(device)
            pred = _decode_components(model, donor_b, recipient_skips, args.amp)[0].float().cpu().numpy()
            pair_rows.append(
                _pair_row(
                    variant=variant,
                    condition=PAIR_CONDITIONS[0],
                    recipient_subject=recipient,
                    donor_subject=subjects[j],
                    true_recipient=true_maps[i],
                    true_donor=true_maps[j],
                    pred=pred,
                    baseline_recipient=baseline_predictions[i],
                    baseline_donor=baseline_predictions[j],
                    common_template=common_template,
                    left_idx=matrix["left_idx"],
                    right_idx=matrix["right_idx"],
                )
            )
            del donor_b, pred
        del x, recipient_skips
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[swap] pass 3/3: recipient bottleneck + donor skips")
    # Donor eligibility is based on the same preselected recipient->donor sets.
    recipient_sets_by_donor = {
        j: [i for i in range(n) if j in donors_by_recipient[i]] for j in range(n)
    }
    for j, donor in enumerate(subjects):
        item = dataset[j]
        x = _to_tensor(item["x"], device)
        _, _, donor_skips = _encode_components(model, x, args.amp)
        for i in recipient_sets_by_donor[j]:
            recipient_b = bottlenecks[i].to(device)
            pred = _decode_components(model, recipient_b, donor_skips, args.amp)[0].float().cpu().numpy()
            pair_rows.append(
                _pair_row(
                    variant=variant,
                    condition=PAIR_CONDITIONS[1],
                    recipient_subject=subjects[i],
                    donor_subject=donor,
                    true_recipient=true_maps[i],
                    true_donor=true_maps[j],
                    pred=pred,
                    baseline_recipient=baseline_predictions[i],
                    baseline_donor=baseline_predictions[j],
                    common_template=common_template,
                    left_idx=matrix["left_idx"],
                    right_idx=matrix["right_idx"],
                )
            )
            del recipient_b, pred
        del x, donor_skips
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[swap] cohort-mean skips condition")
    mean_skips_device = [s.to(device) for s in mean_skips_cpu]
    for i, subject in enumerate(subjects):
        recipient_b = bottlenecks[i].to(device)
        pred = _decode_components(model, recipient_b, mean_skips_device, args.amp)[0].float().cpu().numpy()
        subject_rows.append(
            _subject_condition_row(
                variant=variant,
                condition=SUBJECT_CONDITIONS[2],
                subject=subject,
                true=true_maps[i],
                pred=pred,
                baseline=baseline_predictions[i],
                template=common_template,
                left_idx=matrix["left_idx"],
                right_idx=matrix["right_idx"],
            )
        )
        del recipient_b, pred
    del mean_skips_device

    pair_frame = pd.DataFrame(pair_rows)
    subject_frame = pd.DataFrame(subject_rows)
    recipient_frame = _aggregate_pair_rows(pair_frame)
    condition_summary = _condition_summary(recipient_frame, subject_frame)
    effects = _paired_effects(
        recipient_frame,
        subject_frame,
        np.random.default_rng(args.seed + 1009),
        args.bootstrap,
        args.signflip,
    )

    for frame in (pair_frame, recipient_frame, subject_frame, condition_summary, effects):
        for key, value in metric_provenance(template_metadata).items():
            frame[key] = value
    pair_frame.to_csv(out_dir / "swap_pairwise.csv", index=False)
    recipient_frame.to_csv(out_dir / "swap_per_recipient.csv", index=False)
    subject_frame.to_csv(out_dir / "component_condition_per_subject.csv", index=False)
    condition_summary.to_csv(out_dir / "condition_summary.csv", index=False)
    effects.to_csv(out_dir / "paired_effects.csv", index=False)
    _write_summary_md(out_dir / "mechanism_summary.md", variant, condition_summary, effects)

    np.savez_compressed(
        out_dir / "analysis_reference_maps.npz",
        subjects=np.asarray(subjects, dtype=str),
        common_template=common_template.astype(np.float32),
    )
    manifest = {
        "checkpoint": str(checkpoint),
        "variant": variant,
        "model_name": args.model_name,
        "subjects_file": str(args.subjects),
        "n_subjects": n,
        "n_ordered_pairs": int(len(pair_frame) // max(len(PAIR_CONDITIONS), 1)),
        "max_donors_per_recipient": int(args.max_donors_per_recipient),
        "pair_conditions": list(PAIR_CONDITIONS),
        "subject_conditions": list(SUBJECT_CONDITIONS),
        "unit_of_inference": "recipient subject after averaging across donors",
        "residual_template": "fixed training-only mean of standardized task maps",
        **metric_provenance(template_metadata),
        "amp": args.amp,
        "component_cache_dtype": args.component_cache_dtype,
        "architecture_version": ckpt.get("model_config", {}).get("architecture_version"),
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if args.write_nifti and not pair_frame.empty:
        examples = pair_frame.sort_values("residual_identity_shift_toward_donor", ascending=False).head(
            max(1, int(args.n_example_pairs))
        )
        example_dir = out_dir / "nifti_examples"
        # Recompute only selected examples to avoid retaining thousands of maps.
        for _, row in examples.iterrows():
            i = subjects.index(str(row["recipient_subject"]))
            j = subjects.index(str(row["donor_subject"]))
            condition = str(row["condition"])
            if condition == PAIR_CONDITIONS[0]:
                x = _to_tensor(dataset[i]["x"], device)
                _, _, skips = _encode_components(model, x, args.amp)
                b = bottlenecks[j].to(device)
            else:
                x = _to_tensor(dataset[j]["x"], device)
                _, _, skips = _encode_components(model, x, args.amp)
                b = bottlenecks[i].to(device)
            pred = _decode_components(model, b, skips, args.amp)[0].float().cpu().numpy()
            stem = f"{condition}__recipient-{subjects[i]}__donor-{subjects[j]}"
            write_image_nifti(pred, atlas, example_dir / f"{stem}__hybrid.nii.gz")
            write_image_nifti(true_maps[i], atlas, example_dir / f"{stem}__recipient_true.nii.gz")
            write_image_nifti(true_maps[j], atlas, example_dir / f"{stem}__donor_true.nii.gz")
            del x, skips, b, pred

    print(f"[swap] wrote {out_dir}")
    if not effects.empty:
        print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
