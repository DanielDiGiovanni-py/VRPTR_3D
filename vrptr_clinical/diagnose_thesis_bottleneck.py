from __future__ import annotations

"""Causal and gradient diagnostics for the thesis VRPTR bottleneck."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ClinicalPaths, ImageSubjectDataset, build_subject_matrix_cached, create_image_space, load_subject_list
from .losses import masked_smooth_l1, neg_pearson_loss
from .metrics import subject_metrics, summarize_metric_rows
from .predict_clinical import load_model_checkpoint, prepare_matrix
from .training_common import to_device, write_rows_csv


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Diagnose latent use in thesis-faithful VRPTR.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default=None, help="Override the input data root stored in the checkpoint; leaves the checkpoint unchanged.")
    p.add_argument("--subjects", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-dir", default="")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--group", default="controls")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--signflip", type=int, default=20000)
    return p.parse_args(argv)


def _paired_stats(values: np.ndarray, rng: np.random.Generator, n_boot: int, n_sign: int) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "mean_delta": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "signflip_p": np.nan,
            "better_fraction": np.nan,
        }
    boot = np.empty(max(1, int(n_boot)), dtype=float)
    for i in range(boot.size):
        boot[i] = rng.choice(values, size=values.size, replace=True).mean()
    observed = abs(values.mean())
    extreme = 0
    for _ in range(max(1, int(n_sign))):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=values.size)
        extreme += int(abs((values * signs).mean()) >= observed)
    return {
        "n": int(values.size),
        "mean_delta": float(values.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "signflip_p": float((extreme + 1) / (max(1, int(n_sign)) + 1)),
        "better_fraction": float((values > 0).mean()),
    }


def _effective_rank(matrix: np.ndarray) -> float:
    x = np.asarray(matrix, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    energy = singular * singular
    if energy.sum() <= 0:
        return 0.0
    p = energy / energy.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def _module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            grad = parameter.grad.detach().float()
            total += float(torch.sum(grad * grad).cpu())
    return float(total ** 0.5)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, scaler, ckpt = load_model_checkpoint(args.checkpoint, device)
    paths = ClinicalPaths(**ckpt["paths"])
    if args.data_dir is not None:
        paths.data_dir = args.data_dir
    atlas = create_image_space(paths)
    subjects_requested = load_subject_list(args.subjects)
    cache_dir = args.cache_dir or str(Path(args.checkpoint).parent.parent.parent / "image_cache")
    raw = build_subject_matrix_cached(
        subjects_requested,
        paths,
        atlas,
        cache_dir=cache_dir,
        cache_tag=f"diagnose_{Path(args.subjects).stem}",
        refresh_cache=args.refresh_cache,
    )
    matrix = prepare_matrix(raw, scaler)
    dataset = ImageSubjectDataset(matrix)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    latent_by_subject: dict[str, torch.Tensor] = {}
    latent_rows: list[dict] = []
    mu_flat: list[np.ndarray] = []
    ordered_subjects: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            bd = to_device(batch, device)
            encoded = model.encode_latent(bd["x"])
            mu = encoded["mu_tokens"].detach()
            subject = str(batch["subject"][0])
            latent_by_subject[subject] = mu.cpu()
            ordered_subjects.append(subject)
            flat = mu.cpu().numpy().reshape(-1)
            mu_flat.append(flat)
            latent_rows.append(
                {
                    "subject": subject,
                    "mu_mean": float(flat.mean()),
                    "mu_std": float(flat.std()),
                    "mu_abs_mean": float(np.abs(flat).mean()),
                    "logvar_mean": float(encoded["logvar_tokens"].mean().cpu()),
                    "logvar_std": float(encoded["logvar_tokens"].std().cpu()),
                }
            )
    write_rows_csv(output / "latent_per_subject_summary.csv", latent_rows)

    mu_matrix = np.stack(mu_flat)
    between_feature_std = mu_matrix.std(axis=0, ddof=1)
    within_subject_std = mu_matrix.std(axis=1)
    centered = mu_matrix - mu_matrix.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norm, 1e-8)
    cosine = normalized @ normalized.T
    upper = cosine[np.triu_indices_from(cosine, k=1)]
    latent_geometry = {
        "n": int(mu_matrix.shape[0]),
        "features": int(mu_matrix.shape[1]),
        "between_subject_std_mean": float(between_feature_std.mean()),
        "within_subject_std_mean": float(within_subject_std.mean()),
        "between_to_within_std_ratio": float(between_feature_std.mean() / max(within_subject_std.mean(), 1e-8)),
        "effective_rank": _effective_rank(mu_matrix),
        "pairwise_cosine_mean": float(upper.mean()),
        "pairwise_cosine_std": float(upper.std(ddof=1)),
    }
    write_rows_csv(output / "latent_geometry_summary.csv", [latent_geometry])
    np.savez_compressed(
        output / "latent_tokens.npz",
        subjects=np.asarray(ordered_subjects, dtype=str),
        mu=mu_matrix.astype(np.float32),
    )

    rng = np.random.default_rng(args.seed)
    shuffled_subjects = list(np.asarray(ordered_subjects)[rng.permutation(len(ordered_subjects))])
    cohort_mean = torch.stack([latent_by_subject[s] for s in ordered_subjects], dim=0).mean(dim=0)

    conditions = [
        "normal",
        "zero_latent",
        "shuffled_latent",
        "cohort_mean_latent",
        "zero_bottleneck",
        "bottleneck_scale_0.5",
        "bottleneck_scale_1.5",
    ]
    all_rows: list[dict] = []
    row_lookup: dict[tuple[str, str], dict] = {}
    loaders = {condition: DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) for condition in conditions}

    with torch.no_grad():
        for condition in conditions:
            for index, batch in enumerate(loaders[condition]):
                bd = to_device(batch, device)
                subject = str(batch["subject"][0])
                normal_mu = latent_by_subject[subject].to(device)
                override = None
                bottleneck_scale = 1.0
                if condition == "zero_latent":
                    override = torch.zeros_like(normal_mu)
                elif condition == "shuffled_latent":
                    override = latent_by_subject[shuffled_subjects[index]].to(device)
                elif condition == "cohort_mean_latent":
                    override = cohort_mean.to(device)
                elif condition == "zero_bottleneck":
                    override = normal_mu
                    bottleneck_scale = 0.0
                elif condition == "bottleneck_scale_0.5":
                    override = normal_mu
                    bottleneck_scale = 0.5
                elif condition == "bottleneck_scale_1.5":
                    override = normal_mu
                    bottleneck_scale = 1.5
                out = model(
                    bd["x"],
                    sample_latent=False,
                    latent_override=override,
                    bottleneck_scale=bottleneck_scale,
                )
                pred = out["beta"][0].detach().cpu().numpy()
                true = bd["y"][0].detach().cpu().numpy()
                row = subject_metrics(true, pred, matrix["left_idx"], matrix["right_idx"])
                row.update(
                    {
                        "subject": subject,
                        "group": args.group,
                        "model": ckpt["model_config"].get("variant", "unknown"),
                        "condition": condition,
                        "latent_source_subject": (
                            shuffled_subjects[index] if condition == "shuffled_latent" else subject
                        ),
                    }
                )
                all_rows.append(row)
                row_lookup[(condition, subject)] = row

    write_rows_csv(output / "latent_intervention_per_subject.csv", all_rows)
    summaries = []
    for condition in conditions:
        rows = [row_lookup[(condition, subject)] for subject in ordered_subjects]
        summary = summarize_metric_rows(rows, group=args.group, model=condition)
        summary["condition"] = condition
        summaries.append(summary)
    write_rows_csv(output / "latent_intervention_summary.csv", summaries)

    paired_rows = []
    for condition in conditions:
        if condition == "normal":
            continue
        for metric in ["pearson_r", "spearman_r", "r2", "dice_auc", "dice25"]:
            delta = np.asarray(
                [
                    row_lookup[("normal", subject)].get(metric, np.nan)
                    - row_lookup[(condition, subject)].get(metric, np.nan)
                    for subject in ordered_subjects
                ],
                dtype=float,
            )
            stats = _paired_stats(delta, rng, args.bootstrap, args.signflip)
            paired_rows.append({"reference": "normal", "intervention": condition, "metric": metric, **stats})
    write_rows_csv(output / "latent_intervention_paired_deltas.csv", paired_rows)

    # One-batch gradient audit: the defining bottleneck components must all
    # receive gradient from the full-map objective.
    first_batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    bd = to_device(first_batch, device)
    model.zero_grad(set_to_none=True)
    model.train()
    out = model(bd["x"], sample_latent=False)
    audit_loss = masked_smooth_l1(out["beta"], bd["y"]) + 0.1 * neg_pearson_loss(out["beta"], bd["y"])
    audit_loss = audit_loss + out["logvar_z"].sum() * 0.0
    audit_loss.backward()
    modules = {
        "encoder": model.encoder,
        "spatial_compress": model.bottleneck.spatial_compress,
        "transformer_encoder": model.bottleneck.encoder_transformer,
        "mu_head": model.bottleneck.mu_head,
        "logvar_head": model.bottleneck.logvar_head,
        "transformer_decoder": model.bottleneck.decoder_transformer,
        "spatial_expand": model.bottleneck.spatial_expand,
        "decoder": model.decoder,
        "output_head": model.head,
    }
    gradient_rows = []
    for name, module in modules.items():
        gradient_rows.append(
            {
                "component": name,
                "gradient_l2": _module_grad_norm(module) if module is not None else np.nan,
                "parameter_count": (
                    sum(p.numel() for p in module.parameters()) if module is not None else 0
                ),
            }
        )
    write_rows_csv(output / "gradient_diagnostics.csv", gradient_rows)
    model.eval()

    paired_frame = {(row["intervention"], row["metric"]): row for row in paired_rows}
    gradient_lookup = {row["component"]: row["gradient_l2"] for row in gradient_rows}
    use_transformer = bool(ckpt.get("model_config", {}).get("use_transformer", True))
    gates = [
        {
            "gate": "normal_beats_zero_bottleneck_pearson",
            "applicable": 1,
            "value": paired_frame[("zero_bottleneck", "pearson_r")]["mean_delta"],
            "threshold": 0.005,
            "passed": int(paired_frame[("zero_bottleneck", "pearson_r")]["mean_delta"] > 0.005),
        },
        {
            "gate": "normal_beats_shuffled_latent_pearson",
            "applicable": 1,
            "value": paired_frame[("shuffled_latent", "pearson_r")]["mean_delta"],
            "threshold": 0.002,
            "passed": int(paired_frame[("shuffled_latent", "pearson_r")]["mean_delta"] > 0.002),
        },
        {
            "gate": "normal_beats_cohort_mean_latent_pearson",
            "applicable": 1,
            "value": paired_frame[("cohort_mean_latent", "pearson_r")]["mean_delta"],
            "threshold": 0.002,
            "passed": int(paired_frame[("cohort_mean_latent", "pearson_r")]["mean_delta"] > 0.002),
        },
        {
            "gate": "between_subject_latent_variability",
            "applicable": 1,
            "value": latent_geometry["between_to_within_std_ratio"],
            "threshold": 0.01,
            "passed": int(latent_geometry["between_to_within_std_ratio"] > 0.01),
        },
        {
            "gate": "transformer_encoder_receives_gradient",
            "value": gradient_lookup.get("transformer_encoder", 0.0) if use_transformer else np.nan,
            "threshold": 0.0 if use_transformer else np.nan,
            "passed": int(gradient_lookup.get("transformer_encoder", 0.0) > 0.0) if use_transformer else 1,
            "applicable": int(use_transformer),
        },
        {
            "gate": "transformer_decoder_receives_gradient",
            "value": gradient_lookup.get("transformer_decoder", 0.0) if use_transformer else np.nan,
            "threshold": 0.0 if use_transformer else np.nan,
            "passed": int(gradient_lookup.get("transformer_decoder", 0.0) > 0.0) if use_transformer else 1,
            "applicable": int(use_transformer),
        },
    ]
    applicable_gates = [row for row in gates if int(row.get("applicable", 1)) == 1]
    gates.append(
        {
            "gate": "all_required_gates",
            "value": float(sum(row["passed"] for row in applicable_gates)),
            "threshold": float(len(applicable_gates)),
            "passed": int(all(row["passed"] for row in applicable_gates)),
            "applicable": 1,
        }
    )
    write_rows_csv(output / "smoke_gate_summary.csv", gates)
    print(f"[thesis-diagnostics] wrote {output}")
    for row in gates:
        print(row)


if __name__ == "__main__":
    main()
