from __future__ import annotations

"""Held-out prediction for thesis-faithful VRPTR checkpoints."""

import argparse
from .training_template import add_template_argument, load_for_evaluation, metric_provenance, require_complete_subjects
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    ClinicalPaths,
    ImageSubjectDataset,
    build_subject_matrix_cached,
    create_image_space,
    load_subject_list,
    write_image_nifti,
)
from .metrics import (
    self_vs_other_rows,
    subject_metrics,
    summarize_metric_rows,
    template_residual_analysis_rows,
)
from .model import build_model_from_config
from .representation_ablation import add_input_ablation_args, ablated_model_name, make_eval_dataset
from .training_common import Standardizer, load_state_dict_flexible_, to_device, write_rows_csv


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a thesis-faithful VRPTR checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default=None, help="Override the input data root stored in the checkpoint; leaves the checkpoint unchanged.")
    p.add_argument("--subjects", required=True)
    p.add_argument("--group", choices=["controls", "tumor"], default="controls")
    p.add_argument("--model-name", default="thesis_vrptr")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-dir", default="")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--write-nifti", action="store_true")
    p.add_argument("--save-latents", action="store_true")
    add_template_argument(p)
    p.add_argument("--test-loo-sensitivity", action="store_true")
    add_input_ablation_args(p)
    return p.parse_args(argv)


def load_model_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    if cfg.get("architecture_version") != "thesis_voxel_v1":
        raise RuntimeError(
            f"Checkpoint {path} is not a thesis_voxel_v1 checkpoint. "
            "Old side-branch VRPTR checkpoints are intentionally incompatible."
        )
    model = build_model_from_config(cfg)
    load_state_dict_flexible_(model, ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    scaler = Standardizer.from_state_dict(ckpt["x_scaler"])
    return model, scaler, ckpt


def prepare_matrix(raw: dict, scaler: Standardizer) -> dict:
    matrix = dict(raw)
    is_disk = bool(raw.get("disk_backed", False)) or hasattr(raw.get("x", None), "filename")
    if is_disk:
        matrix["x"] = raw["x"]
        matrix["y"] = raw["y"]
        matrix["x_scaler"] = scaler
    else:
        matrix["x"] = scaler.transform(np.asarray(raw["x"], dtype=np.float32))
        matrix["y"] = np.asarray(raw["y"], dtype=np.float32)
    matrix["input_feature_names"] = list(raw.get("input_feature_names", []))
    return matrix


def predict_dataset(model, dataset, device, left_idx, right_idx, group, model_name, batch_size):
    rows: list[dict] = []
    records: list[tuple[str, np.ndarray, np.ndarray]] = []
    subjects: list[str] = []
    mus: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            bd = to_device(batch, device)
            out = model(bd["x"], sample_latent=False)
            pred = out["beta"].detach().cpu().numpy()
            true = bd["y"].detach().cpu().numpy()
            mu = out["mu_tokens"].detach().cpu().numpy()
            logvar = out["logvar_tokens"].detach().cpu().numpy()
            for j, subject in enumerate(batch["subject"]):
                subject = str(subject)
                row = subject_metrics(true[j], pred[j], left_idx, right_idx)
                row.update({"subject": subject, "group": group, "model": model_name})
                rows.append(row)
                records.append((subject, true[j].copy(), pred[j].copy()))
                subjects.append(subject)
                mus.append(mu[j].copy())
                logvars.append(logvar[j].copy())
    return rows, records, subjects, mus, logvars


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    model, scaler, ckpt = load_model_checkpoint(checkpoint, device)
    paths = ClinicalPaths(**ckpt["paths"])
    if args.data_dir is not None:
        paths.data_dir = args.data_dir
    atlas = create_image_space(paths)
    subject_list = load_subject_list(args.subjects)
    cache_dir = args.cache_dir or str(checkpoint.parent.parent.parent / "image_cache")
    raw = build_subject_matrix_cached(
        subject_list,
        paths,
        atlas,
        cache_dir=cache_dir,
        cache_tag=f"predict_{args.group}_{Path(args.subjects).stem}",
        refresh_cache=args.refresh_cache,
    )
    require_complete_subjects(subject_list, raw["subjects"])
    template, metadata = load_for_evaluation(args.training_template, subject_list, paths, raw["y"].shape[1:])
    matrix = prepare_matrix(raw, scaler)
    dataset = make_eval_dataset(matrix, mode=args.input_ablation, seed=args.ablation_seed)
    effective_name = ablated_model_name(
        args.model_name, args.input_ablation, add_suffix=not bool(args.no_ablation_name_suffix)
    )
    output = Path(args.out_dir) / effective_name / args.group
    output.mkdir(parents=True, exist_ok=True)

    rows, records, latent_subjects, mus, logvars = predict_dataset(
        model,
        dataset,
        device,
        matrix["left_idx"],
        matrix["right_idx"],
        args.group,
        effective_name,
        args.batch_size,
    )
    summary = summarize_metric_rows(rows, group=args.group, model=effective_name)
    write_rows_csv(output / "per_subject_metrics.csv", rows)
    write_rows_csv(output / "summary.csv", [summary])

    if records:
        indiv_subjects = [r[0] for r in records]
        true_maps = [r[1] for r in records]
        pred_maps = [r[2] for r in records]
        self_rows, self_pairs, self_summary = self_vs_other_rows(
            indiv_subjects, true_maps, pred_maps, args.group, effective_name
        )
        write_rows_csv(output / "self_vs_other.csv", self_rows)
        write_rows_csv(output / "self_vs_other_pairwise.csv", self_pairs)
        write_rows_csv(output / "self_vs_other_summary.csv", [self_summary])
        (
            signal_rows,
            signal_summary,
            template_rows,
            template_summary,
            residual_rows,
            residual_pairs,
            residual_summary,
        ) = template_residual_analysis_rows(
            indiv_subjects,
            true_maps,
            pred_maps,
            matrix["left_idx"],
            matrix["right_idx"],
            args.group,
            effective_name,
            training_template=template, template_metadata=metric_provenance(metadata),
        )
        write_rows_csv(output / "signal_recovery.csv", signal_rows)
        write_rows_csv(output / "signal_recovery_summary.csv", [signal_summary])
        write_rows_csv(output / "training_template_per_subject_metrics.csv", template_rows)
        write_rows_csv(output / "training_template_summary.csv", [template_summary])
        write_rows_csv(output / "residual_self_vs_other.csv", residual_rows)
        write_rows_csv(output / "residual_self_vs_other_pairwise.csv", residual_pairs)
        write_rows_csv(output / "residual_self_vs_other_summary.csv", [residual_summary])

    if args.test_loo_sensitivity and records:
        from .template_sensitivity import write_loo_sensitivity
        write_loo_sensitivity(output, indiv_subjects, true_maps, pred_maps,
                              matrix["left_idx"], matrix["right_idx"], args.group, effective_name)

    if args.save_latents and mus:
        np.savez_compressed(
            output / "latent_features.npz",
            subjects=np.asarray(latent_subjects, dtype=str),
            mu=np.stack(mus).astype(np.float32),
            logvar=np.stack(logvars).astype(np.float32),
            architecture_version=np.asarray([ckpt["model_config"]["architecture_version"]]),
        )
        mu_arr = np.stack(mus)
        latent_summary = {
            "group": args.group,
            "model": effective_name,
            "n": len(latent_subjects),
            "mu_between_subject_std_mean": float(mu_arr.std(axis=0, ddof=1).mean()),
            "mu_within_subject_std_mean": float(mu_arr.reshape(mu_arr.shape[0], -1).std(axis=1).mean()),
            "mu_abs_mean": float(np.abs(mu_arr).mean()),
        }
        write_rows_csv(output / "latent_summary.csv", [latent_summary])

    if args.write_nifti:
        nifti_dir = output / "nifti"
        for subject, true, pred in records:
            write_image_nifti(pred, atlas, nifti_dir / f"{subject}_pred.nii.gz")
            write_image_nifti(true, atlas, nifti_dir / f"{subject}_true.nii.gz")

    print(f"[thesis-predict] wrote {output}")
    print(summary)


if __name__ == "__main__":
    main()
