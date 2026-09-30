from __future__ import annotations

"""Posterior-predictive evaluation for the thesis-faithful VRPTR.

The full VRPTR checkpoint learns a Gaussian latent distribution q(z|x). This
module samples that distribution with fixed network weights and fixed U-Net
skip features, then evaluates the induced distribution over task-beta maps.

This is deliberately an evaluation-only analysis. It does not retrain, tune, or
modify any checkpoint. The resulting distribution reflects latent uncertainty,
not full Bayesian uncertainty over model weights.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.stats import norm, spearmanr
from torch.utils.data import DataLoader

from .data import (
    ClinicalPaths,
    build_subject_matrix_cached,
    create_image_space,
    load_subject_list,
    write_image_nifti,
)
from .final_stats_common import write_rows
from .metrics import dominance_class, robust_li, subject_metrics
from .predict_clinical import load_model_checkpoint, prepare_matrix
from .representation_ablation import make_eval_dataset
from .training_common import to_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate VRPTR's latent-induced posterior-predictive distribution.")
    p.add_argument("--analysis", choices=["clean", "corruption", "mc_dropout"], default="clean")
    p.add_argument("--results-root", default="results/thesis_vrptr")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--data-dir", default=None, help="Override the input data root stored in the checkpoint; leaves the checkpoint unchanged.")
    p.add_argument("--deterministic-checkpoint", default="")
    p.add_argument("--subjects", default="splits/test_subjects.txt")
    p.add_argument("--cache-dir", default="results/image_cache")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--sample-batch", type=int, default=2)
    p.add_argument("--max-voxels", type=int, default=25000)
    p.add_argument("--coverage-levels", nargs="+", type=float, default=[0.50, 0.80, 0.90, 0.95])
    p.add_argument("--risk-coverages", nargs="+", type=float, default=[1.00, 0.90, 0.75, 0.50, 0.25])
    p.add_argument("--noise-levels", nargs="+", type=float, default=[0.25, 0.50, 1.00])
    p.add_argument("--channel-dropout-levels", nargs="+", type=float, default=[0.10, 0.25, 0.50])
    p.add_argument("--max-subjects", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--write-nifti", action="store_true")
    p.add_argument("--nifti-max-subjects", type=int, default=5)
    p.add_argument("--out-dir", default="")
    return p.parse_args()


def _autocast(device: torch.device, mode: str):
    enabled = device.type == "cuda" and mode != "none"
    dtype = torch.float16 if mode == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def _stable_seed(base: int, *parts: object) -> int:
    text = "|".join([str(base), *[str(p) for p in parts]])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _valid_flat_indices(y: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isfinite(np.asarray(y).reshape(-1))).astype(np.int64)


def _select_voxels(valid_idx: np.ndarray, max_voxels: int, seed: int) -> np.ndarray:
    valid_idx = np.asarray(valid_idx, dtype=np.int64)
    if max_voxels <= 0 or valid_idx.size <= max_voxels:
        return valid_idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(valid_idx, size=int(max_voxels), replace=False)).astype(np.int64)


def _repeat_skips(skips: list[torch.Tensor], batch: int) -> list[torch.Tensor]:
    if batch == 1:
        return skips
    return [s.expand(batch, *s.shape[1:]).contiguous() for s in skips]


def _decode(model, skips: list[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
    transformed = model.bottleneck.decode(z)
    dec = model.decoder(transformed, _repeat_skips(skips, int(z.shape[0])))
    return torch.nan_to_num(model.head(dec), nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)


def _posterior_samples(
    model,
    x: torch.Tensor,
    n_samples: int,
    sample_batch: int,
    selected_idx: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    device: torch.device,
    amp: str,
    seed: int,
) -> dict:
    """Sample q(z|x) and stream map moments while retaining selected voxels."""
    model.eval()
    with torch.no_grad(), _autocast(device, amp):
        prepared = model._prepare_input(x, 0.0)
        deepest, skips = model.encoder(prepared)
        latent = model.bottleneck.encode(deepest)
        mu = latent["mu_tokens"]
        logvar = latent["logvar_tokens"]
        std_z = torch.exp(0.5 * logvar)
        mu_map = _decode(model, skips, mu)[0].float().cpu().numpy()

    flat_size = int(mu_map.size)
    selected_idx = np.asarray(selected_idx, dtype=np.int64)
    samples_selected = np.empty((int(n_samples), int(selected_idx.size)), dtype=np.float32)
    sum_map = np.zeros(flat_size, dtype=np.float64)
    sumsq_map = np.zeros(flat_size, dtype=np.float64)
    li_samples = np.empty(int(n_samples), dtype=np.float64)

    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))
    offset = 0
    with torch.no_grad():
        while offset < int(n_samples):
            b = min(max(1, int(sample_batch)), int(n_samples) - offset)
            with _autocast(device, amp):
                eps = torch.randn(
                    (b, *mu.shape[1:]),
                    generator=generator,
                    device=device,
                    dtype=mu.dtype,
                )
                z = mu.expand(b, *mu.shape[1:]) + eps * std_z.expand(b, *std_z.shape[1:])
                pred = _decode(model, skips, z).float().cpu().numpy()
            for j in range(b):
                flat = pred[j].reshape(-1).astype(np.float64, copy=False)
                sum_map += flat
                sumsq_map += flat * flat
                samples_selected[offset + j] = flat[selected_idx].astype(np.float32, copy=False)
                li_samples[offset + j] = robust_li(pred[j], left_idx, right_idx)
            offset += b

    mean_flat = sum_map / max(int(n_samples), 1)
    var_flat = np.maximum(sumsq_map / max(int(n_samples), 1) - mean_flat * mean_flat, 0.0)
    sd_flat = np.sqrt(var_flat)
    return {
        "mu_map": mu_map.astype(np.float32),
        "mean_map": mean_flat.reshape(mu_map.shape).astype(np.float32),
        "sd_map": sd_flat.reshape(mu_map.shape).astype(np.float32),
        "samples_selected": samples_selected,
        "li_samples": li_samples,
        "mu_tokens": mu[0].float().cpu().numpy().astype(np.float32),
        "logvar_tokens": logvar[0].float().cpu().numpy().astype(np.float32),
    }


def _enable_dropout_only(model) -> None:
    model.eval()
    dropout_classes = (
        torch.nn.Dropout,
        torch.nn.Dropout1d,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
        torch.nn.AlphaDropout,
    )
    for module in model.modules():
        if isinstance(module, dropout_classes):
            module.train()


def _mc_dropout_samples(
    model,
    x: torch.Tensor,
    n_samples: int,
    selected_idx: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    device: torch.device,
    amp: str,
    seed: int,
) -> dict:
    model.eval()
    with torch.no_grad(), _autocast(device, amp):
        deterministic_map = model(x, sample_latent=False)["beta"][0].float().cpu().numpy()
    _enable_dropout_only(model)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    selected_idx = np.asarray(selected_idx, dtype=np.int64)
    samples_selected = np.empty((int(n_samples), int(selected_idx.size)), dtype=np.float32)
    li_samples = np.empty(int(n_samples), dtype=np.float64)
    sum_map = None
    sumsq_map = None
    first_map = deterministic_map.copy()
    with torch.no_grad():
        for s in range(int(n_samples)):
            with _autocast(device, amp):
                pred_t = model(x, sample_latent=False)["beta"]
            pred = pred_t[0].float().cpu().numpy()
            flat = pred.reshape(-1).astype(np.float64, copy=False)
            if sum_map is None:
                sum_map = np.zeros_like(flat, dtype=np.float64)
                sumsq_map = np.zeros_like(flat, dtype=np.float64)
            sum_map += flat
            sumsq_map += flat * flat
            samples_selected[s] = flat[selected_idx]
            li_samples[s] = robust_li(pred, left_idx, right_idx)
    model.eval()
    mean_flat = sum_map / max(int(n_samples), 1)
    var_flat = np.maximum(sumsq_map / max(int(n_samples), 1) - mean_flat * mean_flat, 0.0)
    return {
        "mu_map": first_map.astype(np.float32),
        "mean_map": mean_flat.reshape(first_map.shape).astype(np.float32),
        "sd_map": np.sqrt(var_flat).reshape(first_map.shape).astype(np.float32),
        "samples_selected": samples_selected,
        "li_samples": li_samples,
        "mu_tokens": np.asarray([], dtype=np.float32),
        "logvar_tokens": np.asarray([], dtype=np.float32),
    }


def _empirical_crps(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """CRPS per voxel using an O(S log S) ensemble formula."""
    x = np.asarray(samples, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64).reshape(1, -1)
    first = np.mean(np.abs(x - y), axis=0)
    xs = np.sort(x, axis=0)
    s = xs.shape[0]
    weights = (2.0 * np.arange(1, s + 1, dtype=np.float64) - s - 1.0).reshape(-1, 1)
    half_pair = np.sum(weights * xs, axis=0) / float(s * s)
    return first - half_pair


def _gaussian_crps(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    sd = np.maximum(np.asarray(sd, dtype=np.float64), 1e-6)
    z = (y - mean) / sd
    return sd * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))


def _gaussian_nll(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd = np.maximum(np.asarray(sd, dtype=np.float64), 1e-6)
    err = np.asarray(y, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return 0.5 * (math.log(2.0 * math.pi) + 2.0 * np.log(sd) + (err / sd) ** 2)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def _risk_curve(abs_error: np.ndarray, uncertainty: np.ndarray, coverages: Iterable[float]) -> tuple[list[dict], float, float]:
    e = np.asarray(abs_error, dtype=np.float64)
    u = np.asarray(uncertainty, dtype=np.float64)
    mask = np.isfinite(e) & np.isfinite(u)
    e = e[mask]
    u = u[mask]
    if e.size == 0:
        return [], float("nan"), float("nan")
    order = np.argsort(u)
    oracle = np.argsort(e)
    cov = sorted(set(float(c) for c in coverages if 0 < float(c) <= 1.0))
    rows = []
    risks, oracle_risks = [], []
    for c in cov:
        n = max(1, int(round(c * e.size)))
        risk = float(e[order[:n]].mean())
        oracle_risk = float(e[oracle[:n]].mean())
        rows.append({"coverage": c, "mae": risk, "oracle_mae": oracle_risk, "random_expected_mae": float(e.mean())})
        risks.append(risk)
        oracle_risks.append(oracle_risk)
    x = np.asarray(cov, dtype=float)
    if x.size == 1:
        aurc = risks[0]
        oracle_aurc = oracle_risks[0]
    else:
        aurc = float(np.trapz(np.asarray(risks), x) / max(x.max() - x.min(), 1e-12))
        oracle_aurc = float(np.trapz(np.asarray(oracle_risks), x) / max(x.max() - x.min(), 1e-12))
    return rows, aurc, oracle_aurc


def _posterior_metrics(
    subject: str,
    condition: str,
    true_map: np.ndarray,
    result: dict,
    selected_idx: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    coverage_levels: list[float],
    risk_coverages: list[float],
    source: str,
) -> tuple[dict, list[dict], list[dict], dict]:
    true_flat = np.asarray(true_map, dtype=np.float64).reshape(-1)
    mean_map = result["mean_map"]
    mu_map = result["mu_map"]
    sd_map = result["sd_map"]
    selected_idx = np.asarray(selected_idx, dtype=np.int64)
    y = true_flat[selected_idx]
    samples = np.asarray(result["samples_selected"], dtype=np.float64)
    mean = np.asarray(mean_map).reshape(-1)[selected_idx].astype(np.float64)
    sd = np.asarray(sd_map).reshape(-1)[selected_idx].astype(np.float64)
    abs_error = np.abs(y - mean)

    sample_mean = samples.mean(axis=0)
    # Streaming map mean and selected sample mean should agree up to rounding.
    mean_consistency = float(np.mean(np.abs(sample_mean - mean)))
    empirical_crps = _empirical_crps(samples, y)
    gaussian_crps = _gaussian_crps(y, mean, sd)
    gaussian_nll = _gaussian_nll(y, mean, sd)
    pit = (np.sum(samples < y[None, :], axis=0) + 0.5 * np.sum(samples == y[None, :], axis=0) + 0.5) / (samples.shape[0] + 1.0)
    hist, _ = np.histogram(pit, bins=np.linspace(0.0, 1.0, 11))
    pit_freq = hist / max(hist.sum(), 1)
    pit_ece = float(0.5 * np.sum(np.abs(pit_freq - 0.1)))

    point_mu = subject_metrics(true_map, mu_map, left_idx, right_idx)
    point_pp = subject_metrics(true_map, mean_map, left_idx, right_idx)
    risk_rows, aurc, oracle_aurc = _risk_curve(abs_error, sd, risk_coverages)
    random_mae = float(abs_error.mean())
    selective_gain = (random_mae - aurc) / max(random_mae, 1e-12)

    li_samples = np.asarray(result["li_samples"], dtype=np.float64)
    li_samples = li_samples[np.isfinite(li_samples)]
    li_true = robust_li(true_map, left_idx, right_idx)
    li_mu = robust_li(mu_map, left_idx, right_idx)
    li_pp = robust_li(mean_map, left_idx, right_idx)
    true_class = dominance_class(li_true)
    class_probs = np.asarray([np.mean([dominance_class(v) == k for v in li_samples]) for k in range(3)], dtype=float) if li_samples.size else np.full(3, np.nan)
    onehot = np.zeros(3, dtype=float)
    onehot[true_class] = 1.0
    brier = float(np.sum((class_probs - onehot) ** 2)) if np.all(np.isfinite(class_probs)) else float("nan")
    log_score = float(-math.log(max(float(class_probs[true_class]), 1e-12))) if np.all(np.isfinite(class_probs)) else float("nan")
    entropy = float(-np.sum(class_probs * np.log(np.maximum(class_probs, 1e-12)))) if np.all(np.isfinite(class_probs)) else float("nan")

    row = {
        "source": source,
        "subject": subject,
        "condition": condition,
        "n_samples": int(samples.shape[0]),
        "n_calibration_voxels": int(samples.shape[1]),
        "posterior_mean_vs_sample_mean_mae": mean_consistency,
        "posterior_sd_mean": float(np.mean(sd)),
        "posterior_sd_median": float(np.median(sd)),
        "posterior_sd_p90": float(np.quantile(sd, 0.90)),
        "posterior_sd_max": float(np.max(sd)),
        "voxel_uncertainty_abs_error_spearman": _safe_spearman(sd, abs_error),
        "empirical_crps_mean": float(np.mean(empirical_crps)),
        "gaussian_crps_raw_mean": float(np.mean(gaussian_crps)),
        "gaussian_nll_raw_mean": float(np.mean(gaussian_nll)),
        "pit_ece": pit_ece,
        "selective_aurc": aurc,
        "oracle_aurc": oracle_aurc,
        "random_expected_mae": random_mae,
        "selective_gain_fraction": selective_gain,
        "li_true": li_true,
        "li_mu": li_mu,
        "li_posterior_mean": li_pp,
        "li_posterior_sd": float(np.std(li_samples, ddof=1)) if li_samples.size > 1 else 0.0,
        "dominance_true": int(true_class),
        "dominance_prob_right": float(class_probs[0]),
        "dominance_prob_bilateral": float(class_probs[1]),
        "dominance_prob_left": float(class_probs[2]),
        "dominance_brier": brier,
        "dominance_log_score": log_score,
        "dominance_entropy": entropy,
    }
    row.update({f"mu_{k}": v for k, v in point_mu.items()})
    row.update({f"posterior_mean_{k}": v for k, v in point_pp.items()})

    coverage_rows = []
    for level in coverage_levels:
        alpha = 1.0 - float(level)
        lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
        covered = (y >= lo) & (y <= hi)
        width = hi - lo
        coverage_rows.append({
            "source": source,
            "subject": subject,
            "condition": condition,
            "nominal_coverage": float(level),
            "empirical_coverage": float(np.mean(covered)),
            "coverage_error": float(np.mean(covered) - level),
            "absolute_coverage_error": float(abs(np.mean(covered) - level)),
            "mean_interval_width": float(np.mean(width)),
            "median_interval_width": float(np.median(width)),
        })
    if li_samples.size:
        for level in coverage_levels:
            alpha = 1.0 - float(level)
            lo, hi = np.quantile(li_samples, [alpha / 2.0, 1.0 - alpha / 2.0])
            row[f"li_interval_{int(round(level * 100))}_low"] = float(lo)
            row[f"li_interval_{int(round(level * 100))}_high"] = float(hi)
            row[f"li_interval_{int(round(level * 100))}_covered"] = float(lo <= li_true <= hi)

    for rr in risk_rows:
        rr.update({"source": source, "subject": subject, "condition": condition})

    arrays = {
        "truth": y.astype(np.float32),
        "posterior_mean": mean.astype(np.float32),
        "posterior_sd": sd.astype(np.float32),
        "mu_prediction": np.asarray(mu_map).reshape(-1)[selected_idx].astype(np.float32),
        "selected_flat_indices": selected_idx.astype(np.int64),
    }
    return row, coverage_rows, risk_rows, arrays


def _condition_specs(args: argparse.Namespace) -> list[tuple[str, str, float]]:
    if args.analysis == "clean" or args.analysis == "mc_dropout":
        return [("clean", "none", 0.0)]
    out = [("clean", "none", 0.0)]
    out.extend((f"gaussian_noise_{v:g}", "gaussian_noise", float(v)) for v in args.noise_levels)
    out.extend((f"channel_dropout_{v:g}", "channel_dropout", float(v)) for v in args.channel_dropout_levels)
    return out


def _apply_corruption(x: torch.Tensor, y: torch.Tensor, family: str, severity: float, seed: int) -> torch.Tensor:
    if family == "none" or severity <= 0:
        return x
    out = x.clone()
    generator = torch.Generator(device=x.device.type if x.device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))
    valid = torch.isfinite(y[:, :1])
    if family == "gaussian_noise":
        noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
        out = torch.where(valid.expand_as(out), out + float(severity) * noise, out)
    elif family == "channel_dropout":
        channels = int(out.shape[1])
        n_drop = min(channels, max(1, int(round(float(severity) * channels))))
        order = torch.randperm(channels, generator=generator, device=out.device)
        out[:, order[:n_drop], ...] = 0.0
    else:
        raise ValueError(f"Unknown corruption family: {family}")
    return torch.nan_to_num(out, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)


def _deterministic_point(model, x: torch.Tensor, device: torch.device, amp: str) -> np.ndarray:
    model.eval()
    with torch.no_grad(), _autocast(device, amp):
        pred = model(x, sample_latent=False)["beta"]
    return pred[0].float().cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    checkpoint = Path(args.checkpoint) if args.checkpoint else root / "checkpoints" / "full" / "final_thesis_vrptr_full.pt"
    deterministic_checkpoint = Path(args.deterministic_checkpoint) if args.deterministic_checkpoint else root / "checkpoints" / "deterministic" / "final_thesis_vrptr_deterministic.pt"
    output = Path(args.out_dir) if args.out_dir else root / "final_analysis" / "posterior_predictive" / args.analysis
    output.mkdir(parents=True, exist_ok=True)
    calibration_dir = output / "calibration_arrays"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, scaler, ckpt = load_model_checkpoint(checkpoint, device)
    if not bool(model.cfg.use_vae):
        raise RuntimeError(f"Posterior sampling requires a VAE checkpoint; got {checkpoint}")
    det_model = None
    det_scaler = None
    if deterministic_checkpoint.exists() and args.analysis != "mc_dropout":
        det_model, det_scaler, _ = load_model_checkpoint(deterministic_checkpoint, device)

    paths = ClinicalPaths(**ckpt["paths"])
    if args.data_dir is not None:
        paths.data_dir = args.data_dir
    atlas = create_image_space(paths)
    subjects = load_subject_list(args.subjects)
    if args.max_subjects > 0:
        subjects = subjects[: int(args.max_subjects)]
    raw = build_subject_matrix_cached(
        subjects,
        paths,
        atlas,
        cache_dir=args.cache_dir,
        cache_tag=f"posterior_predictive_{Path(args.subjects).stem}",
        refresh_cache=False,
    )
    matrix = prepare_matrix(raw, scaler)
    dataset = make_eval_dataset(matrix, mode="none", seed=args.seed)
    det_dataset = make_eval_dataset(prepare_matrix(raw, det_scaler), mode="none", seed=args.seed) if det_model is not None else None
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    det_loader = iter(DataLoader(det_dataset, batch_size=1, shuffle=False, num_workers=0)) if det_dataset is not None else None

    per_subject_rows: list[dict] = []
    coverage_rows: list[dict] = []
    risk_rows: list[dict] = []
    deterministic_rows: list[dict] = []
    condition_specs = _condition_specs(args)

    for subject_index, batch in enumerate(loader):
        bd = to_device(batch, device)
        subject = str(batch["subject"][0])
        true_map = bd["y"][0].float().cpu().numpy()
        valid_idx = _valid_flat_indices(true_map)
        selected_idx = _select_voxels(valid_idx, args.max_voxels, _stable_seed(args.seed, subject, "voxels"))
        det_bd = to_device(next(det_loader), device) if det_loader is not None else None
        if det_bd is not None and str(det_bd["subject"][0]) != subject:
            raise RuntimeError("Full and deterministic datasets are not subject-aligned")

        for condition, family, severity in condition_specs:
            corruption_seed = _stable_seed(args.seed, subject, family, "corruption")
            x = _apply_corruption(bd["x"], bd["y"], family, severity, corruption_seed)
            n_samples = int(args.samples)
            sample_seed = _stable_seed(args.seed, subject, condition, args.analysis)
            if args.analysis == "mc_dropout":
                result = _mc_dropout_samples(
                    model, x, n_samples, selected_idx, matrix["left_idx"], matrix["right_idx"],
                    device, args.amp, sample_seed,
                )
                source = "full_mc_dropout"
            else:
                result = _posterior_samples(
                    model, x, n_samples, args.sample_batch, selected_idx,
                    matrix["left_idx"], matrix["right_idx"], device, args.amp, sample_seed,
                )
                source = "full_vae_latent"

            row, cov, risk, arrays = _posterior_metrics(
                subject, condition, true_map, result, selected_idx,
                matrix["left_idx"], matrix["right_idx"],
                args.coverage_levels, args.risk_coverages, source,
            )
            row.update({"corruption_family": family, "corruption_severity": severity})
            per_subject_rows.append(row)
            coverage_rows.extend(cov)
            risk_rows.extend(risk)

            np.savez_compressed(
                calibration_dir / f"{subject}__{condition}__{source}.npz",
                subject=np.asarray([subject]),
                condition=np.asarray([condition]),
                source=np.asarray([source]),
                **arrays,
            )

            if det_model is not None and det_bd is not None:
                det_x = _apply_corruption(det_bd["x"], det_bd["y"], family, severity, corruption_seed)
                det_pred = _deterministic_point(det_model, det_x, device, args.amp)
                det_metrics = subject_metrics(true_map, det_pred, matrix["left_idx"], matrix["right_idx"])
                det_metrics.update({
                    "subject": subject,
                    "condition": condition,
                    "corruption_family": family,
                    "corruption_severity": severity,
                    "model": "deterministic_vrptr",
                })
                deterministic_rows.append(det_metrics)

            if args.write_nifti and subject_index < int(args.nifti_max_subjects):
                nifti_dir = output / "nifti" / condition
                write_image_nifti(result["mean_map"], atlas, nifti_dir / f"{subject}_posterior_mean.nii.gz")
                write_image_nifti(result["sd_map"], atlas, nifti_dir / f"{subject}_posterior_sd.nii.gz")
                write_image_nifti(result["mu_map"], atlas, nifti_dir / f"{subject}_mu_prediction.nii.gz")

            print(
                f"[posterior] {subject} {condition}: "
                f"r_mu={row.get('mu_pearson_r', float('nan')):.4f} "
                f"r_pp={row.get('posterior_mean_pearson_r', float('nan')):.4f} "
                f"sd={row.get('posterior_sd_mean', float('nan')):.5f} "
                f"rho(sd,error)={row.get('voxel_uncertainty_abs_error_spearman', float('nan')):.3f}"
            )

    write_rows(output / "posterior_per_subject.csv", per_subject_rows)
    write_rows(output / "posterior_coverage_per_subject.csv", coverage_rows)
    write_rows(output / "posterior_risk_coverage.csv", risk_rows)
    write_rows(output / "deterministic_point_per_subject.csv", deterministic_rows)
    manifest = {
        "analysis": args.analysis,
        "checkpoint": str(checkpoint),
        "deterministic_checkpoint": str(deterministic_checkpoint) if deterministic_checkpoint.exists() else "",
        "subjects": str(args.subjects),
        "n_subjects": len(subjects),
        "samples": int(args.samples),
        "sample_batch": int(args.sample_batch),
        "max_voxels": int(args.max_voxels),
        "coverage_levels": args.coverage_levels,
        "risk_coverages": args.risk_coverages,
        "condition_specs": condition_specs,
        "seed": int(args.seed),
        "terminology_note": (
            "Samples are induced by q(z|x) with fixed network weights and fixed skip features. "
            "They quantify latent uncertainty, not full Bayesian weight uncertainty."
        ),
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[posterior] wrote {output}")


if __name__ == "__main__":
    main()
