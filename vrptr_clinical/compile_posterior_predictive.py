from __future__ import annotations

"""Compile posterior-predictive quality, calibration, and utility analyses."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, spearmanr

from .final_stats_common import bh_adjust, paired_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile VRPTR posterior-predictive analyses.")
    p.add_argument("--results-root", default="results/thesis_vrptr")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fit-voxels", type=int, default=500000)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--signflip", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="")
    return p.parse_args()


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={"subject": str, "condition": str, "source": str})
    if "subject" in frame:
        frame["subject"] = frame["subject"].astype(str)
    return frame


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


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def _fit_scale_floor(error: np.ndarray, raw_sd: np.ndarray) -> tuple[float, float, bool, float]:
    e = np.asarray(error, dtype=np.float64)
    s = np.maximum(np.asarray(raw_sd, dtype=np.float64), 0.0)
    mask = np.isfinite(e) & np.isfinite(s)
    e = e[mask]
    s = s[mask]
    if e.size < 10:
        return 1.0, max(float(np.nanstd(e)), 1e-3), False, float("nan")
    rmse = max(float(np.sqrt(np.mean(e * e))), 1e-4)
    s_rms = max(float(np.sqrt(np.mean(s * s))), 1e-6)
    a0 = max(min(rmse / s_rms, 100.0), 1e-3)
    tau0 = max(0.25 * rmse, 1e-4)

    def objective(theta: np.ndarray) -> float:
        a = math.exp(float(theta[0]))
        tau = math.exp(float(theta[1]))
        var = tau * tau + (a * s) ** 2
        return float(np.mean(0.5 * (math.log(2.0 * math.pi) + np.log(var) + (e * e) / var)))

    result = minimize(
        objective,
        np.log([a0, tau0]),
        method="L-BFGS-B",
        bounds=[(-8.0, 8.0), (-10.0, 5.0)],
    )
    theta = result.x if np.all(np.isfinite(result.x)) else np.log([a0, tau0])
    return math.exp(float(theta[0])), math.exp(float(theta[1])), bool(result.success), float(result.fun)


def _crossfit_calibration(clean_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    array_dir = clean_dir / "calibration_arrays"
    files = sorted(array_dir.glob("*__clean__full_vae_latent.npz"))
    if not files:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    records = []
    for path in files:
        arr = _load_arrays(path)
        records.append({
            "subject": str(arr["subject"].reshape(-1)[0]),
            "path": path,
            "truth": arr["truth"].astype(np.float64),
            "mean": arr["posterior_mean"].astype(np.float64),
            "sd": arr["posterior_sd"].astype(np.float64),
        })
    subjects = sorted(r["subject"] for r in records)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(subjects))
    fold_map = {subjects[int(idx)]: int(pos % max(2, args.folds)) for pos, idx in enumerate(perm)}
    per_subject_rows: list[dict] = []
    param_rows: list[dict] = []
    levels = [0.50, 0.80, 0.90, 0.95]

    for fold in sorted(set(fold_map.values())):
        train = [r for r in records if fold_map[r["subject"]] != fold]
        test = [r for r in records if fold_map[r["subject"]] == fold]
        e = np.concatenate([r["truth"] - r["mean"] for r in train])
        s = np.concatenate([r["sd"] for r in train])
        valid = np.isfinite(e) & np.isfinite(s)
        e = e[valid]
        s = s[valid]
        if e.size > int(args.fit_voxels):
            choose = rng.choice(e.size, size=int(args.fit_voxels), replace=False)
            e = e[choose]
            s = s[choose]
        constant_tau = max(float(np.sqrt(np.mean(e * e))), 1e-6)
        scale_a, floor_tau, success, objective = _fit_scale_floor(e, s)
        param_rows.append({
            "fold": fold,
            "n_calibration_subjects": len(train),
            "n_evaluation_subjects": len(test),
            "n_fit_voxels": int(e.size),
            "constant_tau": constant_tau,
            "heteroscedastic_scale_a": scale_a,
            "heteroscedastic_floor_tau": floor_tau,
            "optimizer_success": int(success),
            "training_nll": objective,
        })
        for r in test:
            y = r["truth"]
            m = r["mean"]
            raw = np.maximum(r["sd"], 1e-8)
            const_sd = np.full_like(raw, constant_tau)
            hetero_sd = np.sqrt(floor_tau * floor_tau + (scale_a * raw) ** 2)
            row = {
                "subject": r["subject"],
                "fold": fold,
                "constant_tau": constant_tau,
                "heteroscedastic_scale_a": scale_a,
                "heteroscedastic_floor_tau": floor_tau,
                "raw_sd_mean": float(np.mean(raw)),
                "constant_nll": float(np.mean(_gaussian_nll(y, m, const_sd))),
                "heteroscedastic_nll": float(np.mean(_gaussian_nll(y, m, hetero_sd))),
                "raw_nll": float(np.mean(_gaussian_nll(y, m, raw))),
                "constant_crps": float(np.mean(_gaussian_crps(y, m, const_sd))),
                "heteroscedastic_crps": float(np.mean(_gaussian_crps(y, m, hetero_sd))),
                "raw_crps": float(np.mean(_gaussian_crps(y, m, raw))),
            }
            for level in levels:
                z = float(norm.ppf(0.5 + level / 2.0))
                for name, sd in [("constant", const_sd), ("heteroscedastic", hetero_sd), ("raw", raw)]:
                    covered = np.abs(y - m) <= z * sd
                    row[f"{name}_coverage_{int(level*100)}"] = float(np.mean(covered))
                    row[f"{name}_coverage_abs_error_{int(level*100)}"] = float(abs(np.mean(covered) - level))
                    row[f"{name}_width_{int(level*100)}"] = float(np.mean(2.0 * z * sd))
            per_subject_rows.append(row)

    per = pd.DataFrame(per_subject_rows)
    params = pd.DataFrame(param_rows)
    rng_stats = np.random.default_rng(args.seed + 1701)
    effects: list[dict] = []
    contrasts = [
        ("constant_minus_heteroscedastic_nll", per["constant_nll"] - per["heteroscedastic_nll"], "positive means VAE spatial spread improves cross-fitted NLL"),
        ("constant_minus_heteroscedastic_crps", per["constant_crps"] - per["heteroscedastic_crps"], "positive means VAE spatial spread improves cross-fitted CRPS"),
        ("raw_minus_heteroscedastic_nll", per["raw_nll"] - per["heteroscedastic_nll"], "positive means cross-fitted scaling improves raw VAE NLL"),
        ("raw_minus_heteroscedastic_crps", per["raw_crps"] - per["heteroscedastic_crps"], "positive means cross-fitted scaling improves raw VAE CRPS"),
    ]
    for level in [80, 90, 95]:
        contrasts.append((
            f"constant_minus_heteroscedastic_abs_coverage_error_{level}",
            per[f"constant_coverage_abs_error_{level}"] - per[f"heteroscedastic_coverage_abs_error_{level}"],
            "positive means VAE spatial spread improves interval calibration over a constant-width baseline",
        ))
    for name, values, meaning in contrasts:
        effects.append({"analysis_family": "crossfit_calibration", "contrast": name, "positive_effect_interpretation": meaning, **paired_stats(values.to_numpy(float), rng_stats, args.bootstrap, args.signflip)})
    effects_frame = pd.DataFrame(effects)
    if not effects_frame.empty:
        effects_frame["bh_q"] = bh_adjust(effects_frame["signflip_p_two_sided"].to_numpy(float))
    return per, params, effects_frame


def _clean_effects(clean: pd.DataFrame, coverage: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if clean.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(args.seed + 23)
    rows: list[dict] = []
    contrasts = [
        ("posterior_mean_minus_mu_pearson", clean["posterior_mean_pearson_r"] - clean["mu_pearson_r"], "positive means posterior averaging improves Pearson correlation"),
        ("posterior_mean_minus_mu_r2", clean["posterior_mean_r2"] - clean["mu_r2"], "positive means posterior averaging improves R2"),
        ("posterior_mean_minus_mu_dice_auc", clean["posterior_mean_dice_auc"] - clean["mu_dice_auc"], "positive means posterior averaging improves activation overlap"),
        ("mu_minus_posterior_mean_li_abs_error", clean["mu_li_abs_err"] - clean["posterior_mean_li_abs_err"], "positive means posterior averaging improves laterality error"),
        ("voxel_uncertainty_error_spearman", clean["voxel_uncertainty_abs_error_spearman"], "positive means posterior SD ranks voxelwise error"),
        ("selective_gain_fraction", clean["selective_gain_fraction"], "positive means withholding high-uncertainty voxels lowers error"),
    ]
    for name, values, meaning in contrasts:
        rows.append({"analysis_family": "clean_utility", "contrast": name, "positive_effect_interpretation": meaning, **paired_stats(values.to_numpy(float), rng, args.bootstrap, args.signflip)})
    if not coverage.empty:
        for level, frame in coverage.groupby("nominal_coverage"):
            # This is descriptive: zero is perfect, so test signed error only.
            rows.append({
                "analysis_family": "raw_interval_calibration",
                "contrast": f"raw_coverage_error_{int(round(float(level)*100))}",
                "positive_effect_interpretation": "zero is ideal; negative indicates undercoverage and positive indicates overcoverage",
                **paired_stats(frame["coverage_error"].to_numpy(float), rng, args.bootstrap, args.signflip),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["bh_q"] = bh_adjust(out["signflip_p_two_sided"].to_numpy(float))
    return out


def _summarize_clean(clean: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if clean.empty:
        return pd.DataFrame()
    metrics = [
        "mu_pearson_r", "posterior_mean_pearson_r", "mu_r2", "posterior_mean_r2",
        "mu_dice_auc", "posterior_mean_dice_auc", "posterior_sd_mean",
        "voxel_uncertainty_abs_error_spearman", "empirical_crps_mean", "gaussian_nll_raw_mean",
        "pit_ece", "selective_gain_fraction", "li_posterior_sd", "dominance_brier",
        "dominance_log_score", "dominance_entropy",
    ]
    row = {"n_subjects": int(clean["subject"].nunique())}
    for metric in metrics:
        if metric in clean:
            vals = pd.to_numeric(clean[metric], errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(vals.mean()) if vals.size else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else np.nan
    rows.append(row)
    if not coverage.empty:
        for level, frame in coverage.groupby("nominal_coverage"):
            vals = frame["empirical_coverage"].to_numpy(float)
            rows[0][f"raw_empirical_coverage_{int(round(float(level)*100))}_mean"] = float(np.nanmean(vals))
            rows[0][f"raw_interval_width_{int(round(float(level)*100))}_mean"] = float(np.nanmean(frame["mean_interval_width"].to_numpy(float)))
    return pd.DataFrame(rows)


def _corruption_analysis(corruption: pd.DataFrame, deterministic: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if corruption.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    summary_rows = []
    for condition, frame in corruption.groupby("condition", sort=False):
        row = {
            "condition": condition,
            "corruption_family": str(frame["corruption_family"].iloc[0]),
            "corruption_severity": float(frame["corruption_severity"].iloc[0]),
            "n": int(frame["subject"].nunique()),
        }
        for metric in [
            "mu_pearson_r", "posterior_mean_pearson_r", "posterior_sd_mean",
            "voxel_uncertainty_abs_error_spearman", "empirical_crps_mean",
            "selective_gain_fraction", "dominance_entropy", "li_posterior_sd",
        ]:
            if metric in frame:
                vals = frame[metric].to_numpy(float)
                row[f"{metric}_mean"] = float(np.nanmean(vals))
                row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1))
        if not deterministic.empty:
            d = deterministic[deterministic.condition == condition]
            if not d.empty:
                row["deterministic_pearson_r_mean"] = float(np.nanmean(d["pearson_r"].to_numpy(float)))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    clean = corruption[corruption.condition == "clean"].set_index("subject")
    rng = np.random.default_rng(args.seed + 501)
    effects = []
    for condition in sorted(set(corruption.condition) - {"clean"}):
        other = corruption[corruption.condition == condition].set_index("subject")
        subjects = sorted(set(clean.index.astype(str)) & set(other.index.astype(str)))
        comparisons = [
            ("posterior_sd_increase", other.loc[subjects, "posterior_sd_mean"] - clean.loc[subjects, "posterior_sd_mean"], "positive means uncertainty rises under corruption"),
            ("posterior_mean_pearson_loss", clean.loc[subjects, "posterior_mean_pearson_r"] - other.loc[subjects, "posterior_mean_pearson_r"], "positive means performance degrades under corruption"),
            ("mu_pearson_loss", clean.loc[subjects, "mu_pearson_r"] - other.loc[subjects, "mu_pearson_r"], "positive means z=mu performance degrades under corruption"),
            ("dominance_entropy_increase", other.loc[subjects, "dominance_entropy"] - clean.loc[subjects, "dominance_entropy"], "positive means dominance uncertainty rises under corruption"),
            ("li_posterior_sd_increase", other.loc[subjects, "li_posterior_sd"] - clean.loc[subjects, "li_posterior_sd"], "positive means LI uncertainty rises under corruption"),
        ]
        for metric, values, meaning in comparisons:
            effects.append({"analysis_family": "corruption_vs_clean", "condition": condition, "contrast": metric, "positive_effect_interpretation": meaning, **paired_stats(values.to_numpy(float), rng, args.bootstrap, args.signflip)})
        if not deterministic.empty:
            d_clean = deterministic[deterministic.condition == "clean"].set_index("subject")
            d_other = deterministic[deterministic.condition == condition].set_index("subject")
            common = sorted(set(subjects) & set(d_clean.index.astype(str)) & set(d_other.index.astype(str)))
            vae_loss = clean.loc[common, "posterior_mean_pearson_r"].to_numpy(float) - other.loc[common, "posterior_mean_pearson_r"].to_numpy(float)
            det_loss = d_clean.loc[common, "pearson_r"].to_numpy(float) - d_other.loc[common, "pearson_r"].to_numpy(float)
            effects.append({"analysis_family": "corruption_robustness", "condition": condition, "contrast": "deterministic_minus_vae_performance_loss", "positive_effect_interpretation": "positive means posterior averaging degrades less than deterministic VRPTR", **paired_stats(det_loss - vae_loss, rng, args.bootstrap, args.signflip)})

    slope_rows = []
    for family in sorted(set(corruption.corruption_family) - {"none"}):
        family_frame = corruption[(corruption.corruption_family == family) | (corruption.condition == "clean")].copy()
        for subject, frame in family_frame.groupby("subject"):
            frame = frame.sort_values("corruption_severity")
            severity = frame["corruption_severity"].to_numpy(float)
            if np.unique(severity).size < 3:
                continue
            uncertainty = frame["posterior_sd_mean"].to_numpy(float)
            error = 1.0 - frame["posterior_mean_pearson_r"].to_numpy(float)
            entropy = frame["dominance_entropy"].to_numpy(float)
            slope_rows.append({
                "subject": subject,
                "corruption_family": family,
                "uncertainty_slope": float(np.polyfit(severity, uncertainty, 1)[0]),
                "error_slope": float(np.polyfit(severity, error, 1)[0]),
                "dominance_entropy_slope": float(np.polyfit(severity, entropy, 1)[0]),
                "uncertainty_error_spearman_across_severity": float(spearmanr(uncertainty, error).statistic) if np.std(uncertainty) > 1e-12 and np.std(error) > 1e-12 else np.nan,
            })
    slopes = pd.DataFrame(slope_rows)
    if not slopes.empty:
        for family, frame in slopes.groupby("corruption_family"):
            for metric, meaning in [
                ("uncertainty_slope", "positive means posterior uncertainty rises monotonically with corruption severity"),
                ("error_slope", "positive means prediction error rises monotonically with corruption severity"),
                ("dominance_entropy_slope", "positive means dominance entropy rises with corruption severity"),
                ("uncertainty_error_spearman_across_severity", "positive means uncertainty tracks degradation within subjects"),
            ]:
                effects.append({"analysis_family": "corruption_severity", "condition": family, "contrast": metric, "positive_effect_interpretation": meaning, **paired_stats(frame[metric].to_numpy(float), rng, args.bootstrap, args.signflip)})
    effects_frame = pd.DataFrame(effects)
    if not effects_frame.empty:
        effects_frame["bh_q"] = bh_adjust(effects_frame["signflip_p_two_sided"].to_numpy(float))
    return summary, effects_frame, slopes


def _mc_dropout_comparison(clean: pd.DataFrame, mc: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if clean.empty or mc.empty:
        return pd.DataFrame()
    a = clean.set_index("subject")
    b = mc.set_index("subject")
    subjects = sorted(set(a.index.astype(str)) & set(b.index.astype(str)))
    rng = np.random.default_rng(args.seed + 800)
    rows = []
    comparisons = [
        ("vae_minus_mc_dropout_uncertainty_error_spearman", a.loc[subjects, "voxel_uncertainty_abs_error_spearman"] - b.loc[subjects, "voxel_uncertainty_abs_error_spearman"], "positive means latent posterior ranks error better than MC dropout"),
        ("mc_dropout_minus_vae_empirical_crps", b.loc[subjects, "empirical_crps_mean"] - a.loc[subjects, "empirical_crps_mean"], "positive means latent posterior has lower empirical CRPS"),
        ("vae_minus_mc_dropout_selective_gain", a.loc[subjects, "selective_gain_fraction"] - b.loc[subjects, "selective_gain_fraction"], "positive means latent posterior supports better selective prediction"),
    ]
    for contrast, values, meaning in comparisons:
        rows.append({"analysis_family": "uncertainty_baseline", "contrast": contrast, "positive_effect_interpretation": meaning, **paired_stats(values.to_numpy(float), rng, args.bootstrap, args.signflip)})
    out = pd.DataFrame(rows)
    out["bh_q"] = bh_adjust(out["signflip_p_two_sided"].to_numpy(float))
    return out


def _lookup_effect(effects: pd.DataFrame, contrast: str) -> dict | None:
    if effects.empty or "contrast" not in effects:
        return None
    rows = effects[effects.contrast == contrast]
    return rows.iloc[0].to_dict() if len(rows) else None


def _write_summary(path: Path, clean_summary: pd.DataFrame, all_effects: pd.DataFrame, calibration_params: pd.DataFrame, corruption_summary: pd.DataFrame) -> None:
    lines = [
        "# VRPTR posterior-predictive evaluation",
        "",
        "The predictive distribution is induced by sampling the learned latent q(z|x) while network weights and subject-specific U-Net skip features remain fixed. It therefore quantifies latent uncertainty, not full Bayesian weight uncertainty.",
        "",
    ]
    if not clean_summary.empty:
        r = clean_summary.iloc[0]
        lines.extend([
            "## Clean held-out controls",
            "",
            f"- Mean z=mu Pearson: {r.get('mu_pearson_r_mean', np.nan):.4f}",
            f"- Mean posterior-mean Pearson: {r.get('posterior_mean_pearson_r_mean', np.nan):.4f}",
            f"- Mean posterior SD: {r.get('posterior_sd_mean_mean', np.nan):.5f}",
            f"- Mean voxelwise Spearman(posterior SD, absolute error): {r.get('voxel_uncertainty_abs_error_spearman_mean', np.nan):.4f}",
            f"- Mean selective gain: {r.get('selective_gain_fraction_mean', np.nan):.4f}",
            "",
        ])
    mean_eff = _lookup_effect(all_effects, "posterior_mean_minus_mu_pearson")
    nll_eff = _lookup_effect(all_effects, "constant_minus_heteroscedastic_nll")
    crps_eff = _lookup_effect(all_effects, "constant_minus_heteroscedastic_crps")
    rho_eff = _lookup_effect(all_effects, "voxel_uncertainty_error_spearman")
    lines.extend(["## Decision-oriented interpretation", ""])
    if mean_eff is not None:
        lines.append(
            f"- Posterior averaging effect on Pearson: {mean_eff['mean_effect']:.5f} "
            f"(95% CI {mean_eff['bootstrap_ci95_low']:.5f} to {mean_eff['bootstrap_ci95_high']:.5f}; q={mean_eff.get('bh_q', np.nan):.4g})."
        )
    if rho_eff is not None:
        lines.append(
            f"- Voxelwise uncertainty-error association: mean rho={rho_eff['mean_effect']:.4f} "
            f"(95% CI {rho_eff['bootstrap_ci95_low']:.4f} to {rho_eff['bootstrap_ci95_high']:.4f}; q={rho_eff.get('bh_q', np.nan):.4g})."
        )
    if nll_eff is not None:
        lines.append(
            f"- Cross-fitted heteroscedastic vs constant NLL benefit: {nll_eff['mean_effect']:.5f} "
            f"(95% CI {nll_eff['bootstrap_ci95_low']:.5f} to {nll_eff['bootstrap_ci95_high']:.5f}; q={nll_eff.get('bh_q', np.nan):.4g})."
        )
    if crps_eff is not None:
        lines.append(
            f"- Cross-fitted heteroscedastic vs constant CRPS benefit: {crps_eff['mean_effect']:.5f} "
            f"(95% CI {crps_eff['bootstrap_ci95_low']:.5f} to {crps_eff['bootstrap_ci95_high']:.5f}; q={crps_eff.get('bh_q', np.nan):.4g})."
        )
    if not calibration_params.empty:
        lines.append(f"- Mean fitted use of raw posterior SD (scale a): {calibration_params['heteroscedastic_scale_a'].mean():.4f}; mean residual floor tau: {calibration_params['heteroscedastic_floor_tau'].mean():.4f}.")
    lines.append("")
    supported = []
    unsupported = []
    if nll_eff is not None and float(nll_eff.get("bh_q", 1.0)) < 0.05 and float(nll_eff["mean_effect"]) > 0:
        supported.append("the spatial posterior spread improves a proper probabilistic score beyond a constant-width uncertainty model")
    else:
        unsupported.append("a proper-score advantage over a constant-width uncertainty model")
    if rho_eff is not None and float(rho_eff.get("bh_q", 1.0)) < 0.05 and float(rho_eff["mean_effect"]) > 0:
        supported.append("posterior SD identifies relatively error-prone voxels")
    else:
        unsupported.append("reliable voxelwise error awareness")
    if mean_eff is not None and float(mean_eff.get("bh_q", 1.0)) < 0.05 and float(mean_eff["mean_effect"]) > 0:
        supported.append("posterior averaging improves point prediction")
    if supported:
        lines.append("**Supported probabilistic capabilities:** " + "; ".join(supported) + ".")
    if unsupported:
        lines.append("**Not established by this analysis:** " + "; ".join(unsupported) + ".")
    if not corruption_summary.empty:
        lines.extend(["", "## Controlled corruption", "", "See `corruption_summary.csv` and `corruption_effects.csv` for whether uncertainty increases and tracks degradation under synthetic input shifts."])
    lines.extend([
        "",
        "A positive probabilistic result does not imply full Bayesian uncertainty: model-weight uncertainty and irreducible target noise are not represented unless separately modeled.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    base = root / "final_analysis" / "posterior_predictive"
    output = Path(args.out_dir) if args.out_dir else base / "compiled"
    output.mkdir(parents=True, exist_ok=True)

    clean_dir = base / "clean"
    corruption_dir = base / "corruption"
    mc_dir = base / "mc_dropout"
    clean = _read(clean_dir / "posterior_per_subject.csv")
    coverage = _read(clean_dir / "posterior_coverage_per_subject.csv")
    corruption = _read(corruption_dir / "posterior_per_subject.csv")
    deterministic = _read(corruption_dir / "deterministic_point_per_subject.csv")
    mc = _read(mc_dir / "posterior_per_subject.csv")

    clean_summary = _summarize_clean(clean, coverage)
    clean_effects = _clean_effects(clean, coverage, args)
    crossfit, calibration_params, calibration_effects = _crossfit_calibration(clean_dir, args)
    corruption_summary, corruption_effects, corruption_slopes = _corruption_analysis(corruption, deterministic, args)
    mc_effects = _mc_dropout_comparison(clean, mc, args)

    frames = [f for f in [clean_effects, calibration_effects, corruption_effects, mc_effects] if not f.empty]
    all_effects = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # Preserve family-wise q-values and also provide a global sensitivity q-value.
    if not all_effects.empty:
        all_effects["global_bh_q"] = bh_adjust(all_effects["signflip_p_two_sided"].to_numpy(float))

    clean_summary.to_csv(output / "clean_posterior_summary.csv", index=False)
    clean_effects.to_csv(output / "clean_posterior_effects.csv", index=False)
    crossfit.to_csv(output / "crossfit_calibration_per_subject.csv", index=False)
    calibration_params.to_csv(output / "crossfit_calibration_parameters.csv", index=False)
    calibration_effects.to_csv(output / "crossfit_calibration_effects.csv", index=False)
    corruption_summary.to_csv(output / "corruption_summary.csv", index=False)
    corruption_effects.to_csv(output / "corruption_effects.csv", index=False)
    corruption_slopes.to_csv(output / "corruption_slopes_per_subject.csv", index=False)
    mc_effects.to_csv(output / "vae_vs_mc_dropout_effects.csv", index=False)
    all_effects.to_csv(output / "posterior_predictive_all_effects.csv", index=False)
    _write_summary(output / "posterior_predictive_summary.md", clean_summary, all_effects, calibration_params, corruption_summary)

    manifest = {
        "folds": args.folds,
        "fit_voxels": args.fit_voxels,
        "bootstrap": args.bootstrap,
        "signflip": args.signflip,
        "seed": args.seed,
        "raw_posterior": "latent q(z|x) samples with fixed weights and fixed skip features",
        "crossfit_model": "sigma_calibrated^2 = tau^2 + (a * sigma_raw)^2",
        "constant_baseline": "one homoscedastic tau fitted on the same calibration folds",
    }
    (output / "compile_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[posterior-compile] wrote {output}")


if __name__ == "__main__":
    main()
