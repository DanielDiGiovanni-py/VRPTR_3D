"""Export manuscript-facing checks without changing inferential test families."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .training_template import validate_result_frames


def audit_selective_gain(risk, reported):
    risk = risk[(risk.source == "full_vae_latent") & (risk.condition == "clean")]
    reported = reported[(reported.source == "full_vae_latent") & (reported.condition == "clean")].copy()
    reported["subject"] = reported.subject.astype(str)
    rows = []
    for subject, frame in risk.groupby("subject"):
        frame = frame.sort_values("coverage")
        if frame.coverage.duplicated().any() or len(frame) < 2:
            raise ValueError("Risk-coverage audit needs distinct retained fractions")
        x, y = frame.coverage.to_numpy(float), frame.mae.to_numpy(float)
        baseline = frame.random_expected_mae.to_numpy(float)
        if not np.allclose(baseline, baseline[0]) or baseline[0] <= 0:
            raise ValueError("Inconsistent selected-voxel MAE reference")
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        aurc = float(integrate(y, x) / (x[-1] - x[0]))
        gain = (baseline[0] - aurc) / baseline[0]
        original = reported[reported.subject == str(subject)]
        if len(original) != 1 or not np.isclose(gain, original.selective_gain_fraction.iloc[0], atol=1e-6):
            raise ValueError(f"Selective-gain definition/output mismatch for {subject}")
        rows.append({"subject": str(subject), "retained_fractions": ",".join(map(str, x)),
                     "normalized_risk_coverage_auc": aurc, "all_selected_voxel_mae": baseline[0],
                     "integrated_gain_fraction": gain})
    if set(risk.subject.astype(str)) != set(reported.subject):
        raise ValueError("Selective-scoring tables have different participant sets")
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    args = p.parse_args(argv)
    root = Path(args.root)
    analysis = root / "thesis_vrptr" / "final_analysis"
    destination = root / "revision_audit"
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(analysis / "individualization" / "all_models_individualization_per_subject.csv")
    checksum = validate_result_frames([frame])
    rows = []
    for model, subset in frame.groupby("model"):
        row = {"model": model, "n": subset.subject.nunique(), "template_sha256": checksum}
        for key in ("model_pearson_r", "model_dice_auc", "model_mae", "model_rmse",
                    "template_pearson_r", "template_mae", "template_rmse", "delta_pearson_r",
                    "mae_improvement", "rmse_improvement", "residual_pearson_r",
                    "residual_id_self_vs_best_other_margin", "residual_id_top1_acc"):
            row[key + "_mean"] = float(subset[key].mean())
            row[key + "_sd"] = float(subset[key].std(ddof=1))
        rows.append(row)
    pd.DataFrame(rows).sort_values("model_pearson_r_mean", ascending=False).to_csv(destination / "table1_inputs.csv", index=False)
    summary = {"template_sha256": checksum, "n_evaluated": int(frame.subject.nunique()),
               "primary_outcome": "voxel-map Pearson correlation", "absolute_error_units": "standardized task-map units",
               "mae_rmse_inference": "descriptive; no new p-values added to existing multiple-testing families"}
    clean = analysis / "posterior_predictive" / "clean"
    if (clean / "posterior_risk_coverage.csv").exists():
        checked = audit_selective_gain(pd.read_csv(clean / "posterior_risk_coverage.csv"), pd.read_csv(clean / "posterior_per_subject.csv"))
        checked.to_csv(destination / "selective_prediction_definition_check.csv", index=False)
        summary["selective_integrated_gain_fraction_mean"] = float(checked.integrated_gain_fraction.mean())
    (destination / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[revision-audit] wrote {destination}")


if __name__ == "__main__":
    main()
