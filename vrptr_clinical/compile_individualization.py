from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .training_template import validate_result_frames
from .final_stats_common import bh_adjust, paired_stats

MODEL_SPECS = {
    "thesis_vrptr_full": ("vrptr", "full"),
    "thesis_vrptr_deterministic": ("vrptr", "deterministic"),
    "thesis_vrptr_no_transformer": ("vrptr", "no_transformer"),
    "thesis_vrptr_compression_only": ("vrptr", "compression_only"),
    "BrainSurfCNN-like": ("comparator", "brainsurfcnn_lite"),
    "SWIFUN-like": ("comparator", "swifun_lite"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Compile individualization metrics for all trained models.")
    p.add_argument("--results-root", default="results/thesis_vrptr")
    p.add_argument("--comparisons-dir", default="results/comparisons")
    p.add_argument("--out-dir", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--signflip", type=int, default=20000)
    return p.parse_args()


def _find_comparator_dir(root: Path, model_slug: str) -> Path:
    base = root / model_slug / "individuality"
    candidates = list(base.glob("*/controls")) if base.exists() else []
    exact = [p for p in candidates if p.parent.name.lower() == model_slug.lower()]
    candidates = exact or candidates
    valid = [p for p in candidates if (p / "signal_recovery.csv").exists()]
    if len(valid) != 1:
        raise RuntimeError(
            f"Expected one individualization directory for {model_slug}, found {valid}. "
            "Run the fixed-comparator evaluation stage first."
        )
    return valid[0]


def _model_dir(results_root: Path, comparisons: Path, model: str) -> Path:
    kind, variant = MODEL_SPECS[model]
    if kind == "vrptr":
        return results_root / "predictions" / f"thesis_vrptr_{variant}" / "controls"
    return _find_comparator_dir(comparisons, variant)


def _load_model_tables(directory: Path):
    signal = pd.read_csv(directory / "signal_recovery.csv")
    residual = pd.read_csv(directory / "residual_self_vs_other.csv")
    whole = pd.read_csv(directory / "self_vs_other.csv")
    validate_result_frames([signal, residual])
    for frame in [signal, residual, whole]:
        frame["subject"] = frame["subject"].astype(str)
    merged = signal.copy()
    residual_keep = ["subject", "self_vs_best_other_margin", "self_vs_mean_other_margin", "self_rank", "self_percentile", "top1_acc", "top5_acc"]
    whole_keep = residual_keep.copy()
    residual = residual[residual_keep].rename(columns={c: f"residual_id_{c}" for c in residual_keep if c != "subject"})
    whole = whole[whole_keep].rename(columns={c: f"whole_id_{c}" for c in whole_keep if c != "subject"})
    merged = merged.merge(residual, on="subject", validate="one_to_one").merge(whole, on="subject", validate="one_to_one")
    return merged


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    comparisons = Path(args.comparisons_dir)
    output = Path(args.out_dir) if args.out_dir else results_root / "final_analysis" / "individualization"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tables = {}
    long_rows = []
    summary_rows = []
    for model in MODEL_SPECS:
        directory = _model_dir(results_root, comparisons, model)
        table = _load_model_tables(directory)
        table["model"] = model
        tables[model] = table.set_index("subject")
        long_rows.append(table)
        summary_rows.append({
            "model": model,
            "n": int(len(table)),
            "pearson_r_mean": float(table.model_pearson_r.mean()),
            "mae_mean": float(table.model_mae.mean()),
            "rmse_mean": float(table.model_rmse.mean()),
            "template_mae_mean": float(table.template_mae.mean()),
            "template_rmse_mean": float(table.template_rmse.mean()),
            "mae_improvement_mean": float(table.mae_improvement.mean()),
            "rmse_improvement_mean": float(table.rmse_improvement.mean()),
            "template_source": "training_mean",
            "template_sha256": table.template_sha256.iloc[0],
            "gain_over_template_pearson_mean": float(table.delta_pearson_r.mean()),
            "residual_pearson_r_mean": float(table.residual_pearson_r.mean()),
            "residual_self_vs_best_other_margin_mean": float(table.residual_id_self_vs_best_other_margin.mean()),
            "residual_top1_accuracy": float(table.residual_id_top1_acc.mean()),
            "residual_top5_accuracy": float(table.residual_id_top5_acc.mean()),
            "whole_self_vs_best_other_margin_mean": float(table.whole_id_self_vs_best_other_margin.mean()),
            "whole_top1_accuracy": float(table.whole_id_top1_acc.mean()),
            "li_abs_err_mean": float(table.model_li_abs_err.mean()),
        })
    validate_result_frames(long_rows)
    pd.concat(long_rows, ignore_index=True).to_csv(output / "all_models_individualization_per_subject.csv", index=False)
    pd.DataFrame(summary_rows).sort_values("pearson_r_mean", ascending=False).to_csv(output / "all_models_individualization_summary.csv", index=False)

    metrics = {
        "pearson_r": ("model_pearson_r", True),
        "gain_over_template_pearson": ("delta_pearson_r", True),
        "residual_pearson_r": ("residual_pearson_r", True),
        "residual_dice_auc": ("residual_dice_auc", True),
        "residual_self_vs_best_other_margin": ("residual_id_self_vs_best_other_margin", True),
        "residual_self_vs_mean_other_margin": ("residual_id_self_vs_mean_other_margin", True),
        "whole_self_vs_best_other_margin": ("whole_id_self_vs_best_other_margin", True),
        "li_abs_err": ("model_li_abs_err", False),
    }
    reference = tables["thesis_vrptr_full"]
    paired_rows = []
    for other_model, other in tables.items():
        if other_model == "thesis_vrptr_full":
            continue
        subjects = sorted(set(reference.index) & set(other.index))
        for metric, (column, higher) in metrics.items():
            raw = reference.loc[subjects, column].to_numpy(float) - other.loc[subjects, column].to_numpy(float)
            benefit = raw if higher else -raw
            stats = paired_stats(benefit, rng, args.bootstrap, args.signflip)
            paired_rows.append({
                "reference": "thesis_vrptr_full",
                "comparison": other_model,
                "metric": metric,
                "higher_is_better": int(higher),
                **stats,
            })
    paired = pd.DataFrame(paired_rows)
    paired["bh_q"] = bh_adjust(paired.signflip_p_two_sided.to_numpy())
    paired.to_csv(output / "full_vrptr_vs_all_models_individualization_paired.csv", index=False)

    # Per-subject winners for the two core claims.
    winner_rows = []
    subjects = sorted(set.intersection(*[set(t.index) for t in tables.values()]))
    for subject in subjects:
        for metric, column in [("map_pearson", "model_pearson_r"), ("residual_pearson", "residual_pearson_r"), ("residual_margin", "residual_id_self_vs_best_other_margin")]:
            values = {model: float(table.loc[subject, column]) for model, table in tables.items()}
            winner = max(values, key=values.get)
            winner_rows.append({"subject": subject, "metric": metric, "winner": winner, "winning_value": values[winner]})
    pd.DataFrame(winner_rows).to_csv(output / "per_subject_best_model.csv", index=False)
    print(f"[individualization] wrote {output}")
    print(pd.DataFrame(summary_rows).sort_values("pearson_r_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
