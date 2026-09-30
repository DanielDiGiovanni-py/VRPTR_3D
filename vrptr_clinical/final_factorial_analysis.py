from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .training_template import validate_result_frames
from .final_stats_common import bh_adjust, paired_stats

CELLS = {
    "T1V1": "thesis_vrptr_full",
    "T1V0": "thesis_vrptr_deterministic",
    "T0V1": "thesis_vrptr_no_transformer",
    "T0V0": "thesis_vrptr_compression_only",
}
VARIANT_DIR = {
    "T1V1": "full",
    "T1V0": "deterministic",
    "T0V1": "no_transformer",
    "T0V0": "compression_only",
}


def parse_args():
    p = argparse.ArgumentParser(description="Paired 2x2 Transformer x VAE factorial analysis.")
    p.add_argument("--results-root", default="results/thesis_vrptr")
    p.add_argument("--out-dir", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--signflip", type=int, default=20000)
    return p.parse_args()


def _aligned_values(frames, column):
    subjects = sorted(set.intersection(*[set(frame.index) for frame in frames.values()]))
    return subjects, {key: frame.loc[subjects, column].to_numpy(dtype=float) for key, frame in frames.items()}


def _effects(values):
    tv, t, v, c = values["T1V1"], values["T1V0"], values["T0V1"], values["T0V0"]
    return {
        "transformer_main_effect": 0.5 * ((tv - v) + (t - c)),
        "vae_main_effect": 0.5 * ((tv - t) + (v - c)),
        "transformer_x_vae_interaction": tv - t - v + c,
        "transformer_effect_with_vae": tv - v,
        "transformer_effect_without_vae": t - c,
        "vae_effect_with_transformer": tv - t,
        "vae_effect_without_transformer": v - c,
        "full_minus_compression_only": tv - c,
    }


def _analyse_family(frames, metric_map, family, rng, args):
    rows = []
    cells = []
    for metric, (column, higher_is_better) in metric_map.items():
        subjects, values = _aligned_values(frames, column)
        for cell, arr in values.items():
            cells.append({
                "analysis_family": family,
                "metric": metric,
                "cell": cell,
                "model": CELLS[cell],
                "transformer": int(cell[1] == "1"),
                "vae": int(cell[3] == "1"),
                "n": len(subjects),
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr, ddof=1)),
                "higher_is_better": int(higher_is_better),
            })
        for effect_name, delta in _effects(values).items():
            oriented = delta if higher_is_better else -delta
            stats = paired_stats(oriented, rng, args.bootstrap, args.signflip)
            raw_stats = paired_stats(delta, rng, args.bootstrap, args.signflip)
            rows.append({
                "analysis_family": family,
                "metric": metric,
                "effect": effect_name,
                "higher_is_better": int(higher_is_better),
                "raw_mean_effect": raw_stats["mean_effect"],
                "raw_bootstrap_ci95_low": raw_stats["bootstrap_ci95_low"],
                "raw_bootstrap_ci95_high": raw_stats["bootstrap_ci95_high"],
                "benefit_mean_effect": stats["mean_effect"],
                "benefit_bootstrap_ci95_low": stats["bootstrap_ci95_low"],
                "benefit_bootstrap_ci95_high": stats["bootstrap_ci95_high"],
                "signflip_p_two_sided": stats["signflip_p_two_sided"],
                "beneficial_fraction": stats["positive_effect_fraction"],
                "n": stats["n"],
            })
    frame = pd.DataFrame(rows)
    primary = frame[frame["effect"].isin(["transformer_main_effect", "vae_main_effect", "transformer_x_vae_interaction"])].copy()
    primary["bh_q_within_family"] = bh_adjust(primary["signflip_p_two_sided"].to_numpy())
    frame = frame.merge(primary[["metric", "effect", "bh_q_within_family"]], on=["metric", "effect"], how="left")
    return cells, frame.to_dict("records")


def main():
    args = parse_args()
    root = Path(args.results_root)
    output = Path(args.out_dir) if args.out_dir else root / "final_analysis" / "factorial"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    compiled = pd.read_csv(root / "compiled" / "compiled_thesis_and_fixed_comparators_per_subject.csv")
    compiled["subject"] = compiled["subject"].astype(str)
    prediction_frames = {
        cell: compiled[compiled["model"] == model].set_index("subject") for cell, model in CELLS.items()
    }
    prediction_metrics = {
        "pearson_r": ("pearson_r", True),
        "spearman_r": ("spearman_r", True),
        "r2": ("r2", True),
        "dice_auc": ("dice_auc", True),
        "dice25": ("dice25", True),
        "li_abs_err": ("li_abs_err", False),
    }
    cell_rows, prediction_rows = _analyse_family(prediction_frames, prediction_metrics, "prediction", rng, args)

    signal_frames = {}
    residual_id_frames = {}
    whole_id_frames = {}
    for cell, variant in VARIANT_DIR.items():
        model_dir = root / "predictions" / f"thesis_vrptr_{variant}" / "controls"
        signal = pd.read_csv(model_dir / "signal_recovery.csv"); signal["subject"] = signal["subject"].astype(str)
        residual = pd.read_csv(model_dir / "residual_self_vs_other.csv"); residual["subject"] = residual["subject"].astype(str)
        whole = pd.read_csv(model_dir / "self_vs_other.csv"); whole["subject"] = whole["subject"].astype(str)
        signal_frames[cell] = signal.set_index("subject")
        residual_id_frames[cell] = residual.set_index("subject")
        whole_id_frames[cell] = whole.set_index("subject")

    validate_result_frames(list(signal_frames.values()) + list(residual_id_frames.values()))
    signal_metrics = {
        "gain_over_template_pearson": ("delta_pearson_r", True),
        "residual_pearson_r": ("residual_pearson_r", True),
        "residual_spearman_r": ("residual_spearman_r", True),
        "residual_dice_auc": ("residual_dice_auc", True),
        "residual_dice25": ("residual_dice25", True),
        "li_abs_err_improvement": ("li_abs_err_improvement", True),
    }
    signal_cells, signal_rows = _analyse_family(signal_frames, signal_metrics, "individualized_signal", rng, args)
    residual_metrics = {
        "residual_self_vs_best_other_margin": ("self_vs_best_other_margin", True),
        "residual_self_vs_mean_other_margin": ("self_vs_mean_other_margin", True),
        "residual_self_rank": ("self_rank", False),
        "residual_self_percentile": ("self_percentile", True),
    }
    residual_cells, residual_rows = _analyse_family(residual_id_frames, residual_metrics, "residual_identification", rng, args)
    whole_metrics = {
        "whole_self_vs_best_other_margin": ("self_vs_best_other_margin", True),
        "whole_self_vs_mean_other_margin": ("self_vs_mean_other_margin", True),
        "whole_self_rank": ("self_rank", False),
        "whole_self_percentile": ("self_percentile", True),
    }
    whole_cells, whole_rows = _analyse_family(whole_id_frames, whole_metrics, "whole_map_identification", rng, args)

    all_cells = cell_rows + signal_cells + residual_cells + whole_cells
    all_effects = prediction_rows + signal_rows + residual_rows + whole_rows
    pd.DataFrame(all_cells).to_csv(output / "factorial_cell_summary.csv", index=False)
    effects = pd.DataFrame(all_effects)
    effects.to_csv(output / "factorial_effects_all.csv", index=False)
    effects[effects.analysis_family == "prediction"].to_csv(output / "factorial_effects_prediction.csv", index=False)
    effects[effects.analysis_family != "prediction"].to_csv(output / "factorial_effects_individualization.csv", index=False)

    key = effects[(effects.metric.isin(["pearson_r", "r2", "dice_auc", "residual_pearson_r", "residual_self_vs_best_other_margin"])) &
                  (effects.effect.isin(["transformer_main_effect", "vae_main_effect", "transformer_x_vae_interaction"]))]
    lines = ["# Final VRPTR 2x2 factorial analysis", ""]
    for _, row in key.iterrows():
        lines.append(
            f"- {row['metric']} — {row['effect']}: benefit={row['benefit_mean_effect']:.6f}, "
            f"95% CI [{row['benefit_bootstrap_ci95_low']:.6f}, {row['benefit_bootstrap_ci95_high']:.6f}], "
            f"p={row['signflip_p_two_sided']:.6g}, q={row.get('bh_q_within_family', np.nan):.6g}."
        )
    (output / "factorial_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[factorial] wrote {output}")
    print(key[["metric", "effect", "benefit_mean_effect", "benefit_bootstrap_ci95_low", "benefit_bootstrap_ci95_high", "signflip_p_two_sided", "bh_q_within_family"]].to_string(index=False))


if __name__ == "__main__":
    main()
