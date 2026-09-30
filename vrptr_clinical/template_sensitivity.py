"""Explicitly separate transductive LOO sensitivity from primary results."""
from pathlib import Path
from .metrics import template_residual_analysis_rows
from .final_stats_common import write_rows


def write_loo_sensitivity(output, subjects, truth, predictions, left_idx, right_idx, group, model):
    values = template_residual_analysis_rows(
        subjects, truth, predictions, left_idx, right_idx, group, model,
        template_mode="test_loo_sensitivity",
    )
    names = ["signal_recovery.csv", "signal_recovery_summary.csv",
             "test_loo_template_per_subject_metrics.csv", "test_loo_template_summary.csv",
             "residual_self_vs_other.csv", "residual_self_vs_other_pairwise.csv",
             "residual_self_vs_other_summary.csv"]
    root = Path(output) / "test_loo_sensitivity"
    for name, rows in zip(names, values):
        write_rows(root / name, [rows] if isinstance(rows, dict) else rows)
    (root / "README.txt").write_text(
        "Sensitivity only: each reference uses other evaluation participants' observed task maps. "
        "This is a transductive descriptive analysis, not an independent deployable baseline. "
        "Off-diagonal residual matches use the respective subject's LOO reference.\n")
