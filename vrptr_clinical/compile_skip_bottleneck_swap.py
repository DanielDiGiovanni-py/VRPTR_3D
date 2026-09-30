from __future__ import annotations

"""Combine skip–bottleneck swap outputs across already-trained VRPTR variants."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .training_template import validate_result_frames
from .final_stats_common import bh_adjust, paired_stats


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-root", required=True)
    p.add_argument("--variants", nargs="+", default=["full"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--signflip", type=int, default=20000)
    return p.parse_args(argv)


def _read(root: Path, variant: str, name: str) -> pd.DataFrame:
    path = root / variant / name
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["variant"] = variant
    return frame


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.analysis_root)
    root.mkdir(parents=True, exist_ok=True)
    recipient_frames = []
    subject_frames = []
    effect_frames = []
    summary_frames = []
    for variant in args.variants:
        recipient_frames.append(_read(root, variant, "swap_per_recipient.csv"))
        subject_frames.append(_read(root, variant, "component_condition_per_subject.csv"))
        effect_frames.append(_read(root, variant, "paired_effects.csv"))
        summary_frames.append(_read(root, variant, "condition_summary.csv"))
    validate_result_frames(recipient_frames + subject_frames + effect_frames + summary_frames)
    recipients = pd.concat(recipient_frames, ignore_index=True)
    subjects = pd.concat(subject_frames, ignore_index=True)
    effects = pd.concat(effect_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    recipients.to_csv(root / "all_variants_swap_per_recipient.csv", index=False)
    subjects.to_csv(root / "all_variants_component_condition_per_subject.csv", index=False)
    effects.to_csv(root / "all_variants_paired_effects.csv", index=False)
    summaries.to_csv(root / "all_variants_condition_summary.csv", index=False)

    # Optional matched full-vs-deterministic comparison of the component-swap mechanism.
    cross_rows = []
    variants = set(recipients["variant"].astype(str))
    if {"full", "deterministic"}.issubset(variants):
        rng = np.random.default_rng(args.seed)
        keys = ["condition", "recipient_subject"]
        full = recipients[recipients["variant"].eq("full")].set_index(keys)
        det = recipients[recipients["variant"].eq("deterministic")].set_index(keys)
        common = full.index.intersection(det.index)
        for metric in [
            "recipient_pearson_r",
            "residual_pearson_to_recipient",
            "whole_identity_shift_toward_donor",
            "residual_identity_shift_toward_donor",
            "prediction_rmse_from_recipient_baseline",
        ]:
            if metric not in full or metric not in det:
                continue
            values = full.loc[common, metric].to_numpy(dtype=float) - det.loc[common, metric].to_numpy(dtype=float)
            cross_rows.append(
                {
                    "first_variant": "full",
                    "second_variant": "deterministic",
                    "metric": metric,
                    "positive_effect_interpretation": "positive means larger value in full VRPTR",
                    **paired_stats(values, rng, n_boot=args.bootstrap, n_perm=args.signflip),
                }
            )
    cross = pd.DataFrame(cross_rows)
    if not cross.empty:
        cross["bh_q"] = bh_adjust(cross["signflip_p_two_sided"].to_numpy(dtype=float))
    cross.to_csv(root / "full_vs_deterministic_swap_paired.csv", index=False)
    print(f"[swap-compile] wrote {root}")


if __name__ == "__main__":
    main()
