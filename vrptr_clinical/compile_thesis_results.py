from __future__ import annotations

"""Compile single-run thesis VRPTR variants against fixed comparator results."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import summarize_metric_rows
from .training_common import write_rows_csv


COMPARATOR_NAMES = ("BrainSurfCNN-like", "SWIFUN-like")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare thesis VRPTR with fixed compact-model results.")
    p.add_argument("--predictions-root", required=True)
    p.add_argument("--comparator-per-subject", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--variants", nargs="+", default=["full", "deterministic", "no_transformer", "compression_only"])
    p.add_argument("--seed", type=int, default=42, help="Bootstrap/sign-flip RNG seed only.")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--signflip", type=int, default=20000)
    return p.parse_args(argv)


def _normalise_subject(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def _find_comparators(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    if not path.exists():
        raise FileNotFoundError(f"Fixed comparator file not found: {path}")
    frame = pd.read_csv(path)
    diagnostics: list[dict] = []
    if "subject" not in frame or "model" not in frame:
        raise ValueError(f"Comparator file must contain subject and model columns: {path}")
    frame = frame.copy()
    frame["model"] = frame["model"].replace({"brainsurfcnn_lite": "BrainSurfCNN-like", "BrainSurfCNN-lite": "BrainSurfCNN-like", "swifun_lite": "SWIFUN-like", "SWIFUN-lite": "SWIFUN-like", "SwiFUN-lite": "SWIFUN-like"})
    frame["subject"] = _normalise_subject(frame["subject"])
    if "group" in frame:
        frame = frame[frame["group"].astype(str).str.lower().eq("controls")]
    selected = []
    for model in COMPARATOR_NAMES:
        rows = frame[frame["model"].astype(str).eq(model)].copy()
        diagnostics.append(
            {
                "source": str(path),
                "model": model,
                "rows": int(len(rows)),
                "unique_subjects": int(rows["subject"].nunique()) if len(rows) else 0,
                "status": "ok" if len(rows) else "missing",
            }
        )
        if rows.empty:
            available = sorted(frame["model"].astype(str).unique().tolist())
            raise RuntimeError(f"Missing exact comparator model {model!r}; available={available}")
        if rows["subject"].duplicated().any():
            raise RuntimeError(f"Duplicate subject rows for fixed comparator {model}")
        selected.append(rows)
    return pd.concat(selected, ignore_index=True), diagnostics


def _paired_stats(delta: np.ndarray, rng: np.random.Generator, n_boot: int, n_sign: int) -> dict:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        return {
            "n": 0,
            "mean_delta": np.nan,
            "bootstrap_ci95_low": np.nan,
            "bootstrap_ci95_high": np.nan,
            "signflip_p_two_sided": np.nan,
            "first_model_better_fraction": np.nan,
        }
    boot = np.empty(max(1, int(n_boot)), dtype=float)
    for i in range(boot.size):
        boot[i] = rng.choice(delta, size=delta.size, replace=True).mean()
    observed = abs(delta.mean())
    extreme = 0
    for _ in range(max(1, int(n_sign))):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=delta.size)
        extreme += int(abs((delta * signs).mean()) >= observed)
    return {
        "n": int(delta.size),
        "mean_delta": float(delta.mean()),
        "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
        "signflip_p_two_sided": float((extreme + 1) / (max(1, int(n_sign)) + 1)),
        "first_model_better_fraction": float((delta > 0).mean()),
    }


def _load_vrptr(predictions_root: Path, variants: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    manifest = []
    for variant in variants:
        path = predictions_root / f"thesis_vrptr_{variant}" / "controls" / "per_subject_metrics.csv"
        if not path.exists():
            manifest.append({"role": "vrptr_variant", "model": variant, "path": str(path), "status": "missing", "n": 0})
            continue
        frame = pd.read_csv(path)
        frame["subject"] = _normalise_subject(frame["subject"])
        frame["model"] = f"thesis_vrptr_{variant}"
        frame["variant"] = variant
        frame["family"] = "VRPTR-thesis"
        frames.append(frame)
        manifest.append({"role": "vrptr_variant", "model": variant, "path": str(path), "status": "ok", "n": int(len(frame))})
    if not frames:
        raise RuntimeError(f"No thesis VRPTR prediction files found under {predictions_root}")
    return pd.concat(frames, ignore_index=True), manifest


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparator, comparator_diag = _find_comparators(Path(args.comparator_per_subject))
    comparator["family"] = "fixed comparator"
    comparator["variant"] = comparator["model"]
    vrptr, manifest = _load_vrptr(Path(args.predictions_root), list(args.variants))

    common_subjects = set(comparator[comparator["model"] == COMPARATOR_NAMES[0]]["subject"])
    common_subjects &= set(comparator[comparator["model"] == COMPARATOR_NAMES[1]]["subject"])
    for model in vrptr["model"].unique():
        common_subjects &= set(vrptr[vrptr["model"] == model]["subject"])
    if not common_subjects:
        raise RuntimeError("No common subjects between thesis VRPTR and fixed comparators")
    common_subjects = sorted(common_subjects)
    comparator = comparator[comparator["subject"].isin(common_subjects)].copy()
    vrptr = vrptr[vrptr["subject"].isin(common_subjects)].copy()

    combined = pd.concat([vrptr, comparator], ignore_index=True, sort=False)
    combined.to_csv(out_dir / "compiled_thesis_and_fixed_comparators_per_subject.csv", index=False)

    summaries = []
    for model, frame in combined.groupby("model", sort=False):
        summary = summarize_metric_rows(frame.to_dict("records"), group="controls", model=str(model))
        summary["family"] = str(frame["family"].iloc[0])
        summary["variant"] = str(frame["variant"].iloc[0])
        summary["fixed_comparator"] = int(summary["family"] == "fixed comparator")
        summaries.append(summary)
    write_rows_csv(out_dir / "compiled_thesis_and_fixed_comparators_summary.csv", summaries)

    rng = np.random.default_rng(args.seed)
    metrics = ["pearson_r", "spearman_r", "r2", "dice_auc", "dice25", "li_abs_err"]
    paired_comparator = []
    for variant in sorted(vrptr["variant"].unique()):
        first = vrptr[vrptr["variant"] == variant].set_index("subject")
        for comparator_name in COMPARATOR_NAMES:
            second = comparator[comparator["model"] == comparator_name].set_index("subject")
            subjects = sorted(set(first.index) & set(second.index))
            for metric in metrics:
                if metric not in first or metric not in second:
                    continue
                a = first.loc[subjects, metric].to_numpy(dtype=float)
                b = second.loc[subjects, metric].to_numpy(dtype=float)
                # Lower LI error is better, so orient all deltas as positive=favour thesis VRPTR.
                delta = (b - a) if metric == "li_abs_err" else (a - b)
                paired_comparator.append(
                    {
                        "first_model": f"thesis_vrptr_{variant}",
                        "second_model": comparator_name,
                        "metric": metric,
                        "positive_delta_favours": "thesis_vrptr",
                        **_paired_stats(delta, rng, args.bootstrap, args.signflip),
                    }
                )
    write_rows_csv(out_dir / "thesis_vrptr_vs_fixed_comparators_paired.csv", paired_comparator)

    paired_ablation = []
    if "full" in set(vrptr["variant"]):
        full = vrptr[vrptr["variant"] == "full"].set_index("subject")
        for variant in sorted(set(vrptr["variant"]) - {"full"}):
            other = vrptr[vrptr["variant"] == variant].set_index("subject")
            subjects = sorted(set(full.index) & set(other.index))
            for metric in metrics:
                if metric not in full or metric not in other:
                    continue
                a = full.loc[subjects, metric].to_numpy(dtype=float)
                b = other.loc[subjects, metric].to_numpy(dtype=float)
                delta = (b - a) if metric == "li_abs_err" else (a - b)
                paired_ablation.append(
                    {
                        "first_model": "thesis_vrptr_full",
                        "second_model": f"thesis_vrptr_{variant}",
                        "metric": metric,
                        "positive_delta_favours": "full",
                        **_paired_stats(delta, rng, args.bootstrap, args.signflip),
                    }
                )
    write_rows_csv(out_dir / "thesis_vrptr_component_ablation_paired.csv", paired_ablation)

    manifest.extend(
        {
            "role": "fixed_comparator",
            "model": row["model"],
            "path": row["source"],
            "status": row["status"],
            "n": row["rows"],
        }
        for row in comparator_diag
    )
    write_rows_csv(out_dir / "compile_input_manifest.csv", manifest)
    write_rows_csv(out_dir / "comparator_source_diagnostics.csv", comparator_diag)
    print(f"[thesis-compile] common held-out subjects: {len(common_subjects)}")
    print(f"[thesis-compile] wrote {out_dir}")


if __name__ == "__main__":
    main()
