#!/usr/bin/env python3
from __future__ import annotations

"""Generate the polished main figures for the VRPTR Imaging Neuroscience paper.

The figure set is intentionally restrained. Numerical estimates belong in the
manuscript and captions; the figures show spatial examples, participant-level
distributions, factorial effects, mechanism interventions, and uncertainty
behavior without in-panel statistical annotations.

No model is trained or modified by this script.
"""

import argparse
from .training_template import add_template_argument, load_for_evaluation, validate_result_frames, require_complete_subjects
import csv
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

MODEL_ORDER = [
    "thesis_vrptr_full",
    "thesis_vrptr_deterministic",
    "thesis_vrptr_no_transformer",
    "thesis_vrptr_compression_only",
    "SWIFUN-like",
    "BrainSurfCNN-like",
]

MODEL_SHORT = {
    "thesis_vrptr_full": "Full",
    "thesis_vrptr_deterministic": "Det.",
    "thesis_vrptr_no_transformer": "No T",
    "thesis_vrptr_compression_only": "Comp.",
    "SWIFUN-like": "SWIFUN-like",
    "BrainSurfCNN-like": "BrainSurfCNN-like",
}

COLORS = {
    "full": "#0072B2",
    "deterministic": "#56B4E9",
    "no_transformer": "#E69F00",
    "compression_only": "#D55E00",
    "swifun": "#009E73",
    "brainsurf": "#CC79A7",
    "transformer": "#56B4E9",
    "variational": "#8C6BB1",
    "interaction": "#999999",
    "bottleneck": "#0072B2",
    "skips": "#E66101",
    "raw": "#D55E00",
    "constant": "#777777",
    "hetero": "#0072B2",
    "oracle": "#009E73",
    "random": "#777777",
}

MODEL_COLORS = {
    "thesis_vrptr_full": COLORS["full"],
    "thesis_vrptr_deterministic": COLORS["deterministic"],
    "thesis_vrptr_no_transformer": COLORS["no_transformer"],
    "thesis_vrptr_compression_only": COLORS["compression_only"],
    "SWIFUN-like": COLORS["swifun"],
    "BrainSurfCNN-like": COLORS["brainsurf"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate polished VRPTR paper figures.")
    add_template_argument(p)
    p.add_argument("--root", default="results")
    p.add_argument("--analysis-root", default="", help="Defaults to <root>/thesis_vrptr/final_analysis")
    p.add_argument("--out-dir", default="", help="Defaults to <root>/paper_figures_imaging_neuroscience")
    p.add_argument("--formats", default="pdf,svg,png")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-subjects", default="splits/test_subjects.txt")
    p.add_argument("--cache-dir", default="", help="Defaults to <root>/image_cache")
    p.add_argument("--device", default="cuda")
    p.add_argument("--posterior-samples", type=int, default=128)
    p.add_argument("--posterior-sample-batch", type=int, default=2)
    p.add_argument("--example-subjects", default="", help="Optional comma-separated IDs; default is 25th/50th/75th percentile selection.")
    p.add_argument("--example-quantiles", default="0.25,0.50,0.75")
    p.add_argument("--slice-axis", choices=["x", "y", "z", "auto"], default="z")
    p.add_argument("--skip-uncertainty", action="store_true", help="Generate Figures 1–4 when posterior results are unavailable")
    p.add_argument("--skip-map-inference", action="store_true", help="Generate Figures 1, 3 and 4 only; useful for CPU style checks.")
    p.add_argument("--comparison-module", default="vrptr_clinical.comparison_models")
    p.add_argument("--prediction-module", default="vrptr_clinical.figure_prediction_adapter")
    p.add_argument("--full-checkpoint", default="")
    p.add_argument("--data-dir", default=None, help="Override the input data root stored in the checkpoint; leaves the checkpoint unchanged.")
    p.add_argument("--clean-output", action="store_true", default=True)
    return p.parse_args()


def configure_style(dpi: int) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": dpi,
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required figure input is missing: {path}")
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in formats:
        ext = ext.strip().lower().lstrip(".")
        if not ext:
            continue
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=dpi if ext == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top", ha="left")


def _subject_key(value: object) -> str:
    s = str(value).strip()
    return s.lstrip("0") or s


def parse_subjects(text: str) -> list[str]:
    return [_subject_key(x) for x in text.replace(";", ",").split(",") if x.strip()]


def bootstrap_ci(values: np.ndarray, seed: int = 42, n_boot: int = 5000) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _jitter(n: int, center: float, width: float, rng: np.random.Generator) -> np.ndarray:
    return center + rng.uniform(-width, width, n)


# -----------------------------------------------------------------------------
# Figure 1: concise architecture and factorial design
# -----------------------------------------------------------------------------

def _box(ax: plt.Axes, xywh: tuple[float, float, float, float], title: str, body: str = "", face: str = "white") -> None:
    x, y, w, h = xywh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008", linewidth=0.9,
        edgecolor="#222222", facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * (0.60 if body else 0.50), title, ha="center", va="center", fontweight="bold", fontsize=8.0)
    if body:
        ax.text(x + w / 2, y + h * 0.28, body, ha="center", va="center", fontsize=6.8, linespacing=1.05)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#222222", lw: float = 1.0) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=9, linewidth=lw, color=color))


def plot_figure1(out_dir: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    fig = plt.figure(figsize=(8.7, 4.8))
    outer = GridSpec(2, 1, height_ratios=[2.35, 1.0], hspace=0.30, figure=fig)

    ax = fig.add_subplot(outer[0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_label(ax, "A", x=-0.01, y=1.02)

    boxes = [
        ((0.01, 0.60, 0.135, 0.20), "Resting input", "100 feature maps", "#F7F7F7"),
        ((0.175, 0.60, 0.115, 0.20), "Encoder", "multiscale features", "#F7F7F7"),
        ((0.32, 0.60, 0.125, 0.20), "Compress", "deep map → tokens", "#F7F7F7"),
        ((0.475, 0.60, 0.145, 0.20), "Transformer", "global token mixing", "#DCEAF6"),
        ((0.65, 0.60, 0.105, 0.20), "q(z|x)", "μ, log σ²", "#E9E2F2"),
        ((0.65, 0.20, 0.105, 0.20), "Sample", "z = μ + σ⊙ε", "#E9E2F2"),
        ((0.475, 0.20, 0.145, 0.20), "Transformer", "latent decoding", "#DCEAF6"),
        ((0.32, 0.20, 0.125, 0.20), "Expand", "tokens → deep map", "#F7F7F7"),
        ((0.175, 0.20, 0.115, 0.20), "Decoder", "skip conditioned", "#FCE7D6"),
        ((0.01, 0.20, 0.135, 0.20), "Task map", "Map + raw spread", "#F7F7F7"),
    ]
    for spec in boxes:
        _box(ax, *spec)
    for start, end in [
        ((0.145, 0.70), (0.175, 0.70)), ((0.29, 0.70), (0.32, 0.70)),
        ((0.445, 0.70), (0.475, 0.70)), ((0.62, 0.70), (0.65, 0.70)),
        ((0.7025, 0.60), (0.7025, 0.40)), ((0.65, 0.30), (0.62, 0.30)),
        ((0.475, 0.30), (0.445, 0.30)), ((0.32, 0.30), (0.29, 0.30)),
        ((0.175, 0.30), (0.145, 0.30)),
    ]:
        _arrow(ax, start, end)

    # One unobtrusive skip arc and a crossed deepest bypass.
    ax.add_patch(FancyArrowPatch((0.235, 0.60), (0.235, 0.40), connectionstyle="arc3,rad=0.65", arrowstyle="->", mutation_scale=8, linewidth=1.0, color=COLORS["skips"]))
    ax.text(0.24, 0.50, "multiscale skips", color=COLORS["skips"], fontsize=6.8, ha="left", va="center")
    ax.plot([0.383, 0.383], [0.60, 0.40], linestyle="--", color="#999999", linewidth=0.8)
    ax.text(0.383, 0.50, "×", color="#B2182B", fontsize=12, fontweight="bold", ha="center", va="center")
    ax.text(0.405, 0.50, "no deep bypass", fontsize=6.6, style="italic", va="center")

    ax.text(0.55, 0.88, "latent task representation", color=COLORS["full"], fontsize=7.4, fontweight="bold", ha="center")
    ax.text(0.23, 0.08, "Subject-specific spatial features", color=COLORS["skips"], fontsize=7.4, fontweight="bold", ha="center")
    ax.text(0.70, 0.08, "latent predictive uncertainty", color="#6A3D9A", fontsize=7.4, fontweight="bold", ha="center")

    ax2 = fig.add_subplot(outer[1])
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
    panel_label(ax2, "B", x=-0.01, y=1.03)
    col_x = [0.24, 0.42, 0.60, 0.78]
    names = ["Full", "Deterministic", "No Transformer", "Compression only"]
    for x, name in zip(col_x, names):
        ax2.text(x, 0.88, name, ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax2.text(0.07, 0.57, "Transformer", ha="left", va="center", fontsize=7.4)
    ax2.text(0.07, 0.27, "Variational", ha="left", va="center", fontsize=7.4)
    present = [[1, 1, 0, 0], [1, 0, 1, 0]]
    row_y = [0.57, 0.27]
    for r, y in enumerate(row_y):
        for c, x in enumerate(col_x):
            if present[r][c]:
                ax2.scatter([x], [y], s=70, color=COLORS["full"] if r == 0 else "#6A3D9A", zorder=3)
            else:
                ax2.scatter([x], [y], s=70, facecolors="white", edgecolors="#999999", linewidths=1.0, zorder=3)
    ax2.text(0.92, 0.51, "Fixed comparators\nBrainSurfCNN-like\nSWIFUN-like", ha="center", va="center", fontsize=7.0,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5", edgecolor="#888888", linewidth=0.8))
    ax2.plot([0.03, 0.84], [0.73, 0.73], color="#DDDDDD", linewidth=0.8)
    ax2.plot([0.03, 0.84], [0.42, 0.42], color="#EEEEEE", linewidth=0.8)

    fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.06)
    return save_figure(fig, out_dir, "Figure1_VRPTR_architecture_and_study_design", formats, dpi)


# -----------------------------------------------------------------------------
# Runtime maps
# -----------------------------------------------------------------------------

def choose_quantile_subjects(metrics: pd.DataFrame, quantiles: Sequence[float]) -> pd.DataFrame:
    d = metrics[metrics["model"] == "thesis_vrptr_full"].copy()
    d["subject_key"] = d["subject"].map(_subject_key)
    d["model_pearson_r"] = pd.to_numeric(d["model_pearson_r"], errors="coerce")
    d = d.dropna(subset=["model_pearson_r"]).sort_values("model_pearson_r")
    if d.empty:
        raise ValueError("No full-VRPTR held-out metrics were available for example selection.")
    selected: list[pd.Series] = []
    used: set[str] = set()
    for q in quantiles:
        target = float(d["model_pearson_r"].quantile(q))
        cand = d.assign(distance=(d["model_pearson_r"] - target).abs()).sort_values(["distance", "model_pearson_r"])
        row = next((r for _, r in cand.iterrows() if str(r["subject_key"]) not in used), cand.iloc[0])
        used.add(str(row["subject_key"]))
        rr = row.copy(); rr["selection_quantile"] = q; rr["selection_target_r"] = target
        selected.append(rr)
    return pd.DataFrame(selected)


def _as_volume(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    x = np.squeeze(x)
    if x.ndim != 3:
        raise ValueError(f"Expected a 3-D map after squeeze, got {x.shape}")
    return np.where(np.isfinite(x), x, np.nan)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1); y = np.asarray(b, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _slice_score(vol: np.ndarray, axis: int) -> np.ndarray:
    axes = tuple(i for i in range(3) if i != axis)
    return np.nansum(np.abs(vol), axis=axes)


def choose_informative_slice(target: np.ndarray, axis_name: str) -> tuple[int, int]:
    target = _as_volume(target)
    axes = range(3) if axis_name == "auto" else [{"x": 0, "y": 1, "z": 2}[axis_name]]
    candidates: list[tuple[float, int, int]] = []
    for axis in axes:
        score = _slice_score(target, axis)
        lo = max(0, int(round(score.size * 0.12))); hi = min(score.size, int(round(score.size * 0.88)))
        core = score[lo:hi] if hi > lo else score
        idx = lo + int(np.argmax(core)) if core.size else int(np.argmax(score))
        candidates.append((float(score[idx]), int(axis), int(idx)))
    _, axis, idx = max(candidates, key=lambda z: z[0])
    return axis, idx


def extract_slice(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    vol = _as_volume(vol)
    if axis == 0: sl = vol[index, :, :]
    elif axis == 1: sl = vol[:, index, :]
    else: sl = vol[:, :, index]
    return np.rot90(sl)


def _brain_mask_3d(maps: Sequence[np.ndarray]) -> np.ndarray:
    # Exact observed-map evaluation support; do not fill holes or include background.
    mask = np.isfinite(_as_volume(maps[0]))
    for m in maps[1:]:
        mask &= np.isfinite(_as_volume(m))
    return mask


def _crop_to_mask(data: np.ndarray, mask: np.ndarray, pad: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return data, mask
    r0=max(0,int(rows.min())-pad); r1=min(data.shape[0],int(rows.max())+pad+1)
    c0=max(0,int(cols.min())-pad); c1=min(data.shape[1],int(cols.max())+pad+1)
    return data[r0:r1,c0:c1], mask[r0:r1,c0:c1]


def _transparent_cmap(name: str):
    cmap = mpl.colormaps[name].copy()
    cmap.set_bad((1, 1, 1, 0))
    return cmap


def show_brain(ax: plt.Axes, data: np.ndarray, mask: np.ndarray, cmap: str, vmin: float, vmax: float):
    data_c, mask_c = _crop_to_mask(data, mask)
    masked = np.ma.array(data_c, mask=~mask_c)
    im = ax.imshow(masked, cmap=_transparent_cmap(cmap), vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off(); ax.set_facecolor("white")
    return im


def annotate_orientation(ax, affine, axis):
    affine = np.asarray(affine)
    dimensions = [d for d in range(3) if d != axis]
    positive = ["R", "A", "S"]; negative = ["L", "P", "I"]
    ends = []
    for d in dimensions:
        world = int(np.argmax(np.abs(affine[:3, d])))
        ends.append((negative[world], positive[world]) if affine[world, d] > 0 else (positive[world], negative[world]))
    # np.rot90(slice): first remaining voxel axis increases left-to-right;
    # second remaining voxel axis increases bottom-to-top.
    for x,y,label in [(0.03,0.5,ends[0][0]),(0.97,0.5,ends[0][1]),(0.5,0.03,ends[1][0]),(0.5,0.97,ends[1][1])]:
        ax.text(x,y,label,transform=ax.transAxes,ha="center",va="center",fontsize=6,color="#333333")


def slice_coordinate(affine, shape, axis, index):
    affine = np.asarray(affine)
    voxel = (np.asarray(shape, dtype=float)-1)/2
    voxel[axis] = index
    point = (affine @ np.r_[voxel,1])[:3]
    world = int(np.argmax(np.abs(affine[:3,axis])))
    if np.count_nonzero(np.abs(affine[world,:3]) > 1e-6) != 1:
        return "MNI centre " + ", ".join(f"{v:.1f}" for v in point) + " mm"
    return f"MNI {'xyz'[world]} = {point[world]:.1f} mm"


def _robust_abs_scale(arrays: Sequence[np.ndarray], q: float = 0.995, max_per_map: int = 120000) -> float:
    values: list[np.ndarray] = []
    for arr in arrays:
        x = np.abs(np.asarray(arr, dtype=np.float32).reshape(-1))
        x = x[np.isfinite(x) & (x > 0)]
        if x.size > max_per_map:
            step = max(1, x.size // max_per_map); x = x[::step][:max_per_map]
        if x.size: values.append(x)
    if not values: return 1.0
    return max(float(np.quantile(np.concatenate(values), q)), 1e-5)


def _full_checkpoint(root: Path, args: argparse.Namespace) -> Path:
    return Path(args.full_checkpoint) if args.full_checkpoint else root / "thesis_vrptr" / "checkpoints" / "full" / "final_thesis_vrptr_full.pt"


def load_full_runtime(root: Path, args: argparse.Namespace) -> dict:
    from vrptr_clinical import make_vrptr_imaging_neuroscience_figures as legacy
    checkpoint = _full_checkpoint(root, args)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Full VRPTR checkpoint not found: {checkpoint}")
    subjects_file = Path(args.test_subjects)
    if not subjects_file.exists():
        raise FileNotFoundError(f"Held-out subject list not found: {subjects_file}")
    predmod = legacy.import_prediction_module(args.prediction_module, None)
    spec = legacy.RuntimeModelSpec("Full VRPTR", checkpoint)
    rows, records, atlas, raw = legacy.runtime_predict_for_spec(
        predmod=predmod, spec=spec, subjects_file=subjects_file, group="Control",
        cache_dir=args.cache_dir or str(root / "image_cache"), refresh_cache=False,
        device_str=args.device, batch_size=1, no_data_parallel=True,
        comparison_module_name=args.comparison_module, root=root, data_dir=args.data_dir,
    )
    require_complete_subjects(predmod.load_subject_list(subjects_file), raw["subjects"])
    import torch
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    paths = predmod.ClinicalPaths(**ckpt["paths"])
    if args.data_dir is not None:
        paths.data_dir = args.data_dir
    template, metadata = load_for_evaluation(args.training_template, raw["subjects"], paths, raw["y"].shape[1:])
    truth = {_subject_key(s): _as_volume(t) for s,t,_ in records}
    pred = {_subject_key(s): _as_volume(y) for s,_,y in records}
    return {"legacy": legacy, "predmod": predmod, "spec": spec, "raw": raw, "atlas": atlas, "truth": truth, "pred": pred, "template": _as_volume(template), "template_metadata": metadata}


def _posterior_maps_for_subject(runtime: dict, subject: str, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader
    from vrptr_clinical.training_common import to_device
    from vrptr_clinical import posterior_predictive_analysis as ppa

    legacy=runtime["legacy"]; predmod=runtime["predmod"]; spec=runtime["spec"]; raw=runtime["raw"]
    device=torch.device(args.device)
    model, scaler, ckpt = legacy.load_runtime_model_for_spec(predmod, spec, device, no_data_parallel=True)
    matrix = predmod._prepare_matrix_for_ckpt(raw, ckpt, scaler)
    ds = predmod.make_eval_dataset(matrix, mode="none", seed=args.seed)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    for batch in loader:
        sid=_subject_key(batch["subject"][0])
        if sid != subject: continue
        bd=to_device(batch, device)
        result=ppa._posterior_samples(
            model=model, x=bd["x"], n_samples=int(args.posterior_samples),
            sample_batch=int(args.posterior_sample_batch), selected_idx=np.asarray([0],dtype=np.int64),
            left_idx=matrix["left_idx"], right_idx=matrix["right_idx"], device=device,
            amp="fp16" if device.type=="cuda" else "none", seed=int(args.seed),
        )
        return _as_volume(result["mean_map"]), _as_volume(result["sd_map"])
    raise KeyError(f"Posterior example subject not found in runtime dataset: {subject}")


# -----------------------------------------------------------------------------
# Figure 2: task maps and concise quantitative panels
# -----------------------------------------------------------------------------

def _quant_panel(ax: plt.Axes, data: pd.DataFrame, metric: str, title: str, ylabel: str, panel: str, seed: int) -> None:
    rng=np.random.default_rng(seed)
    d=data[["subject","model",metric]].copy(); d["subject"]=d["subject"].map(_subject_key)
    pivot=d.pivot_table(index="subject", columns="model", values=metric, aggfunc="mean")
    present=[m for m in MODEL_ORDER if m in pivot.columns]
    x=np.arange(len(present),dtype=float)
    for _,row in pivot[present].iterrows():
        y=row.to_numpy(dtype=float)
        ax.plot(x,y,color="#BDBDBD",alpha=0.18,linewidth=0.55,zorder=1)
    for i,m in enumerate(present):
        vals=pivot[m].to_numpy(dtype=float); vals=vals[np.isfinite(vals)]
        ax.scatter(_jitter(vals.size,i,0.11,rng),vals,s=9,color=MODEL_COLORS[m],alpha=0.68,edgecolors="none",zorder=2)
        mean,lo,hi=bootstrap_ci(vals,seed+i)
        ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt="o",markersize=4.8,color="black",mfc=MODEL_COLORS[m],mec="black",capsize=2.3,linewidth=1.0,zorder=4)
    ax.axhline(0,color="#555555",linewidth=0.7,zorder=0) if metric.startswith("delta_") else None
    ax.set_xticks(x); ax.set_xticklabels([MODEL_SHORT[m] for m in present],rotation=28,ha="right",linespacing=0.95)
    ax.tick_params(axis="x",pad=2)
    ax.set_ylabel(ylabel); ax.set_title(title,pad=5)
    panel_label(ax,panel,x=-0.13,y=1.20)


def plot_figure2(analysis_root: Path, runtime: dict, out_dir: Path, args: argparse.Namespace, formats: Sequence[str], dpi: int, seed: int) -> tuple[list[Path], pd.DataFrame]:
    indiv=read_csv(analysis_root/"individualization"/"all_models_individualization_per_subject.csv")
    validate_result_frames([indiv])
    if args.example_subjects:
        ids=parse_subjects(args.example_subjects)
        sel=indiv[(indiv["model"]=="thesis_vrptr_full") & indiv["subject"].map(_subject_key).isin(ids)].copy()
        sel["subject_key"]=sel["subject"].map(_subject_key); sel["selection_quantile"]=np.nan
        sel=sel.set_index("subject_key").loc[ids].reset_index()
    else:
        qs=[float(x) for x in args.example_quantiles.split(",") if x.strip()]
        sel=choose_quantile_subjects(indiv,qs)
    ids=sel["subject_key"].astype(str).tolist()
    missing=[s for s in ids if s not in runtime["truth"] or s not in runtime["pred"]]
    if missing: raise KeyError(f"Selected subjects missing from runtime predictions: {missing}")

    truths=runtime["truth"]; preds=runtime["pred"]
    all_ids=sorted(set(truths)&set(preds))
    template_hash = validate_result_frames([indiv])
    if template_hash != runtime["template_metadata"]["template_sha256"]:
        raise ValueError("Figure maps and compiled metrics use different templates")
    if set(indiv.subject.map(_subject_key)) != set(all_ids):
        raise ValueError("Figure participants differ from compiled results")
    templates={s:runtime["template"] for s in ids}
    obs_res={s:truths[s]-templates[s] for s in ids}
    pred_res={s:preds[s]-templates[s] for s in ids}
    errors={s:np.abs(preds[s]-truths[s]) for s in ids}

    task_scale=_robust_abs_scale([truths[s] for s in all_ids]+[preds[s] for s in all_ids],0.995)
    residual_scale=_robust_abs_scale(list(obs_res.values())+list(pred_res.values()),0.995)
    error_scale=max(_robust_abs_scale(list(errors.values()),0.99),1e-5)

    fig=plt.figure(figsize=(10.6,8.25))
    outer=GridSpec(2,1,height_ratios=[1.25,1.0],hspace=0.30,figure=fig)
    top=GridSpecFromSubplotSpec(4,6,subplot_spec=outer[0],height_ratios=[1,1,1,0.09],hspace=0.05,wspace=0.03)
    col_titles=["Observed","VRPTR prediction","Training template","Observed residual","Predicted residual","Absolute error"]
    row_labels=["25th percentile","Median","75th percentile"]
    selection_records=[]; task_im=resid_im=err_im=None
    for r,sid in enumerate(ids):
        target=truths[sid]; pred=preds[sid]; template=templates[sid]
        axis,idx=choose_informative_slice(target,args.slice_axis)
        mask3=_brain_mask_3d([target,pred,template]); mask2=extract_slice(mask3.astype(np.float32),axis,idx)>0.5
        maps=[target,pred,template,obs_res[sid],pred_res[sid],errors[sid]]
        for c,vol in enumerate(maps):
            ax=fig.add_subplot(top[r,c]); sl=extract_slice(vol,axis,idx)
            if c<=2:
                im=show_brain(ax,sl,mask2,"RdBu_r",-task_scale,task_scale); task_im=task_im or im
            elif c<=4:
                im=show_brain(ax,sl,mask2,"RdBu_r",-residual_scale,residual_scale); resid_im=resid_im or im
            else:
                im=show_brain(ax,sl,mask2,"magma",0,error_scale); err_im=err_im or im
            annotate_orientation(ax, runtime["atlas"].parcel_img.affine, axis)
            if r==0: ax.set_title(col_titles[c],fontsize=7.9,pad=3)
            if c==0:
                ax.text(-0.15,0.5,row_labels[r]+"\n"+slice_coordinate(runtime["atlas"].parcel_img.affine,target.shape,axis,idx),transform=ax.transAxes,ha="right",va="center",fontsize=7.2)
            if r==0 and c==0: panel_label(ax,"A",x=-0.28,y=1.12)
        q=sel.iloc[r].get("selection_quantile",np.nan)
        selection_records.append({
            "quantile":q,"quantile_label":row_labels[r],"subject":sid,
            "selection_rule":"nearest requested quantile of archived full-VRPTR held-out Pearson r; unique subjects; no visual inspection",
            "reported_full_vrptr_pearson_r":float(sel.iloc[r]["model_pearson_r"]),
            "reported_full_vrptr_residual_r":float(sel.iloc[r]["residual_pearson_r"]),
            "runtime_full_vrptr_pearson_r":_safe_corr(target,pred),
            "runtime_full_vrptr_residual_r":_safe_corr(obs_res[sid],pred_res[sid]),
            "slice_axis":["x","y","z"][axis],"slice_index":idx,
            "slice_coordinate":slice_coordinate(runtime["atlas"].parcel_img.affine,target.shape,axis,idx),
            "template_sha256":template_hash, "target_units":"within-participant standardized task map",
            "task_beta_scale":task_scale,"residual_scale":residual_scale,"absolute_error_scale":error_scale,
        })

    # Minimal colorbars: only endpoints and zero where meaningful.
    bar_grid=GridSpecFromSubplotSpec(1,18,subplot_spec=top[3,:],wspace=1.0)
    cax1=fig.add_subplot(bar_grid[0,0:8]); cb1=fig.colorbar(task_im,cax=cax1,orientation="horizontal",ticks=[-task_scale,0,task_scale]); cb1.set_label("Standardized task map",fontsize=7.0); cb1.ax.set_xticklabels([f"{-task_scale:.2f}","0",f"{task_scale:.2f}"]); cb1.ax.tick_params(labelsize=6.2)
    cax2=fig.add_subplot(bar_grid[0,9:14]); cb2=fig.colorbar(resid_im,cax=cax2,orientation="horizontal",ticks=[-residual_scale,0,residual_scale]); cb2.set_label("Residual (standardized units)",fontsize=7.0); cb2.ax.set_xticklabels([f"{-residual_scale:.2f}","0",f"{residual_scale:.2f}"]); cb2.ax.tick_params(labelsize=6.2)
    cax3=fig.add_subplot(bar_grid[0,15:18]); cb3=fig.colorbar(err_im,cax=cax3,orientation="horizontal",ticks=[0,error_scale]); cb3.set_label("|Error| (standardized units)",fontsize=7.0); cb3.ax.set_xticklabels(["0",f"{error_scale:.2f}"]); cb3.ax.tick_params(labelsize=6.2)

    bottom=GridSpecFromSubplotSpec(2,6,subplot_spec=outer[1],hspace=1.05,wspace=0.62)
    axes=[fig.add_subplot(bottom[0,0:2]),fig.add_subplot(bottom[0,2:4]),fig.add_subplot(bottom[0,4:6]),fig.add_subplot(bottom[1,1:3]),fig.add_subplot(bottom[1,3:5])]
    specs=[
        ("model_pearson_r","Whole-map prediction","Pearson r","B"),
        ("model_dice_auc","Activation overlap","Dice AUC","C"),
        ("delta_pearson_r","Template gain","Δ Pearson r","D"),
        ("residual_pearson_r","Residual recovery","Residual r","E"),
        ("residual_id_self_vs_best_other_margin","Residual specificity","Self − best-other r","F"),
    ]
    for i,(metric,title,ylabel,label) in enumerate(specs):
        _quant_panel(axes[i],indiv,metric,title,ylabel,label,seed+10*i)
    fig.subplots_adjust(left=0.095,right=0.985,top=0.985,bottom=0.14)

    sel_df=pd.DataFrame(selection_records)
    for suffix in ("pearson_r", "residual_r"):
        if not np.allclose(sel_df["reported_full_vrptr_"+suffix],sel_df["runtime_full_vrptr_"+suffix],atol=1e-4,rtol=0,equal_nan=True):
            raise ValueError("Runtime figure predictions differ from compiled metrics; check checkpoint, scaling and mask")
    sel_path=out_dir/"Figure2_task_map_example_selection.csv"; out_dir.mkdir(parents=True,exist_ok=True); sel_df.to_csv(sel_path,index=False)
    paths=save_figure(fig,out_dir,"Figure2_prediction_and_individualization",formats,dpi)+[sel_path]
    return paths,sel_df


# -----------------------------------------------------------------------------
# Figure 3: factorial effects
# -----------------------------------------------------------------------------

def _factorial_subject_effects(indiv: pd.DataFrame, metric: str) -> pd.DataFrame:
    d=indiv[indiv["model"].isin(MODEL_ORDER[:4])][["subject","model",metric]].copy(); d["subject"]=d["subject"].map(_subject_key)
    p=d.pivot_table(index="subject",columns="model",values=metric,aggfunc="mean").dropna()
    f=p["thesis_vrptr_full"]; det=p["thesis_vrptr_deterministic"]; nt=p["thesis_vrptr_no_transformer"]; co=p["thesis_vrptr_compression_only"]
    return pd.DataFrame({
        "Transformer":0.5*((f-nt)+(det-co)),
        "Variational":0.5*((f-det)+(nt-co)),
        "Interaction":(f-det)-(nt-co),
    })


def _effect_panel(ax: plt.Axes, effects: pd.DataFrame, title: str, ylabel: str, label: str, seed: int) -> None:
    rng=np.random.default_rng(seed); names=list(effects.columns); colors=[COLORS["transformer"],COLORS["variational"],COLORS["interaction"]]
    for i,(name,color) in enumerate(zip(names,colors)):
        vals=effects[name].to_numpy(dtype=float); vals=vals[np.isfinite(vals)]
        ax.scatter(_jitter(vals.size,i,0.10,rng),vals,s=11,color=color,alpha=0.72,edgecolors="none")
        mean,lo,hi=bootstrap_ci(vals,seed+i)
        ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt="o",markersize=5,color="black",mfc=color,mec="black",capsize=2.5,linewidth=1.0,zorder=4)
    ax.axhline(0,color="#333333",linewidth=0.75)
    ax.set_xticks(range(3)); ax.set_xticklabels(names,rotation=0)
    ax.set_title(title,pad=5); ax.set_ylabel(ylabel)
    panel_label(ax,label,x=-0.12,y=1.06)


def plot_figure3(analysis_root: Path,out_dir: Path,formats: Sequence[str],dpi: int,seed: int)->list[Path]:
    indiv=read_csv(analysis_root/"individualization"/"all_models_individualization_per_subject.csv")
    validate_result_frames([indiv])
    fig,axes=plt.subplots(2,2,figsize=(8.4,6.0))
    specs=[
        ("model_pearson_r","Whole-map prediction","Δ Pearson r","A"),
        ("model_dice_auc","Activation overlap","Δ Dice AUC","B"),
        ("residual_pearson_r","Residual recovery","Δ residual r","C"),
        ("residual_id_self_vs_best_other_margin","Residual specificity","Δ self margin","D"),
    ]
    for i,(metric,title,ylabel,label) in enumerate(specs):
        _effect_panel(axes.flat[i],_factorial_subject_effects(indiv,metric),title,ylabel,label,seed+20*i)
    fig.subplots_adjust(left=0.10,right=0.985,top=0.965,bottom=0.14,wspace=0.28,hspace=0.34)
    return save_figure(fig,out_dir,"Figure3_factorial_component_effects",formats,dpi)


# -----------------------------------------------------------------------------
# Figure 4: mechanism without explanatory text diagram
# -----------------------------------------------------------------------------

def _swap_pair_panel(ax: plt.Axes, swap: pd.DataFrame, metric: str, title: str, ylabel: str, label: str, seed: int) -> None:
    conditions=["donor_bottleneck_recipient_skips","recipient_bottleneck_donor_skips"]
    p=swap.pivot_table(index="recipient_subject",columns="condition",values=metric,aggfunc="mean").dropna()
    x=np.asarray([0,1],dtype=float)
    for _,row in p[conditions].iterrows(): ax.plot(x,row.to_numpy(dtype=float),color="#BDBDBD",alpha=0.55,linewidth=0.65)
    colors=[COLORS["bottleneck"],COLORS["skips"]]; rng=np.random.default_rng(seed)
    for i,(cond,color) in enumerate(zip(conditions,colors)):
        vals=p[cond].to_numpy(dtype=float)
        ax.scatter(_jitter(vals.size,i,0.04,rng),vals,s=14,color=color,alpha=0.78,edgecolors="none")
        mean,lo,hi=bootstrap_ci(vals,seed+i)
        ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt="o",markersize=5,color="black",mfc=color,mec="black",capsize=2.5,linewidth=1.0)
    ax.axhline(0,color="#444444",linewidth=0.7)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Donor\nbottleneck","Donor\nskips"])
    ax.set_title(title,pad=5); ax.set_ylabel(ylabel); panel_label(ax,label,x=-0.14,y=1.06)


def _mean_replacement_panel(ax: plt.Axes, comp: pd.DataFrame, seed: int) -> None:
    p_whole=comp.pivot_table(index="subject",columns="condition",values="pearson_r",aggfunc="mean").dropna()
    p_res=comp.pivot_table(index="subject",columns="condition",values="residual_pearson_r",aggfunc="mean").dropna()
    base="correct_bottleneck_correct_skips"; mb="cohort_mean_bottleneck_recipient_skips"; ms="recipient_bottleneck_cohort_mean_skips"
    series=[p_whole[base]-p_whole[mb],p_whole[base]-p_whole[ms],p_res[base]-p_res[mb],p_res[base]-p_res[ms]]
    labels=["Bottleneck\nwhole map","Skips\nwhole map","Bottleneck\nresidual","Skips\nresidual"]
    colors=[COLORS["bottleneck"],COLORS["skips"],COLORS["bottleneck"],COLORS["skips"]]
    rng=np.random.default_rng(seed)
    for i,(vals,color) in enumerate(zip(series,colors)):
        arr=vals.to_numpy(dtype=float)
        ax.scatter(_jitter(arr.size,i,0.10,rng),arr,s=11,color=color,alpha=0.60,edgecolors="none")
        mean,lo,hi=bootstrap_ci(arr,seed+i)
        ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt="o",markersize=5,color="black",mfc=color,mec="black",capsize=2.5,linewidth=1.0)
    ax.axhline(0,color="#333333",linewidth=0.75)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels,rotation=0)
    ax.set_ylabel("Loss after mean replacement")
    ax.set_title("Cohort-mean replacement",pad=5)
    panel_label(ax,"C",x=-0.10,y=1.06)


def plot_figure4(analysis_root: Path,out_dir: Path,formats: Sequence[str],dpi: int,seed: int)->list[Path]:
    base=analysis_root/"skip_bottleneck_swap"/"full"
    swap=read_csv(base/"swap_per_recipient.csv"); comp=read_csv(base/"component_condition_per_subject.csv")
    fig=plt.figure(figsize=(9.4,3.45)); gs=GridSpec(1,3,width_ratios=[1,1,1.18],wspace=0.38,figure=fig)
    ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[0,2])
    validate_result_frames([swap,comp])
    _swap_pair_panel(ax1,swap,"whole_identity_shift_toward_donor","Whole-map donor shift","Shift toward donor","A",seed)
    _swap_pair_panel(ax2,swap,"residual_identity_shift_toward_donor","Residual donor shift","Shift toward donor","B",seed+10)
    _mean_replacement_panel(ax3,comp,seed+20)
    fig.subplots_adjust(left=0.075,right=0.99,top=0.94,bottom=0.20)
    return save_figure(fig,out_dir,"Figure4_skip_bottleneck_component_swaps",formats,dpi)


# -----------------------------------------------------------------------------
# Figure 5: posterior predictive maps and utility
# -----------------------------------------------------------------------------

def _uncertainty_rank_panel(ax: plt.Axes, clean: pd.DataFrame, seed: int) -> None:
    vals=clean["voxel_uncertainty_abs_error_spearman"].to_numpy(dtype=float); vals=vals[np.isfinite(vals)]
    rng=np.random.default_rng(seed)
    ax.scatter(_jitter(vals.size,0,0.08,rng),vals,s=15,color=COLORS["full"],alpha=0.68,edgecolors="none")
    mean,lo,hi=bootstrap_ci(vals,seed)
    ax.errorbar(0,mean,yerr=[[mean-lo],[hi-mean]],fmt="o",markersize=5,color="black",mfc=COLORS["full"],mec="black",capsize=2.5,linewidth=1.0)
    ax.axhline(0,color="#444444",linewidth=0.7)
    ax.set_xlim(-0.25,0.25); ax.set_xticks([]); ax.set_ylabel("Spearman ρ")
    ax.set_title("Uncertainty ranks error",pad=5); panel_label(ax,"D",x=-0.22,y=1.07)


def _risk_panel(ax: plt.Axes, risk: pd.DataFrame) -> None:
    mean=risk.groupby("coverage",as_index=False)[["mae","oracle_mae","random_expected_mae"]].mean().sort_values("coverage")
    x=100*mean["coverage"].to_numpy(dtype=float)
    ax.plot(x,mean["mae"],marker="o",linewidth=1.4,color=COLORS["full"],label="Posterior SD")
    ax.plot(x,mean["random_expected_mae"],linestyle="--",linewidth=1.1,color=COLORS["random"],label="Random")
    ax.plot(x,mean["oracle_mae"],linestyle=":",linewidth=1.4,color=COLORS["oracle"],label="Oracle")
    ax.set_xlabel("Retained voxels (%)"); ax.set_ylabel("Mean absolute error"); ax.set_title("Selective prediction",pad=5)
    ax.legend(frameon=False,loc="lower right",handlelength=2.3)
    panel_label(ax,"E",x=-0.16,y=1.07)


def _calibration_panel(ax: plt.Axes, calibration: pd.DataFrame) -> None:
    nominal=np.asarray([0.50,0.80,0.90,0.95])
    raw=np.asarray([calibration[f"raw_coverage_{int(x*100)}"].mean() for x in nominal])
    const=np.asarray([calibration[f"constant_coverage_{int(x*100)}"].mean() for x in nominal])
    hetero=np.asarray([calibration[f"heteroscedastic_coverage_{int(x*100)}"].mean() for x in nominal])
    ax.plot(nominal,nominal,color="#222222",linestyle="--",linewidth=1.0,label="Ideal")
    ax.plot(nominal,raw,marker="o",color=COLORS["raw"],linewidth=1.2,label="Raw")
    ax.plot(nominal,const,marker="s",color=COLORS["constant"],linewidth=1.1,label="Constant")
    ax.plot(nominal,hetero,marker="o",color=COLORS["hetero"],linewidth=1.4,label="Spatial")
    ax.set_xlim(0.47,0.98); ax.set_ylim(0,1.02)
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage"); ax.set_title("Cross-fitted calibration",pad=5)
    ax.legend(frameon=False,loc="upper left",ncol=2,handlelength=1.7,columnspacing=0.8)
    panel_label(ax,"F",x=-0.16,y=1.07)


def _corruption_rows(corr: pd.DataFrame,family: str)->pd.DataFrame:
    clean=corr[corr["condition"]=="clean"].copy(); sub=corr[corr["corruption_family"]==family].copy()
    return pd.concat([clean,sub],ignore_index=True).sort_values("corruption_severity")


def _corruption_panel(ax: plt.Axes, corr: pd.DataFrame) -> None:
    for family,label,color,marker in [("gaussian_noise","Gaussian noise",COLORS["full"],"o"),("channel_dropout","Channel dropout",COLORS["skips"],"s")]:
        sub=_corruption_rows(corr,family)
        ax.plot(np.arange(len(sub)),sub["posterior_sd_mean_mean"],marker=marker,color=color,linewidth=1.4,label=label)
    ax.set_xticks([0,1,2,3]); ax.set_xticklabels(["Clean","1","2","3"])
    ax.set_xlabel("Within-family severity step"); ax.set_ylabel("Mean posterior SD"); ax.set_title("Input degradation",pad=5)
    ax.legend(frameon=False,loc="upper left")
    panel_label(ax,"G",x=-0.16,y=1.07)


def plot_figure5(analysis_root: Path,runtime: dict,selection: pd.DataFrame,out_dir: Path,args: argparse.Namespace,formats: Sequence[str],dpi: int,seed: int)->list[Path]:
    clean=read_csv(analysis_root/"posterior_predictive"/"clean"/"posterior_per_subject.csv")
    risk=read_csv(analysis_root/"posterior_predictive"/"clean"/"posterior_risk_coverage.csv")
    calibration=read_csv(analysis_root/"posterior_predictive"/"compiled"/"crossfit_calibration_per_subject.csv")
    corr=read_csv(analysis_root/"posterior_predictive"/"compiled"/"corruption_summary.csv")
    # Use the median performance example selected independently of map appearance.
    if "quantile_label" in selection.columns:
        row=selection.iloc[(selection["quantile_label"]=="Median").to_numpy().nonzero()[0][0]] if (selection["quantile_label"]=="Median").any() else selection.iloc[len(selection)//2]
    else: row=selection.iloc[len(selection)//2]
    sid=_subject_key(row["subject"])
    target=runtime["truth"][sid]
    post_mean,post_sd=_posterior_maps_for_subject(runtime,sid,args)
    error=np.abs(post_mean-target)
    axis,idx=choose_informative_slice(target,args.slice_axis)
    mask3=_brain_mask_3d([target,post_mean]); mask2=extract_slice(mask3.astype(np.float32),axis,idx)>0.5
    task_scale=_robust_abs_scale([target,post_mean],0.995); error_scale=_robust_abs_scale([error],0.99); sd_scale=_robust_abs_scale([post_sd],0.99)

    fig=plt.figure(figsize=(11.2,6.9)); outer=GridSpec(2,1,height_ratios=[0.92,1.0],hspace=0.34,figure=fig)
    top=GridSpecFromSubplotSpec(2,8,subplot_spec=outer[0],height_ratios=[1,0.08],wspace=0.85,hspace=0.12)
    map_axes=[fig.add_subplot(top[0,0:2]),fig.add_subplot(top[0,2:4]),fig.add_subplot(top[0,4:6])]
    slices=[extract_slice(post_mean,axis,idx),extract_slice(error,axis,idx),extract_slice(post_sd,axis,idx)]
    ims=[]
    ims.append(show_brain(map_axes[0],slices[0],mask2,"RdBu_r",-task_scale,task_scale))
    ims.append(show_brain(map_axes[1],slices[1],mask2,"magma",0,error_scale))
    ims.append(show_brain(map_axes[2],slices[2],mask2,"viridis",0,sd_scale))
    for ax,title,label in zip(map_axes,["Posterior mean","Absolute error","Posterior SD"],["A","B","C"]):
        annotate_orientation(ax, runtime["atlas"].parcel_img.affine, axis)
        ax.set_title(title,pad=4); panel_label(ax,label,x=-0.12,y=1.08)
    caxes=[fig.add_subplot(top[1,0:2]),fig.add_subplot(top[1,2:4]),fig.add_subplot(top[1,4:6])]
    cbs=[fig.colorbar(ims[0],cax=caxes[0],orientation="horizontal",ticks=[-task_scale,0,task_scale]),fig.colorbar(ims[1],cax=caxes[1],orientation="horizontal",ticks=[0,error_scale]),fig.colorbar(ims[2],cax=caxes[2],orientation="horizontal",ticks=[0,sd_scale])]
    cbs[0].ax.set_xticklabels([f"{-task_scale:.2f}","0",f"{task_scale:.2f}"])
    cbs[1].ax.set_xticklabels(["0",f"{error_scale:.2f}"])
    cbs[2].ax.set_xticklabels(["0",f"{sd_scale:.3f}"])
    for cb,label in zip(cbs,["Standardized task map","|Error| (standardized)","SD (standardized)"]): cb.set_label(label,fontsize=7.0); cb.ax.tick_params(labelsize=6.2)
    map_axes[0].text(0.5,-0.04,f"Subject {sid}; "+slice_coordinate(runtime["atlas"].parcel_img.affine,target.shape,axis,idx),transform=map_axes[0].transAxes,ha="center",fontsize=6.5)
    axd=fig.add_subplot(top[:,6:8]); _uncertainty_rank_panel(axd,clean,seed)

    bottom=GridSpecFromSubplotSpec(1,3,subplot_spec=outer[1],wspace=0.38)
    axe=fig.add_subplot(bottom[0,0]); axf=fig.add_subplot(bottom[0,1]); axg=fig.add_subplot(bottom[0,2])
    _risk_panel(axe,risk); _calibration_panel(axf,calibration); _corruption_panel(axg,corr)
    fig.subplots_adjust(left=0.075,right=0.985,top=0.965,bottom=0.09)

    sel_path=out_dir/"Figure5_posterior_example_selection.csv"
    pd.DataFrame([{"subject":sid,"selection_rule":"median full-VRPTR held-out Pearson example selected before visual inspection","slice_axis":["x","y","z"][axis],"slice_index":idx,"posterior_samples":int(args.posterior_samples)}]).to_csv(sel_path,index=False)
    return save_figure(fig,out_dir,"Figure5_calibrated_posterior_predictive_uncertainty",formats,dpi)+[sel_path]


# -----------------------------------------------------------------------------
# Captions and manifest
# -----------------------------------------------------------------------------

def write_captions(out_dir: Path) -> Path:
    text="""FIGURE 1. VRPTR architecture and study design. A, The deepest encoder representation is compressed into channel tokens, processed by a Transformer encoder, parameterized as a Gaussian latent posterior, sampled, decoded by a Transformer, and spatially expanded. Shallower multiscale skips condition reconstruction; no deepest-feature bypass is present. B, Matched Transformer-by-variational ablations and fixed learned comparators isolate the contribution of each component.

FIGURE 2. Observed and predicted language-task maps with held-out prediction and secondary residual recovery. A, Observed task maps, full-VRPTR point predictions (z=mu), fixed training-derived population templates, observed and predicted residuals, and absolute errors for participants nearest the 25th, 50th, and 75th percentiles of full-VRPTR held-out Pearson correlation. Examples were selected algorithmically before visual inspection. All panels within a participant use the same selected slice, with orientation and MNI coordinates derived from the image affine; non-brain voxels are masked consistently, and fixed scales are used within map classes. Subject identifiers, slice coordinates, template checksum, and performance values are provided in Figure2_task_map_example_selection.csv and the manuscript caption rather than printed over the images. B is the primary prediction outcome; C–F are complementary or secondary. B–F, Participant-level voxel-map correlation, Dice AUC, gain over the training-derived template, residual correlation, and residual self-versus-best-other margin. Axis abbreviations denote Full VRPTR, deterministic VRPTR (Det.), no-Transformer VRPTR (No T), compression-only VRPTR (Comp.), SWIFUN-like, and BrainSurfCNN-like. Gray lines connect the same participant; colored points show participant values and black symbols show means with percentile bootstrap 95% confidence intervals (5,000 resamples).

FIGURE 3. Factorial effects of Transformer processing and variational sampling. Participant-level Transformer main effects, variational main effects, and interactions for whole-map correlation, activation overlap, residual correlation, and residual self-margin. Points are held-out participants; black symbols show means and percentile bootstrap 95% confidence intervals (5,000 resamples).

FIGURE 4. Skip-bottleneck component swaps. A–B, Changes in similarity toward the participant supplying replacement features, relative to the recipient’s unmodified prediction. C, Performance changes after replacing skips or bottleneck features with their cohort mean. The task-map residual reference is fixed from training; cohort-mean hidden-feature replacements remain descriptive test-cohort interventions. Results are averaged across donors within each recipient before inference. Points represent recipients; summary markers and bars show means and percentile bootstrap 95% confidence intervals (5,000 resamples).

FIGURE 5. Calibrated posterior predictive uncertainty. A–C, Posterior-mean, absolute-error, and posterior-standard-deviation maps for the independently selected median-performance participant, displayed with a common brain mask and numeric standardized-unit colorbars and affine-derived orientation labels. D, Participant-level correspondence between posterior standard deviation and voxelwise absolute error. E, Selective scoring within the sampled output-voxel set, ranked by posterior SD, at retained fractions 25%, 50%, 75%, 90%, and 100%; inputs and predictions are unchanged. The summary gain is 1 minus normalized trapezoidal risk-coverage AUC divided by MAE across all voxels selected for uncertainty evaluation, not an error reduction at one retained fraction. F, Raw and cross-fitted calibrated interval coverage. G, Posterior uncertainty by within-family severity step: Gaussian noise SD 0, 0.25, 0.50, 1.00 in standardized input units; whole-feature-map dropout fraction 0, 0.10, 0.25, 0.50. These are separate perturbation scales. F uses internal cross-fitted calibration on evaluation participants, not independent external validation. Numerical effects and inferential statistics are reported in the text and caption.
"""
    path=out_dir/"main_figure_captions.txt"; path.write_text(text); return path


def write_manifest(out_dir: Path,generated: Iterable[Path],args: argparse.Namespace)->Path:
    path=out_dir/"main_figure_manifest.csv"
    with path.open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["type","key","value"])
        for key in ["root","analysis_root","posterior_samples","posterior_sample_batch","prediction_module","training_template"]:
            w.writerow(["configuration",key,getattr(args,key)])
        for pth in generated: w.writerow(["output",pth.name,str(pth)])
    return path


def _clean_stems(out_dir: Path) -> None:
    stems=["Figure1_VRPTR_architecture_and_study_design","Figure2_prediction_and_individualization","Figure3_factorial_component_effects","Figure4_skip_bottleneck_component_swaps","Figure5_calibrated_posterior_predictive_uncertainty"]
    for stem in stems:
        for ext in ["pdf","svg","png"]:
            p=out_dir/f"{stem}.{ext}"
            if p.exists(): p.unlink()


def main() -> None:
    args=parse_args(); root=Path(args.root).expanduser().resolve()
    analysis_root=Path(args.analysis_root).expanduser().resolve() if args.analysis_root else root/"thesis_vrptr"/"final_analysis"
    out_dir=Path(args.out_dir).expanduser().resolve() if args.out_dir else root/"paper_figures_imaging_neuroscience"
    formats=[x.strip() for x in args.formats.split(",") if x.strip()]
    args.analysis_root=str(analysis_root); configure_style(args.dpi); out_dir.mkdir(parents=True,exist_ok=True)
    if args.clean_output: _clean_stems(out_dir)
    generated=[]
    generated+=plot_figure1(out_dir,formats,args.dpi)
    runtime=None; selection=None
    if not args.skip_map_inference:
        runtime=load_full_runtime(root,args)
        p2,selection=plot_figure2(analysis_root,runtime,out_dir,args,formats,args.dpi,args.seed); generated+=p2
    generated+=plot_figure3(analysis_root,out_dir,formats,args.dpi,args.seed)
    generated+=plot_figure4(analysis_root,out_dir,formats,args.dpi,args.seed)
    if not args.skip_map_inference and not args.skip_uncertainty:
        generated+=plot_figure5(analysis_root,runtime,selection,out_dir,args,formats,args.dpi,args.seed)
    generated.append(write_captions(out_dir)); generated.append(write_manifest(out_dir,generated,args))
    print(f"[figures] Wrote polished main figures to {out_dir}")
    for path in generated: print(f"  - {path}")


if __name__=="__main__": main()
