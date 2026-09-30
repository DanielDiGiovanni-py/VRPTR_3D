#!/usr/bin/env python3
"""Build resting-state features and optional language targets from cleaned images.

Defaults preserve the active resting-state calculations in the supplied study
script. The task stage exposes its previously commented-out invocation and must
be selected explicitly; see README.md for provenance and numerical limitations.
"""

import argparse
import glob
import logging
import os
from pathlib import Path
import time


LOGGER = logging.getLogger(__name__)


def compute_resting_connectivity(rest_imgs, labels_img, output_dir,
                                 t_r=0.72, high_pass=0.01, low_pass=0.1):
    """Save the arithmetic mean of Nilearn correlation estimates across runs.

    Retains ConnectivityMeasure's original default covariance estimator. This
    function saves only the mean matrix, not individual run matrices.
    """
    import numpy as np
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.maskers import NiftiLabelsMasker

    masker = NiftiLabelsMasker(
        labels_img=labels_img, detrend=False, standardize=False,
        low_pass=low_pass, high_pass=high_pass, t_r=t_r,
    )
    time_series = [masker.fit_transform(img) for img in rest_imgs]
    conn_measure = ConnectivityMeasure(kind="correlation")
    conn_matrices = conn_measure.fit_transform(time_series)
    mean_conn = np.mean(conn_matrices, axis=0)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "conn_mean.npy"), mean_conn)
    return mean_conn


def compute_connectivity_fingerprint(rest_imgs, labels_img, output_file,
                                     mask_img=None, t_r=0.72,
                                     high_pass=0.01, low_pass=0.1):
    """Save one voxel-to-parcel connectivity volume per atlas parcel.

    Runs are filtered separately, concatenated, and standardized. The historical
    population-SD normalization and T-1 divisor are preserved exactly.
    """
    import numpy as np
    from nilearn.datasets import load_mni152_gm_mask
    from nilearn.maskers import NiftiLabelsMasker, NiftiMasker

    if mask_img is None:
        mask_img = load_mni152_gm_mask()
    label_masker = NiftiLabelsMasker(
        labels_img=labels_img, detrend=False, standardize=False,
        low_pass=low_pass, high_pass=high_pass, t_r=t_r,
    )
    brain_masker = NiftiMasker(
        mask_img=mask_img, detrend=False, standardize=False,
        low_pass=low_pass, high_pass=high_pass, t_r=t_r,
    )
    roi_ts = np.vstack([label_masker.fit_transform(img) for img in rest_imgs])
    brain_ts = np.vstack([brain_masker.fit_transform(img) for img in rest_imgs])
    roi_ts = (roi_ts - roi_ts.mean(axis=0)) / roi_ts.std(axis=0)
    brain_ts = (brain_ts - brain_ts.mean(axis=0)) / brain_ts.std(axis=0)
    corr = brain_ts.T @ roi_ts / (brain_ts.shape[0] - 1)
    fingerprint_img = brain_masker.inverse_transform(corr.T)
    fingerprint_img.to_filename(output_file)
    return fingerprint_img


def load_hcp_language_events(events_dir):
    """Read story/math onsets and durations; the source weight column is unused."""
    import pandas as pd

    blocks = []
    for condition, filename in [("story", "story.txt"), ("math", "math.txt")]:
        events = pd.read_csv(
            os.path.join(events_dir, filename), sep=r"\s+", header=None,
            names=["onset", "duration", "weight"],
        )
        events["trial_type"] = condition
        blocks.append(events)
    return pd.concat(blocks, ignore_index=True)[["onset", "duration", "trial_type"]]


def compute_language_betas(task_imgs, events_dfs, output_dir,
                           t_r=0.72, hrf_model="spm", drift_model="cosine",
                           high_pass=0.01, smoothing_fwhm=None):
    """Fit the original multirun GLM and save both language contrast effects."""
    import nibabel as nib
    import numpy as np
    from nilearn.datasets import load_mni152_gm_mask
    from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

    if not task_imgs or len(task_imgs) != len(events_dfs):
        raise ValueError("Provide at least one task image and exactly one event table per image.")
    gm_mask = load_mni152_gm_mask()
    imgs = []
    design_matrices = []
    for img_path, events in zip(task_imgs, events_dfs):
        imgs.append(img_path)
        nii = nib.load(img_path)
        frame_times = np.arange(nii.shape[-1]) * t_r
        design_matrix = make_first_level_design_matrix(
            frame_times, events, hrf_model=hrf_model,
            drift_model=drift_model, high_pass=high_pass,
        )
        design_matrices.append(design_matrix)
    first_lvl = FirstLevelModel(
        t_r=t_r, slice_time_ref=0.5, hrf_model=hrf_model, noise_model="ar1",
        smoothing_fwhm=smoothing_fwhm, mask_img=gm_mask,
    ).fit(imgs, design_matrices=design_matrices)
    contrast_defs = {"story_vs_math": "story - math", "math_vs_story": "math - story"}
    os.makedirs(output_dir, exist_ok=True)
    beta_files = {}
    cols = first_lvl.design_matrices_[0].columns
    if "story" not in cols or "math" not in cols:
        LOGGER.warning("First design matrix lacks story/math regressors; no contrasts saved.")
        return beta_files
    for name, contrast in contrast_defs.items():
        beta_map = first_lvl.compute_contrast(contrast, output_type="effect_size")
        out_path = os.path.join(output_dir, f"beta_{name}.nii.gz")
        beta_map.to_filename(out_path)
        beta_files[name] = out_path
    return beta_files


def plot_connectivity_matrix(conn_matrix, output_file, labels=None):
    """Save the original parcel connectivity matrix diagnostic."""
    import matplotlib.pyplot as plt
    from nilearn import plotting

    fig = plt.figure(figsize=(12, 10))
    plotting.plot_matrix(
        conn_matrix, figure=fig, labels=labels, colorbar=True,
        title="Mean Connectivity Matrix", vmax=1.0, vmin=-1.0,
    )
    fig.savefig(output_file)
    plt.close(fig)


def plot_beta_maps(beta_files, output_dir):
    """Save orthogonal-slice diagnostics for each estimated contrast."""
    import matplotlib.pyplot as plt
    from nilearn import plotting

    os.makedirs(output_dir, exist_ok=True)
    for name, path in beta_files.items():
        fig = plt.figure(figsize=(8, 6))
        plotting.plot_stat_map(
            path, bg_img=None, figure=fig, title=name,
            display_mode="ortho", colorbar=True,
        )
        fig.savefig(os.path.join(output_dir, f"{name}.png"))
        plt.close(fig)


def plot_surface(stat_map, output_file, title):
    """Save lateral and medial views using the original plotting settings."""
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_img_on_surf

    plt.figure(figsize=(8, 6))
    plot_img_on_surf(
        stat_map=stat_map, views=["lateral", "medial"],
        hemispheres=["left", "right"], colorbar=True, title=title,
        bg_on_data=True,
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def build_rest_features(subject_dir, tr, rng, force=False, skip_qc=False, atlas_cache=None):
    """Build the 100-parcel features unless the historical QC marker exists."""
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.image import index_img

    output_dir = subject_dir / "output" / "rest_connectivity_100"
    if not force and list(output_dir.glob("fingerprint_surface_100_roi*.png")):
        LOGGER.info("Subject %s: resting QC exists; skipping resting stage.", subject_dir.name)
        return
    rest_imgs = sorted(glob.glob(str(subject_dir / "*REST*_sm_clean.nii.gz")))
    if not rest_imgs:
        LOGGER.info("Subject %s: no resting runs found.", subject_dir.name)
        return
    atlas = fetch_atlas_schaefer_2018(
        n_rois=100, yeo_networks=17, resolution_mm=2, data_dir=atlas_cache,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_conn = compute_resting_connectivity(rest_imgs, atlas.maps, str(output_dir), t_r=tr)
    fingerprint_img = compute_connectivity_fingerprint(
        rest_imgs, atlas.maps, str(output_dir / "fingerprint_100.nii.gz"), t_r=tr,
    )
    if not skip_qc:
        plot_connectivity_matrix(mean_conn, str(output_dir / "mean_conn.png"), labels=atlas.get("labels"))
        index = int(rng.integers(100))
        plot_surface(
            index_img(fingerprint_img, index),
            str(output_dir / f"fingerprint_surface_100_roi{index}.png"),
            f"{subject_dir.name} | n_rois=100, seed={index}",
        )


def build_task_targets(subject_dir, events_root, tr, force=False, skip_qc=False):
    """Run the language stage that was commented out in the historical main."""
    beta_dir = subject_dir / "output" / "task_betas"
    qc_dir = beta_dir / "qc"
    if not force and (qc_dir / "story_vs_math_surface.png").is_file():
        LOGGER.info("Subject %s: task QC exists; skipping task stage.", subject_dir.name)
        return
    lang_pairs = []
    for phase_encoding in ["LR", "RL"]:
        events_dir = (
            events_root / subject_dir.name / "unprocessed" / "3T"
            / f"tfMRI_LANGUAGE_{phase_encoding}" / "LINKED_DATA" / "EPRIME" / "EVs"
        )
        imgs = sorted(glob.glob(str(subject_dir / f"*tfMRI_LANGUAGE_{phase_encoding}*_sm_clean.nii.gz")))
        if not imgs or not events_dir.is_dir():
            continue
        if len(imgs) > 1:
            LOGGER.warning("Multiple task matches for %s %s; using first sorted match, as in the source.", subject_dir.name, phase_encoding)
        lang_pairs.append((imgs[0], load_hcp_language_events(str(events_dir))))
    if not lang_pairs:
        LOGGER.info("Subject %s: no language runs with event directories found.", subject_dir.name)
        return
    betas = compute_language_betas(
        [path for path, _ in lang_pairs], [events for _, events in lang_pairs],
        str(beta_dir), t_r=tr,
    )
    if not skip_qc:
        plot_beta_maps(betas, str(qc_dir))
        if "story_vs_math" in betas:
            plot_surface(
                betas["story_vs_math"], str(qc_dir / "story_vs_math_surface.png"),
                f"{subject_dir.name} Story > Math",
            )


def parse_args(argv=None):
    """Read paths and stage selections without importing imaging dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Root containing numeric subject folders and cleaned NIfTI files.")
    parser.add_argument("--events-root", type=Path, help="Root containing original subject event directories; defaults to DATA_DIR.")
    parser.add_argument("--subjects", nargs="+", help="Numeric subject folder names; default: all numeric folders.")
    parser.add_argument("--subject-id-length", type=int, help="Optional folder-name length filter; the supplied historical main used 3.")
    parser.add_argument("--steps", nargs="+", choices=["rest", "task"], default=["rest"], help="Stages to run (default: rest). Task activation was commented out in the supplied script.")
    parser.add_argument("--tr", type=float, default=0.72, help="Repetition time in seconds (study: 0.72).")
    parser.add_argument("--atlas-cache", type=Path, help="Optional Nilearn atlas cache directory.")
    parser.add_argument("--qc-seed", type=int, default=0, help="Random seed for selecting a fingerprint QC parcel; affects figures only.")
    parser.add_argument("--skip-qc", action="store_true", help="Build arrays/images without figures; no QC completion marker will be written.")
    parser.add_argument("--force", action="store_true", help="Recompute requested stages even when their QC completion markers exist; overwrites generated outputs.")
    args = parser.parse_args(argv)
    if not args.data_dir.is_dir():
        parser.error("--data-dir must be an existing directory")
    if args.tr <= 0 or args.qc_seed < 0:
        parser.error("--tr must be positive and --qc-seed nonnegative")
    if args.subject_id_length is not None and args.subject_id_length < 1:
        parser.error("--subject-id-length must be positive")
    if args.subjects and any(not subject.isdigit() for subject in args.subjects):
        parser.error("--subjects must contain numeric folder names")
    if args.subjects and any(not (args.data_dir / subject).is_dir() for subject in args.subjects):
        parser.error("each requested subject folder must exist under --data-dir")
    args.data_dir = args.data_dir.resolve()
    args.events_root = (args.events_root or args.data_dir).resolve()
    if "task" in args.steps and not args.events_root.is_dir():
        parser.error("the events root must exist when requesting the task stage")
    if args.atlas_cache is not None:
        args.atlas_cache = str(args.atlas_cache.resolve())
    return args


def main(argv=None):
    """Generate selected outputs, retaining the source's QC-based skip logic."""
    args = parse_args(argv)
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    subjects = args.subjects or sorted(
        (entry.name for entry in args.data_dir.iterdir() if entry.is_dir() and entry.name.isdigit()),
        key=int,
    )
    if args.subject_id_length is not None:
        subjects = [subject for subject in subjects if len(subject) == args.subject_id_length]
    rng = np.random.default_rng(args.qc_seed)
    for subject in subjects:
        subject_dir = args.data_dir / subject
        if not list(subject_dir.glob("*_sm_clean.nii.gz")):
            continue
        start = time.monotonic()
        LOGGER.info("Processing subject %s", subject)
        if "rest" in args.steps:
            build_rest_features(subject_dir, args.tr, rng, args.force, args.skip_qc, args.atlas_cache)
        if "task" in args.steps:
            build_task_targets(subject_dir, args.events_root, args.tr, args.force, args.skip_qc)
        LOGGER.info("Subject %s complete in %.1f s", subject, time.monotonic() - start)
    LOGGER.info("Selected subjects complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
