#!/usr/bin/env python3
"""Run the study's historical FSL/Nipype preprocessing workflow.

The computational settings are retained from the supplied study script. This is
an FSL/Nilearn workflow, not an implementation or invocation of fMRIPrep.
"""

import argparse
import logging
import os
from pathlib import Path
import shutil
import tempfile


RUNS = (
    "rfMRI_REST1_LR", "rfMRI_REST1_RL",
    "rfMRI_REST2_LR", "rfMRI_REST2_RL",
    "tfMRI_LANGUAGE_LR", "tfMRI_LANGUAGE_RL",
)
LOGGER = logging.getLogger(__name__)


def smooth_func(in_file):
    """Apply the original 6-mm Gaussian smoothing in a Nipype worker."""
    from nilearn.image import smooth_img

    out = in_file.replace(".nii.gz", "_sm.nii.gz")
    smooth_img(in_file, fwhm=6).to_filename(out)
    return out


def clean_func(in_file, t_r):
    """Detrend, standardize, and high-pass filter in a Nipype worker."""
    from nilearn.image import clean_img

    out = in_file.replace(".nii.gz", "_clean.nii.gz")
    clean_img(
        in_file, detrend=True, standardize=True,
        low_pass=None, high_pass=0.01, t_r=t_r,
    ).to_filename(out)
    return out


def build_workflow(subject, runs, data_dir, output_dir, work_dir, template, tr):
    """Construct the original processing graph for one subject."""
    from nipype import Node, Workflow
    from nipype.interfaces.fsl import BET, ConvertXFM, FLIRT, MCFLIRT, MeanImage, SliceTimer
    from nipype.interfaces.io import DataSink, SelectFiles
    from nipype.interfaces.utility import Function, IdentityInterface

    workflow = Workflow(name=f"preproc_{subject}", base_dir=str(work_dir))
    infosource = Node(IdentityInterface(fields=["subject_id", "run"]), name="infosource")
    infosource.iterables = [("subject_id", [subject]), ("run", runs)]
    templates = {
        "t1": "{subject_id}/unprocessed/3T/T1w_MPR1/{subject_id}_3T_T1w_MPR1.nii.gz",
        "bold": "{subject_id}/unprocessed/3T/{run}/{subject_id}_3T_{run}.nii.gz",
    }
    selectfiles = Node(
        SelectFiles(templates, base_directory=str(data_dir), allow_missing=True),
        name="selectfiles",
    )
    bet = Node(BET(frac=0.3, mask=True), name="bet")
    t1_to_mni = Node(
        FLIRT(reference=str(template), cost="normmi", dof=12, interp="trilinear"),
        name="t1_to_mni",
    )
    slicetimer = Node(SliceTimer(time_repetition=tr, slice_direction=3), name="slicetimer")
    mcflirt = Node(MCFLIRT(save_plots=True), name="mcflirt")
    mean_epi = Node(MeanImage(dimension="T"), name="mean_epi")
    epi_to_t1 = Node(
        FLIRT(cost="normmi", dof=6, bins=640, interp="trilinear"), name="epi_to_t1",
    )
    merge_xfm = Node(ConvertXFM(concat_xfm=True), name="merge_xfm")
    apply_xfm = Node(FLIRT(reference=str(template), apply_xfm=True), name="apply_xfm")
    smooth = Node(
        Function(input_names=["in_file"], output_names=["out_file"], function=smooth_func),
        name="smooth",
    )
    clean = Node(
        Function(input_names=["in_file", "t_r"], output_names=["out_file"], function=clean_func),
        name="clean",
    )
    clean.inputs.t_r = tr
    datasink = Node(DataSink(base_directory=str(output_dir)), name="datasink")
    datasink.inputs.parameterization = False
    workflow.connect([
        (infosource, selectfiles, [("subject_id", "subject_id"), ("run", "run")]),
        (selectfiles, bet, [("t1", "in_file")]),
        (bet, t1_to_mni, [("out_file", "in_file")]),
        (selectfiles, slicetimer, [("bold", "in_file")]),
        (slicetimer, mcflirt, [("slice_time_corrected_file", "in_file")]),
        (mcflirt, mean_epi, [("out_file", "in_file")]),
        (mean_epi, epi_to_t1, [("out_file", "in_file")]),
        (bet, epi_to_t1, [("out_file", "reference")]),
        (epi_to_t1, merge_xfm, [("out_matrix_file", "in_file")]),
        (t1_to_mni, merge_xfm, [("out_matrix_file", "in_file2")]),
        (mcflirt, apply_xfm, [("out_file", "in_file")]),
        (merge_xfm, apply_xfm, [("out_file", "in_matrix_file")]),
        (apply_xfm, smooth, [("out_file", "in_file")]),
        (smooth, clean, [("out_file", "in_file")]),
        (infosource, datasink, [("subject_id", "container")]),
        (clean, datasink, [("out_file", "@cleaned")]),
    ])
    return workflow


def parse_args(argv=None):
    """Read paths and execution options without importing imaging dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Root containing numeric subject folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for cleaned subject images and log.")
    parser.add_argument("--work-dir", type=Path, help="Work root; defaults to OUTPUT_DIR/work.")
    parser.add_argument("--template", type=Path, help="Brain-extracted 2-mm MNI template; defaults to FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz.")
    parser.add_argument("--subjects", nargs="+", help="Numeric subject folder names; default: all numeric folders.")
    parser.add_argument("--runs", nargs="+", choices=RUNS, default=list(RUNS), help="Runs to process when present.")
    parser.add_argument("--tr", type=float, default=0.72, help="Repetition time in seconds (study: 0.72).")
    parser.add_argument("--n-procs", type=int, default=8, help="Nipype worker count (original default: 8).")
    parser.add_argument("--cleanup-work", action="store_true", help="Remove only each newly created work directory after successful execution. Failures retain their files.")
    parser.add_argument("--dry-run", action="store_true", help="List matched runs and completion status without writing or running tools.")
    args = parser.parse_args(argv)
    if not args.data_dir.is_dir():
        parser.error("--data-dir must be an existing directory")
    if args.tr <= 0 or args.n_procs < 1:
        parser.error("--tr and --n-procs must be positive")
    if args.subjects and any(not subject.isdigit() for subject in args.subjects):
        parser.error("--subjects must contain numeric folder names")
    if args.subjects and any(not (args.data_dir / subject).is_dir() for subject in args.subjects):
        parser.error("each requested subject folder must exist under --data-dir")
    if args.template is None:
        fsldir = os.environ.get("FSLDIR")
        if fsldir:
            args.template = Path(fsldir) / "data/standard/MNI152_T1_2mm_brain.nii.gz"
        elif not args.dry_run:
            parser.error("supply --template or configure FSLDIR")
    if args.template is not None and not args.template.is_file() and not args.dry_run:
        parser.error("the selected template does not exist")
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.work_dir = (args.work_dir or args.output_dir / "work").resolve()
    if args.template is not None:
        args.template = args.template.resolve()
    return args


def main(argv=None):
    """Process selected subjects, preserving failed and pre-existing work."""
    args = parse_args(argv)
    handlers = [logging.StreamHandler()]
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.output_dir / "preprocess.log"))
        from nipype import config

        config.update_config({
            "logging": {
                "workflow_level": "DEBUG", "interface_level": "DEBUG",
                "log_to_file": True, "log_to_stdout": False,
            },
            "execution": {"remove_unnecessary_outputs": True},
        })
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    subjects = args.subjects or sorted(
        (entry.name for entry in args.data_dir.iterdir() if entry.is_dir() and entry.name.isdigit()),
        key=int,
    )
    for subject in subjects:
        available_runs = [
            run for run in args.runs
            if (args.data_dir / subject / "unprocessed" / "3T" / run / f"{subject}_3T_{run}.nii.gz").is_file()
        ]
        if not available_runs:
            LOGGER.info("Subject %s: no matching runs; skipping.", subject)
            continue
        expected = [
            args.output_dir / subject / f"{subject}_3T_{run}_st_mcf_flirt_sm_clean.nii.gz"
            for run in available_runs
        ]
        if all(path.is_file() for path in expected):
            LOGGER.info("Subject %s: all expected outputs exist; skipping.", subject)
            continue
        LOGGER.info("Subject %s: selected runs %s", subject, available_runs)
        if args.dry_run:
            continue
        args.work_dir.mkdir(parents=True, exist_ok=True)
        # Unique directories prevent deletion or reuse of unrelated existing work.
        work_dir = Path(tempfile.mkdtemp(prefix=f"preproc_{subject}_", dir=args.work_dir))
        LOGGER.info("Work directory: %s", work_dir)
        try:
            workflow = build_workflow(
                subject, available_runs, args.data_dir, args.output_dir,
                work_dir, args.template, args.tr,
            )
            workflow.run("MultiProc", plugin_args={"n_procs": args.n_procs})
        except Exception:
            LOGGER.exception("Processing failed; retained work directory: %s", work_dir)
            raise
        if args.cleanup_work:
            LOGGER.info("Removing successful task-owned work directory: %s", work_dir)
            shutil.rmtree(work_dir)
    LOGGER.info("Selected subjects complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
