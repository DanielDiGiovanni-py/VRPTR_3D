"""One training-only reference in the exact target space used by evaluation.

NPZ stores the mean, voxel contribution counts, affine and provenance together.
No evaluated task map is ever used to estimate this reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

TARGET_SPACE = "within_subject_mask_zscore_ddof0"


def add_template_argument(parser):
    parser.add_argument("--training-template", default=os.environ.get("TRAINING_TEMPLATE", ""),
                        help="Training-only template NPZ; required for template-relative evaluation.")


def validate_subjects(subjects, label):
    ids = [str(s) for s in subjects]
    if not ids or any(not s for s in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{label}: expected nonempty, unique subject IDs")
    return ids


def assert_disjoint(train, evaluated):
    overlap = set(train) & set(evaluated)
    if overlap:
        raise ValueError(f"Training/template and evaluation IDs overlap: {sorted(overlap)}")


def mean_training_maps(maps):
    """Stream finite-voxel means, preserving unsupported voxels as NaN."""
    total = count = None
    n = 0
    for value in maps:
        value = np.asarray(value, dtype=np.float32)
        if total is None:
            total = np.zeros(value.shape, dtype=np.float64)
            count = np.zeros(value.shape, dtype=np.int32)
        if value.shape != total.shape:
            raise ValueError("Training task-map shapes differ")
        valid = np.isfinite(value)
        total += np.where(valid, value, 0.0)
        count += valid
        n += 1
    if n == 0:
        raise ValueError("No training maps")
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    return mean.astype(np.float32), count, n


def save_template(path, template, counts, affine, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, template_source="training_mean", target_space=TARGET_SPACE)
    # Reproducible content identity, independent of compression/container timestamps.
    digest = hashlib.sha256()
    for array in (template, counts, affine):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(json.dumps(metadata, sort_keys=True).encode())
    metadata["template_sha256"] = digest.hexdigest()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, template=template, counts=counts, affine=affine,
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    temporary.replace(path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def read_template(path, subjects=(), shape=None, affine=None):
    if not path:
        raise ValueError("Pass --training-template or set TRAINING_TEMPLATE; no test-derived fallback is allowed")
    with np.load(path, allow_pickle=False) as saved:
        template = saved["template"].copy()
        counts = saved["counts"].copy()
        grid_affine = saved["affine"].copy()
        meta = json.loads(str(saved["metadata"].item()))
    if meta.get("template_source") != "training_mean" or meta.get("target_space") != TARGET_SPACE:
        raise ValueError("Template provenance/target scaling does not match this evaluator")
    train = validate_subjects(meta["training_subjects"], "template training subjects")
    assert_disjoint(train, [str(s) for s in subjects])
    if meta.get("n_training") != len(train) or counts.shape != template.shape:
        raise ValueError("Invalid template subject count or voxel contribution counts")
    original_hash = meta["template_sha256"]
    digest = hashlib.sha256()
    for array in (template, counts, grid_affine):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(json.dumps({k: v for k, v in meta.items() if k != "template_sha256"}, sort_keys=True).encode())
    if digest.hexdigest() != original_hash:
        raise ValueError("Template content/provenance checksum mismatch")
    if shape is not None and tuple(shape) != template.shape:
        raise ValueError(f"Template shape {template.shape} != target shape {tuple(shape)}")
    if affine is not None and not np.allclose(affine, grid_affine, rtol=0, atol=1e-4):
        raise ValueError("Template and target affines differ; implicit resampling is not allowed")
    return template, meta


def verify_target_grids(paths, subjects, template_shape, affine):
    import nibabel as nib
    for subject in subjects:
        path = Path(paths.data_dir) / str(subject) / paths.beta_pattern
        image = nib.load(path)
        if tuple(image.shape[:3]) != tuple(template_shape[-3:]) or not np.allclose(image.affine, affine, rtol=0, atol=1e-4):
            raise ValueError(f"Task-map grid differs from training reference: {subject}")


def load_for_evaluation(path, subjects, paths, shape=None):
    ids = validate_subjects(subjects, "evaluation subjects")
    template, metadata = read_template(path, ids, shape=shape)
    with np.load(path, allow_pickle=False) as saved:
        affine = saved["affine"]
    verify_target_grids(paths, ids, template.shape, affine)
    return template, metadata


def require_complete_subjects(requested, loaded):
    if set(map(str, requested)) != set(map(str, loaded)) or len(loaded) != len(requested):
        raise ValueError("Some requested participants were skipped or duplicated; fix missing inputs before evaluation")


def metric_provenance(metadata):
    return {"template_source": "training_mean", "template_sha256": metadata["template_sha256"],
            "template_n_training": metadata["n_training"], "target_space": TARGET_SPACE}


def validate_result_frames(frames):
    """Reject stale or mixed-reference tables before aggregation/plotting."""
    hashes = set()
    cohorts = []
    for frame in frames:
        needed = {"template_source", "template_sha256"}
        if not needed <= set(frame.columns) or frame.empty:
            raise ValueError("Missing training-template provenance; rerun evaluation before compiling figures/tables")
        if not frame.template_source.eq("training_mean").all() or frame.template_sha256.isna().any():
            raise ValueError("Mixed or non-training template results")
        hashes.update(frame.template_sha256.astype(str))
        if "subject" in frame:
            cohorts.append(set(frame.subject.astype(str)))
    if len(hashes) != 1 or (cohorts and any(c != cohorts[0] for c in cohorts)):
        raise ValueError("Result tables use different templates or evaluated participant sets")
    return next(iter(hashes))


def main(argv=None):
    from .data import ClinicalPaths, create_image_space, extract_subject_image, load_subject_list
    import nibabel as nib
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-subjects", required=True)
    p.add_argument("--test-subjects", required=True)
    p.add_argument("--checkpoint", default="", help="Read preprocessing paths from an existing trusted VRPTR checkpoint")
    p.add_argument("--data-dir", default="")
    p.add_argument("--fingerprint-pattern", default="")
    p.add_argument("--beta-pattern", default="")
    p.add_argument("--expected-train-count", type=int, default=360)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if Path(args.out).exists():
        raise FileExistsError(f"Template already exists: {args.out}; use a new path to rebuild")
    paths = ClinicalPaths()
    if args.checkpoint:
        import torch
        paths = ClinicalPaths(**torch.load(args.checkpoint, map_location="cpu", weights_only=False)["paths"])
    for name in ("data_dir", "fingerprint_pattern", "beta_pattern"):
        if getattr(args, name):
            setattr(paths, name, getattr(args, name))
    train = validate_subjects(load_subject_list(args.train_subjects), "training list")
    test = validate_subjects(load_subject_list(args.test_subjects), "test list")
    assert_disjoint(train, test)
    if args.expected_train_count > 0 and len(train) != args.expected_train_count:
        raise ValueError(f"Expected {args.expected_train_count} training IDs, found {len(train)}")
    atlas = create_image_space(paths)
    source_files = []
    reference = nib.load(Path(paths.data_dir) / train[0] / paths.beta_pattern)
    verify_target_grids(paths, train + test, (1, *reference.shape[:3]), reference.affine)

    def targets():
        for i, subject in enumerate(train):
            item = extract_subject_image(subject, paths, atlas)
            if item is None:
                raise FileNotFoundError(f"Missing training data: {subject}; refusing a partial template")
            for pattern in (paths.beta_pattern, paths.fingerprint_pattern):
                source = Path(paths.data_dir) / subject / pattern
                stat = source.stat()
                source_files.append({"subject": subject, "path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            print(f"[training-template] {i+1}/{len(train)} {subject}", flush=True)
            yield item["y"]

    template, counts, n = mean_training_maps(targets())
    metadata = save_template(args.out, template, counts, reference.affine, {
        "training_subjects": train, "n_training": n, "paths": asdict(paths),
        "train_list_sha256": hashlib.sha256(Path(args.train_subjects).read_bytes()).hexdigest(),
        "mask_definition": "data._image_mask; mean over finite training targets; no restandardization of the mean",
        "source_files": source_files,
    })
    nifti_path = Path(args.out).with_suffix(".nii.gz")
    nib.save(nib.Nifti1Image(template[0], reference.affine), nifti_path)
    print(f"[training-template] saved {args.out}; n={n}; SHA256={metadata['template_sha256']}")


if __name__ == "__main__":
    main()
