from __future__ import annotations

"""Compatibility adapter for manuscript figure inference.

The legacy montage helper expects the private prediction API used by earlier
VRPTR scripts (``_load_model``, ``_prepare_matrix_for_ckpt``, and
``_predict_dataset``). The final thesis-faithful predictor exposes the same
operations under public names. This module translates between the two APIs
without changing model weights, inputs, predictions, or metrics.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import predict_clinical as _current
from .training_common import Standardizer, load_state_dict_flexible_, maybe_data_parallel

# Re-export the data/runtime helpers required by the legacy figure machinery.
ClinicalPaths = _current.ClinicalPaths
create_image_space = _current.create_image_space
load_subject_list = _current.load_subject_list
build_subject_matrix_cached = _current.build_subject_matrix_cached
make_eval_dataset = _current.make_eval_dataset
write_image_nifti = _current.write_image_nifti


def _load_model(
    ckpt_path: str | Path,
    device: torch.device,
    no_data_parallel: bool = False,
):
    """Load the final thesis-faithful checkpoint using the current predictor."""
    if hasattr(_current, "load_model_checkpoint"):
        # The final predictor already returns a single-GPU eval model. The
        # no_data_parallel argument is retained only for API compatibility.
        return _current.load_model_checkpoint(ckpt_path, device)
    if hasattr(_current, "_load_model"):
        return _current._load_model(  # type: ignore[attr-defined]
            ckpt_path, device, no_data_parallel=no_data_parallel
        )
    raise AttributeError(
        "vrptr_clinical.predict_clinical exposes neither load_model_checkpoint "
        "nor _load_model"
    )


def _prepare_matrix_for_ckpt(
    raw_matrix: dict,
    ckpt: dict,
    scaler: Standardizer,
) -> dict:
    """Apply the checkpoint's training-control standardizer."""
    if hasattr(_current, "prepare_matrix"):
        return _current.prepare_matrix(raw_matrix, scaler)
    if hasattr(_current, "_prepare_matrix_for_ckpt"):
        return _current._prepare_matrix_for_ckpt(  # type: ignore[attr-defined]
            raw_matrix, ckpt, scaler
        )
    raise AttributeError(
        "vrptr_clinical.predict_clinical exposes neither prepare_matrix nor "
        "_prepare_matrix_for_ckpt"
    )


def _predict_dataset(
    model,
    ds,
    device,
    left_idx,
    right_idx,
    group,
    model_name,
    out_dir,
    atlas=None,
    write_nifti: bool = False,
    save_latents: bool = False,
    spatial_chunk_size: int = 0,
    batch_size: int = 8,
):
    """Run deterministic ``z = mu`` inference and return the legacy tuple.

    The final predictor returns ``rows, records, subjects, mus, logvars``.
    The figure helper expects ``rows, latent_rows, records``. No numerical
    operation is changed; only the return structure is adapted.
    """
    if hasattr(_current, "predict_dataset"):
        rows, records, subjects, mus, logvars = _current.predict_dataset(
            model,
            ds,
            device,
            left_idx,
            right_idx,
            group,
            model_name,
            int(batch_size),
        )

        latent_rows: list[dict[str, Any]] = []
        if save_latents and mus:
            for subject, mu, logvar in zip(subjects, mus, logvars):
                mu_arr = np.asarray(mu).reshape(-1)
                logvar_arr = np.asarray(logvar).reshape(-1)
                row: dict[str, Any] = {
                    "subject": str(subject),
                    "group": group,
                    "model": model_name,
                }
                for k, value in enumerate(mu_arr):
                    row[f"mu_{k:04d}"] = float(value)
                for k, value in enumerate(logvar_arr):
                    row[f"logvar_{k:04d}"] = float(value)
                latent_rows.append(row)

        if write_nifti and atlas is not None:
            nifti_dir = Path(out_dir) / "nifti"
            nifti_dir.mkdir(parents=True, exist_ok=True)
            for subject, truth, pred in records:
                write_image_nifti(pred, atlas, nifti_dir / f"{subject}_pred.nii.gz")
                write_image_nifti(truth, atlas, nifti_dir / f"{subject}_true.nii.gz")

        return rows, latent_rows, records

    if hasattr(_current, "_predict_dataset"):
        return _current._predict_dataset(  # type: ignore[attr-defined]
            model,
            ds,
            device,
            left_idx,
            right_idx,
            group,
            model_name,
            out_dir,
            atlas=atlas,
            write_nifti=write_nifti,
            save_latents=save_latents,
            spatial_chunk_size=spatial_chunk_size,
            batch_size=batch_size,
        )
    raise AttributeError(
        "vrptr_clinical.predict_clinical exposes neither predict_dataset nor "
        "_predict_dataset"
    )
