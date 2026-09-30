"""Checkpoint-backed inference helpers used by the manuscript figures.

The module name and runtime APIs are retained for checkpoint and figure-loader
compatibility. Figure generation is provided by ``make_vrptr_current_paper_figures``.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import re
import sys
import tempfile
from pathlib import Path

import numpy as np


class RuntimeModelSpec:
    """Small container for a model checkpoint and optional leave-one-out adapter root."""

    def __init__(self, label: str, checkpoint: Path, adapter_root: Path | None = None):
        self.label = str(label)
        self.checkpoint = Path(checkpoint)
        self.adapter_root = Path(adapter_root) if adapter_root else None

    def __repr__(self) -> str:
        if self.adapter_root:
            return f"RuntimeModelSpec(label={self.label!r}, checkpoint={str(self.checkpoint)!r}, adapter_root={str(self.adapter_root)!r})"
        return f"RuntimeModelSpec(label={self.label!r}, checkpoint={str(self.checkpoint)!r})"


def import_prediction_module(module_name: str | None = None, script_path: Path | None = None):
    """Import the current prediction module without requiring persistent prediction files.

    The prediction script uses package-relative imports, so the safest route is a
    package import. If the figure script is run directly from inside the package,
    we add the package parent to sys.path before trying common module names.
    """
    here = Path(__file__).resolve().parent
    parent = here.parent
    for candidate in [str(parent), str(here)]:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    names = []
    if module_name:
        names.append(module_name)
    names.extend([
        # Current script name in this codebase.
        "vrptr_clinical.predict_clinical",
        "clinical_vrptr.predict_clinical",
        "predict_clinical",
        # Older/fallback names from previous iterations.
        "vrptr_clinical.predict",
        "vrptr_clinical.clinical_vrptr_predict",
        "clinical_vrptr.predict",
        "clinical_vrptr_predict",
        "predict",
    ])
    last_exc = None
    for name in names:
        try:
            mod = importlib.import_module(name)
            required = ["_load_model", "_prepare_matrix_for_ckpt", "_predict_dataset", "ClinicalPaths", "create_image_space", "load_subject_list", "build_subject_matrix_cached", "make_eval_dataset", "write_image_nifti"]
            missing = [x for x in required if not hasattr(mod, x)]
            if missing:
                last_exc = ImportError(f"{name} imported but is missing {missing}")
                continue
            print(f"[montage] Imported prediction module: {name}")
            return mod
        except Exception as exc:
            last_exc = exc

    if script_path:
        # This fallback works only if the script does not depend on package-relative imports.
        spec = importlib.util.spec_from_file_location("_vrptr_prediction_runtime", str(script_path))
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                print(f"[montage] Imported prediction script from path: {script_path}")
                return mod
            except Exception as exc:
                last_exc = exc

    raise ImportError(
        "Could not import the prediction module. Pass --prediction-module with the dotted module name "
        "or run from the package root with the package importable. Last error: " + repr(last_exc)
    )


def import_comparison_module(module_name: str | None = None):
    """Import comparison_model.py for BrainSurfCNN-lite / SWIFUN-lite runtime rendering.

    The comparator checkpoints do not store VRPTR-style ``model_config`` or
    ``x_scaler`` metadata.  Their evaluation script rebuilds the architecture from
    the saved ``args`` and recomputes the training-control standardizer.  Figure 5
    should therefore reuse comparison_model.py rather than trying to force these
    checkpoints through predict_clinical.py.
    """
    here = Path(__file__).resolve().parent
    parent = here.parent
    for candidate in [str(parent), str(here)]:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    names = []
    if module_name:
        names.append(module_name)
    names.extend([
        "vrptr_clinical.comparison_models",
        "clinical_vrptr.comparison_models",
        "comparison_models",
    ])
    last_exc = None
    for name in names:
        try:
            mod = importlib.import_module(name)
            required = [
                "_load_comparison_checkpoint", "_apply_saved_architecture_args",
                "_model_dims_from_matrix", "build_comparison_model", "_eval_model",
                "ClinicalPaths", "create_image_space", "load_subject_list",
                "build_subject_matrix_cached", "ImageSubjectDataset", "Standardizer",
                "split_indices", "load_state_dict_flexible_", "maybe_data_parallel",
            ]
            missing = [x for x in required if not hasattr(mod, x)]
            if missing:
                last_exc = ImportError(f"{name} imported but is missing {missing}")
                continue
            print(f"[montage] Imported comparison module: {name}")
            return mod
        except Exception as exc:
            last_exc = exc
    raise ImportError(
        "Could not import comparison_models.py. Run from the package root or pass --comparison-module. "
        "Last error: " + repr(last_exc)
    )


def _comparison_name_from_label_or_checkpoint(label: str, checkpoint: Path) -> str:
    s = f"{label} {checkpoint}".lower()
    if "brainsurf" in s or "brain_surf" in s:
        return "brainsurfcnn_lite"
    if "swifun" in s or "swi_fun" in s or "swin" in s:
        return "swifun_lite"
    # Fall back to sanitized label; comparison_model will raise if it is unknown.
    return re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")


def _namespace_from_saved_args(saved_args: dict | None, root: Path, checkpoint: Path, device_str: str, cache_dir: str, refresh_cache: bool, batch_size: int, no_data_parallel: bool):
    """Build an argparse-like object for comparison_model helpers.

    comparison_model.py stores the training/evaluation CLI arguments inside each
    comparator checkpoint.  We recover those values, while overriding only runtime
    settings that should follow the current figure-generation call.
    """
    d = dict(saved_args or {})
    # Defaults mirror comparison_model.parse_args sufficiently for eval-only figure rendering.
    d.setdefault("data_dir", "data")
    d.setdefault("train_controls", "splits/train_subjects.txt")
    d.setdefault("test_controls", "splits/test_subjects.txt")
    d.setdefault("test_tumors", "splits/test_tumor.txt")
    d.setdefault("out_dir", str(root / "comparisons"))
    d.setdefault("fingerprint_pattern", "output/rest_connectivity_100/fingerprint_100.nii.gz")
    d.setdefault("beta_pattern", "output/task_betas/beta_story_vs_math.nii.gz")
    d.setdefault("lesion_pattern", "")
    d.setdefault("n_rois", 100)
    d.setdefault("yeo_networks", 17)
    d.setdefault("resolution_mm", 2)
    d.setdefault("representation", "voxel")
    d.setdefault("token_subdivisions", 1)
    d.setdefault("spatial_token_batch_size", 0)
    d.setdefault("full_field_stride", 2)
    d.setdefault("full_res_refine_layers", 0)
    d.setdefault("brainsurf_full_field_stride", 2)
    d.setdefault("swifun_full_field_stride", 2)
    d.setdefault("beta_clip", 8.0)
    d.setdefault("head_init_std", 1e-4)
    d.setdefault("dropout", 0.15)
    d.setdefault("surf_width", 32)
    d.setdefault("surf_depth", 4)
    d.setdefault("swifun_dim", 32)
    d.setdefault("swifun_heads", 4)
    d.setdefault("swifun_depth", 4)
    d.setdefault("swifun_window", 4)
    d.setdefault("val_frac", 0.15)
    d.setdefault("no_val_split", False)
    d.setdefault("seed", 42)
    d.setdefault("ablation_seed", 42)
    d.setdefault("input_ablation", "none")
    d.setdefault("no_ablation_name_suffix", True)
    d.setdefault("num_workers", 0)
    d.setdefault("prefetch_factor", 2)
    d["device"] = device_str
    d["batch_size"] = int(batch_size)
    d["no_data_parallel"] = bool(no_data_parallel)
    d["refresh_cache"] = bool(refresh_cache)
    if cache_dir:
        d["cache_dir"] = cache_dir
    else:
        d.setdefault("cache_dir", str(root / "image_cache"))
    return argparse.Namespace(**d)


def _fit_comparison_standardizer(compmod, raw_train: dict, args) -> tuple[dict, object, np.ndarray]:
    """Reproduce comparison_model.py's scaler fitting for held-out evaluation."""
    if bool(getattr(args, "no_val_split", False)) or float(getattr(args, "val_frac", 0.15)) <= 0.0:
        train_idx = np.arange(len(raw_train["subjects"]), dtype=int)
    else:
        train_idx, _ = compmod.split_indices(len(raw_train["subjects"]), float(args.val_frac), int(args.seed))
    is_disk = bool(raw_train.get("disk_backed", False)) or hasattr(raw_train.get("x", None), "filename")
    if is_disk and hasattr(compmod.Standardizer, "fit_indices_masked"):
        scaler = compmod.Standardizer(exclude_last_n=0).fit_indices_masked(raw_train["x"], raw_train["y"], train_idx)
        train_mat = dict(raw_train)
        train_mat["x_scaler"] = scaler
    elif is_disk and hasattr(compmod.Standardizer, "fit_indices"):
        scaler = compmod.Standardizer(exclude_last_n=0).fit_indices(raw_train["x"], train_idx)
        train_mat = dict(raw_train)
        train_mat["x_scaler"] = scaler
    else:
        scaler = compmod.Standardizer(exclude_last_n=0).fit(raw_train["x"][train_idx])
        train_mat = dict(raw_train)
        train_mat["x"] = scaler.transform(raw_train["x"])
    return train_mat, scaler, train_idx


def _apply_comparison_scaler(compmod, raw_matrix: dict, scaler) -> dict:
    is_disk = bool(raw_matrix.get("disk_backed", False)) or hasattr(raw_matrix.get("x", None), "filename")
    mat = dict(raw_matrix)
    if is_disk:
        mat["x_scaler"] = scaler
    else:
        mat["x"] = scaler.transform(raw_matrix["x"])
    return mat


def runtime_predict_comparison_for_spec(predmod, compmod, spec: RuntimeModelSpec, subjects_file: Path, group: str, cache_dir: str, refresh_cache: bool, device_str: str, batch_size: int, no_data_parallel: bool, root: Path, data_dir: str | None = None):
    """Run BrainSurfCNN-lite/SWIFUN-lite predictions using comparison_model.py logic."""
    import torch

    device = torch.device(device_str)
    ckpt_path = Path(spec.checkpoint)
    ckpt = compmod._load_comparison_checkpoint(ckpt_path, device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Comparator checkpoint must be a dict with model_state_dict and args: {ckpt_path}")
    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    args = _namespace_from_saved_args(saved_args, root, ckpt_path, device_str, cache_dir, refresh_cache, batch_size, no_data_parallel)
    if data_dir is not None:
        args.data_dir = data_dir
    args = compmod._apply_saved_architecture_args(args, saved_args)
    model_name_internal = str(ckpt.get("model_name") or _comparison_name_from_label_or_checkpoint(spec.label, ckpt_path))
    model_name_internal = _comparison_name_from_label_or_checkpoint(model_name_internal, ckpt_path)

    paths = compmod.ClinicalPaths(
        data_dir=args.data_dir,
        fingerprint_pattern=args.fingerprint_pattern,
        beta_pattern=args.beta_pattern,
        lesion_pattern=getattr(args, "lesion_pattern", ""),
        n_rois=int(args.n_rois),
        yeo_networks=int(args.yeo_networks),
        resolution_mm=int(args.resolution_mm),
        representation=args.representation,
        token_subdivisions=int(getattr(args, "token_subdivisions", 1)),
    )
    atlas = compmod.create_image_space(paths)
    cache_root = cache_dir or getattr(args, "cache_dir", "") or str(root / "image_cache")

    # Reproduce comparator eval standardization: fit on training controls, then apply to selected subjects.
    train_subjects = compmod.load_subject_list(str(getattr(args, "train_controls")))
    raw_train = compmod.build_subject_matrix_cached(
        train_subjects, paths, atlas,
        cache_dir=cache_root, cache_tag="train_controls", refresh_cache=bool(refresh_cache),
    )
    train_mat, scaler, _ = _fit_comparison_standardizer(compmod, raw_train, args)

    subjects = compmod.load_subject_list(str(subjects_file))
    cache_tag = f"figures_{group.lower()}_{Path(subjects_file).stem}"
    raw_eval = compmod.build_subject_matrix_cached(
        subjects, paths, atlas,
        cache_dir=cache_root, cache_tag=cache_tag, refresh_cache=bool(refresh_cache),
    )
    eval_mat = _apply_comparison_scaler(compmod, raw_eval, scaler)

    model_tokens, input_dim = compmod._model_dims_from_matrix(train_mat, args)
    model = compmod.build_comparison_model(model_name_internal, model_tokens, input_dim, args).to(device)
    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else ckpt
    if state is None:
        raise RuntimeError(f"Comparator checkpoint did not contain model_state_dict: {ckpt_path}")
    compmod.load_state_dict_flexible_(model, state, strict=False)
    model = compmod.maybe_data_parallel(model, device, disabled=bool(no_data_parallel), context=f"figure_montage_{spec.label}")
    model.eval()

    ds = compmod.ImageSubjectDataset(eval_mat)
    rows, _summary, pred_records = compmod._eval_model(
        model, ds, eval_mat["left_idx"], eval_mat["right_idx"], device,
        group.lower() if str(group).lower().startswith("control") else "tumor",
        spec.label, int(batch_size), spatial_chunk_size=0,
    )
    print(f"[montage] Rendered {spec.label} through comparison_model.py as {model_name_internal} ({len(rows)} subjects).")
    return rows, pred_records, atlas, raw_eval


def _checkpoint_key_summary(ckpt) -> str:
    """Return a compact checkpoint-key summary for diagnostics."""
    try:
        if isinstance(ckpt, dict):
            keys = list(ckpt.keys())
            shown = keys[:40]
            more = "..." if len(keys) > len(shown) else ""
            return f"dict keys={shown}{more}"
        return f"object type={type(ckpt)}"
    except Exception:
        return f"object type={type(ckpt)}"


def _strip_module_prefix_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        kk = str(k)
        if kk.startswith("module."):
            kk = kk[len("module."):]
        out[kk] = v
    return out


def _extract_state_dict_or_model(ckpt):
    """Return (model_object_or_none, state_dict_or_none) from common checkpoint layouts."""
    import torch
    if isinstance(ckpt, torch.nn.Module):
        return ckpt, None
    if isinstance(ckpt, dict):
        for key in ["model", "net", "module", "network"]:
            obj = ckpt.get(key)
            if isinstance(obj, torch.nn.Module):
                return obj, None
        for key in ["model_state_dict", "state_dict", "model", "net", "network", "weights"]:
            obj = ckpt.get(key)
            if isinstance(obj, dict) and obj and all(hasattr(v, "shape") for v in obj.values()):
                return None, _strip_module_prefix_state_dict(obj)
        if ckpt and all(hasattr(v, "shape") for v in ckpt.values()):
            return None, _strip_module_prefix_state_dict(ckpt)
    return None, None


def _infer_encoder_decoder_config_from_state_dict(sd: dict) -> dict | None:
    """Infer the shared Encoder3D/Decoder3D config used by BrainSurfCNN-lite-like checkpoints."""
    stem_key = None
    for k in sd.keys():
        if k.endswith("encoder.stem.0.weight") or k == "encoder.stem.0.weight":
            stem_key = k
            break
    if stem_key is None:
        return None
    w = sd[stem_key]
    if not hasattr(w, "shape") or len(w.shape) < 2:
        return None
    width = int(w.shape[0])
    input_dim = int(w.shape[1])
    levels = set()
    for k in sd.keys():
        m = re.search(r"encoder\.blocks\.(\d+)\.", str(k))
        if m:
            levels.add(int(m.group(1)))
    return {"input_dim": input_dim, "width": width, "levels": max(levels) + 1 if levels else 4}


def _build_encoder_decoder_runtime_model(predmod, sd: dict, label: str):
    """Build a simple encoder-decoder model when a comparator checkpoint lacks model_config.

    This is intended for BrainSurfCNN-lite-like checkpoints that reuse the same
    Encoder3D/Decoder3D classes as VRPTR. It will refuse unknown state-dict
    layouts rather than silently drawing an invalid qualitative panel.
    """
    import importlib
    import torch
    import torch.nn as nn

    cfg = _infer_encoder_decoder_config_from_state_dict(sd)
    if cfg is None:
        return None

    package = getattr(predmod, "__package__", "") or "vrptr_clinical"
    model_mod = None
    for name in [f"{package}.model", "vrptr_clinical.model", "clinical_vrptr.model", "model"]:
        try:
            model_mod = importlib.import_module(name)
            break
        except Exception:
            continue
    if model_mod is None or not hasattr(model_mod, "Encoder3D") or not hasattr(model_mod, "Decoder3D"):
        raise ImportError("Could not import Encoder3D/Decoder3D needed for comparator checkpoint rendering.")

    Encoder3D = model_mod.Encoder3D
    Decoder3D = model_mod.Decoder3D

    class EncoderDecoderRuntime(nn.Module):
        def __init__(self, input_dim: int, width: int, levels: int):
            super().__init__()
            self.encoder = Encoder3D(int(input_dim), width=int(width), levels=int(levels), dropout=0.0)
            self.decoder = Decoder3D(self.encoder.out_channels, width=int(width), levels=int(levels), dropout=0.0)
            self.head = nn.Conv3d(self.decoder.out_channels, 1, 1)
            self.requires_full_field = True

        def forward(self, x, **kwargs):
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
            h, skips = self.encoder(x)
            dec = self.decoder(h, skips)
            beta = torch.nan_to_num(self.head(dec), nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)
            return {"beta": beta, "logvar": torch.zeros_like(beta)}

    model = EncoderDecoderRuntime(**cfg)
    try:
        predmod.load_state_dict_flexible_(model, sd, strict=True)
    except Exception:
        # Fallback to native loading for checkpoints already matching exactly.
        model.load_state_dict(sd, strict=True)
    print(f"[montage] Loaded {label} as inferred shared Encoder3D/Decoder3D model: {cfg}")
    return model


def load_runtime_model_for_spec(predmod, spec: RuntimeModelSpec, device, no_data_parallel: bool = False):
    """Load VRPTR checkpoints via predict_clinical, and comparator checkpoints via flexible loaders.

    predict_clinical._load_model is intentionally VRPTR-specific and requires
    ckpt['model_config']. Comparator checkpoints often lack that key, so the
    montage runtime needs a separate path for BrainSurfCNN/SWIFUN.
    """
    import torch

    ckpt_path = Path(spec.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_config" in ckpt:
        return predmod._load_model(ckpt_path, device, no_data_parallel=bool(no_data_parallel))

    model_obj, sd = _extract_state_dict_or_model(ckpt)
    if model_obj is not None:
        model = model_obj
        print(f"[montage] Loaded {spec.label} from serialized model object in {ckpt_path}")
    elif sd is not None:
        model = _build_encoder_decoder_runtime_model(predmod, sd, spec.label)
        if model is None:
            first_keys = list(sd.keys())[:30]
            raise ValueError(
                f"Cannot render {spec.label}: checkpoint {ckpt_path} lacks 'model_config' and its state_dict "
                "does not match the shared Encoder3D/Decoder3D layout. "
                f"Checkpoint summary: {_checkpoint_key_summary(ckpt)}. First state keys: {first_keys}.\n"
                "This means the figure script needs the comparator prediction/model-loading code for this architecture. "
                "Share the BrainSurfCNN/SWIFUN prediction script or checkpoint-inspection output and I can add the exact loader."
            )
    else:
        raise ValueError(
            f"Cannot render {spec.label}: unsupported checkpoint layout at {ckpt_path}. "
            f"Checkpoint summary: {_checkpoint_key_summary(ckpt)}"
        )

    if not isinstance(ckpt, dict):
        raise ValueError(f"Cannot render {spec.label}: serialized model object checkpoint lacks paths/x_scaler metadata needed to build the data matrix.")
    if "paths" not in ckpt or "x_scaler" not in ckpt:
        raise ValueError(
            f"Cannot render {spec.label}: checkpoint {ckpt_path} loaded, but it lacks paths and/or x_scaler metadata. "
            f"Checkpoint summary: {_checkpoint_key_summary(ckpt)}"
        )

    model.to(device)
    model = predmod.maybe_data_parallel(model, device, disabled=bool(no_data_parallel), context=f"figure_montage_{spec.label}")
    model.eval()
    scaler = predmod.Standardizer.from_state_dict(ckpt["x_scaler"])
    return model, scaler, ckpt


def runtime_predict_for_spec(predmod, spec: RuntimeModelSpec, subjects_file: Path, group: str, cache_dir: str, refresh_cache: bool, device_str: str, batch_size: int, no_data_parallel: bool, comparison_module_name: str | None = None, root: Path | None = None, data_dir: str | None = None):
    """Run the appropriate prediction runtime for a Figure 5 model spec.

    VRPTR checkpoints contain ``model_config`` and are evaluated through
    predict_clinical.py. Comparator checkpoints are evaluated through
    comparison_model.py because their checkpoints store architecture/runtime args
    but not VRPTR-style ``model_config`` or ``x_scaler`` metadata.
    """
    import torch

    ckpt_path = Path(spec.checkpoint)
    try:
        ckpt_probe = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt_probe = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt_probe, dict) and "model_config" not in ckpt_probe:
        compmod = import_comparison_module(comparison_module_name)
        return runtime_predict_comparison_for_spec(
            predmod, compmod, spec, subjects_file, group, cache_dir,
            refresh_cache, device_str, batch_size, no_data_parallel,
            root or Path(spec.checkpoint).resolve().parents[1], data_dir=data_dir,
        )

    # VRPTR path: no persistent prediction maps are written. The only output
    # directory passed to _predict_dataset is a TemporaryDirectory.
    device = torch.device(device_str)
    base_model, base_scaler, base_ckpt = load_runtime_model_for_spec(predmod, spec, device, no_data_parallel=bool(no_data_parallel))
    paths = predmod.ClinicalPaths(**base_ckpt["paths"])
    if data_dir is not None:
        paths.data_dir = data_dir
    atlas = predmod.create_image_space(paths)
    subjects = predmod.load_subject_list(str(subjects_file))
    cache_root = cache_dir or str(Path(spec.checkpoint).parent / "image_cache")
    cache_tag = f"figures_{group.lower()}_{Path(subjects_file).stem}"
    raw_matrix = predmod.build_subject_matrix_cached(subjects, paths, atlas, cache_dir=cache_root, cache_tag=cache_tag, refresh_cache=bool(refresh_cache))

    rows: list[dict] = []
    pred_records: list[tuple[str, np.ndarray, np.ndarray]] = []
    latent_rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="vrptr_fig_runtime_") as td:
        tmp_out = Path(td)
        if spec.adapter_root:
            for i, subj in enumerate(raw_matrix["subjects"]):
                ckpt_path_i = Path(spec.adapter_root) / f"tumor_{subj}.pt"
                if ckpt_path_i.exists():
                    model, scaler, ckpt = load_runtime_model_for_spec(predmod, RuntimeModelSpec(spec.label, ckpt_path_i), device, no_data_parallel=bool(no_data_parallel))
                else:
                    print(f"[montage] missing adapter for {subj} under {spec.adapter_root}; using base checkpoint for {spec.label}")
                    model, scaler, ckpt = base_model, base_scaler, base_ckpt
                matrix_all = predmod._prepare_matrix_for_ckpt(raw_matrix, ckpt, scaler)
                ds = predmod.make_eval_dataset(matrix_all, [i], mode="none", seed=0)
                rr, lr, pr = predmod._predict_dataset(
                    model, ds, device,
                    matrix_all["left_idx"], matrix_all["right_idx"],
                    group.lower() if group.lower().startswith("control") else "tumor",
                    spec.label, tmp_out, atlas=atlas,
                    write_nifti=False, save_latents=False, spatial_chunk_size=0, batch_size=batch_size,
                )
                rows.extend(rr); latent_rows.extend(lr); pred_records.extend(pr)
        else:
            matrix = predmod._prepare_matrix_for_ckpt(raw_matrix, base_ckpt, base_scaler)
            ds = predmod.make_eval_dataset(matrix, mode="none", seed=0)
            rr, lr, pr = predmod._predict_dataset(
                base_model, ds, device,
                matrix["left_idx"], matrix["right_idx"],
                group.lower() if group.lower().startswith("control") else "tumor",
                spec.label, tmp_out, atlas=atlas,
                write_nifti=False, save_latents=False, spatial_chunk_size=0, batch_size=batch_size,
            )
            rows.extend(rr); latent_rows.extend(lr); pred_records.extend(pr)
    return rows, pred_records, atlas, raw_matrix
