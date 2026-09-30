from __future__ import annotations

import json
import os
import hashlib
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from torch.utils.data import Dataset

from .metrics import dominance_class, robust_li


@dataclass
class ClinicalPaths:
    data_dir: str = "data"
    fingerprint_pattern: str = "output/rest_connectivity_100/fingerprint_100.nii.gz"
    beta_pattern: str = "output/task_betas/beta_story_vs_math.nii.gz"
    lesion_pattern: str = ""  # retained only for old CLI compatibility; ignored in v2.
    n_rois: int = 100         # expected fingerprint channels
    yeo_networks: int = 17    # compatibility only
    resolution_mm: int = 2    # compatibility only
    representation: str = "voxel"
    token_subdivisions: int = 1  # compatibility only


@dataclass
class ImageSpace:
    """Image-space metadata for the current v2 architecture.

    The historical name is kept so existing imports/checkpoint payloads do not
    need to change, but this is no longer a Schaefer/parcellation object.
    """
    parcel_img: Optional[nib.Nifti1Image] = None
    parcel_data: Optional[np.ndarray] = None
    parcel_ids: Optional[np.ndarray] = None
    coords_norm: Optional[np.ndarray] = None
    hemi_signed: Optional[np.ndarray] = None
    left_idx: Optional[np.ndarray] = None
    right_idx: Optional[np.ndarray] = None
    fc_dim: int = 100
    source_parcel_ids: Optional[np.ndarray] = None
    network_ids: Optional[np.ndarray] = None
    language_prior: Optional[np.ndarray] = None
    representation: str = "voxel"
    token_ijk: Optional[np.ndarray] = None
    token_mask: Optional[np.ndarray] = None
    crop_slices: Optional[tuple] = None
    crop_shape: Optional[tuple] = None
    mask_flat_idx: Optional[np.ndarray] = None


def load_subject_list(path: str | Path) -> List[str]:
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def _env_rank() -> int:
    """Return torchrun rank from env without importing torch.distributed.

    Cache building happens before all modules necessarily enter a common helper,
    so use the environment directly. Rank 0 is the only process allowed to build
    or refresh sharded image caches; other ranks wait and then open the finished
    manifest. This avoids duplicate extraction and cache corruption under DDP.
    """
    try:
        return int(os.environ.get("RANK", "0"))
    except Exception:
        return 0


def _env_world_size() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        return 1


def create_image_space(paths: ClinicalPaths) -> ImageSpace:
    """Return an empty image-space metadata object.

    v2 assumes all images are already preprocessed into the same normal-space
    coordinate system. No atlas fetching, Schaefer resampling, cropping, or parcel
    averaging is performed. If a subject's fingerprint and beta are in the same
    space but differ in voxel size/grid, the fingerprint is resampled to the beta
    grid subject-wise.
    """
    return ImageSpace(fc_dim=int(paths.n_rois), representation="voxel")


def squeeze_last_dim_if_singleton(img: nib.Nifti1Image) -> nib.Nifti1Image:
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        return nib.Nifti1Image(data[..., 0].astype(np.float32), img.affine, img.header)
    return img


def zscore_within_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.asarray(data, dtype=np.float32).copy()
    m = np.asarray(mask, dtype=bool) & np.isfinite(x)
    if int(m.sum()) == 0:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mu = float(x[m].mean())
    sd = float(x[m].std())
    if sd < 1e-6:
        sd = 1.0
    x[m] = (x[m] - mu) / sd
    x[~np.isfinite(x)] = 0.0
    return x.astype(np.float32)


def _fingerprint_to_channel_last(fp: np.ndarray, expected_c: int, subject: str) -> np.ndarray:
    if fp.ndim != 4:
        raise RuntimeError(f"{subject}: expected 4D fingerprint image, got shape {fp.shape}")
    # Preferred storage is [X,Y,Z,C]. Repair channel-first [C,X,Y,Z] if present.
    if fp.shape[-1] == expected_c:
        return fp.astype(np.float32, copy=False)
    if fp.shape[0] == expected_c:
        return np.moveaxis(fp, 0, -1).astype(np.float32, copy=False)
    raise RuntimeError(f"{subject}: expected {expected_c} fingerprint channels, got shape {fp.shape}")


def _same_voxel_grid(fp_img: nib.Nifti1Image, beta_img: nib.Nifti1Image, expected_c: int) -> bool:
    """Return True if fingerprint spatial axes already match the beta grid."""
    fp_shape = tuple(int(v) for v in fp_img.shape)
    beta_shape = tuple(int(v) for v in beta_img.shape[:3])
    if len(fp_shape) != 4:
        return False
    if fp_shape[-1] == expected_c:
        fp_spatial = fp_shape[:3]
    elif fp_shape[0] == expected_c:
        fp_spatial = fp_shape[1:]
    else:
        return False
    return tuple(fp_spatial) == tuple(beta_shape) and np.allclose(fp_img.affine, beta_img.affine, atol=1e-4)


def _load_fingerprint_on_beta_grid(
    fp_img: nib.Nifti1Image,
    beta_img: nib.Nifti1Image,
    expected_c: int,
    subject: str,
) -> np.ndarray:
    """Load fingerprint as channel-last [X,Y,Z,C] on the beta image grid.

    The current v2 architecture uses full 4D fingerprint images, but it does not
    require the fingerprint and beta to have the same voxel size on disk.  The
    valid assumption is that both are already in the same normal-space coordinate
    system.  When the grids differ, resample each fingerprint channel to the beta
    grid.  This is *not* Schaefer/atlas resampling and does not introduce parcel
    averaging; it only puts input and target tensors on the same voxel lattice.
    """
    if _same_voxel_grid(fp_img, beta_img, expected_c):
        return _fingerprint_to_channel_last(fp_img.get_fdata(dtype=np.float32), expected_c, subject)

    fp_shape = tuple(int(v) for v in fp_img.shape)
    beta_shape = tuple(int(v) for v in beta_img.shape[:3])
    if len(fp_shape) != 4:
        raise RuntimeError(f"{subject}: expected 4D fingerprint image, got shape {fp_shape}")
    if fp_shape[-1] == expected_c:
        channel_axis = -1
        fp_spatial = fp_shape[:3]
    elif fp_shape[0] == expected_c:
        channel_axis = 0
        fp_spatial = fp_shape[1:]
    else:
        raise RuntimeError(f"{subject}: expected {expected_c} fingerprint channels, got shape {fp_shape}")

    print(
        f"[extract-image] {subject}: resampling fingerprint grid {fp_spatial} -> beta grid {beta_shape} "
        f"for {expected_c} channels"
    )
    out = np.empty(tuple(beta_shape) + (int(expected_c),), dtype=np.float32)
    target = (tuple(beta_shape), beta_img.affine)
    dataobj = fp_img.dataobj
    for c in range(int(expected_c)):
        if channel_axis == -1:
            ch = np.asanyarray(dataobj[..., c], dtype=np.float32)
        else:
            ch = np.asanyarray(dataobj[c, ...], dtype=np.float32)
        ch_img = nib.Nifti1Image(ch, fp_img.affine)
        res = resample_from_to(ch_img, target, order=1, mode="constant", cval=0.0)
        out[..., c] = res.get_fdata(dtype=np.float32)
    return out


def _image_mask(fp: np.ndarray, beta: np.ndarray) -> np.ndarray:
    finite_beta = np.isfinite(beta)
    fp_finite = np.all(np.isfinite(fp), axis=-1)
    fp_nonzero = np.any(np.abs(np.nan_to_num(fp, nan=0.0, posinf=0.0, neginf=0.0)) > 1e-8, axis=-1)
    mask = finite_beta & fp_finite & fp_nonzero
    if int(mask.sum()) < 100:
        # Fallback for already masked data where fingerprints may be numerically zero.
        mask = finite_beta & fp_finite
    return mask.astype(bool)


def _update_atlas_from_first_subject(atlas: ImageSpace, ref_img: nib.Nifti1Image, mask: np.ndarray, fc_dim: int) -> None:
    shape = tuple(int(s) for s in mask.shape)
    flat_idx = np.flatnonzero(mask.reshape(-1)).astype(np.int64)
    ijk = np.array(np.where(mask)).T.astype(np.int32)
    world = nib.affines.apply_affine(ref_img.affine, ijk)
    hemi = np.where(world[:, 0] < 0, -1.0, 1.0).astype(np.float32)
    atlas.parcel_img = ref_img
    atlas.parcel_data = mask.astype(np.int16)
    atlas.parcel_ids = (np.arange(int(np.prod(shape)), dtype=np.int64) + 1)
    atlas.coords_norm = np.zeros((int(np.prod(shape)), 3), dtype=np.float32)
    atlas.hemi_signed = np.zeros((int(np.prod(shape)),), dtype=np.float32)
    atlas.hemi_signed[flat_idx] = hemi
    atlas.left_idx = flat_idx[hemi < 0].astype(np.int64)
    atlas.right_idx = flat_idx[hemi > 0].astype(np.int64)
    atlas.fc_dim = int(fc_dim)
    atlas.representation = "voxel"
    atlas.token_mask = mask.astype(bool)
    atlas.crop_slices = None
    atlas.crop_shape = shape
    atlas.mask_flat_idx = flat_idx
    atlas.token_ijk = ijk
    atlas.source_parcel_ids = np.ones((int(np.prod(shape)),), dtype=np.int64)
    atlas.network_ids = None
    atlas.language_prior = None


def extract_subject_image(subject: str, paths: ClinicalPaths, atlas: ImageSpace) -> Optional[dict]:
    """Load one subject as true image tensors.

    Returns x=[C,X,Y,Z] and y=[1,X,Y,Z]. No atlas resampling, cropping, parcel
    averaging, coordinate channels, lesion channels, or flattened token inputs.
    """
    fp_path = os.path.join(paths.data_dir, subject, paths.fingerprint_pattern)
    beta_path = os.path.join(paths.data_dir, subject, paths.beta_pattern)
    if not os.path.exists(fp_path) or not os.path.exists(beta_path):
        print(f"[extract] Missing fp/beta for {subject}; fp={os.path.exists(fp_path)}, beta={os.path.exists(beta_path)}")
        return None

    fp_img = nib.load(fp_path)
    beta_img = squeeze_last_dim_if_singleton(nib.load(beta_path))
    beta = beta_img.get_fdata(dtype=np.float32)
    if beta.ndim == 4 and beta.shape[-1] == 1:
        beta = beta[..., 0]
    if beta.ndim != 3:
        raise RuntimeError(f"{subject}: expected 3D beta image, got shape {beta.shape}")
    fp = _load_fingerprint_on_beta_grid(fp_img, beta_img, int(paths.n_rois), subject)
    if tuple(fp.shape[:3]) != tuple(beta.shape):
        raise RuntimeError(
            f"{subject}: internal grid mismatch after fingerprint-to-beta resampling: fp={fp.shape[:3]} beta={beta.shape}."
        )

    mask = _image_mask(fp, beta)
    beta_z = zscore_within_mask(beta, mask)
    y = beta_z.astype(np.float32, copy=True)
    y[~mask] = np.nan
    fp = np.nan_to_num(fp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    fp[~mask, :] = 0.0
    x = np.moveaxis(fp, -1, 0).astype(np.float32)  # [C,X,Y,Z]
    y = y[None, ...].astype(np.float32)            # [1,X,Y,Z]

    if atlas.parcel_img is None:
        _update_atlas_from_first_subject(atlas, beta_img, mask, x.shape[0])
    else:
        if tuple(y.shape[-3:]) != tuple(atlas.crop_shape):
            raise RuntimeError(f"{subject}: image shape {y.shape[-3:]} differs from cached reference shape {atlas.crop_shape}")

    li = robust_li(y, atlas.left_idx, atlas.right_idx)
    return {
        "subject": str(subject),
        "x": x,
        "y": y,
        "li": np.float32(li),
        "dominance": np.int64(dominance_class(li)),
    }


class ShardedImageArray:
    """Lazy per-subject image array backed by .npy shards.

    shape behaves like a normal [N,...] ndarray, but __getitem__ loads only the
    requested subject. This avoids a single giant memmap/cache file and works
    well with multi-worker DataLoader/DDP.
    """
    def __init__(self, paths: Sequence[str | Path], item_shape: Sequence[int], dtype=np.float32, mmap_mode: str | None = "r"):
        self.paths = [str(Path(p)) for p in paths]
        self.item_shape = tuple(int(x) for x in item_shape)
        self.dtype = np.dtype(dtype)
        self.mmap_mode = mmap_mode
        self.shape = (len(self.paths),) + self.item_shape
        self.ndim = len(self.shape)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return np.asarray(np.load(self.paths[int(idx)], mmap_mode=self.mmap_mode), dtype=self.dtype).copy()
        if isinstance(idx, slice):
            ids = list(range(*idx.indices(len(self))))
            return np.stack([np.asarray(self[i], dtype=self.dtype) for i in ids], axis=0)
        arr = np.asarray(idx)
        if arr.dtype == bool:
            ids = np.flatnonzero(arr)
        else:
            ids = arr.astype(int).reshape(-1)
        return np.stack([np.asarray(self[int(i)], dtype=self.dtype) for i in ids], axis=0)


MATRIX_CACHE_VERSION = "v10.0_image_space_sharded_npy_subject_resample"


def _json_dumps_stable(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _hash_dict(obj) -> str:
    return hashlib.sha256(_json_dumps_stable(obj).encode("utf-8")).hexdigest()


def _safe_cache_tag(tag: str) -> str:
    out = []
    for ch in (tag or "subject_matrix"):
        out.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(out).strip("._") or "subject_matrix"


def _matrix_cache_path(subjects: Sequence[str], paths: ClinicalPaths, cache_dir: Path, cache_tag: str) -> Path:
    spec = {"version": MATRIX_CACHE_VERSION, "subjects": list(subjects), "paths": asdict(paths)}
    h = _hash_dict(spec)[:16]
    return cache_dir / f"{_safe_cache_tag(cache_tag)}_{h}.pkl"


def _source_manifest(subjects: Sequence[str], paths: ClinicalPaths) -> dict:
    rows = []
    for s in subjects:
        fp = Path(paths.data_dir) / s / paths.fingerprint_pattern
        beta = Path(paths.data_dir) / s / paths.beta_pattern
        rows.append({
            "subject": s,
            "fp": str(fp),
            "fp_mtime": fp.stat().st_mtime if fp.exists() else None,
            "fp_size": fp.stat().st_size if fp.exists() else None,
            "beta": str(beta),
            "beta_mtime": beta.stat().st_mtime if beta.exists() else None,
            "beta_size": beta.stat().st_size if beta.exists() else None,
        })
    return {"rows": rows, "paths": asdict(paths), "version": MATRIX_CACHE_VERSION}


def _payload_cache_files_exist(payload: dict) -> bool:
    if "x_paths" in payload and "y_paths" in payload:
        xs = payload.get("x_paths") or []
        ys = payload.get("y_paths") or []
        return bool(xs) and bool(ys) and all(Path(p).exists() for p in xs[:3]) and all(Path(p).exists() for p in ys[:3])
    return Path(payload.get("x_path", "")).exists() and Path(payload.get("y_path", "")).exists()


def _open_memmap_matrix_from_payload(payload: dict) -> dict:
    meta = dict(payload["matrix_meta"])
    if "x_paths" in payload and "y_paths" in payload:
        meta["x"] = ShardedImageArray(payload["x_paths"], payload["x_item_shape"], dtype=np.float32, mmap_mode="r")
        meta["y"] = ShardedImageArray(payload["y_paths"], payload["y_item_shape"], dtype=np.float32, mmap_mode="r")
        meta["x_paths"] = list(payload["x_paths"])
        meta["y_paths"] = list(payload["y_paths"])
        meta["sharded_cache"] = True
        return meta
    # Backward compatibility for older monolithic memmap payloads. The version
    # check normally prevents this path, but keeping it makes manual cache reads safer.
    x_path = Path(payload["x_path"])
    y_path = Path(payload["y_path"])
    meta["x"] = np.lib.format.open_memmap(str(x_path), mode="r+", dtype=np.float32, shape=tuple(payload["x_shape"]))
    meta["y"] = np.lib.format.open_memmap(str(y_path), mode="r+", dtype=np.float32, shape=tuple(payload["y_shape"]))
    meta["x_memmap_path"] = str(x_path)
    meta["y_memmap_path"] = str(y_path)
    meta["sharded_cache"] = False
    return meta


def build_subject_matrix_cached(
    subjects: Sequence[str],
    paths: ClinicalPaths,
    atlas: Optional[ImageSpace] = None,
    cache_dir: str | Path | None = None,
    cache_tag: str = "",
    refresh_cache: bool = False,
) -> dict:
    atlas = atlas or create_image_space(paths)
    subjects = list(subjects)
    if not cache_dir:
        cache_dir = Path("./image_cache")
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _matrix_cache_path(subjects, paths, cache_dir, cache_tag)
    source_sig = _hash_dict(_source_manifest(subjects, paths))

    # In DDP, only rank 0 is allowed to build/refresh caches. Nonzero ranks
    # always wait for the rank-0 manifest and then open the per-subject shards.
    # This matters when REFRESH_CACHE=1: without this guard every rank may
    # rebuild the same cache sequentially after the lock is released.
    rank = _env_rank()
    world = _env_world_size()
    if world > 1 and rank != 0:
        wait_seconds = float(os.environ.get("CACHE_WAIT_SECONDS", "7200"))
        t0 = time.time()
        while True:
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        payload = pickle.load(f)
                    meta = payload.get("metadata", {})
                    if (meta.get("version") == MATRIX_CACHE_VERSION and meta.get("source_signature") == source_sig
                            and _payload_cache_files_exist(payload)):
                        print(f"[cache][rank{rank}] loaded image-space {cache_tag or 'subject_matrix'} after rank-0 build: {cache_path}")
                        return _open_memmap_matrix_from_payload(payload)
                except Exception:
                    pass
            if time.time() - t0 > wait_seconds:
                raise TimeoutError(f"Rank {rank} timed out waiting for rank-0 cache build: {cache_path}")
            time.sleep(2.0)

    if cache_path.exists() and not refresh_cache:
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            meta = payload.get("metadata", {})
            if (meta.get("version") == MATRIX_CACHE_VERSION and meta.get("source_signature") == source_sig
                    and _payload_cache_files_exist(payload)):
                print(f"[cache] loaded image-space {cache_tag or 'subject_matrix'}: {cache_path}")
                matrix = _open_memmap_matrix_from_payload(payload)
                # Update the passed atlas so NIfTI export uses the correct image grid.
                atlas.left_idx = np.asarray(matrix["left_idx"], dtype=np.int64)
                atlas.right_idx = np.asarray(matrix["right_idx"], dtype=np.int64)
                atlas.parcel_ids = np.asarray(matrix["parcel_ids"], dtype=np.int64)
                atlas.crop_shape = tuple(matrix["volume_shape"])
                atlas.fc_dim = int(matrix["fc_dim"])
                ref = matrix.get("reference_image", {})
                if ref:
                    atlas.parcel_img = nib.Nifti1Image(np.zeros(tuple(matrix["volume_shape"]), dtype=np.float32), np.asarray(ref["affine"], dtype=float))
                return matrix
            print(f"[cache] stale image-space cache ignored: {cache_path}")
        except Exception as e:
            print(f"[cache] could not read {cache_path}: {e}; rebuilding")

    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    have_lock = False
    while not have_lock:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            have_lock = True
        except FileExistsError:
            # Another DDP rank/process is building this cache. Wait and then try to load it.
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        payload = pickle.load(f)
                    meta = payload.get("metadata", {})
                    if meta.get("version") == MATRIX_CACHE_VERSION and meta.get("source_signature") == source_sig and _payload_cache_files_exist(payload):
                        print(f"[cache] loaded image-space after waiting {cache_tag or 'subject_matrix'}: {cache_path}")
                        return _open_memmap_matrix_from_payload(payload)
                except Exception:
                    pass
            time.sleep(2.0)
    try:
        # A cache may have appeared while this process was waiting for the lock.
        if cache_path.exists() and not refresh_cache:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            meta = payload.get("metadata", {})
            if meta.get("version") == MATRIX_CACHE_VERSION and meta.get("source_signature") == source_sig and _payload_cache_files_exist(payload):
                print(f"[cache] loaded image-space under lock {cache_tag or 'subject_matrix'}: {cache_path}")
                return _open_memmap_matrix_from_payload(payload)

        print(f"[cache] building sharded image-space {cache_tag or 'subject_matrix'} cache: {cache_path}")
        arr_dir = Path(str(cache_path.with_suffix("")) + "_shards")
        arr_dir.mkdir(parents=True, exist_ok=True)
        valid_subjects: list[str] = []
        x_paths: list[str] = []
        y_paths: list[str] = []
        x_item_shape = None
        y_item_shape = None
        first_y = None
        for i, subj in enumerate(subjects):
            print(f"[extract-image] {i+1}/{len(subjects)} {subj}")
            item = extract_subject_image(subj, paths, atlas)
            if item is None:
                continue
            x = np.asarray(item["x"], dtype=np.float32)
            y = np.asarray(item["y"], dtype=np.float32)
            if x_item_shape is None:
                x_item_shape = tuple(x.shape)
                y_item_shape = tuple(y.shape)
                first_y = y.copy()
            if tuple(x.shape) != tuple(x_item_shape) or tuple(y.shape) != tuple(y_item_shape):
                raise RuntimeError(f"{subj}: inconsistent x/y shape {x.shape}/{y.shape}; expected {x_item_shape}/{y_item_shape}")
            safe = _safe_cache_tag(str(item["subject"]))
            x_file = arr_dir / f"{len(valid_subjects):05d}_{safe}_x.npy"
            y_file = arr_dir / f"{len(valid_subjects):05d}_{safe}_y.npy"
            np.save(x_file, x.astype(np.float32, copy=False), allow_pickle=False)
            np.save(y_file, y.astype(np.float32, copy=False), allow_pickle=False)
            x_paths.append(str(x_file)); y_paths.append(str(y_file))
            valid_subjects.append(str(item["subject"]))
        if len(valid_subjects) == 0 or x_item_shape is None or y_item_shape is None:
            raise RuntimeError("No valid subjects after image extraction.")
    
        x_arr = ShardedImageArray(x_paths, x_item_shape, dtype=np.float32, mmap_mode="r")
        y_arr = ShardedImageArray(y_paths, y_item_shape, dtype=np.float32, mmap_mode="r")
        n_valid = len(valid_subjects)
        left_idx = np.asarray(atlas.left_idx, dtype=np.int64)
        right_idx = np.asarray(atlas.right_idx, dtype=np.int64)
        li = np.asarray([robust_li(y_arr[i], left_idx, right_idx) for i in range(n_valid)], dtype=np.float32)
        dom = np.asarray([dominance_class(v) for v in li], dtype=np.int64)
        vol_shape = tuple(int(s) for s in y_item_shape[-3:])
        reference_image = {
            "affine": atlas.parcel_img.affine.tolist() if atlas.parcel_img is not None else np.eye(4).tolist(),
            "shape": vol_shape,
        }
        matrix_meta = {
            "subjects": valid_subjects,
            "li": li,
            "dominance": dom,
            "parcel_ids": np.arange(1, int(np.prod(vol_shape)) + 1, dtype=np.int64),
            "left_idx": left_idx,
            "right_idx": right_idx,
            "input_feature_names": [f"fp_{k+1:03d}" for k in range(int(x_item_shape[0]))],
            "fc_dim": int(x_item_shape[0]),
            "representation": "voxel_image",
            "volume_shape": vol_shape,
            "token_mask_shape": vol_shape,
            "crop_slices": None,
            "mask_flat_idx": np.flatnonzero(np.isfinite(first_y[0]).reshape(-1)).astype(np.int64),
            "paths": asdict(paths),
            "disk_backed": True,
            "sharded_cache": True,
            "reference_image": reference_image,
        }
        payload = {
            "metadata": {"version": MATRIX_CACHE_VERSION, "source_signature": source_sig, "cache_tag": cache_tag, "subjects": subjects, "paths": asdict(paths)},
            "matrix_meta": matrix_meta,
            "x_paths": x_paths,
            "y_paths": y_paths,
            "x_item_shape": tuple(x_item_shape),
            "y_item_shape": tuple(y_item_shape),
        }
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)
        print(f"[cache] saved sharded image-space {cache_tag or 'subject_matrix'}: {cache_path}")
        return _open_memmap_matrix_from_payload(payload)
    finally:
        if have_lock:
            try:
                Path(lock_path).unlink(missing_ok=True)
            except Exception:
                pass


class ImageSubjectDataset(Dataset):
    def __init__(self, matrix: dict, indices: Optional[Sequence[int]] = None):
        self.matrix = matrix
        n = len(matrix["subjects"])
        self.indices = np.asarray(indices if indices is not None else np.arange(n), dtype=int)
        self.scaler = matrix.get("x_scaler", None)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> dict:
        i = int(self.indices[int(idx)])
        x = np.asarray(self.matrix["x"][i], dtype=np.float32).copy()
        if self.scaler is not None:
            x = self.scaler.transform_subject(x)
        y = np.asarray(self.matrix["y"][i], dtype=np.float32).copy()
        if x.ndim == 4 and y.ndim == 4:
            valid = np.isfinite(y[0])
            x = x.copy()
            x[:, ~valid] = 0.0
        return {
            "subject": self.matrix["subjects"][i],
            "x": x,
            "y": y,
            "li": np.asarray(self.matrix["li"][i], dtype=np.float32),
            "dominance": np.asarray(self.matrix["dominance"][i], dtype=np.int64),
        }


def write_image_nifti(values: np.ndarray, atlas: ImageSpace, out_path: str | Path) -> None:
    vals = np.asarray(values, dtype=np.float32)
    if vals.ndim == 4 and vals.shape[0] == 1:
        vals = vals[0]
    if vals.ndim == 5 and vals.shape[0] == 1 and vals.shape[1] == 1:
        vals = vals[0, 0]
    if vals.ndim != 3:
        vals = vals.reshape(tuple(atlas.crop_shape))
    affine = atlas.parcel_img.affine if atlas.parcel_img is not None else np.eye(4)
    img = nib.Nifti1Image(vals.astype(np.float32), affine)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))


# Backward-compatible aliases for older module imports/checkpoints.
ParcelAtlas = ImageSpace
ParcelSubjectDataset = ImageSubjectDataset
load_schaefer_atlas = create_image_space
extract_subject_parcels = extract_subject_image
write_parcel_nifti = write_image_nifti
