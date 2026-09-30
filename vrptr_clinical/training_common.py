from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler



def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if distributed_is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if distributed_is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed_from_env(device_arg: str = "cuda") -> torch.device:
    """Initialize torch.distributed when launched with torchrun.

    Returns the process-local device. In non-DDP runs this behaves like
    torch.device(device_arg).
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        if is_main_process():
            print(f"[ddp] initialized world_size={world} local_rank={local_rank} device={device}")
        return device
    return torch.device(device_arg)


def cleanup_distributed() -> None:
    if distributed_is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if distributed_is_initialized():
        dist.barrier()


def reduce_mean_scalar(value: float | int) -> float:
    if not distributed_is_initialized():
        return float(value)
    t = torch.tensor(float(value), device=torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu"))
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= float(get_world_size())
    return float(t.detach().cpu())


def make_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0, pin_memory: bool | None = None, prefetch_factor: int = 2, distributed: bool | None = None):
    if distributed is None:
        distributed = distributed_is_initialized()
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False) if distributed else None
    kwargs = dict(
        batch_size=int(batch_size),
        shuffle=(bool(shuffle) and sampler is None),
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(torch.cuda.is_available()) if pin_memory is None else bool(pin_memory),
    )
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


class Standardizer:
    def __init__(
        self,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        exclude_last_n: int = 0,
        clip_value: float = 8.0,
    ):
        self.mean = mean
        self.std = std
        self.exclude_last_n = int(exclude_last_n)
        self.clip_value = float(clip_value)

    def fit(self, x: np.ndarray) -> "Standardizer":
        # Token mode: [N, P, F] -> per-feature mean/std over N,P.
        # Image mode: [N, C, X, Y, Z] -> per-channel mean/std over N,X,Y,Z.
        x = np.asarray(x, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim == 5:
            self.mean = x.mean(axis=(0, 2, 3, 4), keepdims=True).astype(np.float32)
            self.std = x.std(axis=(0, 2, 3, 4), keepdims=True).astype(np.float32)
        else:
            self.mean = x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
            self.std = x.std(axis=(0, 1), keepdims=True).astype(np.float32)
        self.std = np.nan_to_num(self.std, nan=1.0, posinf=1.0, neginf=1.0)
        self.std[self.std < 1e-6] = 1.0
        if self.exclude_last_n > 0 and x.ndim != 5:
            self.mean[..., -self.exclude_last_n:] = 0.0
            self.std[..., -self.exclude_last_n:] = 1.0
        return self

    def fit_indices(self, x: np.ndarray, indices: Sequence[int] | np.ndarray, subject_chunk: int = 1) -> "Standardizer":
        """Fit feature scaling without materializing a full [N, P, F] copy.

        This is important for the v2 voxel/fingerprint representation, where
        ``x`` may be a disk-backed memmap with many spatial tokens.  The method
        streams selected subjects and accumulates sums/squares per feature.
        """
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            idx = np.arange(int(x.shape[0]), dtype=int)
        if x.ndim == 5:
            c = int(x.shape[1])
            total = 0
            s1 = np.zeros((c,), dtype=np.float64)
            s2 = np.zeros((c,), dtype=np.float64)
            for start in range(0, idx.size, max(1, int(subject_chunk))):
                chunk_idx = idx[start:start + max(1, int(subject_chunk))]
                for ii in chunk_idx:
                    arr = np.asarray(x[int(ii)], dtype=np.float32)
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    s1 += arr.sum(axis=(1,2,3), dtype=np.float64)
                    s2 += np.square(arr, dtype=np.float64).sum(axis=(1,2,3), dtype=np.float64)
                    total += int(arr.shape[1] * arr.shape[2] * arr.shape[3])
            if total <= 0:
                raise RuntimeError("Cannot fit Standardizer on zero voxels.")
            mean = s1 / float(total)
            var = np.maximum(s2 / float(total) - mean * mean, 0.0)
            std = np.sqrt(var); std[std < 1e-6] = 1.0
            self.mean = mean.reshape(1, c, 1, 1, 1).astype(np.float32)
            self.std = std.reshape(1, c, 1, 1, 1).astype(np.float32)
            return self

        f = int(x.shape[-1])
        total = 0
        s1 = np.zeros((f,), dtype=np.float64)
        s2 = np.zeros((f,), dtype=np.float64)
        for start in range(0, idx.size, max(1, int(subject_chunk))):
            chunk_idx = idx[start:start + max(1, int(subject_chunk))]
            for ii in chunk_idx:
                arr = np.asarray(x[int(ii)], dtype=np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                s1 += arr.sum(axis=0, dtype=np.float64)
                s2 += np.square(arr, dtype=np.float64).sum(axis=0, dtype=np.float64)
                total += int(arr.shape[0])
        if total <= 0:
            raise RuntimeError("Cannot fit Standardizer on zero spatial tokens.")
        mean = s1 / float(total)
        var = np.maximum(s2 / float(total) - mean * mean, 0.0)
        std = np.sqrt(var)
        std[std < 1e-6] = 1.0
        self.mean = mean.reshape(1, 1, f).astype(np.float32)
        self.std = std.reshape(1, 1, f).astype(np.float32)
        if self.exclude_last_n > 0:
            self.mean[..., -self.exclude_last_n:] = 0.0
            self.std[..., -self.exclude_last_n:] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer has not been fit.")
        y = ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip standardized features. This prevents a single malformed subject/
        # parcel from blowing up the matrix encoder or attention logits.
        if self.clip_value > 0:
            if self.exclude_last_n > 0:
                keep = y[..., -self.exclude_last_n:].copy()
                y = np.clip(y, -self.clip_value, self.clip_value)
                y[..., -self.exclude_last_n:] = keep
            else:
                y = np.clip(y, -self.clip_value, self.clip_value)
        return y.astype(np.float32)

    def transform_subject(self, x: np.ndarray) -> np.ndarray:
        """Transform one subject/token field without allocating the whole dataset."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer has not been fit.")
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 4 and self.mean is not None and self.mean.ndim == 5:
            mean = self.mean.reshape(-1, 1, 1, 1)
            std = self.std.reshape(-1, 1, 1, 1)
        else:
            mean = self.mean.reshape(-1)
            std = self.std.reshape(-1)
        y = ((x_arr - mean) / std).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if self.clip_value > 0:
            if self.exclude_last_n > 0:
                keep = y[..., -self.exclude_last_n:].copy()
                y = np.clip(y, -self.clip_value, self.clip_value)
                y[..., -self.exclude_last_n:] = keep
            else:
                y = np.clip(y, -self.clip_value, self.clip_value)
        return y.astype(np.float32)


    def fit_indices_masked(self, x: np.ndarray, y: np.ndarray, indices: Sequence[int] | np.ndarray, subject_chunk: int = 1) -> "Standardizer":
        """Fit per-channel image scaling using only valid target voxels.

        In v2 image mode, y is [N,1,X,Y,Z] with NaNs outside the brain/mask.
        This avoids outside-mask zeros dominating fingerprint normalization.
        """
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            idx = np.arange(int(x.shape[0]), dtype=int)
        if x.ndim != 5:
            return self.fit_indices(x, idx, subject_chunk=subject_chunk)
        c = int(x.shape[1])
        total = 0
        s1 = np.zeros((c,), dtype=np.float64)
        s2 = np.zeros((c,), dtype=np.float64)
        for ii in idx:
            arr = np.asarray(x[int(ii)], dtype=np.float32)
            yy = np.asarray(y[int(ii)], dtype=np.float32)
            mask = np.isfinite(yy[0]) if yy.ndim == 4 else np.isfinite(yy.reshape(yy.shape[-3:]))
            if int(mask.sum()) == 0:
                continue
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            vals = arr[:, mask]
            s1 += vals.sum(axis=1, dtype=np.float64)
            s2 += np.square(vals, dtype=np.float64).sum(axis=1, dtype=np.float64)
            total += int(mask.sum())
        if total <= 0:
            raise RuntimeError("Cannot fit Standardizer on zero valid image voxels.")
        mean = s1 / float(total)
        var = np.maximum(s2 / float(total) - mean * mean, 0.0)
        std = np.sqrt(var); std[std < 1e-6] = 1.0
        self.mean = mean.reshape(1, c, 1, 1, 1).astype(np.float32)
        self.std = std.reshape(1, c, 1, 1, 1).astype(np.float32)
        return self
    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "exclude_last_n": self.exclude_last_n, "clip_value": self.clip_value}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Standardizer":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
            exclude_last_n=int(state.get("exclude_last_n", 0)),
            clip_value=float(state.get("clip_value", 8.0)),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac))) if n >= 5 else 0
    return idx[n_val:], idx[:n_val]


def to_device(batch: dict, device: torch.device) -> dict:
    # Stage tensors on CPU first, then transfer.  For very large full-field
    # batches this is more robust than constructing CUDA tensors directly from
    # NumPy/memmap-backed arrays, and it gives DDP a normal subject
    # batch to scatter.
    x_cpu = torch.as_tensor(batch["x"], dtype=torch.float32)
    y_cpu = torch.as_tensor(batch["y"], dtype=torch.float32)
    li_cpu = torch.as_tensor(batch["li"], dtype=torch.float32)
    dom_cpu = torch.as_tensor(batch["dominance"], dtype=torch.long)
    non_blocking = bool(device.type == "cuda")
    x = x_cpu.to(device=device, non_blocking=non_blocking)
    y = y_cpu.to(device=device, non_blocking=non_blocking)
    li = li_cpu.to(device=device, non_blocking=non_blocking)
    dom = dom_cpu.to(device=device, non_blocking=non_blocking)
    return {
        "subject": batch["subject"],
        "x": torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        # Keep NaNs in y: in voxel-image mode outside-atlas voxels are NaN so
        # losses/metrics can restrict to valid brain-mask voxels.
        "y": y,
        "li": torch.nan_to_num(li, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0),
        "dominance": dom.clamp(0, 2),
    }


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when a model is wrapped in DDP."""
    return model.module if isinstance(model, DDP) else model


def model_state_dict_cpu(model: torch.nn.Module) -> dict:
    """State dict without DDP's `module.` prefix, stored on CPU."""
    base = unwrap_model(model)
    return {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in base.state_dict().items()}


def load_state_dict_flexible_(model: torch.nn.Module, state: dict, strict: bool = True):
    """Load checkpoints into wrapped/unwrapped models and tolerate `module.` prefixes."""
    base = unwrap_model(model)
    if not isinstance(state, dict):
        return base.load_state_dict(state, strict=strict)
    # If a previous wrapped checkpoint was saved by mistake, strip the prefix.
    if any(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
        state = {str(k).replace("module.", "", 1): v for k, v in state.items()}
    device = next(base.parameters()).device
    return base.load_state_dict({k: v.to(device) if torch.is_tensor(v) else v for k, v in state.items()}, strict=strict)


def maybe_data_parallel(model: torch.nn.Module, device: torch.device, disabled: bool = False, context: str = "model") -> torch.nn.Module:
    """Wrap model with DDP only when launched by torchrun.

    The lean v2 code intentionally does not use torch.nn.DataParallel. For full
    3D volumes, DataParallel introduced GPU-0 gather/synchronization bottlenecks
    and watchdog timeouts. Multi-GPU training should use torchrun/DDP.
    """
    if distributed_is_initialized():
        if isinstance(model, DDP):
            return model
        if device.type == "cuda":
            if is_main_process():
                print(f"[ddp] wrapping {context} with DistributedDataParallel on local device {device}")
            return DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
        return DDP(model, find_unused_parameters=False)
    return model


def restore_state_dict_(model: torch.nn.Module, state: dict) -> None:
    load_state_dict_flexible_(model, state, strict=True)


def predict_beta_in_chunks(model, x: torch.Tensor, max_tokens: int = 0) -> dict:
    """Predict beta/logvar for a spatial field.

    v2 full-field neural models must see the complete fingerprint field before
    the encoder/bottleneck.  For such models, chunking the input would change
    the latent representation and is therefore disabled.  The legacy chunked
    path remains only for older token-independent checkpoints.
    """
    p = int(x.shape[1])
    mt = int(max_tokens or 0)
    if bool(getattr(unwrap_model(model), "requires_full_field", False)):
        return model(x, sample_latent=False)
    if mt <= 0 or mt >= p:
        return model(x, sample_latent=False)
    betas, logvars = [], []
    for start in range(0, p, mt):
        xb = x[:, start:start + mt, :]
        out = model(xb, sample_latent=False)
        betas.append(out["beta"])
        logvars.append(out.get("logvar", torch.zeros_like(out["beta"])))
    beta = torch.cat(betas, dim=1)
    logvar = torch.cat(logvars, dim=1)
    ret = {
        "beta": beta,
        "logvar": logvar,
    }
    return ret


def write_rows_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = []
    seen = set()
    preferred = ["group", "model", "subject", "n"]
    for k in preferred:
        if any(k in r for r in rows) and k not in seen:
            keys.append(k); seen.add(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k); seen.add(k)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
