from __future__ import annotations

import argparse
from .training_template import add_template_argument, load_for_evaluation, metric_provenance, require_complete_subjects
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ClinicalPaths, ImageSubjectDataset, build_subject_matrix_cached, create_image_space, load_subject_list
from .losses import neg_pearson_loss, subject_contrastive_loss
from .metrics import subject_metrics, summarize_metric_rows, self_vs_other_rows, template_residual_analysis_rows
from .representation_ablation import add_input_ablation_args, ablated_model_name, make_eval_dataset
from .model import Encoder3D, Decoder3D, _safe_heads, _gn
from .training_common import (
    Standardizer, barrier, cleanup_distributed, distributed_is_initialized, get_rank, is_main_process, make_dataloader,
    load_state_dict_flexible_, maybe_data_parallel, model_state_dict_cpu, reduce_mean_scalar,
    set_seed, setup_distributed_from_env, split_indices, to_device, unwrap_model,
    write_rows_csv, save_checkpoint, predict_beta_in_chunks, restore_state_dict_,
)


def _read_rows_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _merge_group_model_rows(new_rows: List[dict], existing_path: Path) -> List[dict]:
    """Keep fresh rows first but preserve untouched group/model rows.

    Evaluation-only calls may be limited to --eval-groups tumor or controls.  The
    old implementation overwrote model_dir/summary.csv with only the groups from
    the most recent partial run, which made later compilation drop untouched rows.
    """
    rows = list(new_rows) + _read_rows_csv(existing_path)
    out: List[dict] = []
    seen = set()
    for r in rows:
        key = (str(r.get("group", "")), str(r.get("model", "")))
        if not key[0].strip() or not key[1].strip() or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


class BrainSurfCNNLite(nn.Module):
    """3D full-fingerprint BrainSurfCNN-like comparator.

    This is the shared backbone comparator for v2: a real 3D Conv encoder sees
    the full fingerprint image [B,C,X,Y,Z], then a 3D decoder predicts the full
    beta map [B,1,X,Y,Z]. No flattened mask tokens or non-voxel metadata.
    """

    def __init__(self, n_tokens: int, input_dim: int, width: int = 32, depth: int = 4, dropout: float = 0.10, full_field_stride: int = 2, full_res_refine_layers: int = 1, beta_clip: float = 8.0, head_init_std: float = 1e-4):
        super().__init__()
        self.requires_full_field = True
        self.beta_clip = float(beta_clip)
        levels = max(2, int(depth))
        width = max(8, int(width))
        self.encoder = Encoder3D(int(input_dim), width=width, levels=levels, dropout=dropout)
        self.decoder = Decoder3D(self.encoder.out_channels, width=width, levels=levels, dropout=dropout)
        self.head = nn.Conv3d(self.decoder.out_channels, 1, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=float(head_init_std))
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, sample_latent: bool | None = None, **kwargs) -> Dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise RuntimeError(f"BrainSurfCNN-like v2 expects [B,C,X,Y,Z], got {tuple(x.shape)}")
        h, skips = self.encoder(x)
        dec = self.decoder(h, skips)
        beta = torch.nan_to_num(self.head(dec), nan=0.0, posinf=0.0, neginf=0.0).clamp(-self.beta_clip, self.beta_clip)
        return {
            "beta": beta,
            "logvar": torch.zeros_like(beta),
        }


class WindowAttention3D(nn.Module):
    """3D window self-attention used by the SwiFUN-like comparator.

    This is intentionally lightweight but follows the Swin-UNETR idea: attention
    is local within 3D windows, and consecutive blocks alternate regular and
    shifted windows.  It avoids global attention over the entire volume.
    """
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.10):
        super().__init__()
        self.dim = int(dim)
        self.heads = _safe_heads(dim, heads)
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.proj = nn.Linear(self.dim, self.dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [num_windows * B, window_volume, C]
        b_windows, n, c = x.shape
        qkv = self.qkv(x).reshape(b_windows, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if mask is not None:
            # mask: [num_windows, N, N]
            n_win = int(mask.shape[0])
            attn = attn.view(b_windows // n_win, n_win, self.heads, n, n)
            attn = attn + mask.to(dtype=attn.dtype, device=attn.device).unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.heads, n, n)
        attn = self.drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b_windows, n, c)
        return self.proj(out)


def _window_partition_3d(x: torch.Tensor, window_size: tuple[int, int, int]) -> torch.Tensor:
    # x: [B, D, H, W, C]
    b, d, h, w, c = x.shape
    wd, wh, ww = window_size
    x = x.view(b, d // wd, wd, h // wh, wh, w // ww, ww, c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wd * wh * ww, c)
    return windows


def _window_reverse_3d(windows: torch.Tensor, window_size: tuple[int, int, int], b: int, d: int, h: int, w: int) -> torch.Tensor:
    wd, wh, ww = window_size
    c = windows.shape[-1]
    x = windows.view(b, d // wd, h // wh, w // ww, wd, wh, ww, c)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, c)
    return x


def _make_shift_mask_3d(d: int, h: int, w: int, window_size: tuple[int, int, int], shift_size: tuple[int, int, int], device: torch.device) -> torch.Tensor | None:
    sd, sh, sw = shift_size
    if sd == 0 and sh == 0 and sw == 0:
        return None
    wd, wh, ww = window_size
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    d_slices = (slice(0, -wd), slice(-wd, -sd), slice(-sd, None)) if sd > 0 else (slice(0, d),)
    h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None)) if sh > 0 else (slice(0, h),)
    w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None)) if sw > 0 else (slice(0, w),)
    cnt = 0
    for ds in d_slices:
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, ds, hs, ws, :] = cnt
                cnt += 1
    mask_windows = _window_partition_3d(img_mask, window_size).view(-1, wd * wh * ww)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
    return attn_mask


class SwinBlock3D(nn.Module):
    """A minimal Swin-style 3D block with shifted local windows."""
    def __init__(self, dim: int, heads: int = 4, window_size: int = 4, shift: bool = False, dropout: float = 0.10, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = int(dim)
        self.window_size_base = max(1, int(window_size))
        self.shift = bool(shift)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(round(dim * mlp_ratio))
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        b, c, d0, h0, w0 = x.shape
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()  # [B,D,H,W,C]
        shortcut = x_cl
        x_norm = self.norm1(x_cl)

        wd = min(self.window_size_base, max(1, d0))
        wh = min(self.window_size_base, max(1, h0))
        ww = min(self.window_size_base, max(1, w0))
        pd = (wd - d0 % wd) % wd
        ph = (wh - h0 % wh) % wh
        pw = (ww - w0 % ww) % ww
        x_pad = F.pad(x_norm, (0, 0, 0, pw, 0, ph, 0, pd))
        _, d, h, w, _ = x_pad.shape
        window_size = (wd, wh, ww)
        shift_size = (wd // 2 if self.shift and d > wd else 0,
                      wh // 2 if self.shift and h > wh else 0,
                      ww // 2 if self.shift and w > ww else 0)
        if any(s > 0 for s in shift_size):
            shifted = torch.roll(x_pad, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = _make_shift_mask_3d(d, h, w, window_size, shift_size, x.device)
        else:
            shifted = x_pad
            attn_mask = None
        windows = _window_partition_3d(shifted, window_size)
        attn_windows = self.attn(windows, mask=attn_mask)
        shifted = _window_reverse_3d(attn_windows, window_size, b, d, h, w)
        if any(s > 0 for s in shift_size):
            shifted = torch.roll(shifted, shifts=shift_size, dims=(1, 2, 3))
        if pd or ph or pw:
            shifted = shifted[:, :d0, :h0, :w0, :].contiguous()
        x_cl = shortcut + shifted
        x_cl = x_cl + self.mlp(self.norm2(x_cl))
        return torch.nan_to_num(x_cl.permute(0, 4, 1, 2, 3).contiguous(), nan=0.0, posinf=0.0, neginf=0.0)


class SwinBottleneck3D(nn.Module):
    """Alternating regular/shifted-window attention blocks at the U-Net bottleneck."""
    def __init__(self, channels: int, heads: int = 4, depth: int = 4, window_size: int = 4, dropout: float = 0.10, mlp_ratio: float = 2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock3D(channels, heads=heads, window_size=window_size, shift=(i % 2 == 1), dropout=dropout, mlp_ratio=mlp_ratio)
            for i in range(max(1, int(depth)))
        ])
        self.norm = _gn(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class SwiFUNLite(nn.Module):
    """Swin-UNETR-inspired volumetric task-map comparator.

    The model uses the same full 4D connectivity-fingerprint input and the same
    3D U-Net encoder/decoder scale as the other neural models.  Its distinctive
    component is a stack of alternating regular and shifted 3D window-attention
    blocks at the compressed bottleneck.  This is the intended SwiFUN-like
    comparator: local convolutional encoding/decoding plus Swin-style local
    self-attention, without VRPTR's variational bottleneck.
    """

    def __init__(self, n_tokens: int, input_dim: int, dim: int = 32, heads: int = 4, depth: int = 4,
                 window_size: int = 4, dropout: float = 0.10, full_field_stride: int = 2,
                 full_res_refine_layers: int = 1, beta_clip: float = 8.0, head_init_std: float = 1e-4):
        super().__init__()
        self.requires_full_field = True
        self.beta_clip = float(beta_clip)
        levels = max(2, int(depth))
        width = max(8, int(dim))
        self.encoder = Encoder3D(int(input_dim), width=width, levels=levels, dropout=dropout)
        self.swin_bottleneck = SwinBottleneck3D(
            self.encoder.out_channels,
            heads=heads,
            depth=max(1, int(depth)),
            window_size=max(1, int(window_size)),
            dropout=dropout,
        )
        self.decoder = Decoder3D(self.encoder.out_channels, width=width, levels=levels, dropout=dropout)
        self.head = nn.Conv3d(self.decoder.out_channels, 1, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=float(head_init_std))
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, sample_latent: bool | None = None, **kwargs) -> Dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise RuntimeError(f"SWIFUN-like v2 expects [B,C,X,Y,Z], got {tuple(x.shape)}")
        h, skips = self.encoder(x)
        h = self.swin_bottleneck(h)
        dec = self.decoder(h, skips)
        beta = torch.nan_to_num(self.head(dec), nan=0.0, posinf=0.0, neginf=0.0).clamp(-self.beta_clip, self.beta_clip)
        return {
            "beta": beta,
            "logvar": torch.zeros_like(beta),
        }

def set_comparison_adapter_tuning(model: nn.Module, mode: str = "decoder") -> None:
    """Freeze/unfreeze comparator parameters for memory-safe tumor adaptation."""
    base = unwrap_model(model)
    mode = str(mode or "decoder").lower().replace("-", "_")
    for p in base.parameters():
        p.requires_grad = False
    if mode == "full":
        for p in base.parameters():
            p.requires_grad = True
        return
    modules = []
    if mode in {"bottleneck_decoder", "latent_decoder"}:
        for name in ["swin", "bottleneck", "transformer"]:
            mod = getattr(base, name, None)
            if mod is not None:
                modules.append(mod)
    if mode in {"decoder", "bottleneck_decoder", "latent_decoder"}:
        mod = getattr(base, "decoder", None)
        if mod is not None:
            modules.append(mod)
    if mode in {"head", "decoder", "bottleneck_decoder", "latent_decoder"}:
        mod = getattr(base, "head", None)
        if mod is not None:
            modules.append(mod)
    if mode not in {"head", "decoder", "bottleneck_decoder", "latent_decoder"}:
        raise ValueError(f"Unknown comparator adapter mode: {mode}")
    for mod in modules:
        for p in mod.parameters():
            p.requires_grad = True

def build_comparison_model(name: str, n_tokens: int, input_dim: int, args) -> nn.Module:
    name = name.lower()
    if name in {"brainsurfcnn_lite", "brainsurfcnn", "surfcnn"}:
        return BrainSurfCNNLite(n_tokens, input_dim, width=args.surf_width, depth=args.surf_depth, dropout=args.dropout, full_field_stride=max(args.full_field_stride, args.brainsurf_full_field_stride), full_res_refine_layers=args.full_res_refine_layers, beta_clip=args.beta_clip, head_init_std=args.head_init_std)
    if name in {"swifun_lite", "swifun", "swin_unet_lite", "swinunet_lite"}:
        return SwiFUNLite(n_tokens, input_dim, dim=args.swifun_dim, heads=args.swifun_heads, depth=args.swifun_depth,
                         window_size=args.swifun_window, dropout=args.dropout, full_field_stride=max(args.full_field_stride, args.swifun_full_field_stride), full_res_refine_layers=args.full_res_refine_layers, beta_clip=args.beta_clip, head_init_std=args.head_init_std)
    raise ValueError(f"Unknown comparison model: {name}")


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "pct_trainable": float(100.0 * trainable / max(total, 1))}


def _loss_for_epoch(out: Dict[str, torch.Tensor], y: torch.Tensor, epoch_1based: int, args) -> tuple[torch.Tensor, Dict[str, float]]:
    """Shared BrainSurfCNN-style reconstructive-contrastive loss for all comparators.

    This deliberately matches VRPTR's supervised objective except that comparator
    models do not have the VRPTR KL term. The loss is stable and simple:
    SmoothL1 beta-map reconstruction from the start, then subject-specific
    contrastive loss after the map-only phase.
    """
    pred = out["beta"]
    mask = torch.isfinite(pred) & torch.isfinite(y)
    rec = F.smooth_l1_loss(pred[mask], y[mask]) if bool(mask.any()) else pred.new_tensor(0.0)
    corr = neg_pearson_loss(pred, y) if float(args.w_corr) > 0 else pred.new_tensor(0.0)
    contrastive_scale = 1.0 if int(epoch_1based) > int(args.map_only_epochs) else 0.0
    contrast = subject_contrastive_loss(
        pred, y,
        temperature=args.contrastive_temperature,
        mse_weight=args.contrastive_mse_weight,
    ) if float(args.w_contrastive) > 0 and contrastive_scale > 0 else pred.new_tensor(0.0)
    total = float(args.w_mse) * rec + float(args.w_corr) * corr + float(args.w_contrastive) * contrastive_scale * contrast
    total = torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0)
    return total, {
        "total": float(total.detach().cpu()),
        "reconstruct": float(rec.detach().cpu()),
        "corr": float(corr.detach().cpu()),
        "contrast": float(contrast.detach().cpu()),
        "contrastive_scale": float(contrastive_scale),
    }

def _eval_model(model: nn.Module, dataset: ImageSubjectDataset, left_idx, right_idx, device, group: str, model_name: str, batch_size: int, spatial_chunk_size: int = 0, individuality_dir: Path | None = None, training_template_path: str = "", template_paths=None, test_loo_sensitivity: bool = False):
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, distributed=False)
    eval_model = unwrap_model(model)
    eval_model.eval()
    rows = []
    pred_records = []
    with torch.no_grad():
        for batch in loader:
            bd = to_device(batch, device)
            out = predict_beta_in_chunks(eval_model, bd["x"], max_tokens=spatial_chunk_size)
            pred = out["beta"].detach().cpu().numpy()
            true = bd["y"].detach().cpu().numpy()
            for i, subj in enumerate(list(batch["subject"])):
                m = subject_metrics(true[i], pred[i], left_idx, right_idx)
                m.update({"subject": subj, "group": group, "model": model_name})
                rows.append(m)
                pred_records.append((str(subj), true[i].copy(), pred[i].copy()))
    summary = summarize_metric_rows(rows, group=group, model=model_name)
    if individuality_dir is not None and pred_records:
        individuality_dir.mkdir(parents=True, exist_ok=True)
        subjects = [r[0] for r in pred_records]
        true_maps = [r[1] for r in pred_records]
        pred_maps = [r[2] for r in pred_records]
        template, metadata = load_for_evaluation(training_template_path, subjects, template_paths, true_maps[0].shape)
        indiv_rows, indiv_pairwise, indiv_summary = self_vs_other_rows(subjects, true_maps, pred_maps, group, model_name)
        write_rows_csv(individuality_dir / "self_vs_other.csv", indiv_rows)
        write_rows_csv(individuality_dir / "self_vs_other_pairwise.csv", indiv_pairwise)
        write_rows_csv(individuality_dir / "self_vs_other_summary.csv", [indiv_summary])
        (
            signal_rows, signal_summary, template_rows, template_summary,
            residual_rows, residual_pairwise, residual_summary,
        ) = template_residual_analysis_rows(subjects, true_maps, pred_maps, left_idx, right_idx, group, model_name, training_template=template, template_metadata=metric_provenance(metadata))
        if test_loo_sensitivity:
            from .template_sensitivity import write_loo_sensitivity
            write_loo_sensitivity(individuality_dir, subjects, true_maps, pred_maps, left_idx, right_idx, group, model_name)
        write_rows_csv(individuality_dir / "signal_recovery.csv", signal_rows)
        write_rows_csv(individuality_dir / "signal_recovery_summary.csv", [signal_summary])
        write_rows_csv(individuality_dir / "training_template_per_subject_metrics.csv", template_rows)
        write_rows_csv(individuality_dir / "training_template_summary.csv", [template_summary])
        write_rows_csv(individuality_dir / "residual_self_vs_other.csv", residual_rows)
        write_rows_csv(individuality_dir / "residual_self_vs_other_pairwise.csv", residual_pairwise)
        write_rows_csv(individuality_dir / "residual_self_vs_other_summary.csv", [residual_summary])
    return rows, summary, pred_records


def _model_dims_from_matrix(matrix: dict, args=None) -> tuple[int, int]:
    """Infer model input dimensions without materializing disk-backed shards.

    ``matrix["x"]`` may be a ShardedImageArray. Calling ``np.asarray`` on it
    stacks all subjects into RAM; for 360 controls that is ~148 GiB. Only use
    its lightweight ``shape`` metadata here.
    """
    x_shape = tuple(int(v) for v in matrix["x"].shape)
    y_shape = tuple(int(v) for v in matrix["y"].shape)
    rep = str(matrix.get("representation", getattr(args, "representation", "voxel") if args is not None else "voxel"))
    if rep.startswith("voxel") or len(x_shape) == 5:
        vol_shape = tuple(matrix.get("volume_shape", y_shape[-3:]))
        n_tokens = int(np.prod(vol_shape))
        input_dim = int(x_shape[1])
    else:
        n_tokens = int(x_shape[1])
        input_dim = int(x_shape[2])
    return n_tokens, input_dim


def train_one_model(name: str, train_ds, eval_ds, test_ds, tumor_ds, matrix: dict, args, out_dir: Path):
    device = torch.device(args.device)
    model_tokens, input_dim = _model_dims_from_matrix(matrix, args)
    model = build_comparison_model(name, model_tokens, input_dim, args).to(device)
    model = maybe_data_parallel(model, device, disabled=bool(args.no_data_parallel), context=f"comparison:{name}")
    if is_main_process():
        print(f"[comparison] model={name} params={count_parameters(unwrap_model(model))} ddp={distributed_is_initialized()}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_dataloader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    best_score = -1e18
    best_state = None
    history = []
    ddp_training = distributed_is_initialized()
    if ddp_training and is_main_process():
        print(f"[comparison] {name}: DDP training active; epoch/final evaluation is deferred to a single-process eval-only pass to avoid rank timeout during long full-volume evaluation.")
    model_dir = out_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(args.epochs)):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        if int(args.lr_step_size) > 0 and epoch > 0 and epoch % int(args.lr_step_size) == 0:
            for g in opt.param_groups:
                g["lr"] = max(float(g["lr"]) * float(args.lr_gamma), float(args.lr_min))
        sums = {}
        n_batches = 0
        for batch in train_loader:
            bd = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            x_in, y_in = bd["x"], bd["y"]
            # v2 full-field: no spatial token sampling before the encoder.
            out = model(x_in, sample_latent=False)
            loss, parts = _loss_for_epoch(out, y_in, epoch + 1, args)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=False)
            opt.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n_batches += 1
        eval_every = max(1, int(args.eval_every))
        eval_this_epoch = False if ddp_training else (((epoch + 1) % eval_every == 0) or ((epoch + 1) == int(args.epochs)))
        if eval_this_epoch and is_main_process():
            rows_eval, summ_val, _ = _eval_model(model, eval_ds, matrix["left_idx"], matrix["right_idx"], device, "controls_train_eval" if args.no_val_split or args.val_frac <= 0 else "controls_val", name, args.batch_size, spatial_chunk_size=0)
            score = float(summ_val.get("pearson_r_mean", -1e9)) + args.score_r2_weight * float(summ_val.get("r2_mean", 0.0)) + args.score_dice25_weight * float(summ_val.get("dice25_mean", 0.0))
        else:
            rows_eval, summ_val = [], {}
            score = float("nan")
        barrier()
        row = {"model": name, "epoch": epoch + 1, "lr": float(opt.param_groups[0]["lr"]), "valid_batches": n_batches, "selection_score": score, "eval_ran": int(eval_this_epoch), "eval_every": int(eval_every)}
        row.update({f"train_{k}": reduce_mean_scalar(v / max(n_batches, 1)) for k, v in sums.items()})
        n_batches_global = int(round(reduce_mean_scalar(n_batches)))
        row["valid_batches"] = n_batches_global
        row.update(summ_val)
        if is_main_process():
            history.append(row)
        if eval_this_epoch and is_main_process() and score > best_score:
            best_score = score
            best_state = model_state_dict_cpu(model)
            save_checkpoint(model_dir / "best_model.pt", {"model_name": name, "model_state_dict": best_state, "params": count_parameters(unwrap_model(model)), "args": vars(args), "best_summary": summ_val})
        if is_main_process():
            if eval_this_epoch:
                print(f"[comparison] {name} epoch={epoch+1:03d} train={row.get('train_total', 0.0):.4f} eval_r={summ_val.get('pearson_r_mean', float('nan')):.3f} eval_r2={summ_val.get('r2_mean', float('nan')):.3f} eval_dice25={summ_val.get('dice25_mean', float('nan')):.3f} score={score:.4f}")
            else:
                print(f"[comparison] {name} epoch={epoch+1:03d} train={row.get('train_total', 0.0):.4f} eval=skipped")
            write_rows_csv(model_dir / "training_history.csv", history)

    if ddp_training:
        # Do not run long full-volume evaluation inside a DDP training job. Nonzero
        # ranks would sit in a collective/barrier while rank 0 evaluates and can hit
        # the NCCL watchdog timeout. Save the final checkpoint on rank 0, then let the
        # launcher run a normal single-process --eval-only pass.
        if is_main_process():
            save_checkpoint(model_dir / "final_model.pt", {"model_name": name, "model_state_dict": model_state_dict_cpu(model), "params": count_parameters(unwrap_model(model)), "args": vars(args), "final_epoch": int(args.epochs), "evaluation_deferred": True})
        barrier()
        return [], []
    if not is_main_process():
        barrier()
        return [], []
    if best_state is not None and not (bool(args.no_val_split) or float(args.val_frac) <= 0.0):
        restore_state_dict_(model, best_state)
    save_checkpoint(model_dir / "final_model.pt", {"model_name": name, "model_state_dict": model_state_dict_cpu(model), "params": count_parameters(unwrap_model(model)), "args": vars(args), "final_epoch": int(args.epochs)})
    all_rows = []
    summaries = []
    eval_groups = set(args.eval_groups or ["controls"])
    if "controls" in eval_groups:
        if test_ds is None:
            raise RuntimeError("eval_groups includes controls but control test dataset was not loaded")
        rows, summ, _ = _eval_model(model, test_ds, matrix["left_idx"], matrix["right_idx"], device, "controls", name, args.batch_size, spatial_chunk_size=0, individuality_dir=model_dir / "individuality" / name / "controls", training_template_path=args.training_template, template_paths=ClinicalPaths(data_dir=args.data_dir, beta_pattern=args.beta_pattern), test_loo_sensitivity=args.test_loo_sensitivity)
        all_rows.extend(rows); summaries.append(summ)
    if "tumor" in eval_groups:
        if tumor_ds is None:
            raise RuntimeError("eval_groups includes tumor but tumor dataset was not loaded; pass --include-tumors")
        rows, summ, _ = _eval_model(model, tumor_ds, matrix["left_idx"], matrix["right_idx"], device, "tumor", f"{name}_zero_shot", args.batch_size, spatial_chunk_size=0, individuality_dir=model_dir / "individuality" / f"{name}_zero_shot" / "tumor", training_template_path=args.training_template, template_paths=ClinicalPaths(data_dir=args.data_dir, beta_pattern=args.beta_pattern), test_loo_sensitivity=args.test_loo_sensitivity)
        all_rows.extend(rows); summaries.append(summ)
    write_rows_csv(model_dir / "per_subject_metrics.csv", all_rows)
    summary_path = model_dir / "summary.csv"
    merged_summaries = _merge_group_model_rows(summaries, summary_path)
    write_rows_csv(summary_path, merged_summaries)
    barrier()
    return all_rows, summaries


def _load_comparison_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison checkpoint: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _apply_saved_architecture_args(args, saved_args: dict | None):
    # Evaluation-only should not require the user to remember the exact comparator
    # width/depth/dropout used during training. Keep current data/runtime paths,
    # but recover architecture-defining fields from the checkpoint when present.
    if not saved_args:
        return args
    arch_fields = [
        "dropout", "surf_width", "surf_depth", "swifun_dim", "swifun_heads",
        "swifun_depth", "swifun_window", "representation", "spatial_token_batch_size", "full_field_stride", "full_res_refine_layers", "brainsurf_full_field_stride", "swifun_full_field_stride", "beta_clip", "head_init_std",
    ]
    for key in arch_fields:
        if key in saved_args:
            setattr(args, key, saved_args[key])
    return args


def evaluate_saved_model(name: str, test_ds, tumor_ds, matrix: dict, args, out_dir: Path):
    device = torch.device(args.device)
    model_dir = out_dir / name
    ckpt_name = "best_model.pt" if args.checkpoint_kind == "best" else "final_model.pt"
    ckpt_path = model_dir / ckpt_name
    ckpt = _load_comparison_checkpoint(ckpt_path, device)
    _apply_saved_architecture_args(args, ckpt.get("args") if isinstance(ckpt, dict) else None)
    model_tokens, input_dim = _model_dims_from_matrix(matrix, args)
    model = build_comparison_model(name, model_tokens, input_dim, args).to(device)
    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else ckpt
    if state is None:
        raise RuntimeError(f"Checkpoint did not contain model_state_dict: {ckpt_path}")
    load_state_dict_flexible_(model, state, strict=False)
    model = maybe_data_parallel(model, device, disabled=bool(args.no_data_parallel), context=f"comparison_eval:{name}")
    print(f"[comparison] eval-only model={name} checkpoint={ckpt_path} ddp={distributed_is_initialized()}")

    all_rows = []
    summaries = []
    eval_groups = set(args.eval_groups or ["controls"])
    if "controls" in eval_groups:
        if test_ds is None:
            raise RuntimeError("eval_groups includes controls but control test dataset was not loaded")
        model_name_controls = ablated_model_name(name, args.input_ablation, add_suffix=not bool(args.no_ablation_name_suffix))
        eval_ds_controls = make_eval_dataset(test_ds.matrix, test_ds.indices, mode=args.input_ablation, seed=args.ablation_seed)
        rows, summ, _ = _eval_model(model, eval_ds_controls, matrix["left_idx"], matrix["right_idx"], device, "controls", model_name_controls, args.batch_size, spatial_chunk_size=0, individuality_dir=model_dir / "individuality" / model_name_controls / "controls", training_template_path=args.training_template, template_paths=ClinicalPaths(data_dir=args.data_dir, beta_pattern=args.beta_pattern), test_loo_sensitivity=args.test_loo_sensitivity)
        all_rows.extend(rows); summaries.append(summ)
    if "tumor" in eval_groups:
        if tumor_ds is None:
            raise RuntimeError("eval_groups includes tumor but tumor dataset was not loaded")
        model_name_tumor = ablated_model_name(f"{name}_zero_shot", args.input_ablation, add_suffix=not bool(args.no_ablation_name_suffix))
        eval_ds_tumor = make_eval_dataset(tumor_ds.matrix, tumor_ds.indices, mode=args.input_ablation, seed=args.ablation_seed)
        rows, summ, _ = _eval_model(model, eval_ds_tumor, matrix["left_idx"], matrix["right_idx"], device, "tumor", model_name_tumor, args.batch_size, spatial_chunk_size=0, individuality_dir=model_dir / "individuality" / model_name_tumor / "tumor", training_template_path=args.training_template, template_paths=ClinicalPaths(data_dir=args.data_dir, beta_pattern=args.beta_pattern), test_loo_sensitivity=args.test_loo_sensitivity)
        all_rows.extend(rows); summaries.append(summ)
    write_rows_csv(model_dir / "per_subject_metrics.csv", all_rows)
    summary_path = model_dir / "summary.csv"
    merged_summaries = _merge_group_model_rows(summaries, summary_path)
    write_rows_csv(summary_path, merged_summaries)
    barrier()
    return all_rows, summaries


def _save_comparison_adapter_checkpoint(path: Path, model_name: str, model: nn.Module, args, base_checkpoint: Path,
                                        target: str, train_subjects: List[str], val_subjects: List[str], summary: dict):
    save_checkpoint(path, {
        "model_name": model_name,
        "model_state_dict": model_state_dict_cpu(model),
        "params": count_parameters(unwrap_model(model)),
        "args": vars(args),
        "base_checkpoint": str(base_checkpoint),
        "target_tumor": target,
        "train_subjects": train_subjects,
        "inner_val_subjects": val_subjects,
        "best_inner_summary": summary,
    })


def _safe_split_train_val(indices: np.ndarray, val_frac: float, seed: int, target_i: int):
    indices = np.asarray(indices, dtype=int)
    if indices.size <= 2 or float(val_frac) <= 0.0:
        return indices, np.asarray([], dtype=int)
    rng = np.random.default_rng(int(seed) + int(target_i))
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(indices.size * float(val_frac))))
    n_val = min(n_val, max(1, indices.size - 1))
    val_idx = np.sort(shuffled[:n_val])
    train_idx = np.sort(shuffled[n_val:])
    if train_idx.size == 0:
        train_idx = indices
    return train_idx, val_idx


def adapt_saved_model_to_tumors(name: str, tumor_ds_all: ImageSubjectDataset, tumor_mat: dict, matrix: dict, args, out_dir: Path):
    """Leave-one-tumor-out fine-tuning for neural comparator models.

    This starts from the control-trained comparator checkpoint, fine-tunes on N-1 tumor
    patients, evaluates the held-out tumor patient, and repeats for all tumor patients.
    It is intentionally analogous to the VRPTR tumor_adapter path.
    """
    device = torch.device(args.device)
    model_dir = out_dir / name
    adapter_dir = model_dir / "tumor_adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = "best_model.pt" if args.checkpoint_kind == "best" else "final_model.pt"
    ckpt_path = model_dir / ckpt_name
    ckpt = _load_comparison_checkpoint(ckpt_path, device)
    _apply_saved_architecture_args(args, ckpt.get("args") if isinstance(ckpt, dict) else None)
    base_state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else ckpt
    if base_state is None:
        raise RuntimeError(f"Checkpoint did not contain model_state_dict: {ckpt_path}")
    base_state = {k: v.detach().cpu().clone() for k, v in base_state.items()}

    subjects = list(tumor_mat["subjects"])
    all_rows: List[dict] = []
    summaries_per_target: List[dict] = []
    history_all: List[dict] = []
    adapted_pred_records: List[tuple[str, np.ndarray, np.ndarray]] = []
    model_name = ablated_model_name(f"{name}_tumor_adapted", args.input_ablation, add_suffix=not bool(args.no_ablation_name_suffix))
    ddp_training = distributed_is_initialized()

    for target_i, target in enumerate(subjects):
        target_ckpt = adapter_dir / f"tumor_{target}.pt"
        train_all = np.asarray([i for i in range(len(subjects)) if i != target_i], dtype=int)
        if train_all.size == 0:
            print(f"[comparison][adapt] target={target}: no other tumor subjects; evaluating control checkpoint only")
            train_idx = val_idx = train_all
        else:
            train_idx, val_idx = _safe_split_train_val(train_all, args.adapter_inner_val_frac, args.seed, target_i)

        model_tokens, input_dim = _model_dims_from_matrix(matrix, args)
        model = build_comparison_model(name, model_tokens, input_dim, args).to(device)
        load_state_dict_flexible_(model, base_state, strict=False)
        mode = "head" if bool(getattr(args, "adapter_head_only", False)) else str(getattr(args, "adapter_mode", "decoder"))
        needs_training = (not (target_ckpt.exists() and not args.overwrite_adapters)) and train_all.size > 0 and int(args.adapter_epochs) > 0
        if needs_training:
            # Critical for DDP decoder-only adaptation: freeze the encoder before wrapping.
            # DDP only tracks parameters with requires_grad=True at construction time.
            set_comparison_adapter_tuning(model, mode=mode)
        model = maybe_data_parallel(model, device, disabled=bool(args.no_data_parallel), context=f"comparison_adapter:{name}:{target}")

        if is_main_process():
            print(f"[comparison][adapt] model={name} target={target} mode={mode} ddp={ddp_training} trainable={count_parameters(unwrap_model(model))}")

        if target_ckpt.exists() and not args.overwrite_adapters:
            if is_main_process():
                print(f"[comparison][adapt] exists, loading {target_ckpt}")
            saved = _load_comparison_checkpoint(target_ckpt, device)
            state = saved.get("model_state_dict") if isinstance(saved, dict) else saved
            load_state_dict_flexible_(model, state, strict=False)
            best_summary = saved.get("best_inner_summary", {}) if isinstance(saved, dict) else {}
        elif bool(getattr(args, "require_existing_adapters", False)) and int(args.adapter_epochs) <= 0:
            raise FileNotFoundError(
                f"Required existing adapter checkpoint is missing for {name} target={target}: {target_ckpt}. "
                "Run with --train tumor_adapter to create adapters, or disable --require-existing-adapters."
            )
        elif train_all.size > 0 and int(args.adapter_epochs) > 0:
            # Memory-safe full-volume tumor adaptation. The trainable subset was
            # selected before DDP wrapping so DDP sees only parameters expected to
            # receive gradients.
            train_ds = ImageSubjectDataset(tumor_mat, train_idx)
            use_inner_val = val_idx.size > 0
            val_ds = ImageSubjectDataset(tumor_mat, val_idx) if use_inner_val else None
            train_loader = make_dataloader(train_ds, batch_size=args.adapter_batch_size, shuffle=True, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.adapter_lr, weight_decay=args.adapter_weight_decay)
            best_score = -1e18
            best_state = model_state_dict_cpu(model)
            best_summary = {"note": "fixed_schedule_no_inner_val"} if not use_inner_val else {}
            patience_left = int(args.adapter_patience)
            final_epoch = 0
            for epoch in range(int(args.adapter_epochs)):
                final_epoch = epoch + 1
                if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)
                model.train()
                sums: Dict[str, float] = {}
                nb = 0
                for batch in train_loader:
                    bd = to_device(batch, device)
                    opt.zero_grad(set_to_none=True)
                    x_in, y_in = bd["x"], bd["y"]
                    # v2 full-field: no spatial token sampling before the encoder.
                    out = model(x_in, sample_latent=False)
                    # Use the same map-reconstruction objective as clean comparator training,
                    # but keep contrastive off by default unless adapter_map_only_epochs is lower.
                    loss, parts = _loss_for_epoch(out, y_in, epoch + 1, args)
                    if not torch.isfinite(loss):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.adapter_grad_clip, error_if_nonfinite=False)
                    opt.step()
                    for k, v in parts.items():
                        sums[k] = sums.get(k, 0.0) + float(v)
                    nb += 1
                eval_every = max(1, int(args.adapter_eval_every))
                run_eval = use_inner_val and (not ddp_training) and (((epoch + 1) % eval_every == 0) or ((epoch + 1) == int(args.adapter_epochs)) or epoch == 0)
                row = {"model": name, "target": target, "epoch": epoch + 1, "valid_batches": nb, "lr": float(opt.param_groups[0]["lr"])}
                row.update({f"train_{k}": v / max(nb, 1) for k, v in sums.items()})
                if run_eval:
                    _, summ_val, _ = _eval_model(model, val_ds, matrix["left_idx"], matrix["right_idx"], device, "tumor_inner_val", model_name, args.adapter_batch_size, spatial_chunk_size=0)
                    score = float(summ_val.get("pearson_r_mean", -1e9)) + float(args.score_r2_weight) * float(summ_val.get("r2_mean", 0.0)) + float(args.score_dice25_weight) * float(summ_val.get("dice25_mean", 0.0)) - float(args.adapter_score_li_weight) * float(summ_val.get("li_abs_err_mean", 0.0))
                    row.update({"selection_score": score, **summ_val})
                    if np.isfinite(score) and score > best_score + float(args.adapter_min_delta):
                        best_score = float(score)
                        best_state = model_state_dict_cpu(model)
                        best_summary = summ_val
                        patience_left = int(args.adapter_patience)
                    else:
                        patience_left -= 1
                    print(f"[comparison][adapt] {name} target={target} epoch={epoch+1:03d} train={row.get('train_total', 0.0):.4f} val_r={summ_val.get('pearson_r_mean', float('nan')):.3f} val_r2={summ_val.get('r2_mean', float('nan')):.3f} score={score:.4f}")
                history_all.append(row)
                if use_inner_val and int(args.adapter_patience) > 0 and patience_left <= 0:
                    print(f"[comparison][adapt] {name} target={target}: early stopping at epoch={epoch+1:03d}")
                    break
            if ddp_training:
                best_state = model_state_dict_cpu(model)
                best_summary = {"note": "DDP adapter train-only; adapted evaluation deferred"}
            elif not use_inner_val:
                best_state = model_state_dict_cpu(model)
                best_summary = {"note": "fixed_schedule_no_inner_val", "final_epoch": int(final_epoch)}
            restore_state_dict_(model, best_state)
            if is_main_process():
                _save_comparison_adapter_checkpoint(
                    target_ckpt, name, model, args, ckpt_path, target,
                    [subjects[int(i)] for i in train_idx], [subjects[int(i)] for i in val_idx], best_summary,
                )
            barrier()
        else:
            best_summary = {}

        if ddp_training:
            barrier()
            continue
        target_ds = make_eval_dataset(tumor_mat, np.asarray([target_i], dtype=int), mode=args.input_ablation, seed=args.ablation_seed)
        rows, summ, pr = _eval_model(model, target_ds, matrix["left_idx"], matrix["right_idx"], device, "tumor", model_name, 1, spatial_chunk_size=0)
        all_rows.extend(rows)
        adapted_pred_records.extend(pr)
        summaries_per_target.append({"target": target, **summ})

    if ddp_training:
        if is_main_process():
            print(f"[comparison][adapt] DDP adapter training finished for {name}; adapted evaluation is deferred to single-process eval.")
        return [], []

    summary = summarize_metric_rows(all_rows, group="tumor", model=model_name)
    write_rows_csv(model_dir / "tumor_adapted_per_subject_metrics.csv", all_rows)
    write_rows_csv(model_dir / "tumor_adapted_summary.csv", [summary])
    # Also place the adapted result under predictions/ so compile_results includes it naturally.
    pred_dir = out_dir.parent / "predictions" / model_name / "tumor"
    write_rows_csv(pred_dir / "per_subject_metrics.csv", all_rows)
    write_rows_csv(pred_dir / "summary.csv", [summary])
    if adapted_pred_records:
        subjects_indiv = [r[0] for r in adapted_pred_records]
        true_maps = [r[1] for r in adapted_pred_records]
        pred_maps = [r[2] for r in adapted_pred_records]
        indiv_rows, indiv_pairwise, indiv_summary = self_vs_other_rows(subjects_indiv, true_maps, pred_maps, "tumor", model_name)
        write_rows_csv(pred_dir / "self_vs_other.csv", indiv_rows)
        write_rows_csv(pred_dir / "self_vs_other_pairwise.csv", indiv_pairwise)
        write_rows_csv(pred_dir / "self_vs_other_summary.csv", [indiv_summary])
        write_rows_csv(model_dir / "individuality" / model_name / "tumor" / "self_vs_other.csv", indiv_rows)
        write_rows_csv(model_dir / "individuality" / model_name / "tumor" / "self_vs_other_pairwise.csv", indiv_pairwise)
        write_rows_csv(model_dir / "individuality" / model_name / "tumor" / "self_vs_other_summary.csv", [indiv_summary])
        template, metadata = load_for_evaluation(args.training_template, subjects_indiv, ClinicalPaths(data_dir=args.data_dir, beta_pattern=args.beta_pattern), true_maps[0].shape)
        (
            signal_rows, signal_summary, template_rows, template_summary,
            residual_rows, residual_pairwise, residual_summary,
        ) = template_residual_analysis_rows(subjects_indiv, true_maps, pred_maps, matrix["left_idx"], matrix["right_idx"], "tumor", model_name,
                training_template=template, template_metadata=metric_provenance(metadata))
        for base in [pred_dir, model_dir / "individuality" / model_name / "tumor"]:
            write_rows_csv(base / "signal_recovery.csv", signal_rows)
            write_rows_csv(base / "signal_recovery_summary.csv", [signal_summary])
            write_rows_csv(base / "training_template_per_subject_metrics.csv", template_rows)
            write_rows_csv(base / "training_template_summary.csv", [template_summary])
            write_rows_csv(base / "residual_self_vs_other.csv", residual_rows)
            write_rows_csv(base / "residual_self_vs_other_pairwise.csv", residual_pairwise)
            write_rows_csv(base / "residual_self_vs_other_summary.csv", [residual_summary])
    write_rows_csv(adapter_dir / "adapter_inner_val_summary.csv", summaries_per_target)
    write_rows_csv(adapter_dir / "adapter_training_history.csv", history_all)
    print(f"[comparison][adapt] wrote {pred_dir / 'summary.csv'}")
    return all_rows, [summary]


def parse_args():
    p = argparse.ArgumentParser(description="Train BrainSurfCNN-like and SOTA-lite neural comparators on cached clinical/HCP token matrices.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--train-controls", default="splits/train_subjects.txt")
    p.add_argument("--test-controls", default="splits/test_subjects.txt")
    p.add_argument("--test-tumors", default="splits/test_tumor.txt")
    p.add_argument("--out-dir", default="results/comparisons")
    p.add_argument("--fingerprint-pattern", default="output/rest_connectivity_100/fingerprint_100.nii.gz")
    p.add_argument("--beta-pattern", default="output/task_betas/beta_story_vs_math.nii.gz")
    p.add_argument("--lesion-pattern", default="")
    p.add_argument("--n-rois", type=int, default=100)
    p.add_argument("--yeo-networks", type=int, default=17)
    p.add_argument("--resolution-mm", type=int, default=2)
    p.add_argument("--representation", choices=["voxel"], default="voxel", help="Current v2 image representation only: x=[B,100,X,Y,Z], y=[B,1,X,Y,Z].")
    p.add_argument("--token-subdivisions", type=int, default=1)
    p.add_argument("--spatial-token-batch-size", type=int, default=0, help="Legacy/debug only. v2 full-field neural models ignore token sampling and feed the full fingerprint field through the encoder.")
    p.add_argument("--eval-spatial-chunk-size", type=int, default=0, help="Legacy/debug only. v2 full-field neural models ignore chunking.")
    p.add_argument("--full-field-stride", type=int, default=2, help="Compress the full spatial field by this stride before expensive comparator/VRPTR bottleneck blocks. Encoder still sees the full field through the stem/skip path.")
    p.add_argument("--full-res-refine-layers", type=int, default=0, help="Optional shallow full-resolution refinement layers after decoding. Keep 0-1 for large voxel fields.")
    p.add_argument("--brainsurf-full-field-stride", type=int, default=2, help="Minimum compression stride for BrainSurfCNN-like bottleneck in voxel/full-field mode.")
    p.add_argument("--swifun-full-field-stride", type=int, default=2, help="Minimum compression stride for SWIFUN-like bottleneck in voxel/full-field mode; use larger values to avoid long CUDA kernels/timeouts.")
    p.add_argument("--beta-clip", type=float, default=8.0, help="Clamp predicted beta values during comparator training to avoid early full-field explosions.")
    p.add_argument("--head-init-std", type=float, default=1e-4, help="Small init std for comparator beta heads in full-field mode.")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--models", nargs="+", default=["brainsurfcnn_lite", "swifun_lite"])
    p.add_argument("--include-tumors", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="Load saved comparator checkpoints and evaluate without retraining.")
    p.add_argument("--eval-groups", nargs="+", choices=["controls", "tumor"], default=["controls"], help="Groups to evaluate in --eval-only mode.")
    p.add_argument("--checkpoint-kind", choices=["final", "best"], default="final", help="Comparator checkpoint to load for --eval-only.")
    p.add_argument("--adapt-tumor", action="store_true", help="After loading control-trained comparator checkpoints, run leave-one-tumor-out fine-tuning and evaluate adapted tumor predictions.")
    p.add_argument("--adapt-only", action="store_true", help="Run adapter training only. In DDP this saves adapter checkpoints and defers adapted evaluation to a single-process call.")
    p.add_argument("--adapter-epochs", type=int, default=80)
    p.add_argument("--adapter-batch-size", type=int, default=4)
    p.add_argument("--adapter-lr", type=float, default=5e-5)
    p.add_argument("--adapter-weight-decay", type=float, default=1e-4)
    p.add_argument("--adapter-inner-val-frac", type=float, default=0.0)
    p.add_argument("--adapter-eval-every", type=int, default=10)
    p.add_argument("--adapter-patience", type=int, default=20)
    p.add_argument("--adapter-min-delta", type=float, default=1e-4)
    p.add_argument("--adapter-grad-clip", type=float, default=0.5)
    p.add_argument("--adapter-score-li-weight", type=float, default=0.10)
    p.add_argument("--adapter-mode", choices=["full", "bottleneck_decoder", "decoder", "head"], default="decoder", help="Trainable comparator parameters during tumor adaptation. Default freezes the encoder and adapts decoder/heads.")
    p.add_argument("--adapter-head-only", action="store_true", help="Backward-compatible alias for --adapter-mode head.")
    p.add_argument("--overwrite-adapters", action="store_true", help="Recreate existing comparator tumor adapter checkpoints.")
    p.add_argument("--require-existing-adapters", action="store_true", help="When adapter_epochs <= 0, fail instead of silently evaluating the control checkpoint if a tumor adapter is missing.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--no-val-split", action="store_true", help="Train comparators on all training controls; epoch metrics are train-set diagnostics and final checkpoint is evaluated on held-out controls.")
    p.add_argument("--eval-every", type=int, default=20, help="Run epoch-level diagnostic evaluation every N epochs. Final epoch is always evaluated.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--map-only-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-step-size", type=int, default=0)
    p.add_argument("--lr-gamma", type=float, default=1.0)
    p.add_argument("--lr-min", type=float, default=1e-7)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--surf-width", type=int, default=32)
    p.add_argument("--surf-depth", type=int, default=4)
    p.add_argument("--swifun-dim", type=int, default=32)
    p.add_argument("--swifun-heads", type=int, default=4)
    p.add_argument("--swifun-depth", type=int, default=4)
    p.add_argument("--swifun-window", type=int, default=4, help="3D shifted-window size for the SwiFUN-like bottleneck attention.")
    p.add_argument("--w-mse", type=float, default=1.0)
    p.add_argument("--w-r2", type=float, default=0.0)
    p.add_argument("--r2-loss-clip", type=float, default=25.0, help="Clip the R2-loss component so early full-field prediction explosions do not dominate training.")
    p.add_argument("--w-calibration", type=float, default=0.0)
    p.add_argument("--w-distribution", type=float, default=0.0)
    p.add_argument("--w-centered", type=float, default=0.0)
    p.add_argument("--w-topk", type=float, default=0.0)
    p.add_argument("--topk-weight", type=float, default=5.0)
    p.add_argument("--w-multires", type=float, default=0.0)
    p.add_argument("--w-corr", type=float, default=0.0)
    p.add_argument("--w-contrastive", type=float, default=0.08)
    p.add_argument("--contrastive-temperature", type=float, default=0.20)
    p.add_argument("--contrastive-mse-weight", type=float, default=0.15)
    p.add_argument("--score-r2-weight", type=float, default=1.75)
    p.add_argument("--score-dice25-weight", type=float, default=0.35)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-data-parallel", action="store_true", help="Compatibility no-op; multi-GPU training uses torchrun/DDP.")
    add_template_argument(p)
    p.add_argument("--test-loo-sensitivity", action="store_true")
    add_input_ablation_args(p)
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per process for sharded image cache.")
    p.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when num-workers > 0.")
    return p.parse_args()


def main():
    args = parse_args()
    # Restore the original training split before reconstructing the input scaler.
    # Legacy comparator checkpoints store training args but no x_scaler.
    if args.eval_only:
        split_settings = []
        for name in args.models:
            checkpoint_name = "best_model.pt" if args.checkpoint_kind == "best" else "final_model.pt"
            saved = _load_comparison_checkpoint(Path(args.out_dir) / name / checkpoint_name, torch.device("cpu"))
            training_args = saved.get("args", {})
            required = ("no_val_split", "val_frac", "seed")
            if any(key not in training_args for key in required):
                raise ValueError("Comparator checkpoint lacks scaler-training split provenance")
            split_settings.append(tuple(training_args[key] for key in required))
        if len(set(split_settings)) != 1:
            raise ValueError("Comparator training splits differ; evaluate one model per command")
        args.no_val_split, args.val_frac, args.seed = split_settings[0]
        print(f"[comparison] restored training split for scaler: no_val_split={args.no_val_split}, val_frac={args.val_frac}, seed={args.seed}")
    device = setup_distributed_from_env(args.device)
    args.device = str(device)
    set_seed(int(args.seed) + get_rank())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = ClinicalPaths(
        data_dir=args.data_dir,
        fingerprint_pattern=args.fingerprint_pattern,
        beta_pattern=args.beta_pattern,
        lesion_pattern=args.lesion_pattern,
        n_rois=args.n_rois,
        yeo_networks=args.yeo_networks,
        resolution_mm=args.resolution_mm,
        representation=args.representation,
        token_subdivisions=args.token_subdivisions,
    )
    atlas = create_image_space(paths)
    cache_dir = args.cache_dir or str(out_dir.parent / "image_cache")
    raw_train = build_subject_matrix_cached(load_subject_list(args.train_controls), paths, atlas, cache_dir=cache_dir, cache_tag="train_controls", refresh_cache=args.refresh_cache)
    if bool(args.adapt_only) and not bool(args.adapt_tumor):
        raise RuntimeError("--adapt-only requires --adapt-tumor")
    eval_group_set = set([] if bool(args.adapt_only) else (args.eval_groups or ["controls"]))
    ddp_train_only = distributed_is_initialized() and (not bool(args.eval_only))
    if ddp_train_only:
        # In torchrun/DDP mode this module must be training-only.  Do not even
        # load/build held-out control or tumor caches here: rank 0 can spend many
        # minutes extracting them while rank 1 reaches DDP(model) and times out
        # waiting for NCCL communicator setup.  The launcher runs a separate
        # single-process --eval-only pass after DDP training to build/load eval
        # caches and write held-out summaries.
        need_controls = False
        need_tumor = False
        if is_main_process():
            print("[comparison] DDP train-only mode: skipping held-out control/tumor cache loading; evaluation is deferred outside torchrun.")
    else:
        if bool(args.adapt_only):
            need_controls = False
            need_tumor = True
        else:
            need_controls = (not bool(args.eval_only)) or ("controls" in eval_group_set)
            need_tumor = bool(args.include_tumors) or bool(args.adapt_tumor) or (bool(args.eval_only) and "tumor" in eval_group_set)
    raw_test = None
    if need_controls:
        raw_test = build_subject_matrix_cached(load_subject_list(args.test_controls), paths, atlas, cache_dir=cache_dir, cache_tag="test_controls", refresh_cache=args.refresh_cache)
        require_complete_subjects(load_subject_list(args.test_controls), raw_test["subjects"])
    raw_tumor = None
    if need_tumor:
        raw_tumor = build_subject_matrix_cached(load_subject_list(args.test_tumors), paths, atlas, cache_dir=cache_dir, cache_tag="test_tumors", refresh_cache=args.refresh_cache)
    if bool(args.no_val_split) or float(args.val_frac) <= 0.0:
        train_idx = np.arange(len(raw_train["subjects"]), dtype=int)
        val_idx = np.asarray([], dtype=int)
        eval_idx = train_idx
        print("[comparison] no validation split: training neural comparators on all training controls; epoch metrics are train-set diagnostics only.")
    else:
        train_idx, val_idx = split_indices(len(raw_train["subjects"]), args.val_frac, args.seed)
        eval_idx = val_idx
    is_disk = bool(raw_train.get("disk_backed", False)) or hasattr(raw_train.get("x", None), "filename")
    if is_disk and hasattr(Standardizer, "fit_indices"):
        scaler = Standardizer(exclude_last_n=0).fit_indices_masked(raw_train["x"], raw_train["y"], train_idx)
        train_mat = dict(raw_train); train_mat["x_scaler"] = scaler
        test_mat = None
        if raw_test is not None:
            test_mat = dict(raw_test); test_mat["x_scaler"] = scaler
        tumor_mat = None
        if raw_tumor is not None:
            tumor_mat = dict(raw_tumor); tumor_mat["x_scaler"] = scaler
    else:
        scaler = Standardizer(exclude_last_n=0).fit(raw_train["x"][train_idx])
        train_mat = dict(raw_train); train_mat["x"] = scaler.transform(raw_train["x"])
        test_mat = None
        if raw_test is not None:
            test_mat = dict(raw_test); test_mat["x"] = scaler.transform(raw_test["x"])
        tumor_mat = None
        if raw_tumor is not None:
            tumor_mat = dict(raw_tumor); tumor_mat["x"] = scaler.transform(raw_tumor["x"])
    train_ds = ImageSubjectDataset(train_mat, train_idx)
    eval_ds = ImageSubjectDataset(train_mat, eval_idx)
    test_ds = ImageSubjectDataset(test_mat) if test_mat is not None else None
    tumor_ds = ImageSubjectDataset(tumor_mat) if tumor_mat is not None else None

    all_rows: List[dict] = []
    all_summaries: List[dict] = []
    for name in args.models:
        if bool(args.adapt_only):
            rows, summaries = [], []
        elif args.eval_only:
            rows, summaries = evaluate_saved_model(name, test_ds, tumor_ds, train_mat, args, out_dir)
        else:
            rows, summaries = train_one_model(name, train_ds, eval_ds, test_ds, tumor_ds, train_mat, args, out_dir)
        all_rows.extend(rows)
        all_summaries.extend(summaries)
        if args.adapt_tumor:
            if tumor_ds is None or tumor_mat is None:
                raise RuntimeError("--adapt-tumor requires --test-tumors and a loadable tumor dataset")
            rows_adapt, summaries_adapt = adapt_saved_model_to_tumors(name, tumor_ds, tumor_mat, train_mat, args, out_dir)
            if is_main_process():
                all_rows.extend(rows_adapt)
                all_summaries.extend(summaries_adapt)
            barrier()
    if is_main_process():
        write_rows_csv(out_dir / "comparison_per_subject_metrics.csv", all_rows)
        comparison_summary_path = out_dir / "comparison_summary.csv"
        write_rows_csv(comparison_summary_path, _merge_group_model_rows(all_summaries, comparison_summary_path))
        mode = "evaluated" if args.eval_only else "wrote"
        print(f"[comparison] {mode} {out_dir}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
