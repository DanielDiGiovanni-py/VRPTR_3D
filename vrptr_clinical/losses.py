from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist


def _flatten_like_map(x: torch.Tensor, valid_from: torch.Tensor | None = None) -> torch.Tensor:
    if x.ndim > 2:
        return x.reshape(x.shape[0], -1)
    return x


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = _flatten_like_map(pred); t = _flatten_like_map(target)
    mask = torch.isfinite(p) & torch.isfinite(t)
    if not bool(mask.any()):
        return p.new_tensor(0.0)
    return F.smooth_l1_loss(p[mask], t[mask])


def _finite_flat(x: torch.Tensor, ref: torch.Tensor | None = None) -> torch.Tensor:
    z = _flatten_like_map(x)
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def neg_pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = _flatten_like_map(pred); t = _flatten_like_map(target)
    mask = torch.isfinite(p) & torch.isfinite(t)
    p = torch.where(mask, p, torch.zeros_like(p))
    t = torch.where(mask, t, torch.zeros_like(t))
    count = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    pm = p.sum(dim=1, keepdim=True) / count
    tm = t.sum(dim=1, keepdim=True) / count
    p = torch.where(mask, p - pm, torch.zeros_like(p))
    t = torch.where(mask, t - tm, torch.zeros_like(t))
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p * p).sum(dim=1) * (t * t).sum(dim=1)).clamp_min(eps)
    corr = num / den
    return torch.nan_to_num(1.0 - corr.mean(), nan=0.0, posinf=2.0, neginf=0.0)


def _compact_embedding(x: torch.Tensor, max_features: int = 8192) -> torch.Tensor:
    z = _finite_flat(x)
    if z.shape[1] > int(max_features):
        # Deterministic subsampling keeps DDP all-gather small. This is used
        # only for contrastive identity; reconstruction still uses full maps.
        idx = torch.linspace(0, z.shape[1] - 1, steps=int(max_features), device=z.device).long()
        z = z.index_select(1, idx)
    z = z - z.mean(dim=1, keepdim=True)
    z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    world = dist.get_world_size()
    if world <= 1:
        return x
    bufs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(bufs, x.detach().contiguous())
    return torch.cat(bufs, dim=0)


def subject_contrastive_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.20, mse_weight: float = 0.0) -> torch.Tensor:
    # In DDP each process may have local batch size 1. Gather compact target
    # embeddings across ranks so the global batch still supplies negatives.
    p = _compact_embedding(pred)
    t = _compact_embedding(target)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        t_all = _gather_no_grad(t)
        rank = dist.get_rank()
        b = p.shape[0]
        labels = torch.arange(b, device=pred.device) + rank * b
        sim = torch.nan_to_num(p @ t_all.t(), nan=0.0, posinf=0.0, neginf=0.0)
        logits = (sim / max(float(temperature), 1e-3)).clamp(-30.0, 30.0)
        return torch.nan_to_num(F.cross_entropy(logits, labels), nan=0.0, posinf=1e3, neginf=0.0)
    if pred.shape[0] < 2:
        return _flatten_like_map(pred).new_tensor(0.0)
    sim = torch.nan_to_num(p @ t.t(), nan=0.0, posinf=0.0, neginf=0.0)
    if mse_weight and float(mse_weight) > 0:
        # MSE term on compact normalized embeddings; capped for stability.
        mse = (p[:, None, :] - t[None, :, :]).pow(2).mean(dim=-1).clamp(0.0, 25.0)
        sim = sim - float(mse_weight) * mse
    logits = (sim / max(float(temperature), 1e-3)).clamp(-30.0, 30.0)
    labels = torch.arange(pred.shape[0], device=pred.device)
    return torch.nan_to_num(0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)), nan=0.0, posinf=1e3, neginf=0.0)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if mu is None or logvar is None:
        return torch.as_tensor(0.0)
    mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    logvar = torch.nan_to_num(logvar, nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)
    return torch.nan_to_num(-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean(), nan=0.0, posinf=1e3, neginf=0.0)


def clinical_vrptr_loss(
    outputs: Dict[str, torch.Tensor],
    x: torch.Tensor,
    target_beta: torch.Tensor,
    target_li: torch.Tensor,
    target_dom: torch.Tensor,
    left_idx: torch.Tensor,
    right_idx: torch.Tensor,
    fc_dim: int,
    w_mse: float = 1.0,
    w_r2: float = 0.0,
    w_calibration: float = 0.0,
    w_distribution: float = 0.0,
    w_centered: float = 0.0,
    w_topk_mse: float = 0.0,
    topk_mse_weight: float = 4.0,
    w_multires: float = 0.0,
    w_smoothness: float = 0.0,
    w_latent_map_l2: float = 0.0,
    w_corr: float = 0.0,
    w_rank: float = 0.0,
    w_contrastive: float = 0.10,
    contrastive_temperature: float = 0.20,
    contrastive_mse_weight: float = 0.0,
    w_language: float = 0.0,
    language_top_q: float = 0.80,
    w_li: float = 0.0,
    w_cls: float = 0.0,
    w_uncertainty: float = 0.0,
    w_fc: float = 0.0,
    w_kl: float = 0.0,
    w_kl_capacity: float = 0.0,
    kl_capacity: float = 0.0,
    w_latent_cov: float = 0.0,
    w_latent_map_l1: float = 0.0,
    w_latent_map_orth: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Shared BrainSurfCNN-style neural loss.

    All neural models use reconstructive loss + subject-specific contrastive loss.
    VRPTR adds only the KL term when mu/logvar_z are present and w_kl > 0.
    """
    pred = outputs["beta"]
    rec = masked_smooth_l1(pred, target_beta)
    corr = neg_pearson_loss(pred, target_beta) if float(w_corr) > 0 else _flatten_like_map(pred).new_tensor(0.0)
    contrast = subject_contrastive_loss(pred, target_beta, temperature=contrastive_temperature, mse_weight=contrastive_mse_weight) if float(w_contrastive) > 0 else _flatten_like_map(pred).new_tensor(0.0)
    kl = kl_divergence(outputs.get("mu"), outputs.get("logvar_z")) if (float(w_kl) > 0 and "mu" in outputs and "logvar_z" in outputs) else _flatten_like_map(pred).new_tensor(0.0)
    total = float(w_mse) * rec + float(w_corr) * corr + float(w_contrastive) * contrast + float(w_kl) * kl
    total = torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0)
    return total, {
        "total": float(total.detach().cpu()),
        "reconstruct": float(rec.detach().cpu()),
        "corr": float(corr.detach().cpu()),
        "contrast": float(contrast.detach().cpu()),
        "kl": float(kl.detach().cpu()),
    }
