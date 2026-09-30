from __future__ import annotations

"""Train one thesis-faithful volumetric VRPTR variant on control subjects.

This program deliberately trains only VRPTR. BrainSurfCNN-lite and SWIFUN-lite
are fixed external references and are never instantiated or modified here.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import (
    ClinicalPaths,
    ImageSubjectDataset,
    build_subject_matrix_cached,
    create_image_space,
    load_subject_list,
    save_json,
)
from .losses import clinical_vrptr_loss
from .model import ClinicalVRPTR, ClinicalVRPTRConfig, count_parameters
from .training_common import Standardizer, barrier, cleanup_distributed, is_main_process, make_dataloader, model_state_dict_cpu, reduce_mean_scalar, save_checkpoint, set_seed, setup_distributed_from_env, to_device, unwrap_model, write_rows_csv


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train thesis-faithful volumetric VRPTR on controls.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--train-subjects", default="splits/train_subjects.txt")
    p.add_argument("--out-dir", default="results/thesis_vrptr/full")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--fingerprint-pattern", default="output/rest_connectivity_100/fingerprint_100.nii.gz")
    p.add_argument("--beta-pattern", default="output/task_betas/beta_story_vs_math.nii.gz")
    p.add_argument("--n-rois", type=int, default=100)

    p.add_argument("--variant", choices=["full", "deterministic", "no_transformer", "compression_only"], default="full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=1, help="Per-process batch size under DDP.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    p.add_argument("--amp-init-scale", type=float, default=1024.0, help="Initial FP16 GradScaler scale. Lower than PyTorch default to reduce first-batch overflow on the thesis model.")
    p.add_argument("--resume", default="")
    p.add_argument("--save-every", type=int, default=10)

    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--latent-spatial-dim", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--transformer-layers", type=int, default=6)
    p.add_argument("--transformer-heads", type=int, default=8)
    p.add_argument("--transformer-mlp-ratio", type=float, default=2.0)
    p.add_argument("--head-init-std", type=float, default=1e-3)
    p.add_argument("--stem-input-chunk-size", type=int, default=20, help="Evaluate the first 100-channel Conv3d in exact input-channel chunks; 0 disables chunking.")
    p.add_argument("--disable-activation-checkpointing", action="store_true", help="Disable activation checkpointing in convolutional blocks.")

    p.add_argument("--w-reconstruct", type=float, default=1.0)
    p.add_argument("--w-corr", type=float, default=0.0)
    p.add_argument("--w-contrastive", type=float, default=0.08)
    p.add_argument("--contrastive-temperature", type=float, default=0.20)
    p.add_argument("--contrastive-after-epoch", type=int, default=51)
    p.add_argument("--w-kl", type=float, default=1e-4)
    p.add_argument("--deterministic-warmup-epochs", type=int, default=20)
    p.add_argument("--kl-warmup-epochs", type=int, default=30)
    p.add_argument("--noise-warmup-epochs", type=int, default=10)
    p.add_argument("--initial-noise-scale", type=float, default=0.10)
    p.add_argument("--mask-fc-prob", type=float, default=0.0)
    return p.parse_args(argv)


def _prepare_matrix(raw: dict, scaler: Standardizer | None = None):
    indices = np.arange(len(raw["subjects"]), dtype=int)
    if scaler is None:
        scaler = Standardizer(exclude_last_n=0).fit_indices_masked(raw["x"], raw["y"], indices)
    matrix = dict(raw)
    is_disk = bool(raw.get("disk_backed", False)) or hasattr(raw.get("x", None), "filename")
    if is_disk:
        matrix["x"] = raw["x"]
        matrix["y"] = raw["y"]
        matrix["x_scaler"] = scaler
    else:
        matrix["x"] = scaler.transform(np.asarray(raw["x"], dtype=np.float32))
        matrix["y"] = np.asarray(raw["y"], dtype=np.float32)
    matrix["input_feature_names"] = list(raw.get("input_feature_names", []))
    return matrix, scaler


def _amp_dtype(mode: str):
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    return None


def _schedule(epoch_1based: int, args) -> tuple[bool, float, float, float]:
    use_vae = args.variant in {"full", "no_transformer"}
    if not use_vae or epoch_1based <= int(args.deterministic_warmup_epochs):
        return False, 0.0, 0.0, 0.0
    latent_epoch = epoch_1based - int(args.deterministic_warmup_epochs)
    kl_scale = min(1.0, latent_epoch / max(float(args.kl_warmup_epochs), 1.0))
    noise_progress = min(1.0, latent_epoch / max(float(args.noise_warmup_epochs), 1.0))
    start = float(np.clip(args.initial_noise_scale, 0.0, 1.0))
    noise_scale = start + (1.0 - start) * noise_progress
    return True, noise_scale, float(args.w_kl) * kl_scale, kl_scale


def _checkpoint_payload(model, cfg, scaler, paths, subjects, args, epoch, history, note):
    return {
        "model_state_dict": model_state_dict_cpu(model),
        "model_config": asdict(cfg),
        "x_scaler": scaler.state_dict(),
        "paths": asdict(paths),
        "train_subjects": list(subjects),
        "val_subjects": [],
        "eval_subjects": [],
        "validation_mode": "train_all_controls_final_checkpoint",
        "args": vars(args),
        "epoch": int(epoch),
        "training_history_tail": history[-10:],
        "checkpoint_note": note,
        "architecture_contract": {
            "direct_deepest_bottleneck_bypass": False,
            "ordinary_shallow_unet_skips": True,
            "target": "full_task_beta_map",
            "transformer_order": "spatial_compress->transformer_encoder->mu_logvar->z->transformer_decoder->spatial_expand",
        },
    }


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = setup_distributed_from_env(args.device)
    set_seed(int(args.seed))

    paths = ClinicalPaths(
        data_dir=args.data_dir,
        fingerprint_pattern=args.fingerprint_pattern,
        beta_pattern=args.beta_pattern,
        lesion_pattern="",
        n_rois=args.n_rois,
        yeo_networks=17,
        resolution_mm=2,
        representation="voxel",
        token_subdivisions=1,
    )
    atlas = create_image_space(paths)
    subjects = load_subject_list(args.train_subjects)
    cache_dir = args.cache_dir or str(out_dir.parent.parent / "image_cache")
    raw = build_subject_matrix_cached(
        subjects,
        paths,
        atlas,
        cache_dir=cache_dir,
        cache_tag="train_controls",
        refresh_cache=args.refresh_cache,
    )
    matrix, scaler = _prepare_matrix(raw)
    dataset = ImageSubjectDataset(matrix)
    loader = make_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    input_shape = tuple(int(v) for v in matrix["x"].shape[-3:])
    cfg = ClinicalVRPTRConfig(
        input_dim=int(matrix["x"].shape[1]),
        fc_dim=int(matrix.get("fc_dim", args.n_rois)),
        input_shape=input_shape,
        d_model=args.width,
        n_layers=args.depth,
        dropout=args.dropout,
        stem_input_chunk_size=args.stem_input_chunk_size,
        activation_checkpointing=not args.disable_activation_checkpointing,
        latent_spatial_dim=args.latent_spatial_dim,
        latent_dim=args.latent_dim,
        latent_transformer_layers=args.transformer_layers,
        latent_transformer_heads=args.transformer_heads,
        latent_transformer_mlp_ratio=args.transformer_mlp_ratio,
        use_transformer=args.variant in {"full", "deterministic"},
        use_vae=args.variant in {"full", "no_transformer"},
        direct_head_init_std=args.head_init_std,
        variant=args.variant,
    )
    model = ClinicalVRPTR(cfg).to(device)
    from .training_common import maybe_data_parallel
    model = maybe_data_parallel(model, device, context=f"thesis_vrptr_{args.variant}")
    if is_main_process():
        print(f"[thesis-train] variant={args.variant} subjects={len(subjects)} input={input_shape}")
        print(f"[thesis-train] bottleneck_shape={unwrap_model(model).bottleneck_shape}")
        print(f"[thesis-train] parameters={count_parameters(unwrap_model(model))}")
        print("[thesis-train] direct deepest-bottleneck bypass: DISABLED")
        print(f"[thesis-train] stem_input_chunk_size={cfg.stem_input_chunk_size} activation_checkpointing={cfg.activation_checkpointing}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    history: list[dict] = []
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        unwrap_model(model).load_state_dict(resume["model_state_dict"], strict=True)
        if "optimizer_state_dict" in resume:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume.get("epoch", 0))
        history = list(resume.get("history", []))

    amp_mode = str(args.amp)
    if device.type == "cuda" and amp_mode == "bf16" and not torch.cuda.is_bf16_supported():
        if is_main_process():
            print("[thesis-train][warn] BF16 is not supported on this GPU; falling back to FP16.")
        amp_mode = "fp16"
    amp_dtype = _amp_dtype(amp_mode)
    use_amp = device.type == "cuda" and amp_dtype is not None
    scaler_enabled = bool(use_amp and amp_mode == "fp16")
    # torch.cuda.amp.GradScaler is deprecated. Use the device-aware API when
    # available, while retaining compatibility with older PyTorch releases.
    try:
        scaler_amp = torch.amp.GradScaler(
            "cuda",
            enabled=scaler_enabled,
            init_scale=float(args.amp_init_scale),
        )
    except (AttributeError, TypeError):
        scaler_amp = torch.cuda.amp.GradScaler(
            enabled=scaler_enabled,
            init_scale=float(args.amp_init_scale),
        )
    if args.resume and "amp_scaler_state_dict" in resume and scaler_amp.is_enabled():
        scaler_amp.load_state_dict(resume["amp_scaler_state_dict"])
    if device.type == "cuda":
        # Release temporary allocator blocks created during model/optimizer setup.
        torch.cuda.empty_cache()
    left_t = torch.as_tensor(matrix["left_idx"], dtype=torch.long, device=device)
    right_t = torch.as_tensor(matrix["right_idx"], dtype=torch.long, device=device)

    for epoch in range(start_epoch, int(args.epochs)):
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        model.train()
        sample_latent, noise_scale, effective_w_kl, kl_scale = _schedule(epoch + 1, args)
        contrast_scale = 1.0 if (epoch + 1) >= int(args.contrastive_after_epoch) else 0.0
        sums: dict[str, float] = {}
        valid_batches = 0
        skipped_nonfinite_loss_batches = 0
        skipped_nonfinite_grad_batches = 0
        grad_norm_sum = 0.0

        for batch in loader:
            bd = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp,
            ) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
            with autocast_ctx:
                out = model(
                    bd["x"],
                    mask_fc_prob=args.mask_fc_prob,
                    sample_latent=sample_latent,
                    latent_noise_scale=noise_scale,
                )
                loss, parts = clinical_vrptr_loss(
                    out,
                    bd["x"],
                    bd["y"],
                    bd["li"],
                    bd["dominance"],
                    left_t,
                    right_t,
                    fc_dim=cfg.fc_dim,
                    w_mse=args.w_reconstruct,
                    w_corr=args.w_corr,
                    w_contrastive=float(args.w_contrastive) * contrast_scale,
                    contrastive_temperature=args.contrastive_temperature,
                    w_kl=effective_w_kl,
                )
                # During deterministic warm-up the full model's log-variance
                # head still participates in DDP's graph, avoiding unused-parameter
                # failures while contributing exactly zero to the objective.
                loss = loss + out["logvar_z"].sum() * 0.0

            if not torch.isfinite(loss):
                skipped_nonfinite_loss_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            if scaler_amp.is_enabled():
                scaler_amp.scale(loss).backward()
                # unscale_ may be called at most once between scaler.update()
                # calls. Always finish this scaler iteration below, including
                # when the gradient norm is non-finite.
                scaler_amp.unscale_(optimizer)
            else:
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(args.grad_clip),
                error_if_nonfinite=False,
            )
            grad_norm_is_finite = bool(torch.isfinite(torch.as_tensor(grad_norm)).item())

            if scaler_amp.is_enabled():
                if grad_norm_is_finite:
                    scaler_amp.step(optimizer)
                else:
                    skipped_nonfinite_grad_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                # update() must run after every unscale_(), even if the
                # optimizer step is skipped. Otherwise GradScaler remains in
                # the UNSCALED state and the next batch raises:
                # "unscale_() has already been called ... since the last update".
                scaler_amp.update()
            else:
                if not grad_norm_is_finite:
                    skipped_nonfinite_grad_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()

            if not grad_norm_is_finite:
                continue

            valid_batches += 1
            grad_norm_sum += float(torch.as_tensor(grad_norm).detach().cpu())
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            with torch.no_grad():
                mu_tokens = out["mu_tokens"]
                sums["mu_abs_mean"] = sums.get("mu_abs_mean", 0.0) + float(mu_tokens.abs().mean().cpu())
                sums["mu_within_std"] = sums.get("mu_within_std", 0.0) + float(mu_tokens.std().cpu())
                sums["pred_std"] = sums.get("pred_std", 0.0) + float(out["beta"].std().cpu())

        global_batches = max(1.0, reduce_mean_scalar(valid_batches))
        row = {
            "epoch": epoch + 1,
            "variant": args.variant,
            "valid_batches": int(round(global_batches)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "sample_latent": int(sample_latent),
            "latent_noise_scale": float(noise_scale),
            "effective_w_kl": float(effective_w_kl),
            "kl_schedule_scale": float(kl_scale),
            "contrastive_active": int(contrast_scale > 0),
            "grad_norm": reduce_mean_scalar(grad_norm_sum / max(valid_batches, 1)),
            "skipped_nonfinite_loss_batches": int(round(reduce_mean_scalar(skipped_nonfinite_loss_batches))),
            "skipped_nonfinite_grad_batches": int(round(reduce_mean_scalar(skipped_nonfinite_grad_batches))),
            "amp_scale": float(scaler_amp.get_scale()) if scaler_amp.is_enabled() else 1.0,
        }
        for key, value in sums.items():
            row[f"train_{key}"] = reduce_mean_scalar(value / max(valid_batches, 1))
        history.append(row)

        if is_main_process():
            write_rows_csv(out_dir / "training_history.csv", history)
            print(
                f"[thesis-train] variant={args.variant} epoch={epoch+1:03d}/{args.epochs} "
                f"loss={row.get('train_total', float('nan')):.5f} "
                f"rec={row.get('train_reconstruct', float('nan')):.5f} "
                f"kl={row.get('train_kl', 0.0):.5f} "
                f"sample={row['sample_latent']} noise={row['latent_noise_scale']:.2f} "
                f"batches={row['valid_batches']}"
            )
            payload = _checkpoint_payload(
                model, cfg, scaler, paths, matrix["subjects"], args, epoch + 1, history,
                note=f"thesis VRPTR {args.variant} epoch {epoch + 1}",
            )
            payload["optimizer_state_dict"] = optimizer.state_dict()
            payload["amp_scaler_state_dict"] = scaler_amp.state_dict() if scaler_amp.is_enabled() else {}
            payload["history"] = history
            save_checkpoint(out_dir / "latest.pt", payload)
            if (epoch + 1) % max(1, int(args.save_every)) == 0:
                save_checkpoint(out_dir / f"epoch_{epoch+1:03d}.pt", payload)
        barrier()

    if is_main_process():
        final_payload = _checkpoint_payload(
            model, cfg, scaler, paths, matrix["subjects"], args, int(args.epochs), history,
            note=f"final thesis VRPTR {args.variant}; held-out evaluation deferred",
        )
        final_payload["optimizer_state_dict"] = optimizer.state_dict()
        final_payload["amp_scaler_state_dict"] = scaler_amp.state_dict() if scaler_amp.is_enabled() else {}
        final_payload["history"] = history
        final_path = out_dir / f"final_thesis_vrptr_{args.variant}.pt"
        save_checkpoint(final_path, final_payload)
        save_json(
            {
                "variant": args.variant,
                "seed": args.seed,
                "checkpoint": str(final_path),
                "architecture_version": cfg.architecture_version,
                "direct_deepest_bottleneck_bypass": False,
                "fixed_comparators_retrained": False,
                "args": vars(args),
            },
            out_dir / "manifest.json",
        )
        print(f"[thesis-train] final checkpoint: {final_path}")
    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
