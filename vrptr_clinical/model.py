from __future__ import annotations

"""Thesis-faithful volumetric VRPTR.

The defining architectural rule is that the deepest convolutional feature map
has exactly one route to the decoder:

    encoder bottleneck
      -> spatial compression
      -> Transformer encoder
      -> variational latent (mu/logvar/reparameterization)
      -> Transformer decoder
      -> spatial expansion
      -> decoder

The untreated convolutional bottleneck is never concatenated with, added to,
or passed around the Transformer/VAE module. Ordinary U-Net skips from
shallower encoder resolutions remain available to the decoder, matching the
thesis architecture.
"""

from dataclasses import dataclass
from typing import Dict, Iterable
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ClinicalVRPTRConfig:
    # Input / shared convolutional backbone.
    input_dim: int = 100
    fc_dim: int = 100
    input_shape: tuple[int, int, int] = (99, 117, 95)
    d_model: int = 32                 # 3-D encoder base width.
    n_layers: int = 4                # Number of encoder resolutions.
    dropout: float = 0.15
    # Memory controls. These do not change the mathematical architecture.
    # The chunked stem evaluates the same 100->width 3x3x3 convolution in
    # input-channel slices to avoid a multi-gigabyte cuDNN workspace on 12 GB GPUs.
    stem_input_chunk_size: int = 20
    activation_checkpointing: bool = True

    # Thesis bottleneck. The original mesh model compressed the spatial axis,
    # treated convolutional channels as Transformer tokens, then expanded the
    # spatial axis again. This is the direct voxel-space analogue.
    latent_spatial_dim: int = 32
    latent_dim: int = 32
    latent_transformer_layers: int = 6
    latent_transformer_heads: int = 8
    latent_transformer_mlp_ratio: float = 2.0
    use_transformer: bool = True
    use_vae: bool = True
    latent_logvar_min: float = -8.0
    latent_logvar_max: float = 4.0

    # Output.
    direct_head_init_std: float = 1e-3
    architecture_version: str = "thesis_voxel_v1"
    variant: str = "full"

    # Compatibility fields accepted from older launchers/checkpoints. They do
    # not alter the thesis bottleneck and may be removed from future configs.
    n_parcels: int = 0
    n_heads: int = 8
    adapter_dim: int = 0
    decoder_layers: int = 0
    decoder_skip_scale: float = 1.0
    direct_scale: float = 1.0
    n_latent_factors: int = 0
    latent_factor_dim: int = 32
    latent_map_scale: float = 1.0
    latent_film_scale: float = 0.0
    factor_loading_init_std: float = 0.0
    full_field_stride: int = 2
    full_res_decoder_layers: int = 0
    use_uncertainty: bool = True
    latent_dropout: float = 0.0
    encoder_kernels: tuple[int, ...] = (3,)
    encoder_global_context: bool = True
    use_edge_bias: bool = False
    use_matrix_encoder: bool = False
    matrix_encoder_channels: int = 0
    fc_clip: float = 8.0
    attention_logit_clip: float = 40.0
    edge_topk_frac: float = 1.0
    token_mixer_kernel: int = 0
    token_mixer_scale: float = 0.0
    use_graph_mixer: bool = False
    graph_mixer_scale: float = 0.0
    graph_bias_scale: float = 0.0
    use_multiscale_skip_heads: bool = False
    skip_head_scale: float = 0.0
    skip_head_init_std: float = 0.0
    use_index_pos_embed: bool = False

    def __post_init__(self) -> None:
        self.input_shape = tuple(int(v) for v in self.input_shape)
        if len(self.input_shape) != 3 or any(v <= 0 for v in self.input_shape):
            raise ValueError(f"input_shape must contain three positive integers, got {self.input_shape}")
        self.variant = str(self.variant).lower().replace("-", "_")
        if self.variant == "deterministic":
            self.use_vae = False
        elif self.variant == "no_transformer":
            self.use_transformer = False
        elif self.variant == "compression_only":
            self.use_transformer = False
            self.use_vae = False
        elif self.variant != "full":
            raise ValueError(f"Unknown VRPTR variant: {self.variant}")


def _safe_heads(dim: int, requested: int) -> int:
    requested = max(1, int(requested))
    dim = int(dim)
    if dim % requested == 0:
        return requested
    for h in range(min(requested, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


def _gn(channels: int) -> nn.GroupNorm:
    groups = 8 if channels >= 8 else 1
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def infer_bottleneck_shape(input_shape: Iterable[int], levels: int) -> tuple[int, int, int]:
    """Shape after ``levels - 1`` stride-2 convolutions with k=3,p=1."""
    shape = [int(v) for v in input_shape]
    for _ in range(max(0, int(levels) - 1)):
        shape = [(v + 1) // 2 for v in shape]
    return tuple(shape)


class ChunkedInputConv3d(nn.Conv3d):
    """Exactly evaluate a groups=1 Conv3d in input-channel chunks.

    This retains one weight tensor and the same convolution equation as
    ``nn.Conv3d``. It only changes the execution schedule, reducing the peak
    cuDNN workspace required by the 100-channel full-resolution stem.
    """

    def __init__(self, *args, input_chunk_size: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_chunk_size = int(input_chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunk = int(self.input_chunk_size)
        if chunk <= 0 or self.in_channels <= chunk or self.groups != 1:
            return super().forward(x)
        if x.shape[1] != self.in_channels:
            raise RuntimeError(
                f"ChunkedInputConv3d expected {self.in_channels} channels, got {x.shape[1]}"
            )
        out = None
        for start in range(0, self.in_channels, chunk):
            end = min(start + chunk, self.in_channels)
            piece = F.conv3d(
                x[:, start:end],
                self.weight[:, start:end],
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=1,
            )
            out = piece if out is None else out + piece
        if out is None:
            raise RuntimeError("ChunkedInputConv3d produced no channel chunks")
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)
        return out


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            _gn(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.Dropout3d(float(dropout)),
            _gn(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x + self.net(x), nan=0.0, posinf=0.0, neginf=0.0)


class Encoder3D(nn.Module):
    """Shared 3-D encoder used by VRPTR and the retained compact CNN code."""

    def __init__(self, in_ch: int, width: int = 32, levels: int = 4, dropout: float = 0.1, stem_input_chunk_size: int = 0, activation_checkpointing: bool = False):
        super().__init__()
        self.levels = max(2, int(levels))
        self.activation_checkpointing = bool(activation_checkpointing)
        self.stem = nn.Sequential(
            ChunkedInputConv3d(int(in_ch), int(width), 3, padding=1, input_chunk_size=int(stem_input_chunk_size)),
            _gn(int(width)),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = int(width)
        for li in range(self.levels):
            self.blocks.append(ResidualBlock3D(ch, dropout=float(dropout)))
            if li < self.levels - 1:
                nxt = ch * 2
                self.downs.append(
                    nn.Sequential(
                        nn.Conv3d(ch, nxt, 3, stride=2, padding=1),
                        _gn(nxt),
                        nn.GELU(),
                    )
                )
                ch = nxt
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        h = self.stem(x)
        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            if self.activation_checkpointing and self.training and h.requires_grad:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
            skips.append(h)
            if i < len(self.downs):
                down = self.downs[i]
                if self.activation_checkpointing and self.training and h.requires_grad:
                    h = checkpoint(down, h, use_reentrant=False)
                else:
                    h = down(h)
        return h, skips


class Decoder3D(nn.Module):
    """U-Net decoder with only the ordinary shallower encoder skips."""

    def __init__(self, bottleneck_ch: int, width: int = 32, levels: int = 4, dropout: float = 0.1, activation_checkpointing: bool = False):
        super().__init__()
        self.levels = max(2, int(levels))
        self.activation_checkpointing = bool(activation_checkpointing)
        self.ups = nn.ModuleList()
        self.fuse = nn.ModuleList()
        ch = int(bottleneck_ch)
        skip_channels = [int(width) * (2 ** i) for i in range(self.levels)]
        for skip_ch in reversed(skip_channels[:-1]):
            self.ups.append(
                nn.Sequential(nn.Conv3d(ch, skip_ch, 1), _gn(skip_ch), nn.GELU())
            )
            self.fuse.append(
                nn.Sequential(
                    nn.Conv3d(skip_ch * 2, skip_ch, 3, padding=1),
                    _gn(skip_ch),
                    nn.GELU(),
                    ResidualBlock3D(skip_ch, float(dropout)),
                )
            )
            ch = skip_ch
        self.out_channels = ch

    def forward(self, h: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        # The deepest skip is deliberately excluded: the only deepest path is
        # through the thesis Transformer/VAE bottleneck.
        usable = list(reversed(skips[:-1]))
        for up, fuse, skip in zip(self.ups, self.fuse, usable):
            h = F.interpolate(h, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            h = up(h)
            fused_input = torch.cat([h, skip], dim=1)
            if self.activation_checkpointing and self.training and fused_input.requires_grad:
                h = checkpoint(fuse, fused_input, use_reentrant=False)
            else:
                h = fuse(fused_input)
        return torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)


class SinusoidalChannelPosition(nn.Module):
    """Sinusoidal positions over convolutional-channel tokens."""

    def __init__(self, dim: int, max_tokens: int, dropout: float):
        super().__init__()
        dim = int(dim)
        max_tokens = int(max_tokens)
        pe = torch.zeros(max_tokens, dim)
        position = torch.arange(max_tokens, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        if dim > 1:
            pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=True)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.pe.shape[1]:
            raise RuntimeError(
                f"Transformer token count {x.shape[1]} exceeds positional capacity {self.pe.shape[1]}"
            )
        return self.dropout(x + self.pe[:, : x.shape[1]].to(dtype=x.dtype, device=x.device))


class TransformerStage(nn.Module):
    def __init__(
        self,
        dim: int,
        n_tokens: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
        enabled: bool,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.position = SinusoidalChannelPosition(dim, n_tokens, dropout)
        if self.enabled:
            layer = nn.TransformerEncoderLayer(
                d_model=int(dim),
                nhead=_safe_heads(int(dim), int(heads)),
                dim_feedforward=max(int(dim), int(round(int(dim) * float(mlp_ratio)))),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.stack = nn.TransformerEncoder(
                layer,
                num_layers=max(1, int(layers)),
                enable_nested_tensor=False,
            )
            self.norm = nn.LayerNorm(int(dim))
        else:
            self.stack = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return self.norm(self.stack(self.position(x)))


class ThesisTransformerVAEBottleneck3D(nn.Module):
    """Voxel adaptation of the thesis mesh bottleneck.

    Input/output shape: [B, C, D, H, W]
    Transformer shape:  [B, C, latent_spatial_dim]

    The spatial axis is compressed independently for each convolutional channel,
    exactly mirroring the thesis repository's ``Linear(vertices, latent)``.
    """

    def __init__(self, channels: int, bottleneck_shape: tuple[int, int, int], cfg: ClinicalVRPTRConfig):
        super().__init__()
        self.channels = int(channels)
        self.bottleneck_shape = tuple(int(v) for v in bottleneck_shape)
        self.spatial_size = int(math.prod(self.bottleneck_shape))
        self.latent_spatial_dim = int(cfg.latent_spatial_dim)
        self.latent_dim = int(cfg.latent_dim)
        self.use_vae = bool(cfg.use_vae)
        self.use_transformer = bool(cfg.use_transformer)
        self.logvar_min = float(cfg.latent_logvar_min)
        self.logvar_max = float(cfg.latent_logvar_max)

        self.spatial_compress = nn.Linear(self.spatial_size, self.latent_spatial_dim)
        self.encoder_transformer = TransformerStage(
            dim=self.latent_spatial_dim,
            n_tokens=self.channels,
            layers=cfg.latent_transformer_layers,
            heads=cfg.latent_transformer_heads,
            mlp_ratio=cfg.latent_transformer_mlp_ratio,
            dropout=cfg.dropout,
            enabled=self.use_transformer,
        )
        self.mu_head = nn.Linear(self.latent_spatial_dim, self.latent_dim)
        self.logvar_head = nn.Linear(self.latent_spatial_dim, self.latent_dim) if self.use_vae else None
        self.latent_to_decoder = (
            nn.Identity() if self.latent_dim == self.latent_spatial_dim
            else nn.Linear(self.latent_dim, self.latent_spatial_dim)
        )
        self.decoder_transformer = TransformerStage(
            dim=self.latent_spatial_dim,
            n_tokens=self.channels,
            layers=cfg.latent_transformer_layers,
            heads=cfg.latent_transformer_heads,
            mlp_ratio=cfg.latent_transformer_mlp_ratio,
            dropout=cfg.dropout,
            enabled=self.use_transformer,
        )
        self.spatial_expand = nn.Linear(self.latent_spatial_dim, self.spatial_size)

    def encode(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        if tuple(h.shape[-3:]) != self.bottleneck_shape:
            raise RuntimeError(
                "Unexpected encoder bottleneck shape. "
                f"Configured={self.bottleneck_shape}, observed={tuple(h.shape[-3:])}. "
                "The checkpoint/input grid or encoder depth does not match."
            )
        compressed = self.spatial_compress(h.flatten(2))   # [B,C,L]
        encoded = self.encoder_transformer(compressed)
        mu = self.mu_head(encoded)
        if self.logvar_head is None:
            logvar = torch.zeros_like(mu)
        else:
            logvar = self.logvar_head(encoded).clamp(self.logvar_min, self.logvar_max)
        return {
            "compressed": compressed,
            "encoded": encoded,
            "mu_tokens": mu,
            "logvar_tokens": logvar,
        }

    def decode(self, z_tokens: torch.Tensor, bottleneck_scale: float = 1.0) -> torch.Tensor:
        z = self.latent_to_decoder(z_tokens)
        decoded = self.decoder_transformer(z)
        expanded = self.spatial_expand(decoded)
        h = expanded.reshape(z_tokens.shape[0], self.channels, *self.bottleneck_shape)
        return float(bottleneck_scale) * h

    def forward(
        self,
        h: torch.Tensor,
        sample_latent: bool = False,
        latent_noise_scale: float = 1.0,
        latent_override: torch.Tensor | None = None,
        latent_scale: float = 1.0,
        bottleneck_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        latent = self.encode(h)
        mu = latent["mu_tokens"]
        logvar = latent["logvar_tokens"]
        if latent_override is not None:
            z = latent_override.to(device=mu.device, dtype=mu.dtype)
            if z.shape != mu.shape:
                raise RuntimeError(f"latent_override shape {tuple(z.shape)} != expected {tuple(mu.shape)}")
        elif self.use_vae and bool(sample_latent):
            noise_scale = float(max(0.0, latent_noise_scale))
            z = mu + noise_scale * torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        z = float(latent_scale) * torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        reconstructed = self.decode(z, bottleneck_scale=bottleneck_scale)
        latent.update({"z_tokens": z, "reconstructed_bottleneck": reconstructed})
        return reconstructed, latent


class ClinicalVRPTR(nn.Module):
    """Thesis-faithful volumetric VRPTR with a single transformed bottleneck."""

    def __init__(self, cfg: ClinicalVRPTRConfig):
        super().__init__()
        self.cfg = cfg
        levels = max(2, int(cfg.n_layers))
        width = max(8, int(cfg.d_model))
        self.encoder = Encoder3D(
            cfg.input_dim, width=width, levels=levels, dropout=cfg.dropout,
            stem_input_chunk_size=cfg.stem_input_chunk_size,
            activation_checkpointing=cfg.activation_checkpointing,
        )
        self.bottleneck_shape = infer_bottleneck_shape(cfg.input_shape, levels)
        self.bottleneck = ThesisTransformerVAEBottleneck3D(
            self.encoder.out_channels, self.bottleneck_shape, cfg
        )
        # Compatibility aliases used by some analysis code.
        self.latent_bottleneck = self.bottleneck
        self.latent_transformer = self.bottleneck.encoder_transformer
        self.decoder = Decoder3D(
            self.encoder.out_channels, width=width, levels=levels, dropout=cfg.dropout,
            activation_checkpointing=cfg.activation_checkpointing,
        )
        self.head = nn.Conv3d(self.decoder.out_channels, 1, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=float(cfg.direct_head_init_std))
        nn.init.zeros_(self.head.bias)
        self.requires_full_field = True

    def encode_latent(self, x: torch.Tensor, mask_fc_prob: float = 0.0) -> dict[str, torch.Tensor]:
        x = self._prepare_input(x, mask_fc_prob)
        h, _ = self.encoder(x)
        return self.bottleneck.encode(h)

    @staticmethod
    def _prepare_input(x: torch.Tensor, mask_fc_prob: float) -> torch.Tensor:
        if x.ndim != 5:
            raise RuntimeError(f"VRPTR expects [B,C,X,Y,Z], got {tuple(x.shape)}")
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if float(mask_fc_prob) > 0:
            mask = torch.rand_like(x[:, :1]) < float(mask_fc_prob)
            x = x.masked_fill(mask, 0.0)
        return x

    def set_adapter_tuning(self, mode: str = "bottleneck_decoder", **_: object) -> None:
        """Compatibility helper; not used by the control-only experiment."""
        mode = str(mode or "bottleneck_decoder").lower().replace("-", "_")
        for p in self.parameters():
            p.requires_grad = False
        if mode == "full":
            for p in self.parameters():
                p.requires_grad = True
            return
        groups: list[nn.Module] = []
        if mode in {"bottleneck", "bottleneck_decoder", "latent_decoder"}:
            groups.append(self.bottleneck)
        if mode in {"decoder", "bottleneck_decoder", "latent_decoder"}:
            groups.append(self.decoder)
        groups.append(self.head)
        if mode not in {"head", "decoder", "bottleneck", "bottleneck_decoder", "latent_decoder"}:
            raise ValueError(f"Unsupported adapter mode for thesis VRPTR: {mode}")
        for module in groups:
            for p in module.parameters():
                p.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        mask_fc_prob: float = 0.0,
        sample_latent: bool = False,
        latent_noise_scale: float = 1.0,
        latent_override: torch.Tensor | None = None,
        latent_scale: float = 1.0,
        bottleneck_scale: float = 1.0,
        **_: object,
    ) -> Dict[str, torch.Tensor]:
        x = self._prepare_input(x, mask_fc_prob)
        deepest, skips = self.encoder(x)
        transformed, latent = self.bottleneck(
            deepest,
            sample_latent=sample_latent,
            latent_noise_scale=latent_noise_scale,
            latent_override=latent_override,
            latent_scale=latent_scale,
            bottleneck_scale=bottleneck_scale,
        )
        # No direct deepest-bottleneck bypass is present here.
        dec = self.decoder(transformed, skips)
        beta = torch.nan_to_num(self.head(dec), nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)
        mu = latent["mu_tokens"]
        logvar = latent["logvar_tokens"]
        z = latent["z_tokens"]
        return {
            "beta": beta,
            "logvar": torch.zeros_like(beta),
            "mu": mu.reshape(mu.shape[0], -1),
            "logvar_z": logvar.reshape(logvar.shape[0], -1),
            "z": z.reshape(z.shape[0], -1),
            "mu_tokens": mu,
            "logvar_tokens": logvar,
            "z_tokens": z,
            "compressed_tokens": latent["compressed"],
            "encoded_tokens": latent["encoded"],
            "transformed_bottleneck": transformed,
            "attention": torch.zeros((x.shape[0], 1, 1), dtype=x.dtype, device=x.device),
        }


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "pct_trainable": float(100.0 * trainable / max(total, 1)),
    }


def build_model_from_config(config: dict | ClinicalVRPTRConfig) -> ClinicalVRPTR:
    if isinstance(config, ClinicalVRPTRConfig):
        cfg = config
    else:
        valid = set(ClinicalVRPTRConfig.__dataclass_fields__.keys())
        payload = {k: v for k, v in dict(config).items() if k in valid}
        if "input_shape" in payload:
            payload["input_shape"] = tuple(int(v) for v in payload["input_shape"])
        cfg = ClinicalVRPTRConfig(**payload)
    return ClinicalVRPTR(cfg)
