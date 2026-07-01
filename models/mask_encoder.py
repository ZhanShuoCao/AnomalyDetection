"""
Frozen Layout Encoder for defect masks.

Encodes binary masks through a frozen VAE encoder to produce layout tokens.
Can share a VAE instance with FrozenVAE to avoid loading weights twice.

Usage (conda env):
    conda activate omg
"""

import torch
import torch.nn as nn
from typing import Optional
from diffusers import AutoencoderKL

from utils.freeze import freeze_module


class FrozenLayoutEncoder(nn.Module):
    """
    Encode binary defect masks into latent layout tokens.

    Steps:
      1. Repeat single-channel mask to 3 channels.
      2. Encode via frozen VAE encoder (shared or standalone).
      3. Patchify to layout token sequence.

    If `shared_vae` is provided, reuses that VAE instance (saves ~300 MB).
    Otherwise loads a standalone VAE.
    """

    def __init__(
        self,
        patch_size: int = 2,
        image_size: int = 256,
        vae_model_name: str = "stabilityai/sd-vae-ft-mse",
        shared_vae: Optional[AutoencoderKL] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size

        # VAE: reuse shared instance or load standalone
        if shared_vae is not None:
            self.vae = shared_vae
            print(f"[FrozenLayoutEncoder] Reusing shared VAE instance "
                  f"(already frozen, latent_channels=4)")
        else:
            self.vae = AutoencoderKL.from_pretrained(vae_model_name)
            freeze_module(self.vae)
            print(f"[FrozenLayoutEncoder] Loaded standalone VAE: {vae_model_name}")

        # Latent dimensions
        self.latent_size = image_size // 8                           # 32
        self.num_patches = (self.latent_size // patch_size) ** 2     # 256
        self.latent_channels = 4

        print(f"[FrozenLayoutEncoder] image_size={image_size}, latent_size={self.latent_size}, "
              f"patch_size={patch_size}, num_patches={self.num_patches}")

    @property
    def output_dim(self) -> int:
        """Token dimension per patch: C·P² = 4·2·2 = 16."""
        return self.latent_channels * self.patch_size * self.patch_size

    @torch.no_grad()
    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Encode binary masks to layout tokens.

        Args:
            masks: (B, H, W) in [0, 1], single-channel binary masks.

        Returns:
            layout_tokens: (B, num_patches, output_dim), i.e. (B, 256, 16).
        """
        # 1. Single channel → 3 channels
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)                      # (B, 1, H, W)
        masks_3ch = masks.repeat(1, 3, 1, 1)                # (B, 3, H, W)

        # 2. Encode via frozen VAE (Conv+GroupNorm, fp16-safe)
        latents = self.vae.encode(masks_3ch).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor  # (B, 4, 32, 32)

        # 3. Patchify
        layout_tokens = self._patchify(latents)              # (B, 256, 16)
        return layout_tokens

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, H/p·W/p, C·p²)."""
        B, C, H, W = latents.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, \
            f"Latent size {H}×{W} must be divisible by patch_size {p}"

        tokens = latents.reshape(B, C, H // p, p, W // p, p)
        tokens = tokens.permute(0, 2, 4, 1, 3, 5).contiguous()
        tokens = tokens.reshape(B, (H // p) * (W // p), C * p * p)
        return tokens
