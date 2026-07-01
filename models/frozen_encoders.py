"""
Frozen encoder modules: Text Encoder, VAE, Visual Encoder.

All modules are frozen on initialization and use torch.no_grad() in forward.

Usage (conda env):
    conda activate omg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKL
import timm

from utils.freeze import freeze_module


# ---------------------------------------------------------------------------
# FrozenTextEncoder
# ---------------------------------------------------------------------------

class FrozenTextEncoder(nn.Module):
    """
    Frozen text encoder. Supports CLIP or T5.
    Default: openai/clip-vit-large-patch14
    """

    SUPPORTED_MODELS = {
        "openai/clip-vit-large-patch14": "clip",
        "openai/clip-vit-base-patch32": "clip",
        "google/t5-v1_1-base": "t5",
        "google/t5-v1_1-large": "t5",
    }

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", max_length: int = 77):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        model_type = self.SUPPORTED_MODELS.get(model_name, "clip")

        if model_type == "clip":
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.encoder = CLIPTextModel.from_pretrained(model_name)
        elif model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.encoder = T5EncoderModel.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported text encoder: {model_name}")

        # Freeze immediately
        freeze_module(self.encoder)
        print(f"[FrozenTextEncoder] Loaded and frozen: {model_name} "
              f"(type={model_type}, output_dim={self.output_dim})")

    @property
    def output_dim(self) -> int:
        """Output hidden dimension."""
        return self.encoder.config.hidden_size

    @torch.no_grad()
    def forward(self, texts: list) -> torch.Tensor:
        """
        Encode a list of text prompts.

        Args:
            texts: List of prompt strings.

        Returns:
            Text embeddings, shape (B, seq_len, hidden_dim).
            For CLIP: (B, 77, 768)
        """
        device = next(self.encoder.parameters()).device

        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        # Force fp32: T5-v1_1-base is notoriously unstable in fp16 — its
        # attention QK^T overflows fp16's ~65504 max on 512-token sequences,
        # producing inf → softmax → NaN.  CLIP is generally fp16-safe but
        # we force fp32 anyway (frozen → no grad mem, only activations).
        with torch.amp.autocast('cuda', enabled=False):
            if hasattr(self.encoder, "config") and "t5" in self.model_name.lower():
                outputs = self.encoder(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
                hidden_states = outputs.last_hidden_state
            else:
                # CLIP-style: return pooled + hidden
                outputs = self.encoder(**tokens)
                hidden_states = outputs.last_hidden_state  # (B, seq_len, hidden_dim)

        # Cast to fp32 if the encoder returned in a lower precision.
        # Autocast downstream will handle the fp32→fp16 transition safely
        # because the hidden values are bounded by LayerNorm (roughly ±10).
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()

        return hidden_states


# ---------------------------------------------------------------------------
# FrozenVAE
# ---------------------------------------------------------------------------

class FrozenVAE(nn.Module):
    """
    Frozen VAE (encoder + decoder) from Stable Diffusion.
    Default: stabilityai/sd-vae-ft-mse
    """

    def __init__(self, model_name: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self.model_name = model_name
        self.vae = AutoencoderKL.from_pretrained(model_name)

        # Freeze immediately
        freeze_module(self.vae)
        print(f"[FrozenVAE] Loaded and frozen: {model_name} (latent_channels=4)")

    @property
    def latent_channels(self) -> int:
        return 4

    def get_scaling_factor(self) -> float:
        """Get the latent scaling factor (0.18215 for SD VAE)."""
        return self.vae.config.scaling_factor

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space.

        Args:
            images: (B, 3, H, W) in [-1, 1] (SD VAE native range).

        Returns:
            latents: (B, 4, H/8, W/8), scaled by vae.config.scaling_factor (0.18215).
        """
        # SD VAE uses Conv + GroupNorm — fp16-safe.  Only T5 needs fp32.
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back to images.

        Args:
            latents: (B, 4, H/8, W/8), scaled by vae.config.scaling_factor.

        Returns:
            images: (B, 3, H, W) in [-1, 1].
        """
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample
        # SD VAE output is in [-1, 1]; clamp for numerical stability
        images = torch.clamp(images, -1.0, 1.0)
        return images


# ---------------------------------------------------------------------------
# FrozenVisualEncoder
# ---------------------------------------------------------------------------

class FrozenVisualEncoder(nn.Module):
    """
    Frozen self-supervised visual encoder.
    Uses DINOv2 (timm) as default, similar in spirit to iBOT.

    Default: dinov2_vitb14 (ViT-B/14 DINOv2, 768-dim output).
    """

    SUPPORTED_MODELS = {
        "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
        "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
        "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    }

    def __init__(self, model_name: str = "dinov2_vitb14", image_size: int = 256):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size

        timm_name = self.SUPPORTED_MODELS.get(model_name)
        if timm_name is None:
            raise ValueError(
                f"Unsupported visual encoder: {model_name}. "
                f"Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.encoder = timm.create_model(timm_name, pretrained=True, num_classes=0)
        self.encoder.reset_classifier(0)  # Remove classifier head

        # Freeze immediately
        freeze_module(self.encoder)
        print(f"[FrozenVisualEncoder] Loaded and frozen: {model_name} "
              f"(timm={timm_name}, output_dim={self.output_dim})")

    @property
    def output_dim(self) -> int:
        """Feature dimension (768 for ViT-B)."""
        if hasattr(self.encoder, "embed_dim"):
            return self.encoder.embed_dim
        return 768

    @torch.no_grad()
    def forward(self, images: torch.Tensor, return_tokens: bool = False) -> torch.Tensor:
        """
        Extract visual embeddings.

        Args:
            images: (B, 3, H, W) in [-1, 1].
            return_tokens: If True, return patch tokens. If False, return [CLS] token.

        Returns:
            If return_tokens=False: (B, embed_dim)  -- global [CLS] token.
            If return_tokens=True:  (B, num_patches + 1, embed_dim)  -- all tokens.
        """
        # DINOv2 ViT with LayerNorm is fp16-safe.  Only T5 needs fp32.
        # DINOv2 expects images normalized with ImageNet stats
        # Input [-1,1] -> convert to [0,1] -> apply ImageNet normalization
        images_01 = (images + 1.0) / 2.0  # [-1,1] -> [0,1]

        # DINOv2 ViT expects a fixed input size (e.g. 518x518); resize if needed
        target_size = self.encoder.patch_embed.img_size
        if isinstance(target_size, (tuple, list)):
            target_size = target_size[0]
        if images_01.shape[2] != target_size:
            images_01 = F.interpolate(images_01, size=(target_size, target_size),
                                       mode='bilinear', align_corners=False)

        # ImageNet normalization
        mean = torch.tensor([0.4850, 0.4560, 0.4060], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2290, 0.2240, 0.2250], device=images.device).view(1, 3, 1, 1)
        images_norm = (images_01 - mean) / std

        if return_tokens:
            # Use forward_features to get patch tokens
            features = self.encoder.forward_features(images_norm)
            # timm DINOv2 returns dict with 'x_norm_patchtokens' or 'x_patchtokens'
            if isinstance(features, dict):
                tokens = features.get("x_norm_patchtokens", features.get("x_patchtokens", None))
                if tokens is not None:
                    return tokens
            # Fallback: return global feature repeated
            global_feat = self.encoder(images_norm)
            return global_feat.unsqueeze(1)  # (B, 1, embed_dim)
        else:
            # Global [CLS] token
            return self.encoder(images_norm)  # (B, embed_dim)
