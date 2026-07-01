"""
IC-DiT Defect Generator: Top-level model integrating all frozen + trainable components.

Architecture (matches paper Fig. 3):

  FROZEN (all .eval(), requires_grad=False, torch.no_grad()):
    text_encoder   — T5 (or CLIP), encodes prompt → text tokens
    vae            — SD VAE for image encode/decode
    layout_encoder — binary mask → dedicated VAE → patchified layout tokens
    visual_encoder — DINOv2, extracts dense visual embeddings

  TRAINABLE:
    text_projector    — MLP: text_dim → hidden_dim
    layout_projector  — MLP: layout_patch_dim → hidden_dim
    visual_projector  — MLP: visual_dim → hidden_dim
    latent_dit        — DiT backbone with concat-QKV MM-Attention (Eqs. 6-8)
                        Returns ALL FOUR updated token sets (image, text,
                        layout, visual) — matching paper Fig. 3.
    layout_head       — MLP: hidden_dim → 1 (per-patch defect logits)
                        Enables explicit spatial grounding loss.

Usage (conda env):
    conda activate omg
"""

import torch
import torch.nn as nn
from typing import Optional

from .frozen_encoders import FrozenTextEncoder, FrozenVAE, FrozenVisualEncoder
from .mask_encoder import FrozenLayoutEncoder
from .latent_dit import LatentDiT
from utils.freeze import freeze_module, print_trainable_parameters


class ICDiTDefectGenerator(nn.Module):
    """
    IC-DiT for layout-guided controllable defect generation.

    Config keys (model.*):
        text_encoder_name, vae_name, visual_encoder_name
        latent_channels, hidden_dim, num_layers, num_heads, patch_size
        training_mode, freeze_*, visual_embedding_source
    """

    def __init__(self, config: dict, image_size: int = 256):
        super().__init__()
        cfg = config["model"]
        self.hidden_dim = cfg["hidden_dim"]
        self.training_mode = cfg.get("training_mode", "full_generator")
        self.visual_embedding_source = cfg.get("visual_embedding_source", "reference_normal")
        self.image_size = image_size
        self.latent_channels = cfg.get("latent_channels", 4)

        # =====================================================================
        # FROZEN ENCODERS
        # =====================================================================

        self.text_encoder = FrozenTextEncoder(
            model_name=cfg.get("text_encoder_name", "openai/clip-vit-large-patch14"),
            max_length=cfg.get("text_max_length", 77),
        )

        # Shared VAE: one instance used for image encode/decode
        self.vae = FrozenVAE(
            model_name=cfg.get("vae_name", "stabilityai/sd-vae-ft-mse"),
        )

        # Layout encoder — dedicated VAE (paper Fig. 3: separate VAE encoder
        # for layout masks, NOT shared with the image VAE).
        # Controlled by config key "dedicated_layout_vae" (default True).
        _dedicated = cfg.get("dedicated_layout_vae", True)
        self.layout_encoder = FrozenLayoutEncoder(
            patch_size=cfg.get("patch_size", 2),
            image_size=image_size,
            shared_vae=None if _dedicated else self.vae.vae,
            vae_model_name=cfg.get("vae_name", "stabilityai/sd-vae-ft-mse"),
        )

        self.visual_encoder = FrozenVisualEncoder(
            model_name=cfg.get("visual_encoder_name", "dinov2_vitb14"),
            image_size=image_size,
        )

        # Belt-and-suspenders: re-freeze all encoders
        if cfg.get("freeze_text_encoder", True):
            freeze_module(self.text_encoder)
        if cfg.get("freeze_vae", True):
            freeze_module(self.vae)
        if cfg.get("freeze_layout_encoder", True):
            freeze_module(self.layout_encoder)
        if cfg.get("freeze_visual_encoder", True):
            freeze_module(self.visual_encoder)

        # =====================================================================
        # TRAINABLE PROJECTORS (2-layer MLP: Linear→LN→GELU→Linear)
        # =====================================================================

        text_input_dim = self.text_encoder.output_dim
        self.text_projector = nn.Sequential(
            nn.Linear(text_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        layout_input_dim = self.layout_encoder.output_dim  # 16
        self.layout_projector = nn.Sequential(
            nn.Linear(layout_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        visual_input_dim = self.visual_encoder.output_dim  # 768
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # =====================================================================
        # TRAINABLE LATENT DiT BACKBONE (concat-QKV MM-Attention, Eqs. 6-8)
        # =====================================================================

        self.latent_dit = LatentDiT(
            latent_channels=self.latent_channels,
            hidden_dim=self.hidden_dim,
            num_layers=cfg.get("num_layers", 12),
            num_heads=cfg.get("num_heads", 12),
            patch_size=cfg.get("patch_size", 2),
            image_size=image_size,
            dropout=cfg.get("dropout", 0.1),
        )

        # =====================================================================
        # TRAINABLE LAYOUT HEAD — mask reconstruction from updated layout tokens
        # =====================================================================
        # After MM-Attention, layout tokens carry bidirectional image↔layout
        # information.  This head decodes them back to a per-patch defect
        # probability, enabling an explicit spatial grounding loss (Dice/BCE)
        # that forces the model to preserve the mask's spatial structure
        # through the attention stack.
        self.layout_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),  # binary logit per patch
        )

    # ------------------------------------------------------------------
    # Convenience encode/decode helpers (all frozen, torc.no_grad)
    # ------------------------------------------------------------------

    def _safe_project(self, projector: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        """Project encoder output and clamp to fp16-safe range.

        T5/VAE encoders can produce values >100× larger than expected for fp16.
        The LayerNorm inside the projector bounds the INTERNAL values, but the
        final Linear layer can still push the output past fp16 max (~65504).
        This clamp is a last-resort safety net that only activates for extreme
        values; in normal operation it's a no-op.
        """
        out = projector(hidden)
        if out.dtype == torch.float16:
            out = torch.clamp(out, min=-60000.0, max=60000.0)
        return out

    def encode_text(self, prompts: list) -> torch.Tensor:
        """Frozen text encode → projected tokens (B, seq_len, hidden_dim)."""
        with torch.no_grad():
            text_hidden = self.text_encoder(prompts)
        return self._safe_project(self.text_projector, text_hidden)

    def encode_layout(self, masks: torch.Tensor) -> torch.Tensor:
        """Frozen layout encode → projected tokens (B, 256, hidden_dim)."""
        with torch.no_grad():
            layout_hidden = self.layout_encoder(masks)
        return self._safe_project(self.layout_projector, layout_hidden)

    def encode_visual(self, reference_images: torch.Tensor) -> torch.Tensor:
        """Frozen visual encode → projected tokens (B, N_patches, hidden_dim).

        Uses dense patch tokens (matching paper Eq.8) so the MM-Attention
        can perform spatial image↔visual alignment, enhancing local
        morphological detail as described in Section 3.5.
        """
        with torch.no_grad():
            visual_feat = self.visual_encoder(reference_images, return_tokens=True)  # (B, N, 768)
        return self._safe_project(self.visual_projector, visual_feat)

    def encode_image_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """Frozen VAE encode → latent z_0 (B, 4, H/8, W/8)."""
        with torch.no_grad():
            return self.vae.encode(images)

    def decode_latent_to_image(self, latents: torch.Tensor) -> torch.Tensor:
        """Frozen VAE decode → image (B, 3, H, W) in [-1, 1]."""
        with torch.no_grad():
            return self.vae.decode(latents)

    def decode_layout(self, layout_tokens: torch.Tensor) -> torch.Tensor:
        """Predict per-patch defect probability from updated layout tokens.

        The layout_head maps each of the 256 layout tokens (corresponding to
        a 16×16 spatial grid) to a single logit.  The result can be compared
        with a down-sampled version of the input mask for spatial grounding.

        Args:
            layout_tokens: (B, 256, hidden_dim) — updated layout tokens
                           after MM-Attention.

        Returns:
            logits: (B, 256) — per-patch defect logits (clamped to fp16-safe range).
        """
        logits = self.layout_head(layout_tokens).squeeze(-1)  # (B, 256)
        if logits.dtype == torch.float16:
            logits = torch.clamp(logits, min=-60000.0, max=60000.0)
        return logits

    # ------------------------------------------------------------------
    # Forward — full training step (used by training loop)
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompts: list,
        masks: torch.Tensor,
        reference_images: torch.Tensor,
    ):
        """
        Predict epsilon noise + return all updated multimodal tokens.

        This is the canonical interface matching the paper's Eq. 9:
            ε_θ(z_t, t, text, layout, visual_embedding)

        The MM-Attention block (Eqs. 6-8) performs BIDIRECTIONAL fusion:
        image tokens are updated with information from text/layout/visual,
        AND each condition modality is updated with information from the
        image.  All updated tokens are returned so the training loop can
        apply auxiliary alignment losses.

        Args:
            noisy_latents:    (B, 4, H/8, W/8) — z_t.
            timesteps:        (B,) diffusion timestep indices.
            prompts:          List[str] text prompts.
            masks:            (B, H, W) binary anomaly masks in [0, 1].
            reference_images: (B, 3, H, W) reference normal images in [-1, 1].

        Returns:
            eps_pred:              (B, 4, H/8, W/8) predicted noise ε_θ.
            text_tokens_updated:   (B, N_txt, hidden_dim).
            layout_tokens_updated: (B, 256, hidden_dim).
            visual_tokens_updated: (B, N_vis, hidden_dim).
            layout_logits:         (B, 256) per-patch defect logits
                                   (for spatial grounding loss).
            text_tokens_init:      (B, N_txt, hidden_dim) initial projected
                                   text tokens (target for consistency loss).
            visual_tokens_init:    (B, N_vis, hidden_dim) initial projected
                                   visual tokens (target for consistency loss).
        """
        # Encode all conditions
        text_tokens = self.encode_text(prompts)
        layout_tokens = self.encode_layout(masks)
        visual_tokens = self.encode_visual(reference_images)

        # Predict noise through DiT with concat-QKV MM-Attention
        # All four token sets are returned (bidirectionally updated)
        eps_pred, text_tokens_upd, layout_tokens_upd, visual_tokens_upd = \
            self.latent_dit(
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                text_tokens=text_tokens,
                layout_tokens=layout_tokens,
                visual_tokens=visual_tokens,
            )

        # Auxiliary output: reconstruct mask from updated layout tokens
        layout_logits = self.decode_layout(layout_tokens_upd)

        return (eps_pred,
                text_tokens_upd, layout_tokens_upd, visual_tokens_upd,
                layout_logits,
                text_tokens, visual_tokens)  # initial tokens for consistency loss

    # ------------------------------------------------------------------
    # Generate — iterative DDIM sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        masks: torch.Tensor,
        prompts: list,
        reference_images: torch.Tensor,
        noise_scheduler,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        return_aux: bool = False,
    ):
        """
        Generate defect images via iterative DDIM denoising with CFG.

        When ``return_aux=True``, also returns the updated multimodal tokens
        from the last denoising step.  These can be used to compute the
        evaluation metrics from the paper (Table 1-3):

            - layout_logits → Mask Faithfulness (Dice) vs input mask
            - text_tokens    → PLIP / CLIP-FID image-text similarity
            - visual_tokens  → Embedding Similarity (cosine) vs reference

        Args:
            masks:            (B, H, W) in [0, 1].
            prompts:          List[str].
            reference_images: (B, 3, H, W) in [-1, 1].
            noise_scheduler:  NoiseScheduler instance.
            num_inference_steps: Number of DDIM steps.
            guidance_scale:   CFG scale (>1 enables CFG).
            return_aux:       If True, return (images, aux_dict).

        Returns:
            If return_aux=False: images (B, 3, H, W) in [-1, 1].
            If return_aux=True:  (images, aux) where aux is a dict with keys
                                 "text_tokens", "layout_tokens", "visual_tokens",
                                 "layout_logits".
        """
        device = next(self.parameters()).device
        B = masks.shape[0]
        H_lat = W_lat = self.image_size // 8

        # Encode conditions once (frozen, no grad needed)
        text_tokens = self.encode_text(prompts)
        layout_tokens = self.encode_layout(masks)
        visual_tokens = self.encode_visual(reference_images)

        # Pre-compute null (unconditional) text tokens for CFG
        if guidance_scale > 1.0:
            null_text_tokens = self.encode_text([""] * B)

        # Start from pure Gaussian noise
        latents = torch.randn(B, self.latent_channels, H_lat, W_lat, device=device)
        latents = latents * noise_scheduler.init_noise_sigma

        # Set inference timesteps
        noise_scheduler.set_timesteps(num_inference_steps, device=device)

        # Placeholders for last-step condition tokens (captured when return_aux=True)
        text_upd = layout_upd = visual_upd = None

        for i, t in enumerate(noise_scheduler.timesteps):
            is_last = (i == len(noise_scheduler.timesteps) - 1)

            # CFG: double batch (cond + uncond)
            if guidance_scale > 1.0:
                latent_input = torch.cat([latents] * 2, dim=0)
                t_batch = t.unsqueeze(0).repeat(2 * B).to(device)
                text_input = torch.cat([text_tokens, null_text_tokens], dim=0)
                layout_input = torch.cat([layout_tokens] * 2, dim=0)
                visual_input = torch.cat([visual_tokens] * 2, dim=0)
            else:
                latent_input = latents
                t_batch = t.unsqueeze(0).repeat(B).to(device)
                text_input = text_tokens
                layout_input = layout_tokens
                visual_input = visual_tokens

            # Predict noise
            noise_pred, t_upd_d, l_upd_d, v_upd_d = self.latent_dit(
                noisy_latents=latent_input,
                timesteps=t_batch,
                text_tokens=text_input,
                layout_tokens=layout_input,
                visual_tokens=visual_input,
            )

            # CFG combination
            if guidance_scale > 1.0:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                # Capture conditional-half tokens on the last step
                if return_aux and is_last:
                    text_upd   = t_upd_d.chunk(2, dim=0)[0]
                    layout_upd = l_upd_d.chunk(2, dim=0)[0]
                    visual_upd = v_upd_d.chunk(2, dim=0)[0]
            else:
                noise_pred = noise_pred
                if return_aux and is_last:
                    text_upd, layout_upd, visual_upd = t_upd_d, l_upd_d, v_upd_d

            # DDIM step
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode
        images = self.decode_latent_to_image(latents)

        if return_aux:
            aux = {
                "text_tokens":    text_upd,
                "layout_tokens":  layout_upd,
                "visual_tokens":  visual_upd,
                "layout_logits":  self.decode_layout(layout_upd),
            }
            return images, aux

        return images

    def print_stats(self):
        """Print frozen/trainable parameter statistics."""
        print_trainable_parameters(self, prefix="ICDiTDefectGenerator")
