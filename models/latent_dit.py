"""
Latent Diffusion Transformer (DiT) backbone for noise prediction.

Operates on latent image patches with MM-Attention fusion.

Architecture:
  1. Patchify noisy latent  (B, 4, 32, 32) → (B, 256, 16)
  2. Patch embedding MLP    (B, 256, 16)    → (B, 256, dim)
  3. Add positional embedding
  4. Prepend timestep token
  5. MM-Attention stack with text/layout/visual (concat-QKV, Eqs. 6-8)
  6. Remove timestep token
  7. Final norm + output projection
  8. Unpatchify              (B, 256, 16)    → (B, 4, 32, 32)

Usage (conda env):
    conda activate omg
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mm_attention import MMAttentionStack


# ---------------------------------------------------------------------------
# Timestep utilities (sinusoidal, same as DDPM / DiT)
# ---------------------------------------------------------------------------

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embedding.

    Args:
        timesteps: (B,) integer timestep indices.
        embedding_dim: Output dimension.

    Returns:
        (B, embedding_dim)
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )

    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


# ---------------------------------------------------------------------------
# Latent DiT
# ---------------------------------------------------------------------------

class LatentDiT(nn.Module):
    """
    Latent Diffusion Transformer backbone for epsilon prediction.

    Takes noisy latent z_t + timestep t + multimodal condition tokens,
    predicts the noise component ε_θ(z_t, t, text, layout, visual).
    """

    def __init__(
        self,
        latent_channels: int = 4,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        patch_size: int = 2,
        image_size: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.latent_size = image_size // 8                # 32 for image_size=256
        self.num_patches = (self.latent_size // patch_size) ** 2  # (32//2)^2 = 256
        self.patch_dim = latent_channels * patch_size * patch_size  # 4*2*2 = 16

        # ---- Patch embedding: latent patch → hidden_dim (MLP matching other projectors) ----
        self.patch_embed = nn.Sequential(
            nn.Linear(self.patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ---- Learnable positional embedding ----
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

        # ---- Timestep embedding (sinusoidal → MLP) ----
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # ---- MM-Attention stack (concat-QKV, Eqs. 6-8) ----
        self.mm_attention_stack = MMAttentionStack(
            num_layers=num_layers,
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_self_attn=True,
        )

        # ---- Output head ----
        self.norm_final = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.patch_dim)

        self._init_weights()

        print(f"[LatentDiT] Built DiT backbone:")
        print(f"  latent_channels={latent_channels}, hidden_dim={hidden_dim}")
        print(f"  num_layers={num_layers}, num_heads={num_heads}, patch_size={patch_size}")
        print(f"  latent_size={self.latent_size}, num_patches={self.num_patches}")

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)

        # Xavier for Linear layers, zeros for biases
        for module in self.patch_embed:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    # ------------------------------------------------------------------
    # Patchify / Unpatchify
    # ------------------------------------------------------------------

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, C·P²)."""
        B, C, H, W = latents.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, f"Latent {H}×{W} not divisible by patch_size {p}"

        tokens = latents.reshape(B, C, H // p, p, W // p, p)
        tokens = tokens.permute(0, 2, 4, 1, 3, 5).contiguous()
        tokens = tokens.reshape(B, (H // p) * (W // p), C * p * p)
        return tokens

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, C·P²) → (B, C, H, W)."""
        B, N, _ = tokens.shape
        p = self.patch_size
        H = W = int(N ** 0.5)
        C = self.latent_channels
        assert H * W == N, f"Num patches {N} not a perfect square"

        tokens = tokens.reshape(B, H, W, C, p, p)
        tokens = tokens.permute(0, 3, 1, 4, 2, 5).contiguous()
        tokens = tokens.reshape(B, C, H * p, W * p)
        return tokens

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_tokens: torch.Tensor,
        layout_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
    ):
        """
        Predict epsilon noise from noisy latent + multimodal conditions.

        The MM-Attention stack performs BIDIRECTIONAL fusion (paper Eqs. 6-8):
        every condition modality is updated alongside the image tokens so that
        all four token sets carry mutually aligned information after the stack.

        Args:
            noisy_latents: (B, 4, H_lat, W_lat) — z_t.
            timesteps:     (B,) diffusion timestep indices.
            text_tokens:   (B, N_txt, hidden_dim) — projected text tokens.
            layout_tokens: (B, N_lay, hidden_dim) — projected layout tokens.
            visual_tokens: (B, N_vis, hidden_dim) — projected visual tokens.

        Returns:
            eps_pred:              (B, 4, H_lat, W_lat) — predicted noise ε_θ.
            text_tokens_updated:   (B, N_txt, hidden_dim) — text tokens after
                                   bidirectional fusion with image.
            layout_tokens_updated: (B, N_lay, hidden_dim) — layout tokens after
                                   bidirectional fusion with image.
            visual_tokens_updated: (B, N_vis, hidden_dim) — visual tokens after
                                   bidirectional fusion with image.
        """
        B = noisy_latents.shape[0]

        # --- NaN-tracing helper -------------------------------------------------
        def _check(name: str, tensor: torch.Tensor):
            """Log first appearance of NaN; returns True if clean."""
            if torch.isnan(tensor).any():
                import sys
                print(f"[LatentDiT NaN @ {name}] shape={tensor.shape} "
                      f"min={tensor.min().item():.4f} max={tensor.max().item():.4f}",
                      file=sys.stderr, flush=True)
                return False
            return True

        # 1. Patchify noisy latent
        image_tokens = self._patchify(noisy_latents)             # (B, 256, 16)
        if not _check("patchify", image_tokens): pass
        image_tokens = self.patch_embed(image_tokens)            # (B, 256, hidden_dim)
        if not _check("patch_embed", image_tokens): pass
        if image_tokens.dtype == torch.float16:
            image_tokens = torch.clamp(image_tokens, min=-60000.0, max=60000.0)

        # 2. Add positional embedding
        image_tokens = image_tokens + self.pos_embed
        if not _check("pos_embed_add", image_tokens): pass

        # 3. Compute timestep embedding and prepend as extra token
        t_emb = get_timestep_embedding(timesteps, self.hidden_dim)
        t_emb = self.time_embed(t_emb)                           # (B, hidden_dim)
        if t_emb.dtype == torch.float16:
            t_emb = torch.clamp(t_emb, min=-60000.0, max=60000.0)
        t_token = t_emb.unsqueeze(1)                             # (B, 1, hidden_dim)
        image_tokens = torch.cat([t_token, image_tokens], dim=1) # (B, 1+256, hidden_dim)

        # 4. Pre-check condition tokens
        for name, tok in [("text", text_tokens), ("layout", layout_tokens), ("visual", visual_tokens)]:
            _check(f"cond_{name}_in", tok)

        # 5. MM-Attention fusion (concat-QKV, bidirectional, paper Eqs. 6-8)
        image_tokens, text_tokens, layout_tokens, visual_tokens = \
            self.mm_attention_stack(
                image_tokens=image_tokens,
                text_tokens=text_tokens,
                layout_tokens=layout_tokens,
                visual_tokens=visual_tokens,
            )
        for name, tok in [("img_out", image_tokens), ("txt_out", text_tokens),
                          ("lay_out", layout_tokens), ("vis_out", visual_tokens)]:
            _check(f"mmattn_{name}", tok)

        # 5. Remove timestep token from image tokens
        image_tokens = image_tokens[:, 1:, :]                    # (B, 256, hidden_dim)

        # 6. Final norm + output projection → patch_dim
        image_tokens = self.norm_final(image_tokens)
        image_tokens = self.output_proj(image_tokens)            # (B, 256, 16)

        # 7. Unpatchify → latent shape
        eps_pred = self._unpatchify(image_tokens)                # (B, 4, 32, 32)

        return eps_pred, text_tokens, layout_tokens, visual_tokens
