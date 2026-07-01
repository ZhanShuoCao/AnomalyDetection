"""
Multi-Modal Attention (MM-Attention) blocks — Concat QKV Fusion.

Implements IC-DiT paper Eqs. 6-8: bidirectional concat-QKV joint attention.

    h^{z'}, h^{p'} = Attention([Q^z, Q^p], [K^z, K^p], [V^z, V^p])   (Eq.6: image<->text)
    h^{z'}, h^{l'} = Attention([Q^z, Q^l], [K^z, K^l], [V^z, V^l])   (Eq.7: image<->layout)
    h^{z'}, h^{e'} = Attention([Q^z, Q^e], [K^z, K^e], [V^z, V^e])   (Eq.8: image<->visual)

Key design:
  - Image tokens AND condition tokens are concatenated along the sequence dim.
  - A single joint self-attention is computed over the concatenated sequence.
  - The output is split back: updated image tokens + updated condition tokens.
  - Image tokens carry forward; condition tokens are also updated for the next layer.

Each MM-Attention block also includes:
  - Self-attention among image tokens (ResNet-style pre-norm residual)
  - Three modality concat-QKV fusions (image↔text, image↔layout, image↔visual)
  - Feed-forward network (MLP)

Usage (conda env):
    conda activate omg
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Concat QKV Joint Attention — the core mechanism from Eqs. 6-8
# ---------------------------------------------------------------------------

class ConcatQKVAttention(nn.Module):
    """
    Bidirectional joint attention via concatenated QKV.

    Implements a single modality fusion from the paper:
        h^{img'}, h^{cond'} = Attention([Q_img, Q_cond], [K_img, K_cond], [V_img, V_cond])

    Both image and condition tokens participate as joint queries and are
    updated together. Only image tokens are returned (condition tokens are
    updated in-place for subsequent layers).
    """

    def __init__(self, dim: int, num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.scale = self.head_dim ** -0.5

        # Modality-specific QKV projections (Fig. 3: "Linear" blocks per modality)
        self.q_proj_img = nn.Linear(dim, dim)
        self.k_proj_img = nn.Linear(dim, dim)
        self.v_proj_img = nn.Linear(dim, dim)

        self.q_proj_cond = nn.Linear(dim, dim)
        self.k_proj_cond = nn.Linear(dim, dim)
        self.v_proj_cond = nn.Linear(dim, dim)

        # Output projection (applied to the image-portion of the result)
        self.out_proj_img = nn.Linear(dim, dim)
        self.out_proj_cond = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        image_tokens: torch.Tensor,      # (B, N_img, dim)
        condition_tokens: torch.Tensor,   # (B, N_cond, dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint concat-QKV attention: image <-> condition.

        Returns:
            updated_image:     (B, N_img, dim)
            updated_condition: (B, N_cond, dim)
        """
        B, N_img, _ = image_tokens.shape
        _, N_cond, _ = condition_tokens.shape

        # --- Project image tokens ---
        Q_img = self.q_proj_img(image_tokens)  # (B, N_img, dim)
        K_img = self.k_proj_img(image_tokens)
        V_img = self.v_proj_img(image_tokens)

        # --- Project condition tokens ---
        Q_cond = self.q_proj_cond(condition_tokens)  # (B, N_cond, dim)
        K_cond = self.k_proj_cond(condition_tokens)
        V_cond = self.v_proj_cond(condition_tokens)

        # --- Concatenate along token dimension ---
        Q = torch.cat([Q_img, Q_cond], dim=1)  # (B, N_img+N_cond, dim)
        K = torch.cat([K_img, K_cond], dim=1)
        V = torch.cat([V_img, V_cond], dim=1)

        # --- Multi-head reshape ---
        N_total = N_img + N_cond
        Q = Q.reshape(B, N_total, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, h, N, d)
        K = K.reshape(B, N_total, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, N_total, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # --- Joint attention (memory-efficient, flash attention on Ampere+) ---
        # The old path cast Q/K/V to fp32 (~2 GB for visual fusion's
        # 1626×1626 attn matrix × batch 8 × 12 heads) and OOM'd a 24 GB GPU.
        #
        # T5 now runs in fp32 (frozen_encoders.py), so NaN is fixed at the
        # source and Q/K/V values are bounded by the projector clamp.
        # F.scaled_dot_product_attention stays in fp16 and uses < 1/2 the
        # memory of the old fp32 path.
        out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )
        # Safety clamp (fp16 max is 65504; keep 2σ margin).
        # Under normal conditions this is a no-op — attention output is a
        # convex combination of V, so it inherits V's range.
        if out.dtype == torch.float16:
            out = torch.clamp(out, min=-60000.0, max=60000.0)

        # --- Reshape back ---
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, N_total, self.dim)

        # --- Split output back ---
        out_img = out[:, :N_img, :]      # (B, N_img, dim)
        out_cond = out[:, N_img:, :]     # (B, N_cond, dim)

        # --- Output projections ---
        out_img = self.out_proj_img(out_img)
        out_cond = self.out_proj_cond(out_cond)

        return out_img, out_cond


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Standard MLP with GELU activation and fp16 safety clamp."""

    def __init__(self, dim: int, expansion_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # Safety: GELU × Linear can exceed fp16 max (65504) in deep stacks
        if x.dtype == torch.float16:
            out = torch.clamp(out, min=-60000.0, max=60000.0)
        return out


# ---------------------------------------------------------------------------
# MM-Attention Block — one layer of multimodal fusion
# ---------------------------------------------------------------------------

class MMAttentionBlock(nn.Module):
    """
    Single MM-Attention block (one transformer layer).

    Pipeline (pre-norm residual):
        1. Self-attention among image tokens
        2. Concat-QKV: image <-> text          (Eq.6)
        3. Concat-QKV: image <-> layout        (Eq.7)
        4. Concat-QKV: image <-> visual embed  (Eq.8)
        5. Feed-forward network

    Image tokens carry forward; condition tokens (text/layout/visual) are
    also updated after each concat-QKV fusion and passed to the next block.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        use_self_attn: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_self_attn = use_self_attn

        # --- Layer norms (pre-norm style) ---
        self.norm_self = nn.LayerNorm(dim) if use_self_attn else None
        self.norm_text_in = nn.LayerNorm(dim)
        self.norm_layout_in = nn.LayerNorm(dim)
        self.norm_visual_in = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

        # --- Condition token norms (for fp16 stability across residual layers) ---
        self.norm_text_cond = nn.LayerNorm(dim)
        self.norm_layout_cond = nn.LayerNorm(dim)
        self.norm_visual_cond = nn.LayerNorm(dim)

        # --- Condition token OUTPUT norms (bounds values between blocks) ---
        # Without these, 12 layers of residual updates can drive token values
        # past fp16 max even with internal attention clamps.
        self.norm_img_out = nn.LayerNorm(dim)
        self.norm_text_out = nn.LayerNorm(dim)
        self.norm_layout_out = nn.LayerNorm(dim)
        self.norm_visual_out = nn.LayerNorm(dim)

        # --- Self-attention among image tokens ---
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, batch_first=True
            )

        # --- Concat-QKV fusions (Eqs. 6-8) ---
        self.fusion_text = ConcatQKVAttention(dim, num_heads, dropout)
        self.fusion_layout = ConcatQKVAttention(dim, num_heads, dropout)
        self.fusion_visual = ConcatQKVAttention(dim, num_heads, dropout)

        # --- Feed-forward ---
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(
        self,
        image_tokens: torch.Tensor,    # (B, N_img, dim)
        text_tokens: torch.Tensor,     # (B, N_txt, dim)
        layout_tokens: torch.Tensor,   # (B, N_lay, dim)
        visual_tokens: torch.Tensor,   # (B, N_vis, dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns updated (image_tokens, text_tokens, layout_tokens, visual_tokens).
        """

        # ---- 1. Self-attention among image tokens ----
        if self.use_self_attn:
            x = self.norm_self(image_tokens)
            x, _ = self.self_attn(x, x, x)
            if x.dtype == torch.float16:
                x = torch.clamp(x, min=-60000.0, max=60000.0)
            image_tokens = image_tokens + x * 0.5

        # ---- 2. Concat-QKV: image <-> text (Eq.6) ----
        x_img = self.norm_text_in(image_tokens)
        x_txt = self.norm_text_cond(text_tokens)      # normalize cond tokens for fp16 safety
        delta_img, delta_txt = self.fusion_text(x_img, x_txt)
        # Scale residual to prevent fp16 overflow across 12 layers
        image_tokens = image_tokens + delta_img * 0.5
        text_tokens = text_tokens + delta_txt * 0.5

        # ---- 3. Concat-QKV: image <-> layout (Eq.7) ----
        x_img = self.norm_layout_in(image_tokens)
        x_lay = self.norm_layout_cond(layout_tokens)  # normalize cond tokens for fp16 safety
        delta_img, delta_lay = self.fusion_layout(x_img, x_lay)
        image_tokens = image_tokens + delta_img * 0.5
        layout_tokens = layout_tokens + delta_lay * 0.5

        # ---- 4. Concat-QKV: image <-> visual embedding (Eq.8) ----
        x_img = self.norm_visual_in(image_tokens)
        x_vis = self.norm_visual_cond(visual_tokens)  # normalize cond tokens for fp16 safety
        delta_img, delta_vis = self.fusion_visual(x_img, x_vis)
        image_tokens = image_tokens + delta_img * 0.5
        visual_tokens = visual_tokens + delta_vis * 0.5

        # ---- 5. Feed-forward ----
        x = self.norm_ffn(image_tokens)
        x = self.ffn(x)
        image_tokens = image_tokens + x

        # ---- 6. Output safety clamp + norms ----
        if image_tokens.dtype == torch.float16:
            image_tokens = torch.clamp(image_tokens, min=-60000.0, max=60000.0)
        image_tokens   = self.norm_img_out(image_tokens)
        text_tokens    = self.norm_text_out(text_tokens)
        layout_tokens  = self.norm_layout_out(layout_tokens)
        visual_tokens  = self.norm_visual_out(visual_tokens)

        return image_tokens, text_tokens, layout_tokens, visual_tokens


# ---------------------------------------------------------------------------
# MM-Attention Stack — multiple layers
# ---------------------------------------------------------------------------

class MMAttentionStack(nn.Module):
    """Stack of N MM-Attention blocks."""

    def __init__(
        self,
        num_layers: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        use_self_attn: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MMAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                use_self_attn=use_self_attn,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        layout_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Each block updates all four token sets (image, text, layout, visual).
        Only image tokens are used downstream for noise prediction.
        """
        for block in self.blocks:
            image_tokens, text_tokens, layout_tokens, visual_tokens = block(
                image_tokens, text_tokens, layout_tokens, visual_tokens
            )
        return image_tokens, text_tokens, layout_tokens, visual_tokens
