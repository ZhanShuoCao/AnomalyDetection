#!/usr/bin/env python
"""
Sampling/Inference script for IC-DiT defect generation.

Usage (conda env):
    conda activate omg

    # Generate with mask from file:
    python sample.py \
      --ckpt checkpoints/best.pt \
      --category bottle \
      --defect_type broken_large \
      --mask_path demo/mask.png \
      --reference_image demo/normal.png \
      --prompt "an industrial image of a bottle with a broken large defect localized in the provided mask region"

    # Generate with all-zero mask (defect-free):
    python sample.py \
      --ckpt checkpoints/best.pt \
      --category bottle \
      --defect_type good \
      --reference_image demo/normal.png \
      --prompt "a defect-free industrial image of a bottle"
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.icdit_defect_generator import ICDiTDefectGenerator
from diffusion.scheduler import NoiseScheduler
from utils.image_utils import tensor_to_pil, pil_to_tensor
from utils.train_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="IC-DiT Defect Generation Sampling")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--category", type=str, required=True,
                        help="Product category (bottle, capsule, etc.)")
    parser.add_argument("--defect_type", type=str, required=True,
                        help="Defect type (scratch, crack, hole, etc.)")
    parser.add_argument("--mask_path", type=str, default=None,
                        help="Path to binary mask image (PNG). If not provided, uses all-zero mask.")
    parser.add_argument("--reference_image", type=str, default=None,
                        help="Path to reference normal image (PNG)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt. If not provided, generates automatically.")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of DDIM inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size (must match training config)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--return_aux", action="store_true", default=False,
                        help="Also return multimodal tokens for evaluation metrics "
                             "(Mask Faithfulness, CLIP-FID, Embedding Similarity)")
    return parser.parse_args()


def load_model(ckpt_path: str, device: torch.device, image_size: int):
    """Load model from checkpoint."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    # If config not in checkpoint, use default
    if not config:
        print("[WARNING] Config not found in checkpoint. Using default config.")
        config = {
            "model": {
                "text_encoder_name": "openai/clip-vit-large-patch14",
                "vae_name": "stabilityai/sd-vae-ft-mse",
                "visual_encoder_name": "dinov2_vitb14",
                "hidden_dim": 768,
                "num_layers": 12,
                "num_heads": 12,
                "patch_size": 2,
                "latent_channels": 4,
                "training_mode": "full_generator",
                "visual_embedding_source": "reference_normal",
                "freeze_text_encoder": True,
                "freeze_vae": True,
                "freeze_layout_encoder": True,
                "freeze_visual_encoder": True,
            }
        }

    model = ICDiTDefectGenerator(config, image_size=image_size)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    print(f"[Model] Loaded from: {ckpt_path}")
    if "val_loss" in checkpoint:
        print(f"[Model] Checkpoint val_loss: {checkpoint['val_loss']:.6f}")
    if "global_step" in checkpoint:
        print(f"[Model] Checkpoint step: {checkpoint['global_step']}")

    return model, config


def load_or_create_mask(mask_path: str, image_size: int, device: torch.device) -> torch.Tensor:
    """Load mask from file or create all-zero mask."""
    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((image_size, image_size), Image.NEAREST)
        mask_arr = np.array(mask).astype(np.float32) / 255.0
        mask_arr = (mask_arr > 0.5).astype(np.float32)  # Binarize
        print(f"[Mask] Loaded from: {mask_path}, anomaly area: {mask_arr.sum():.0f} pixels")
    else:
        mask_arr = np.zeros((image_size, image_size), dtype=np.float32)
        print(f"[Mask] Using all-zero mask (defect-free generation)")

    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).to(device)  # (1, H, W)
    return mask_tensor


def load_reference_image(ref_path: str, image_size: int, device: torch.device) -> torch.Tensor:
    """Load reference image or create blank."""
    if ref_path and os.path.exists(ref_path):
        img = Image.open(ref_path).convert("RGB")
        tensor = pil_to_tensor(img).unsqueeze(0).to(device)
        # Resize if needed
        if tensor.shape[2] != image_size:
            tensor = F.interpolate(tensor, size=(image_size, image_size), mode='bilinear')
        print(f"[Reference] Loaded from: {ref_path}")
    else:
        # Default: all-gray image
        tensor = torch.zeros(1, 3, image_size, image_size, device=device)
        print(f"[Reference] No reference image provided. Using gray image.")
    return tensor


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # Load model
    model, config = load_model(args.ckpt, device, args.image_size)

    # Build noise scheduler
    diffusion_cfg = config.get("diffusion", {})
    noise_scheduler = NoiseScheduler(
        num_train_timesteps=diffusion_cfg.get("num_train_timesteps", 1000),
        beta_schedule=diffusion_cfg.get("beta_schedule", "linear"),
        prediction_type=diffusion_cfg.get("prediction_type", "epsilon"),
    )

    # Prepare inputs
    masks = load_or_create_mask(args.mask_path, args.image_size, device)
    # Repeat for num_samples
    masks = masks.repeat(args.num_samples, 1, 1)  # (B, H, W)

    reference = load_reference_image(args.reference_image, args.image_size, device)
    reference = reference.repeat(args.num_samples, 1, 1, 1)  # (B, 3, H, W)

    # Generate prompts
    if args.prompt:
        prompts = [args.prompt] * args.num_samples
    else:
        defect_name = args.defect_type.replace("_", " ")
        if args.defect_type == "good":
            prompts = [f"a defect-free industrial image of a {args.category}"] * args.num_samples
        else:
            prompts = [
                f"an industrial image of a {args.category} with a {defect_name} defect "
                f"localized in the provided mask region"
            ] * args.num_samples
    print(f"[Prompt] {prompts[0]}")

    # Generate
    print(f"[Generation] Running {args.num_inference_steps} DDIM steps...")
    with torch.no_grad():
        result = model.generate(
            masks=masks,
            prompts=prompts,
            reference_images=reference,
            noise_scheduler=noise_scheduler,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            return_aux=args.return_aux,
        )

    if args.return_aux:
        generated, aux = result
    else:
        generated = result

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.num_samples):
        img_tensor = generated[i]  # (3, H, W) in [-1, 1]
        img = tensor_to_pil(img_tensor)
        output_path = os.path.join(args.output_dir, f"generated_{i:02d}.png")
        img.save(output_path)
        print(f"[Saved] {output_path}")

    # ---- Compute evaluation metrics from auxiliary tokens ----
    if args.return_aux:
        B = args.num_samples
        # Mask Faithfulness (Dice coefficient)
        mask_16 = F.interpolate(
            masks.unsqueeze(1), size=(16, 16), mode='area'
        ).squeeze(1).reshape(B, 256)
        layout_prob = torch.sigmoid(aux["layout_logits"])  # (B, 256)
        layout_pred = (layout_prob > 0.5).float()
        intersection = (layout_pred * mask_16).sum(dim=1)
        union = layout_pred.sum(dim=1) + mask_16.sum(dim=1)
        dice = (2 * intersection / (union + 1e-8)).mean().item()
        print(f"\n[Metrics] Mask Faithfulness (Dice): {dice:.4f}")

        # Embedding Similarity (cosine between visual tokens and initial encoding)
        # visual tokens are (B, 1369, 768); pool to global then cosine
        vis_upd_pooled = aux["visual_tokens"].mean(dim=1)  # (B, 768)
        vis_init_pooled = model.encode_visual(reference).mean(dim=1)  # (B, 768)
        emb_sim = F.cosine_similarity(vis_upd_pooled, vis_init_pooled, dim=1).mean().item()
        print(f"[Metrics] Embedding Similarity:    {emb_sim:.4f}")

        # Text-Image alignment (cosine between text tokens and visual tokens)
        text_pooled = aux["text_tokens"].mean(dim=1)  # (B, 768)
        text_sim = F.cosine_similarity(text_pooled, vis_upd_pooled, dim=1).mean().item()
        print(f"[Metrics] Text-Image Similarity:   {text_sim:.4f}")

    print(f"\n[Done] Generated {args.num_samples} images in {args.output_dir}")


if __name__ == "__main__":
    main()
