#!/usr/bin/env python
"""Quick test: can we download models from HuggingFace?"""
import os

# Set mirror first — safest for mainland China servers
if os.environ.get("HF_ENDPOINT") is None:
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

print(f"HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', 'default')}")

# Test 1: CLIP
print("\n[1/3] Testing CLIP...")
try:
    from transformers import CLIPTextModel, CLIPTokenizer
    model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    print("  ✓ CLIP OK")
except Exception as e:
    print(f"  ✗ CLIP FAILED: {e}")

# Test 2: SD VAE
print("\n[2/3] Testing SD VAE...")
try:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    print("  ✓ SD VAE OK")
except Exception as e:
    print(f"  ✗ SD VAE FAILED: {e}")

# Test 3: DINOv2
print("\n[3/3] Testing DINOv2...")
try:
    import timm
    encoder = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
    print("  ✓ DINOv2 OK")
except Exception as e:
    print(f"  ✗ DINOv2 FAILED: {e}")

print("\nDone.")
