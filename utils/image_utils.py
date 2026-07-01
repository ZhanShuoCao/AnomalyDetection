"""
Image utility functions.

Usage (conda env):
    conda activate omg
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union, Optional


def load_image(path: str) -> Image.Image:
    """Load an image from path, convert to RGB."""
    img = Image.open(path).convert("RGB")
    return img


def resize_image(image: Union[Image.Image, np.ndarray], size: int) -> np.ndarray:
    """Resize image to (size, size) and return as numpy array."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.resize((size, size), Image.BILINEAR)
    return np.array(image)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image from [0,255] to [-1, 1]."""
    image = image.astype(np.float32) / 127.5 - 1.0
    return image


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image from [-1, 1] to [0, 255]."""
    image = (image + 1.0) * 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor in [-1, 1] to PIL Image."""
    tensor = tensor.detach().cpu()
    # Assume tensor shape: (C, H, W) or (1, C, H, W)
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)
    tensor = (tensor * 255).to(torch.uint8)
    # CHW -> HWC
    arr = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor in [-1, 1], shape (C, H, W)."""
    arr = np.array(image).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
    return tensor


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Dilate a binary mask."""
    if kernel_size <= 0:
        return mask
    import cv2
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def random_mask_augment(mask: np.ndarray) -> np.ndarray:
    """Apply random augmentation to binary mask (small translation, scaling).

    Accepts both [0, 1] float and [0, 255] uint8 masks, always returns
    the same dtype and range as the input.
    """
    import cv2
    was_float = mask.dtype == np.float32 or mask.dtype == np.float64
    # Work internally in uint8 [0, 255]
    if was_float:
        mask_u8 = (mask * 255).astype(np.uint8)
    else:
        mask_u8 = mask.astype(np.uint8)

    h, w = mask_u8.shape
    # Random translation
    tx = np.random.randint(-5, 5)
    ty = np.random.randint(-5, 5)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    mask_u8 = cv2.warpAffine(mask_u8, M, (w, h),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)

    # Random scale
    scale = np.random.uniform(0.9, 1.1)
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Pad or crop back to original size
    if scale < 1.0:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        mask_u8[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled
    else:
        crop_h = (new_h - h) // 2
        crop_w = (new_w - w) // 2
        mask_u8 = scaled[crop_h:crop_h+h, crop_w:crop_w+w]

    mask_u8 = (mask_u8 > 127).astype(np.uint8) * 255

    if was_float:
        return mask_u8.astype(np.float32) / 255.0
    return mask_u8


def save_image_grid(images: list, save_path: str, nrow: int = 4):
    """Save a grid of PIL images."""
    from torchvision.utils import make_grid

    tensors = []
    for img in images:
        if isinstance(img, np.ndarray):
            img = pil_to_tensor(Image.fromarray(img))
        elif isinstance(img, Image.Image):
            img = pil_to_tensor(img)
        tensors.append(img)

    grid = make_grid(tensors, nrow=nrow, normalize=True, value_range=(-1, 1))
    grid_img = tensor_to_pil(grid)
    grid_img.save(save_path)
    print(f"Saved image grid to {save_path}")
