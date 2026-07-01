"""
MVTec AD dataset for layout-guided defect generation.

Usage (conda env):
    conda activate omg

Expected directory structure:
    mvtec/
      bottle/
        train/
          good/
            xxx.png
        test/
          good/
          broken_large/
          broken_small/
          contamination/
          ...
        ground_truth/
          broken_large/
            xxx_mask.png
          broken_small/
            xxx_mask.png
          contamination/
            xxx_mask.png
"""

import os
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils.image_utils import dilate_mask, random_mask_augment
from datasets.prompt_templates import build_prompt


class MVTecDefectDataset(Dataset):
    """MVTec AD dataset for defect generation training."""

    # All MVTec AD categories
    ALL_CATEGORIES = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
    ]

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        categories: str = "all",
        use_good_images: bool = True,
        use_defective_images: bool = True,
        use_reference_normal: bool = True,
        mask_dilation: int = 0,
        mask_random_augment: bool = True,
        split: str = "train",
        descriptions_path: Optional[str] = None,
    ):
        """
        Args:
            root: Root directory of MVTec AD dataset.
            image_size: Target image size (square).
            categories: "all" or list of category names.
            use_good_images: Include normal (good) images.
            use_defective_images: Include defective images.
            use_reference_normal: Sample a reference normal image per sample.
            mask_dilation: Dilate mask by this many pixels.
            mask_random_augment: Apply random augmentation to masks.
            split: "train" or "val" (MVTec test set used as val).
            descriptions_path: Path to mvtec_descriptions.json (VLM-generated
                              per-image descriptions). If None, falls back to
                              hand-written DEFECT_DESCRIPTIONS.
        """
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.use_good_images = use_good_images
        self.use_defective_images = use_defective_images
        self.use_reference_normal = use_reference_normal
        self.mask_dilation = mask_dilation
        self.mask_random_augment = mask_random_augment
        self.split = split

        # ---- Load VLM descriptions (per-image) ----
        self._vlm_lookup = {}  # {(cat, defect_type, image_id): "The defect shape is ..."}
        if descriptions_path is not None and os.path.exists(descriptions_path):
            loaded = 0
            with open(descriptions_path, "r", encoding="utf-8") as f:
                vlm_data = json.load(f)
            for entry in vlm_data:
                desc = entry.get("description", "")
                if desc:
                    key = (entry["category"], entry["defect_type"], entry["image_id"])
                    self._vlm_lookup[key] = desc
                    loaded += 1
            print(f"[MVTec] Loaded {loaded} VLM descriptions from {descriptions_path}")

        # Resolve categories
        if categories == "all":
            self.categories = self.ALL_CATEGORIES
        elif isinstance(categories, str):
            self.categories = [c.strip() for c in categories.split(",")]
        else:
            self.categories = categories

        # Verify categories exist
        for cat in self.categories:
            cat_path = self.root / cat
            if not cat_path.exists():
                raise FileNotFoundError(f"Category directory not found: {cat_path}")

        # Image transforms
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [0,1] -> [-1,1]
        ])

        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        ])

        # Collect all samples
        self.samples = []
        self._reference_pool = {}  # category -> list of normal image paths

        for cat in self.categories:
            train_good_dir = self.root / cat / "train" / "good"
            if train_good_dir.exists():
                self._reference_pool[cat] = sorted(list(train_good_dir.glob("*.png")))

            # Use test set for both train and val (MVTec has no val split)
            test_dir = self.root / cat / "test"
            gt_dir = self.root / cat / "ground_truth"

            if not test_dir.exists():
                print(f"[WARNING] Test directory not found for {cat}: {test_dir}")
                continue

            for defect_type_dir in sorted(test_dir.iterdir()):
                if not defect_type_dir.is_dir():
                    continue

                defect_type = defect_type_dir.name

                # Skip based on use flags
                if defect_type == "good" and not use_good_images:
                    continue
                if defect_type != "good" and not use_defective_images:
                    continue

                image_paths = sorted(list(defect_type_dir.glob("*.png")))

                for img_path in image_paths:
                    # Find corresponding mask
                    mask_path = None
                    if defect_type != "good":
                        gt_defect_dir = gt_dir / defect_type
                        if gt_defect_dir.exists():
                            # Mask filenames usually end with _mask.png
                            base_name = img_path.stem
                            candidate_mask = gt_defect_dir / f"{base_name}_mask.png"
                            if candidate_mask.exists():
                                mask_path = candidate_mask

                    self.samples.append({
                        "image_path": str(img_path),
                        "mask_path": str(mask_path) if mask_path else None,
                        "category": cat,
                        "defect_type": defect_type,
                    })

        # ---- Train/Val split (stratified-ish: shuffle each category independently) ----
        # MVTec has no official split; we create an 80/20 random split here.
        # Fixed seed ensures reproducibility across runs.
        rng = random.Random(42)
        train_samples, val_samples = [], []
        for cat in self.categories:
            cat_samples = [s for s in self.samples if s["category"] == cat]
            rng.shuffle(cat_samples)
            n_train = int(len(cat_samples) * 0.8)
            train_samples.extend(cat_samples[:n_train])
            val_samples.extend(cat_samples[n_train:])

        if split == "train":
            self.samples = train_samples
        elif split == "val":
            self.samples = val_samples
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        print(f"[MVTec] Loaded {len(self.samples)} samples for split='{split}' (total pool: {len(train_samples)+len(val_samples)})")
        for cat in self.categories:
            cat_samples = [s for s in self.samples if s["category"] == cat]
            defect_types = set(s["defect_type"] for s in cat_samples)
            print(f"  {cat}: {len(cat_samples)} samples, defects: {defect_types}")

    def _get_reference_image(self, category: str, exclude_path: Optional[str] = None) -> str:
        """Get a random reference normal image from the same category."""
        pool = self._reference_pool.get(category, [])
        if not pool:
            # Fallback: return the same image
            return None

        if len(pool) == 1:
            return str(pool[0])

        # Sample randomly, optionally excluding the target image
        candidates = pool if exclude_path is None else [p for p in pool if str(p) != exclude_path]
        if not candidates:
            return str(pool[0])

        return str(random.choice(candidates))

    def _generate_prompt(self, category: str, defect_type: str, image_path: str = "") -> str:
        """
        Generate text prompt. Uses per-image VLM description if available,
        otherwise falls back to WinCLIP-style CPE template pool.
        CFG null prompt is NOT handled here — done in training loop.
        """
        # Try per-image VLM description
        vlm_desc = None
        if image_path and self._vlm_lookup:
            image_id = self._extract_image_id(image_path, category, defect_type)
            if image_id:
                vlm_desc = self._vlm_lookup.get((category, defect_type, image_id))

        return build_prompt(category, defect_type, null_prompt_prob=0.0, vlm_description=vlm_desc)

    def _extract_image_id(self, image_path: str, category: str, defect_type: str) -> str:
        """Extract image_id in the same format as mvtec_describe.py parse_mvtc_path.

        e.g. /root/bottle/test/broken_large/001.png → "001.png"
             /root/bottle/test/broken_large/sub/001.png → "sub/001.png"
        """
        try:
            rel = str(Path(image_path).relative_to(self.root))
            parts = Path(rel).parts
            # parts: ("bottle", "test", "broken_large", "001.png") or deeper
            if len(parts) >= 4:
                return str(Path(*parts[3:]))
            elif len(parts) >= 3:
                return str(Path(*parts[2:]))
        except ValueError:
            pass
        return ""

    def _load_mask(self, mask_path: Optional[str]) -> np.ndarray:
        """Load and preprocess mask. Returns numpy array (H, W) in [0, 1]."""
        if mask_path is None:
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)

        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)

        # Binarize
        mask = (mask > 127).astype(np.uint8) * 255

        # Resize
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize(
            (self.image_size, self.image_size),
            Image.NEAREST
        )
        mask = np.array(mask_img).astype(np.float32) / 255.0

        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load target image
        image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = self.image_transform(image)

        # Load mask
        mask = self._load_mask(sample["mask_path"])
        is_anomaly = sample["defect_type"] != "good"

        # Dilate mask if configured
        if self.mask_dilation > 0 and mask.sum() > 0:
            mask = dilate_mask(mask, self.mask_dilation)

        # Random mask augmentation (only during training)
        if self.mask_random_augment and self.split == "train" and mask.sum() > 0:
            mask = random_mask_augment(mask)

        mask_tensor = torch.from_numpy(mask).float()  # (H, W)

        # Generate prompt (uses per-image VLM description if available)
        prompt = self._generate_prompt(sample["category"], sample["defect_type"], sample["image_path"])

        # Get reference image
        reference_tensor = None
        if self.use_reference_normal:
            ref_path = self._get_reference_image(
                sample["category"],
                exclude_path=sample["image_path"] if sample["defect_type"] == "good" else None
            )
            if ref_path:
                ref_image = Image.open(ref_path).convert("RGB")
                reference_tensor = self.image_transform(ref_image)
            else:
                # Fallback: use the image itself
                reference_tensor = image_tensor.clone()

        return {
            "image": image_tensor,               # (3, H, W) in [-1, 1]
            "mask": mask_tensor,                 # (H, W) in [0, 1]
            "category": sample["category"],      # str
            "defect_type": sample["defect_type"],# str
            "prompt": prompt,                    # str
            "reference_image": reference_tensor, # (3, H, W) in [-1, 1] or None
            "is_anomaly": is_anomaly,            # bool
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for the dataset."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    categories = [item["category"] for item in batch]
    defect_types = [item["defect_type"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    is_anomaly = [item["is_anomaly"] for item in batch]

    # Reference images may be None
    ref_images = []
    for item in batch:
        ref = item["reference_image"]
        if ref is not None:
            ref_images.append(ref)
        else:
            # Use the target image as fallback
            ref_images.append(item["image"])
    reference_images = torch.stack(ref_images, dim=0)

    return {
        "image": images,
        "mask": masks,
        "category": categories,
        "defect_type": defect_types,
        "prompt": prompts,
        "reference_image": reference_images,
        "is_anomaly": is_anomaly,
    }
