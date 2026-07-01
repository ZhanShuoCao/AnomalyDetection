
import numpy as np
from PIL import Image
from pathlib import Path

root = Path('/home/czs/Desktop/MVTec')
for cat, dtype in [('cable','cable_swap'), ('tile','oil'),
                    ('cable','poke_insulation'), ('grid','metal_contamination')]:
    gt_dir = root / cat / 'ground_truth' / dtype
    masks = sorted(gt_dir.glob('*.png'))[:2]
    for m in masks:
        arr = np.array(Image.open(m))
        unique = np.unique(arr)
        nonzero = (arr > 0).sum()
        total = arr.size
        print(f'{cat}/{dtype}/{m.name}: shape={arr.shape}, unique_values={unique}, '
            f'nonzero={nonzero}/{total} ({100*nonzero/total:.4f}%)')