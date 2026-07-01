#!/usr/bin/env python
"""
Environment checker for IC-DiT defect generation training.
Run this first on any new server to verify everything is ready.

Usage:
    python check_env.py
"""

import sys
import importlib

# ── Package list: (import_name, pip_name, required)
#    required=False means optional / nice-to-have
REQUIRED = [
    ("torch",        "torch",        True),
    ("torchvision",  "torchvision",  True),
    ("diffusers",    "diffusers",    True),
    ("transformers", "transformers", True),
    ("accelerate",   "accelerate",   True),
    ("timm",         "timm",         True),
    ("cv2",          "opencv-python", True),
    ("PIL",          "Pillow",        True),
    ("yaml",         "PyYAML",        True),
    ("numpy",        "numpy",         True),
    ("tqdm",         "tqdm",          True),
]

OPTIONAL = [
    ("tensorboard",  "tensorboard",   False),
    ("einops",       "einops",        False),
    ("skimage",      "scikit-image",  False),
]


def check_package(import_name: str, pip_name: str) -> dict:
    """Try to import a package and collect version info."""
    result = {"pip_name": pip_name, "available": False, "version": None, "error": None}
    try:
        mod = importlib.import_module(import_name)
        result["available"] = True
        result["version"] = getattr(mod, "__version__", "?")
        # Handle cv2
        if import_name == "cv2":
            result["version"] = getattr(mod, "__version__", getattr(mod, "getVersionString", lambda: "?")())
        if import_name == "PIL":
            from PIL import Image
            result["version"] = getattr(Image, "__version__", "?")
    except ImportError as e:
        result["error"] = str(e)
    return result


def main():
    ok = 0
    missing_required = 0
    total = len(REQUIRED) + len(OPTIONAL)

    print("=" * 65)
    print("  IC-DiT Environment Check")
    print("=" * 65)

    # ── Python ──
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  Python        {py_ver}")

    # ── PyTorch / CUDA ──
    results = {}
    for import_name, pip_name, required in REQUIRED + OPTIONAL:
        results[import_name] = check_package(import_name, pip_name)

    torch_info = results["torch"]
    if torch_info["available"]:
        import torch
        print(f"  PyTorch       {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available {cuda_available}")
        if cuda_available:
            print(f"  CUDA version  {torch.version.cuda}")
            print(f"  cuDNN version {torch.backends.cudnn.version()}")
            print(f"  GPU count     {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024 ** 3)
                print(f"  GPU[{i}]        {props.name} ({vram_gb:.1f} GB)")
        else:
            print("  ⚠ GPU not available — training will be very slow on CPU")
    else:
        print("  ✗ PyTorch NOT FOUND — install first")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    # ── Required packages ──
    print(f"\n{'─' * 50}")
    print("  Required packages")
    print(f"{'─' * 50}")
    for import_name, pip_name, _ in REQUIRED:
        r = results[import_name]
        status = "✓" if r["available"] else "✗ MISSING"
        ver = f"({r['version']})" if r["available"] else ""
        line = f"  {status:12s} {pip_name:20s} {ver}"
        print(line)
        if not r["available"]:
            missing_required += 1
        else:
            ok += 1

    # ── Optional packages ──
    print(f"\n{'─' * 50}")
    print("  Optional packages")
    print(f"{'─' * 50}")
    for import_name, pip_name, _ in OPTIONAL:
        r = results[import_name]
        status = "✓" if r["available"] else "- not installed"
        ver = f"({r['version']})" if r["available"] else ""
        print(f"  {status:12s} {pip_name:20s} {ver}")
        if r["available"]:
            ok += 1

    # ── Summary ──
    print(f"\n{'─' * 50}")
    if missing_required == 0:
        print(f"  ✅ All required packages available ({ok}/{total})")
    else:
        print(f"  ❌ {missing_required} required package(s) missing")
        print(f"  Install with: pip install {' '.join(r['pip_name'] for i, p, rq in REQUIRED if not results[i]['available'])}")
    print(f"{'─' * 50}\n")

    return missing_required


if __name__ == "__main__":
    sys.exit(main())
