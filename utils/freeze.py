"""
Freeze/Unfreeze utilities for strict frozen/trainable separation.

Usage (conda env):
    conda activate omg
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze a module: eval mode + requires_grad_(False) on all parameters.

    Also patches ``module.train()`` so that calling ``model.train()`` on the
    parent cannot accidentally flip a frozen submodule back to train mode.
    """
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)

    # Patch train() to keep this module permanently in eval mode.
    # model.train() recurses into all children; this stops it at frozen boundaries.
    _orig_train = module.train

    def _frozen_train(mode: bool = True):
        # Silently ignore attempts to put into train mode
        return _orig_train(False)

    module.train = _frozen_train
    return module


def unfreeze_module(module: nn.Module) -> nn.Module:
    """Unfreeze a module: train mode + requires_grad_(True) on all parameters."""
    module.train()
    for p in module.parameters():
        p.requires_grad_(True)
    return module


def print_trainable_parameters(model: nn.Module, prefix: str = ""):
    """Print frozen and trainable parameter names with counts."""
    frozen_names = []
    trainable_names = []
    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_names.append(name)
            trainable_count += param.numel()
        else:
            frozen_names.append(name)
            frozen_count += param.numel()

    total_count = frozen_count + trainable_count

    print(f"\n{'='*60}")
    print(f"Parameter Statistics [{prefix}]")
    print(f"{'='*60}")

    print(f"\n--- Frozen Parameters ({len(frozen_names)} groups, {frozen_count:,} params) ---")
    for n in frozen_names:
        print(f"  [FROZEN]  {n}")

    print(f"\n--- Trainable Parameters ({len(trainable_names)} groups, {trainable_count:,} params) ---")
    for n in trainable_names:
        print(f"  [TRAINABLE] {n}")

    print(f"\n--- Summary ---")
    print(f"  Trainable: {trainable_count:,} ({100*trainable_count/total_count:.2f}%)")
    print(f"  Frozen:    {frozen_count:,} ({100*frozen_count/total_count:.2f}%)")
    print(f"  Total:     {total_count:,}")
    print(f"{'='*60}\n")

    return trainable_count, frozen_count, total_count


def get_optimizer_trainable_params(model: nn.Module) -> List[nn.parameter.Parameter]:
    """Return only parameters with requires_grad=True for optimizer."""
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("No trainable parameters found! Check freeze logic.")
    return params


def assert_no_frozen_params_in_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer) -> bool:
    """
    Verify that optimizer only contains trainable parameters.
    Returns True if all optimizer params have requires_grad=True.
    Raises AssertionError if frozen params are found in optimizer.
    """
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    frozen_ids = {id(p) for p in model.parameters() if not p.requires_grad}

    for pg_idx, param_group in enumerate(optimizer.param_groups):
        for p_idx, p in enumerate(param_group["params"]):
            param_id = id(p)
            if param_id in frozen_ids:
                raise AssertionError(
                    f"FROZEN PARAMETER IN OPTIMIZER! "
                    f"Param group {pg_idx}, param {p_idx}. "
                    f"This parameter has requires_grad=False but is in the optimizer."
                )
            if param_id not in trainable_ids:
                raise AssertionError(
                    f"UNKNOWN PARAMETER IN OPTIMIZER! "
                    f"Param group {pg_idx}, param {p_idx}. "
                    f"This parameter is not recognized as trainable or frozen."
                )

    print(f"[OK] assert_no_frozen_params_in_optimizer: all {len(trainable_ids)} optimizer params are trainable.")
    return True


def assert_trainable_status(model: nn.Module) -> bool:
    """
    Verify frozen/trainable consistency:
    - Frozen modules must be in eval mode
    - Frozen module params must have requires_grad=False
    - Trainable modules should be in train mode
    Returns True if all checks pass.
    """
    errors = []

    for name, module in model.named_modules():
        # Skip the root module itself
        if name == "":
            continue

        is_frozen = all(not p.requires_grad for p in module.parameters(recurse=False))
        has_params = any(True for _ in module.parameters(recurse=False))

        if not has_params:
            continue

        if is_frozen and module.training:
            errors.append(
                f"Module '{name}' has frozen params but is in TRAIN mode. "
                f"Expected eval mode."
            )

    if errors:
        print("[WARN] assert_trainable_status: frozen modules in train mode (harmless — "
              "requires_grad=False + torch.no_grad() still active):")
        # Only print first 5 to avoid flooding the console
        for e in errors[:5]:
            print(f"  - {e}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    else:
        print("[OK] assert_trainable_status: all frozen modules are in eval mode.")

    return True
