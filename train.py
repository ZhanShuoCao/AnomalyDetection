#!/usr/bin/env python
"""
Training script for IC-DiT layout-guided defect generation on MVTec AD.

Usage (conda env):
    conda activate omg
    accelerate launch train.py --config configs/mvtec_icdit.yaml
"""

import os
import sys
import copy
import random
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts
from accelerate import Accelerator

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.freeze import (
    get_optimizer_trainable_params,
    print_trainable_parameters,
    assert_no_frozen_params_in_optimizer,
    assert_trainable_status,
)
from utils.train_utils import set_seed, load_config, save_config, AverageMeter
from utils.logging_utils import Logger
from utils.image_utils import tensor_to_pil, save_image_grid
from datasets.mvtec import MVTecDefectDataset, collate_fn
from models.icdit_defect_generator import ICDiTDefectGenerator
from diffusion.scheduler import NoiseScheduler


# ---------------------------------------------------------------------------
# Exponential Moving Average (EMA) for stable sampling
# ---------------------------------------------------------------------------

class EMAModel:
    """
    Exponential Moving Average of model weights (shadows stored on CPU to save GPU memory).

    Keeps a shadow copy of trainable parameters on CPU, updated with:
        shadow = decay * shadow + (1 - decay) * param

    During sampling/evaluation, apply_shadow() temporarily swaps in the
    EMA weights (moving them to GPU); restore() puts the raw training weights back.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}  # stored on CPU
        self.backup: dict[str, torch.Tensor] = {}
        self._registered = False

    def register(self):
        """Snapshot current trainable parameters as initial shadow values (on CPU)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach().cpu()
        self._registered = True

    def update(self):
        """Update shadow weights (kept on CPU to save GPU memory)."""
        if not self._registered:
            self.register()
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # shadow = decay * shadow + (1-decay) * param
                    # Move param to CPU for update, keep shadow on CPU
                    new_val = param.data.cpu()
                    self.shadow[name].mul_(self.decay).add_(
                        new_val, alpha=1.0 - self.decay
                    )

    def apply_shadow(self):
        """Temporarily replace model weights with EMA shadow weights (CPU→GPU)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device))

    def restore(self):
        """Restore original training weights after apply_shadow()."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict: dict):
        self.shadow = {k: v.cpu() for k, v in state_dict["shadow"].items()}
        self.decay = state_dict.get("decay", self.decay)
        self._registered = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train IC-DiT for defect generation")
    parser.add_argument("--config", type=str, default="configs/mvtec_icdit.yaml",
                        help="Path to config YAML")
    return parser.parse_args()


def validate(model, val_loader, noise_scheduler, device):
    """Compute validation loss (with dropout OFF)."""
    model.eval()
    val_loss = AverageMeter()

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            prompts = batch["prompt"]
            reference_images = batch["reference_image"].to(device)

            B = images.shape[0]
            H_lat, W_lat = images.shape[2] // 8, images.shape[3] // 8

            # Encode → add noise → predict (using model's component methods)
            latents = model.encode_image_to_latent(images)
            noise = torch.randn(B, 4, H_lat, W_lat, device=device)
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (B,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # forward() returns 7 values; we only need eps_pred for val loss
            result = model.forward(
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                prompts=prompts,
                masks=masks,
                reference_images=reference_images,
            )
            eps_pred = result[0]

            loss = F.mse_loss(eps_pred, noise)
            val_loss.update(loss.item(), B)

    model.train()
    return val_loss.avg


# ---------------------------------------------------------------------------
# Fixed sample set: one defective image per category (screw, zipper, pill, transistor)
# so generated samples are always comparable across training runs.
# ---------------------------------------------------------------------------
FIXED_SAMPLE_CATEGORIES = ["screw", "zipper", "pill", "transistor"]

def build_fixed_sample_set(dataset: MVTecDefectDataset, image_size: int, device: torch.device):
    """Pick the first defective image from each FIXED_SAMPLE_CATEGORY."""
    fixed = []
    for cat in FIXED_SAMPLE_CATEGORIES:
        found = None
        for idx, sample in enumerate(dataset.samples):
            if sample["category"] == cat and sample["defect_type"] != "good":
                found = dataset[idx]
                break
        if found is None:
            raise RuntimeError(f"No defective sample found for category: {cat}")
        fixed.append(found)

    collated = collate_fn(fixed)
    images = collated["image"].to(device)
    masks = collated["mask"].to(device)
    prompts = collated["prompt"]
    reference_images = collated["reference_image"].to(device)
    defect_types = collated["defect_type"]
    categories = collated["category"]
    return images, masks, prompts, reference_images, defect_types, categories


@torch.no_grad()
def generate_samples(model, fixed_sample_set, noise_scheduler, num_inference_steps,
                     output_dir, step, logger=None):
    """Generate sample images from a FIXED sample set and save a comparison grid."""
    device = next(model.parameters()).device
    images, masks, prompts, reference_images, defect_types, categories = fixed_sample_set

    n = len(images)
    # Resize masks to model input size if needed
    if masks.shape[1] != model.image_size:
        masks = F.interpolate(
            masks.unsqueeze(1), size=(model.image_size, model.image_size),
            mode='nearest'
        ).squeeze(1)

    # Diagnostic
    for i in range(n):
        mask_frac = masks[i].sum().item() / (masks.shape[1] * masks.shape[2])
        msg = (f"  Sample[{i}] {categories[i]:12s}/{defect_types[i]:15s} "
               f"mask_coverage={mask_frac*100:5.2f}%")
        if logger:
            logger.info(msg)
        else:
            print(msg)

    generated = model.generate(
        masks=masks,
        prompts=prompts,
        reference_images=reference_images,
        noise_scheduler=noise_scheduler,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.0,
    )

    # Build comparison grid: [real, generated, mask_overlay, reference] per sample
    comparison = []
    for i in range(n):
        comparison.append(images[i])
        comparison.append(generated[i])
        mask_rgb = masks[i].unsqueeze(0).repeat(3, 1, 1) * 2 - 1  # [0,1] → [-1,1]
        comparison.append(mask_rgb)
        comparison.append(reference_images[i])

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"samples_step_{step:06d}.png")
    save_image_grid(comparison, save_path, nrow=4)
    return save_path


def main():
    args = parse_args()
    config = load_config(args.config)

    # ---- Single-GPU enforcement (set via config or default) -------------------
    gpu_id = config["train"].get("gpu_id", 0)
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU] CUDA_VISIBLE_DEVICES={gpu_id}")

    # ---- Accelerator --------------------------------------------------------
    accelerator = Accelerator(
        mixed_precision=config["train"].get("mixed_precision", "fp16"),
        gradient_accumulation_steps=config["train"].get("gradient_accumulation_steps", 1),
        log_with="tensorboard",
        project_dir=config["train"]["output_dir"],
    )

    # ---- Seed ---------------------------------------------------------------
    seed = config.get("seed", 42)
    set_seed(seed)

    # ---- Logger -------------------------------------------------------------
    logger = Logger(
        log_dir=os.path.join(config["train"]["output_dir"], "logs"),
        use_tensorboard=True,
    )
    logger.info(f"Project: {config['project_name']}  |  Seed: {seed}")

    save_config(config, config["train"]["output_dir"])

    # =========================================================================
    # Build model
    # =========================================================================
    logger.info("Building IC-DiT model...")
    image_size = config["data"].get("image_size", 256)
    model = ICDiTDefectGenerator(config, image_size=image_size)
    model.print_stats()

    # =========================================================================
    # Build noise scheduler
    # =========================================================================
    noise_scheduler = NoiseScheduler(
        num_train_timesteps=config["diffusion"].get("num_train_timesteps", 1000),
        beta_schedule=config["diffusion"].get("beta_schedule", "linear"),
        prediction_type=config["diffusion"].get("prediction_type", "epsilon"),
    )
    logger.info(f"Noise scheduler: T={noise_scheduler.num_train_timesteps}, "
                f"schedule={config['diffusion']['beta_schedule']}")

    # =========================================================================
    # Optimizer — ONLY trainable parameters (enforced)
    # =========================================================================
    trainable_params = get_optimizer_trainable_params(model)
    lr = float(config["train"].get("lr", 1e-4))
    wd = float(config["train"].get("weight_decay", 1e-2))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=wd,
    )
    logger.info(f"Optimizer: AdamW(lr={lr}, wd={wd}), params={len(trainable_params)}")

    # Hard check: no frozen params leaked into optimizer
    assert_no_frozen_params_in_optimizer(model, optimizer)

    # =========================================================================
    # Datasets (train uses augmentation, val does not)
    # =========================================================================
    logger.info("Building datasets...")
    descriptions_path = config["data"].get("descriptions_path", None)
    if descriptions_path:
        logger.info(f"Using VLM descriptions: {descriptions_path}")
    train_dataset = MVTecDefectDataset(
        root=config["data"]["root"],
        image_size=image_size,
        categories=config["data"].get("categories", "all"),
        use_good_images=config["data"].get("use_good_images", True),
        use_defective_images=config["data"].get("use_defective_images", True),
        use_reference_normal=config["data"].get("use_reference_normal", True),
        mask_dilation=config["data"].get("mask_dilation", 0),
        mask_random_augment=config["data"].get("mask_random_augment", True),
        split="train",
        descriptions_path=descriptions_path,
    )

    # NOTE: MVTec AD has no standard train/val split. Val uses same test set
    # with mask_augment=False. For rigorous evaluation, an 80/20 random split
    # should be implemented in future iterations.
    val_dataset = MVTecDefectDataset(
        root=config["data"]["root"],
        image_size=image_size,
        categories=config["data"].get("categories", "all"),
        use_good_images=config["data"].get("use_good_images", True),
        use_defective_images=config["data"].get("use_defective_images", True),
        use_reference_normal=config["data"].get("use_reference_normal", True),
        mask_dilation=0,
        mask_random_augment=False,  # no augmentation for val
        split="val",
        descriptions_path=descriptions_path,
    )

    batch_size = config["train"].get("batch_size", 8)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=False,
    )
    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # =========================================================================
    # Accelerator prepare
    # =========================================================================
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # =========================================================================
    # LR scheduler: Linear warmup → CosineAnnealingWarmRestarts
    # =========================================================================
    # Rationale: CosineAnnealingLR decays monotonically to eta_min=1e-6, which
    # traps the model in local minima once the LR becomes negligible.
    # CosineAnnealingWarmRestarts periodically jumps back to base_lr, giving the
    # optimizer momentum to escape poor basins. Combined with linear warmup to
    # avoid early instability.
    num_epochs = config["train"].get("num_epochs", 100)
    steps_per_epoch = len(train_loader) // accelerator.gradient_accumulation_steps

    warmup_steps = config["train"].get("warmup_steps", 1000)
    restart_T_0_epochs = config["train"].get("lr_restart_T_0", 50)
    restart_T_mult = config["train"].get("lr_restart_T_mult", 2)
    restart_T_0_steps = restart_T_0_epochs * steps_per_epoch

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=restart_T_0_steps,
        T_mult=restart_T_mult,
        eta_min=1e-6,
    )
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # =========================================================================
    # EMA (Exponential Moving Average)
    # =========================================================================
    ema_decay = config["train"].get("ema_decay", 0.0)
    ema_model = EMAModel(model, decay=ema_decay) if ema_decay > 0 else None
    if ema_model is not None:
        logger.info(f"EMA enabled: decay={ema_decay}")

    # =========================================================================
    # Resume from checkpoint (if configured)
    # =========================================================================
    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    resume_from = config["train"].get("resume_from", None)
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))

        # Scheduler state: only restore if checkpoint uses the same scheduler type
        if "lr_scheduler_state_dict" in checkpoint:
            try:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                logger.info("  Restored LR scheduler state")
            except Exception as e:
                logger.info(f"  Could not restore LR scheduler (old format?): {e}")
                logger.info(f"  Starting fresh scheduler from step {global_step}")

        # EMA state: restore if present
        if ema_model is not None and "ema_state_dict" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_state_dict"])
            logger.info("  Restored EMA state")

        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        logger.info(f"  Resumed at epoch {start_epoch}, step {global_step}, "
                    f"best_val_loss={best_val_loss:.6f}")
        logger.info(f"  Current LR: {lr_scheduler.get_last_lr()[0]:.2e}")

    total_steps = num_epochs * steps_per_epoch

    # =========================================================================
    # Training loop
    # =========================================================================
    remaining_epochs = num_epochs - start_epoch
    remaining_steps = remaining_epochs * steps_per_epoch
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: epoch {start_epoch+1} → {num_epochs} "
                f"({remaining_epochs} remaining), ~{remaining_steps} steps "
                f"(warmup={warmup_steps} steps, T_0={restart_T_0_steps} steps)")
    logger.info(f"Batch size: {batch_size}, "
                f"Accumulation: {accelerator.gradient_accumulation_steps}")
    logger.info(f"Cosine schedule: {config['diffusion']['beta_schedule']}, "
                f"Dropout: {config['model'].get('dropout', 0.0)}")
    logger.info(f"{'='*60}\n")
    log_every = config["train"].get("log_every", 50)
    save_every = config["train"].get("save_every", 1000)
    sample_every = config["train"].get("sample_every", 1000)
    keep_last_n = config["train"].get("keep_last_n_ckpts", 3)
    num_inference_steps = config["diffusion"].get("num_inference_steps", 50)
    cfg_dropout_prob = config["train"].get("cfg_dropout_prob", 0.1)

    # ---- Auxiliary loss weights (0 = disable) ---------------------------------
    aux_w_text   = float(config["train"].get("aux_loss_weight_text", 0.01))
    aux_w_layout = float(config["train"].get("aux_loss_weight_layout", 0.1))
    aux_w_visual = float(config["train"].get("aux_loss_weight_visual", 0.01))
    _use_aux = (aux_w_text > 0 or aux_w_layout > 0 or aux_w_visual > 0)
    if _use_aux:
        logger.info(f"Auxiliary losses: text={aux_w_text}, layout={aux_w_layout}, visual={aux_w_visual}")

    # ---- Fixed sample set (same images every run) ---------------------------
    logger.info("Building fixed sample set (screw, zipper, pill, transistor)...")
    fixed_sample_set = build_fixed_sample_set(
        val_dataset, image_size, accelerator.device
    )
    for i, (cat, dt) in enumerate(zip(fixed_sample_set[5], fixed_sample_set[4])):
        logger.info(f"  Fixed[{i}] {cat}/{dt}")

    for epoch in range(start_epoch, num_epochs):
        model.train()

        if epoch == 0:
            assert_trainable_status(model)

        epoch_loss = AverageMeter()
        epoch_grad_norm = AverageMeter()
        L_text = L_layout = L_visual = torch.tensor(0.0)  # default, overwritten each step
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}",
                     disable=not accelerator.is_local_main_process)

        for batch in pbar:
            with accelerator.accumulate(model):
                images = batch["image"]
                masks = batch["mask"]
                prompts = batch["prompt"]
                reference_images = batch["reference_image"]
                B = images.shape[0]

                # ---- CFG dropout: randomly drop text condition ----
                prompts = [
                    "" if random.random() < cfg_dropout_prob else p
                    for p in prompts
                ]

                # ---- 1. Encode image to latent ----
                latents = model.encode_image_to_latent(images)

                # ---- 2. Sample noise & timesteps ----
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (B,),
                    device=latents.device
                ).long()

                # ---- 3. Add noise via scheduler (correct diffusion forward) ----
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # ---- 4. Predict noise (canonical ε_θ interface) + all updated tokens ----
                result = model(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    prompts=prompts,
                    masks=masks,
                    reference_images=reference_images,
                )
                eps_pred, text_upd, layout_upd, visual_upd, layout_logits, \
                    text_init, visual_init = result

                # ---- 5. Losses ----
                # Primary: noise prediction (Eq.9)
                L_noise = F.mse_loss(eps_pred, noise)
                loss = L_noise

                # NaN detection: identify which component breaks first
                _nan_msg = None
                if torch.isnan(L_noise):
                    _nan_msg = (f"L_noise NaN! eps_pred min/max: "
                                f"{eps_pred.min().item():.2f}/{eps_pred.max().item():.2f}")
                elif torch.isinf(L_noise):
                    _nan_msg = "L_noise Inf!"

                # Auxiliary: text consistency (prevent semantic drift)
                if aux_w_text > 0:
                    L_text = F.mse_loss(text_upd, text_init.detach())
                    if torch.isnan(L_text) and _nan_msg is None:
                        _nan_msg = (f"L_text NaN! text_upd max={text_upd.abs().max().item():.1f}, "
                                    f"text_init max={text_init.abs().max().item():.1f}")
                    loss = loss + aux_w_text * L_text

                # Auxiliary: spatial grounding
                if aux_w_layout > 0:
                    mask_16 = F.interpolate(
                        masks.unsqueeze(1), size=(16, 16), mode='area'
                    ).squeeze(1).reshape(B, 256)
                    L_layout = F.binary_cross_entropy_with_logits(
                        layout_logits, mask_16
                    )
                    if torch.isnan(L_layout) and _nan_msg is None:
                        _nan_msg = (f"L_layout NaN! logits max={layout_logits.abs().max().item():.1f}")
                    loss = loss + aux_w_layout * L_layout

                # Auxiliary: visual consistency
                if aux_w_visual > 0:
                    L_visual = F.mse_loss(visual_upd, visual_init.detach())
                    if torch.isnan(L_visual) and _nan_msg is None:
                        _nan_msg = (f"L_visual NaN! visual_upd max={visual_upd.abs().max().item():.1f}, "
                                    f"visual_init max={visual_init.abs().max().item():.1f}")
                    loss = loss + aux_w_visual * L_visual

                if _nan_msg is not None:
                    logger.error(f"[NaN DETECTED at step {global_step}] {_nan_msg}")
                    logger.error(f"  L_noise={L_noise.item():.4f}, "
                                 f"eps_pred stats: min={eps_pred.min().item():.2f} max={eps_pred.max().item():.2f}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # Compute grad norm BEFORE clipping (for logging)
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    epoch_grad_norm.update(total_norm, 1)

                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        config["train"].get("max_grad_norm", 1.0),
                    )

                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # ---- EMA update (after optimizer step) ----
                if ema_model is not None and accelerator.sync_gradients:
                    ema_model.update()

            epoch_loss.update(loss.detach().item(), B)

            # ---- Logging ----
            if accelerator.sync_gradients:
                global_step += 1

                if global_step % log_every == 0:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    _msg = (f"Step {global_step} | Loss: {epoch_loss.avg:.6f} | "
                            f"LR: {current_lr:.2e} | GradNorm: {epoch_grad_norm.avg:.4f}")
                    if _use_aux:
                        _l_t = L_text.item() if aux_w_text > 0 else 0
                        _l_l = L_layout.item() if aux_w_layout > 0 else 0
                        _l_v = L_visual.item() if aux_w_visual > 0 else 0
                        _msg += f" | Aux: text={_l_t:.4f} layout={_l_l:.4f} visual={_l_v:.4f}"
                    logger.info(_msg)
                    logger.log_scalar("train/loss", epoch_loss.avg, global_step)
                    logger.log_scalar("train/lr", current_lr, global_step)
                    logger.log_scalar("train/grad_norm", epoch_grad_norm.avg, global_step)
                    if _use_aux:
                        if aux_w_text > 0:
                            logger.log_scalar("train/aux_text", L_text.item(), global_step)
                        if aux_w_layout > 0:
                            logger.log_scalar("train/aux_layout", L_layout.item(), global_step)
                        if aux_w_visual > 0:
                            logger.log_scalar("train/aux_visual", L_visual.item(), global_step)

                # ---- Save checkpoint ----
                if global_step % save_every == 0:
                    ckpt_dir = os.path.join(config["train"]["output_dir"], "checkpoints")
                    os.makedirs(ckpt_dir, exist_ok=True)

                    # Save to temp file first, then rename (atomic — no corrupt ckpt from OOM)
                    ckpt_path = os.path.join(ckpt_dir, f"step_{global_step:07d}.pt")
                    tmp_path = ckpt_path + ".tmp"
                    unwrapped = accelerator.unwrap_model(model)
                    ckpt_dict = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "model_state_dict": unwrapped.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                        "config": config,
                        "val_loss": best_val_loss,
                    }
                    if ema_model is not None:
                        ckpt_dict["ema_state_dict"] = ema_model.state_dict()
                    torch.save(ckpt_dict, tmp_path)
                    os.rename(tmp_path, ckpt_path)
                    logger.info(f"Checkpoint saved: {ckpt_path}")

                    # Rotate: keep only last N step checkpoints (always keep best.pt)
                    step_ckpts = sorted(
                        [f for f in os.listdir(ckpt_dir) if f.startswith("step_") and f.endswith(".pt")],
                        key=lambda x: int(x.replace("step_", "").replace(".pt", "")),
                    )
                    for old in step_ckpts[:-keep_last_n]:
                        os.remove(os.path.join(ckpt_dir, old))
                        logger.info(f"Removed old checkpoint: {old}")

                # ---- Generate samples (from fixed set, using EMA if available) ----
                if global_step % sample_every == 0:
                    sample_dir = os.path.join(config["train"]["output_dir"], "samples")
                    if ema_model is not None:
                        ema_model.apply_shadow()
                    model.eval()  # disable dropout for deterministic sampling
                    try:
                        sample_path = generate_samples(
                            model=model,
                            fixed_sample_set=fixed_sample_set,
                            noise_scheduler=noise_scheduler,
                            num_inference_steps=num_inference_steps,
                            output_dir=sample_dir,
                            step=global_step,
                            logger=logger,
                        )
                        logger.info(f"Samples saved: {sample_path}")
                    finally:
                        model.train()  # restore training mode (dropout ON)
                        if ema_model is not None:
                            ema_model.restore()

                pbar.set_postfix({
                    "loss": f"{epoch_loss.avg:.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.1e}",
                    "step": global_step,
                })

        # ---- End-of-epoch validation (use EMA if available) ----
        if ema_model is not None:
            ema_model.apply_shadow()
        try:
            val_loss = validate(model, val_loader, noise_scheduler, accelerator.device)
        finally:
            if ema_model is not None:
                ema_model.restore()

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_loss.avg:.6f} | Val Loss: {val_loss:.6f} | "
            f"GradNorm: {epoch_grad_norm.avg:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.2e}"
        )
        logger.log_scalar("val/loss", val_loss, global_step)
        logger.log_scalar("train/epoch_loss", epoch_loss.avg, epoch + 1)

        # ---- Save best (EMA weights as model_state_dict for direct inference) ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dir = os.path.join(config["train"]["output_dir"], "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            best_path = os.path.join(ckpt_dir, "best.pt")
            unwrapped = accelerator.unwrap_model(model)

            # Snapshot EMA shadow → model weights for best.pt so inference
            # can load it directly without special EMA handling.
            if ema_model is not None:
                ema_model.apply_shadow()
            model_state = copy.deepcopy(unwrapped.state_dict())
            if ema_model is not None:
                ema_model.restore()

            ckpt_dict = {
                "global_step": global_step,
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "config": config,
                "val_loss": best_val_loss,
            }
            torch.save(ckpt_dict, best_path)
            logger.info(f"Best model saved: {best_path} (val_loss={best_val_loss:.6f})")

        if epoch == 0:
            assert_trainable_status(model)

    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete! Best val loss: {best_val_loss:.6f}")
    logger.info(f"Outputs: {config['train']['output_dir']}")
    logger.info(f"{'='*60}")
    logger.close()


if __name__ == "__main__":
    main()
