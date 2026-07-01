"""
Noise scheduler: DDPM training + DDIM sampling.

Usage (conda env):
    conda activate omg
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def make_beta_schedule(schedule: str = "linear", num_timesteps: int = 1000,
                       beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """
    Create beta schedule for diffusion.

    Args:
        schedule: "linear" or "cosine".
        num_timesteps: Number of diffusion timesteps.
        beta_start: Start value for linear schedule.
        beta_end: End value for linear schedule.

    Returns:
        betas: (num_timesteps,) tensor.
    """
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    elif schedule == "cosine":
        # Cosine schedule as in improved DDPM
        s = 0.008
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos((t / num_timesteps + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clamp(betas, max=0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")

    return betas.float()


class NoiseScheduler:
    """
    DDPM-style noise scheduler for training, with DDIM sampling.

    Pre-computes all alpha/beta/alpha_cumprod values.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
    ):
        """
        Args:
            num_train_timesteps: Total diffusion steps (T).
            beta_schedule: "linear" or "cosine".
            prediction_type: "epsilon" (predict noise) or "v_prediction".
        """
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        # Betas
        self.betas = make_beta_schedule(beta_schedule, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # For q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # DDIM
        self.init_noise_sigma = 1.0
        self.timesteps = None

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0).

        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise

        Args:
            original: (B, C, H, W) clean latent.
            noise:    (B, C, H, W) Gaussian noise.
            timesteps:(B,) integer timestep indices.

        Returns:
            noisy:    (B, C, H, W) noisy latent.
        """
        device = original.device
        sqrt_alpha = self.sqrt_alphas_cumprod.to(device)[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps]

        # Reshape for broadcasting
        while sqrt_alpha.dim() < original.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        noisy = sqrt_alpha * original + sqrt_one_minus_alpha * noise
        return noisy

    def set_timesteps(self, num_inference_steps: int, device: torch.device = None):
        """
        Set timesteps for DDIM sampling.

        Args:
            num_inference_steps: Number of inference steps (<= num_train_timesteps).
            device: Target device.
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        eta: float = 0.0,
    ) -> "DDIMSchedulerOutput":
        """
        Single DDIM denoising step.

        Args:
            model_output: (B, C, H, W) predicted noise.
            timestep:     Scalar timestep index.
            sample:       (B, C, H, W) current noisy latent x_t.
            eta:          DDIM stochasticity (0=deterministic, 1=DDPM).

        Returns:
            DDIMSchedulerOutput with prev_sample.
        """
        device = sample.device
        t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep

        # Get prev timestep
        if len(self.timesteps) > 0 and hasattr(self, 'timesteps'):
            idx = (self.timesteps == t).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                prev_idx = idx.item() + 1
                if prev_idx < len(self.timesteps):
                    prev_t = self.timesteps[prev_idx].item()
                else:
                    prev_t = -1
            else:
                prev_t = t - (self.num_train_timesteps // len(self.timesteps))
                prev_t = max(prev_t, 0)
        else:
            prev_t = max(t - 1, 0)

        alpha_prod_t = self.alphas_cumprod[t].to(device)
        alpha_prod_t_prev = self.alphas_cumprod[prev_t].to(device) if prev_t >= 0 else torch.tensor(1.0, device=device)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # Predicted original sample
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # Direction pointing to x_t
        pred_sample_direction = (1 - alpha_prod_t_prev).sqrt() * model_output

        # x_{t-1} = sqrt(alpha_prod_t_prev) * x_0 + sqrt(1 - alpha_prod_t_prev) * eps_pred
        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction

        # Add noise (eta=0 -> deterministic DDIM)
        if eta > 0:
            noise = torch.randn_like(sample)
            variance = (eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) *
                              (1 - alpha_prod_t / alpha_prod_t_prev))).sqrt()
            prev_sample = prev_sample + variance * noise

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


class DDIMSchedulerOutput:
    """Output of a DDIM scheduler step."""
    def __init__(self, prev_sample: torch.Tensor, pred_original_sample: Optional[torch.Tensor] = None):
        self.prev_sample = prev_sample
        self.pred_original_sample = pred_original_sample
