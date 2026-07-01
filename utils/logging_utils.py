"""
Logging utilities with TensorBoard and console output.

Usage (conda env):
    conda activate omg
"""

import os
import logging
from torch.utils.tensorboard import SummaryWriter
from typing import Optional


class Logger:
    """Unified logger with TensorBoard and console support."""

    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        log_level: int = logging.INFO,
    ):
        os.makedirs(log_dir, exist_ok=True)

        # Console logger
        self.logger = logging.getLogger("ICDiT")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # TensorBoard
        self.use_tensorboard = use_tensorboard
        self.writer = None
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
            self.logger.info(f"TensorBoard logging to: {log_dir}")

        self.global_step = 0

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log scalar to TensorBoard."""
        if step is None:
            step = self.global_step
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_value_dict: dict, step: Optional[int] = None):
        """Log multiple scalars to TensorBoard."""
        if step is None:
            step = self.global_step
        if self.writer:
            self.writer.add_scalars(main_tag, tag_value_dict, step)

    def log_images(self, tag: str, images, step: Optional[int] = None):
        """Log images to TensorBoard. images: torch.Tensor (B, C, H, W) in [-1,1]."""
        if step is None:
            step = self.global_step
        if self.writer:
            self.writer.add_images(tag, images, step, dataformats="NCHW")

    def set_step(self, step: int):
        self.global_step = step

    def step_increment(self):
        self.global_step += 1

    def close(self):
        if self.writer:
            self.writer.close()
