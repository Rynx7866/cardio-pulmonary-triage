"""
Device Management Utility
=========================
Detects available compute devices (CPU / CUDA GPU) and provides
a consistent interface for selecting the device across all models.

WHY THIS EXISTS:
- We need to gracefully handle both GPU and CPU environments.
- Google Colab has GPU; local laptop may only have CPU PyTorch.
- Every model training and inference script should call get_device()
  instead of hardcoding 'cuda' or 'cpu'.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Select the best available compute device.

    Logic:
        1. If prefer_gpu=True and CUDA is available → use CUDA.
        2. Otherwise → use CPU.

    Args:
        prefer_gpu: Whether to prefer GPU if available. Default True.

    Returns:
        torch.device: The selected device ('cuda' or 'cpu').

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
        >>> tensor = tensor.to(device)
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        logger.info(f"Using GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)")
    else:
        device = torch.device("cpu")
        if prefer_gpu:
            logger.info("GPU requested but CUDA not available. Using CPU.")
        else:
            logger.info("Using CPU (GPU not requested).")

    return device


def get_device_info() -> dict:
    """
    Return a dictionary with detailed device information.
    Useful for logging and environment verification.

    Returns:
        dict: Device information including PyTorch version,
              CUDA availability, GPU name, and VRAM.
    """
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": None,
        "gpu_vram_mb": None,
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1
        )

    return info
