"""
File I/O Utilities
==================
Helper functions for loading and saving models, configs, and data.

WHY THIS EXISTS:
- Centralises file operations so every module uses the same
  conventions for paths, formats, and error handling.
- Avoids scattered open()/torch.save() calls with inconsistent paths.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import yaml
import torch
import logging

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict:
    """
    Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        dict: Parsed YAML content.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {path}")
    return config


def save_yaml(data: dict, path: str | Path) -> None:
    """Save a dictionary as a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {path}")


def save_torch_model(
    model: torch.nn.Module,
    path: str | Path,
    metadata: dict | None = None,
) -> None:
    """
    Save a PyTorch model's state dict along with optional metadata.

    WHY state_dict instead of the whole model?
    - state_dict saves only the learned parameters (weights and biases).
    - It's smaller, more portable, and less fragile across Python versions.
    - To reload, you recreate the model architecture and load the state_dict.

    Args:
        model: The PyTorch model to save.
        path: Destination file path (typically .pth).
        metadata: Optional dict of metadata (e.g., epoch, metrics).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, str(path))
    logger.info(f"Model saved to {path}")


def load_torch_model(
    model: torch.nn.Module,
    path: str | Path,
    device: torch.device | None = None,
) -> dict | None:
    """
    Load a PyTorch model's state dict from a checkpoint file.

    Args:
        model: The model architecture (must match the saved state_dict).
        path: Path to the checkpoint file.
        device: Device to map the weights to.

    Returns:
        dict or None: Metadata from the checkpoint, if any.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    map_location = device if device else torch.device("cpu")
    checkpoint = torch.load(str(path), map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model loaded from {path}")

    return checkpoint.get("metadata")


def save_json(data: Any, path: str | Path) -> None:
    """Save data as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved JSON to {path}")


def load_json(path: str | Path) -> Any:
    """Load data from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded JSON from {path}")
    return data


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save an object using pickle (e.g., sklearn models)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)

    logger.info(f"Saved pickle to {path}")


def load_pickle(path: str | Path) -> Any:
    """Load an object from a pickle file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    logger.info(f"Loaded pickle from {path}")
    return obj


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it doesn't exist and return the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
