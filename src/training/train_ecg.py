"""Training pipeline for the PTB-XL ECG CNN."""

from pathlib import Path
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.ecg_loader import (
    load_ptbxl_metadata,
    load_scp_statements,
    build_label_matrix,
    create_fold_splits,
    DEFAULT_CLASSES,
)
from src.data.ecg_dataset import PTBXLDataset
from src.models.ecg_cnn import ECGCNN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_pos_weights(metadata, label_columns):
    """Calculate capped positive-class weights from training labels only."""
    labels = metadata[label_columns].to_numpy(dtype=np.float32)

    positives = labels.sum(axis=0)
    negatives = len(labels) - positives

    weights = negatives / np.maximum(positives, 1.0)
    weights = np.clip(weights, 1.0, 20.0)

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(signals)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = signals.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        logits = model(signals)
        loss = criterion(logits, labels)

        batch_size = signals.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def main():
    set_seed(42)

    data_dir = os.environ.get("PTBXL_DIR")

    if not data_dir:
        raise EnvironmentError(
            "PTBXL_DIR is not set. Set it to your PTB-XL dataset directory."
        )

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    output_dir = Path("checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Training device:", device)

    metadata = load_ptbxl_metadata(str(data_dir), sampling_rate=100)
    statements = load_scp_statements(str(data_dir))
    metadata = build_label_matrix(metadata, statements)

    train_metadata, val_metadata, _ = create_fold_splits(metadata)

    # Optional limits for local smoke testing.
    train_limit = int(os.environ.get("ECG_TRAIN_LIMIT", "0"))
    val_limit = int(os.environ.get("ECG_VAL_LIMIT", "0"))

    if train_limit > 0:
        train_metadata = train_metadata.sample(
            n=min(train_limit, len(train_metadata)),
            random_state=42,
        )

    if val_limit > 0:
        val_metadata = val_metadata.sample(
            n=min(val_limit, len(val_metadata)),
            random_state=42,
        )

    print("Training recordings:", len(train_metadata))
    print("Validation recordings:", len(val_metadata))

    label_columns = [f"label_{name}" for name in DEFAULT_CLASSES]

    train_dataset = PTBXLDataset(
        train_metadata,
        data_dir,
        sampling_rate=100,
    )

    val_dataset = PTBXLDataset(
        val_metadata,
        data_dir,
        sampling_rate=100,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = ECGCNN(num_classes=len(DEFAULT_CLASSES)).to(device)

    pos_weights = calculate_pos_weights(
        train_metadata,
        label_columns,
    ).to(device)

    print("Positive class weights:", pos_weights.cpu().tolist())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    epochs = int(os.environ.get("ECG_EPOCHS", "10"))
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint_path = output_dir / "best_ecg_cnn.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": val_loss,
                    "classes": DEFAULT_CLASSES,
                },
                checkpoint_path,
            )

            print("Saved best checkpoint:", checkpoint_path)

    print("\nECG TRAINING FINISHED")
    print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
    main()