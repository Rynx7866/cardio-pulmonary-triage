
"""PyTorch Dataset and DataLoader for PTB-XL ECG recordings."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.ecg_loader import (
    load_ecg_record,
    DEFAULT_CLASSES,
)

from src.data.ecg_preprocessing import preprocess_ecg


class PTBXLDataset(Dataset):
    """Lazy-loading PyTorch dataset for PTB-XL."""

    def __init__(
        self,
        metadata,
        data_dir,
        sampling_rate=100,
        class_names=DEFAULT_CLASSES,
    ):
        self.metadata = metadata.reset_index(drop=True).copy()
        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.class_names = list(class_names)

        self.label_columns = [
            f"label_{name}" for name in self.class_names
        ]

        filename_column = (
            "filename_lr" if sampling_rate == 100
            else "filename_hr"
        )

        required = {filename_column, *self.label_columns}
        missing = required - set(self.metadata.columns)

        if missing:
            raise ValueError(
                f"Missing required metadata columns: {missing}"
            )

        if self.metadata.empty:
            raise ValueError("Dataset metadata is empty.")

        if self.metadata[self.label_columns].isna().any().any():
            raise ValueError("Missing values found in labels.")

        self.filename_column = filename_column

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        relative_path = row[self.filename_column]
        signal, _ = load_ecg_record(
            self.data_dir,
            relative_path,
            channels_first=False,
        )

        # Preprocessing expects (time_steps, 12).
        # It filters the signal and converts it to (12, time_steps).
        signal = preprocess_ecg(
            signal,
            sampling_rate=self.sampling_rate,
        )




        signal = np.asarray(signal, dtype=np.float32)

        labels = row[self.label_columns].to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(signal).all():
            raise ValueError(
                f"Non-finite ECG signal at index {index}"
            )

        if not np.isfinite(labels).all():
            raise ValueError(
                f"Non-finite labels at index {index}"
            )

        if signal.ndim != 2 or signal.shape[0] != 12:
            raise ValueError(
                f"Expected ECG shape (12, time), got {signal.shape}"
            )

        signal_tensor = torch.from_numpy(signal)
        label_tensor = torch.from_numpy(labels)

        return signal_tensor, label_tensor


def create_ecg_dataloader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
):
    """Create a PyTorch DataLoader."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
