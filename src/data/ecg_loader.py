
"""
PTB-XL ECG data loader.

Supports:
- Loading PTB-XL metadata
- Loading diagnostic SCP statements
- Mapping SCP codes to five diagnostic superclasses
- Creating multilabel targets
- Creating official patient-safe train/validation/test splits
- Loading individual ECG recordings using WFDB
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def parse_scp_codes(value):
    """Convert the scp_codes metadata field into a Python dictionary."""
    if isinstance(value, dict):
        return value

    if pd.isna(value):
        return {}

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        raise ValueError(f"Could not parse SCP codes: {value}")

    if not isinstance(parsed, dict):
        raise ValueError("SCP codes must contain a dictionary.")

    return parsed


def load_ptbxl_metadata(data_dir, sampling_rate=100):
    """
    Load PTB-XL metadata.

    sampling_rate=100 uses filename_lr.
    sampling_rate=500 uses filename_hr.
    """
    data_dir = Path(data_dir)

    metadata_path = data_dir / "ptbxl_database.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"PTB-XL metadata not found: {metadata_path}"
        )

    if sampling_rate not in (100, 500):
        raise ValueError("Sampling rate must be 100 or 500.")

    metadata = pd.read_csv(metadata_path)

    filename_column = (
        "filename_lr" if sampling_rate == 100 else "filename_hr"
    )

    required_columns = {
        "ecg_id",
        "patient_id",
        "strat_fold",
        "scp_codes",
        filename_column,
    }

    missing = required_columns - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Missing required metadata columns: {sorted(missing)}"
        )

    metadata["scp_codes"] = metadata["scp_codes"].apply(
        parse_scp_codes
    )

    return metadata


def load_scp_statements(data_dir):
    """Load diagnostic SCP-code definitions."""
    path = Path(data_dir) / "scp_statements.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"SCP statements file not found: {path}"
        )

    statements = pd.read_csv(path, index_col=0)

    required = {"diagnostic", "diagnostic_class"}

    missing = required - set(statements.columns)

    if missing:
        raise ValueError(
            f"Missing SCP statement columns: {sorted(missing)}"
        )

    return statements


def aggregate_diagnostic_labels(scp_codes, statements):
    """
    Map individual SCP codes to diagnostic superclasses.

    A recording may belong to multiple classes.
    """
    labels = set()

    for code in scp_codes:
        if code not in statements.index:
            continue

        row = statements.loc[code]

        if not bool(row["diagnostic"]):
            continue

        diagnostic_class = row["diagnostic_class"]

        if pd.notna(diagnostic_class):
            labels.add(str(diagnostic_class))

    return sorted(labels)


def build_label_matrix(
    metadata,
    statements,
    class_names=DEFAULT_CLASSES,
):
    """Create five-column multilabel targets."""
    result = metadata.copy()

    for class_name in class_names:
        result[f"label_{class_name}"] = 0

    for index, scp_codes in result["scp_codes"].items():
        labels = aggregate_diagnostic_labels(
            scp_codes, statements
        )

        for label in labels:
            if label in class_names:
                result.at[index, f"label_{label}"] = 1

    return result


def create_fold_splits(
    metadata,
    train_folds=tuple(range(1, 9)),
    validation_folds=(9,),
    test_folds=(10,),
):
    """
    Split using the official PTB-XL strat_fold assignments.

    Do not randomly split individual recordings.
    """
    train_folds = set(train_folds)
    validation_folds = set(validation_folds)
    test_folds = set(test_folds)

    if (
        train_folds & validation_folds
        or train_folds & test_folds
        or validation_folds & test_folds
    ):
        raise ValueError("Train, validation and test folds overlap.")

    train = metadata[
        metadata["strat_fold"].isin(train_folds)
    ].copy()

    validation = metadata[
        metadata["strat_fold"].isin(validation_folds)
    ].copy()

    test = metadata[
        metadata["strat_fold"].isin(test_folds)
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "One or more splits are empty. Check strat_fold values."
        )

    patient_sets = [
        set(train["patient_id"].dropna()),
        set(validation["patient_id"].dropna()),
        set(test["patient_id"].dropna()),
    ]

    if (
        patient_sets[0] & patient_sets[1]
        or patient_sets[0] & patient_sets[2]
        or patient_sets[1] & patient_sets[2]
    ):
        raise ValueError(
            "Patient leakage detected between data splits."
        )

    return train, validation, test


def load_ecg_record(data_dir, relative_path, channels_first=True):
    """
    Load one ECG recording using WFDB.

    Returns float32 array:
    channels_first=True:  (12, time_steps)
    channels_first=False: (time_steps, 12)
    """
    import wfdb

    record_path = Path(data_dir) / relative_path

    signal, metadata = wfdb.rdsamp(str(record_path))

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 2:
        raise ValueError("Expected a two-dimensional ECG signal.")

    if channels_first:
        signal = signal.T

    return signal, metadata
