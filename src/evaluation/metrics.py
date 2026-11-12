"""
Shared Evaluation Metrics
=========================
Centralised functions for computing model evaluation metrics.

WHY THIS EXISTS:
- Every model (X-ray, ECG, cardiac, respiratory, NLP, fusion) must
  be evaluated with the same consistent set of metrics.
- Centralising prevents copy-paste errors and ensures we always
  report: Accuracy, Precision, Recall, F1, Specificity, ROC-AUC.
- Special attention to false negatives (missed emergencies).

USAGE:
    from src.evaluation.metrics import compute_classification_metrics
    results = compute_classification_metrics(y_true, y_pred, y_prob)
"""

import numpy as np
from typing import Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import logging

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "weighted",
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground truth labels. Shape (n_samples,).
        y_pred: Predicted labels. Shape (n_samples,).
        y_prob: Predicted probabilities. Shape (n_samples,) for binary
                or (n_samples, n_classes) for multiclass. Optional.
        average: Averaging strategy for multiclass ('weighted', 'macro', 'micro').
        class_names: Optional list of class names for the report.

    Returns:
        dict: Dictionary containing all computed metrics.
    """
    metrics = {}

    # --- Core Metrics ---
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(
        precision_score(y_true, y_pred, average=average, zero_division=0)
    )
    metrics["recall"] = float(
        recall_score(y_true, y_pred, average=average, zero_division=0)
    )
    # Recall IS sensitivity — they are the same thing.
    # We store both names for clarity in medical context.
    metrics["sensitivity"] = metrics["recall"]
    metrics["f1_score"] = float(
        f1_score(y_true, y_pred, average=average, zero_division=0)
    )

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # --- Specificity (for binary classification) ---
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    else:
        # Per-class specificity for multiclass
        specificities = []
        for i in range(len(cm)):
            tp_i = cm[i, i]
            fn_i = cm[i, :].sum() - tp_i
            fp_i = cm[:, i].sum() - tp_i
            tn_i = cm.sum() - tp_i - fn_i - fp_i
            spec_i = float(tn_i / (tn_i + fp_i)) if (tn_i + fp_i) > 0 else 0.0
            specificities.append(spec_i)
        metrics["specificity_per_class"] = specificities
        metrics["specificity"] = float(np.mean(specificities))

    # --- ROC-AUC (requires probabilities) ---
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary: y_prob should be 1D (probability of positive class)
                prob = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
            else:
                # Multiclass: y_prob should be (n_samples, n_classes)
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
                )
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    # --- Classification Report (text) ---
    target_names = class_names if class_names else None
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )

    return metrics


def log_metrics(metrics: dict, model_name: str = "Model") -> None:
    """
    Log computed metrics in a readable format.

    Args:
        metrics: Dictionary from compute_classification_metrics().
        model_name: Name of the model for the log header.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  {model_name} — Evaluation Results")
    logger.info(f"{'='*60}")
    logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:   {metrics['precision']:.4f}")
    logger.info(f"  Recall:      {metrics['recall']:.4f}")
    logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"  Specificity: {metrics['specificity']:.4f}")
    logger.info(f"  F1 Score:    {metrics['f1_score']:.4f}")

    if metrics.get("roc_auc") is not None:
        logger.info(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")

    if metrics.get("false_negative_rate") is not None:
        logger.info(f"  FN Rate:     {metrics['false_negative_rate']:.4f}  ⚠️ (missed cases)")

    logger.info(f"\n{metrics['classification_report']}")
    logger.info(f"{'='*60}")
