"""
Shared Evaluation Metrics
=========================
Supports binary, multiclass and multilabel classification.

Designed for ECG, chest X-ray, cardiac risk and respiratory
research models. Undefined probability metrics are returned as None.
"""

import logging
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    multilabel_confusion_matrix,
)

logger = logging.getLogger(__name__)


def _safe_auc(y_true, y_prob):
    """Return ROC-AUC or None when it is undefined."""
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return None


def _safe_auprc(y_true, y_prob):
    """Return average precision or None when undefined."""
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return None


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    average: str = "weighted",
    class_names: list[str] | None = None,
) -> dict:
    """Calculate classification metrics for binary, multiclass and multilabel tasks."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes.")

    if y_true.ndim not in (1, 2):
        raise ValueError("Inputs must be 1D or 2D arrays.")

    if y_true.size == 0:
        raise ValueError("Inputs must not be empty.")

    # A single-column 2D array represents ordinary binary labels.
    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Labels and predictions must be finite.")

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)

        if not np.all(np.isfinite(y_prob)):
            raise ValueError("Probabilities must be finite.")

        if np.any((y_prob < 0) | (y_prob > 1)):
            raise ValueError("Probabilities must be between 0 and 1.")

    metrics = {}

    # --------------------------------------------------
    # MULTILABEL CLASSIFICATION
    # --------------------------------------------------
    if y_true.ndim == 2:

        n_classes = y_true.shape[1]

        if n_classes < 2:
            raise ValueError("Multilabel inputs require at least two labels.")

        if not np.all(np.isin(y_true, [0, 1])) or not np.all(
            np.isin(y_pred, [0, 1])
        ):
            raise ValueError("Multilabel targets and predictions must be binary.")

        if class_names is None:
            class_names = [f"Class_{i}" for i in range(n_classes)]

        if len(class_names) != n_classes:
            raise ValueError("class_names length must match label count.")

        metrics["task_type"] = "multilabel"
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["accuracy_definition"] = "subset_accuracy"

        metrics["precision"] = float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        )
        metrics["sensitivity"] = metrics["recall"]
        metrics["f1_score"] = float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        )

        matrices = multilabel_confusion_matrix(y_true, y_pred)
        per_class = {}
        specificity_scores = []

        for i, name in enumerate(class_names):
            tn, fp, fn, tp = matrices[i].ravel()

            sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
            specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
            fn_rate = float(fn / (fn + tp)) if (fn + tp) else 0.0

            specificity_scores.append(specificity)

            per_class[name] = {
                "precision": float(
                    precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
                ),
                "recall": sensitivity,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "false_negative_rate": fn_rate,
                "false_positive_rate": (
                    float(fp / (fp + tn)) if (fp + tn) else 0.0
                ),
                "f1_score": float(
                    f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
                ),
                "confusion_matrix": matrices[i].tolist(),
            }

        metrics["specificity"] = float(np.mean(specificity_scores))
        metrics["per_class"] = per_class

        if y_prob is not None:
            if y_prob.shape != y_true.shape:
                raise ValueError(
                    "For multilabel tasks, y_prob must match y_true shape."
                )

            auc_scores = []
            ap_scores = []

            for i, name in enumerate(class_names):
                auc = _safe_auc(y_true[:, i], y_prob[:, i])
                ap = _safe_auprc(y_true[:, i], y_prob[:, i])

                per_class[name]["roc_auc"] = auc
                per_class[name]["auprc"] = ap

                if auc is not None:
                    auc_scores.append(auc)
                if ap is not None:
                    ap_scores.append(ap)

            metrics["roc_auc"] = (
                float(np.mean(auc_scores)) if auc_scores else None
            )
            metrics["auprc"] = (
                float(np.mean(ap_scores)) if ap_scores else None
            )

            metrics["micro_roc_auc"] = _safe_auc(
                y_true.ravel(), y_prob.ravel()
            )
            metrics["micro_auprc"] = _safe_auprc(
                y_true.ravel(), y_prob.ravel()
            )

        else:
            metrics["roc_auc"] = None
            metrics["auprc"] = None
            metrics["micro_roc_auc"] = None
            metrics["micro_auprc"] = None

        metrics["classification_report"] = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )

    # --------------------------------------------------
    # BINARY AND MULTICLASS CLASSIFICATION
    # --------------------------------------------------
    else:

        labels = np.unique(y_true)
        task_type = "binary" if len(labels) == 2 else "multiclass"

        if class_names is not None and len(class_names) != len(labels):
            raise ValueError("class_names length must match the number of classes.")

        metrics["task_type"] = task_type
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        )
        metrics["f1_score"] = float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        )

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        metrics["confusion_matrix"] = cm.tolist()

        if task_type == "binary":
            positive_label = labels[1]
            tn, fp, fn, tp = cm.ravel()

            sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
            specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

            metrics["sensitivity"] = sensitivity
            metrics["recall"] = sensitivity
            metrics["specificity"] = specificity
            metrics["false_negative_rate"] = (
                float(fn / (fn + tp)) if (fn + tp) else 0.0
            )
            metrics["false_positive_rate"] = (
                float(fp / (fp + tn)) if (fp + tn) else 0.0
            )

        else:
            metrics["sensitivity"] = metrics["recall"]
            specificities = []

            for i in range(len(labels)):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - tp - fn - fp

                specificities.append(
                    float(tn / (tn + fp)) if (tn + fp) else 0.0
                )

            metrics["specificity"] = float(np.mean(specificities))

        if y_prob is not None:
            if task_type == "binary":
                if y_prob.ndim == 1:
                    positive_prob = y_prob
                elif y_prob.ndim == 2 and y_prob.shape == (len(y_true), 1):
                    positive_prob = y_prob[:, 0]
                elif y_prob.ndim == 2 and y_prob.shape == (len(y_true), 2):
                    positive_prob = y_prob[:, 1]
                else:
                    raise ValueError("Binary probabilities must have shape (n,), (n, 1), or (n, 2).")

                if len(positive_prob) != len(y_true):
                    raise ValueError("Probability sample count must match y_true.")

                metrics["roc_auc"] = _safe_auc(
                    (y_true == positive_label).astype(int), positive_prob
                )
                metrics["auprc"] = _safe_auprc(
                    (y_true == positive_label).astype(int), positive_prob
                )

            else:
                if y_prob.shape != (len(y_true), len(labels)):
                    raise ValueError("Multiclass probabilities must have shape (n_samples, n_classes).")

                if class_names is None:
                    class_names = [str(label) for label in labels]

                auc_scores = []
                ap_scores = []

                for i, label in enumerate(labels):
                    binary_true = (y_true == label).astype(int)
                    auc = _safe_auc(binary_true, y_prob[:, i])
                    ap = _safe_auprc(binary_true, y_prob[:, i])

                    if auc is not None:
                        auc_scores.append(auc)
                    if ap is not None:
                        ap_scores.append(ap)

                metrics["roc_auc"] = (
                    float(np.mean(auc_scores)) if auc_scores else None
                )
                metrics["auprc"] = (
                    float(np.mean(ap_scores)) if ap_scores else None
                )

        else:
            metrics["roc_auc"] = None
            metrics["auprc"] = None

        metrics["classification_report"] = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )

    return metrics


def log_metrics(metrics: dict, model_name: str = "Model") -> None:
    """Log evaluation metrics in a readable format."""

    logger.info("=" * 60)
    logger.info("%s Evaluation Results", model_name)
    logger.info("=" * 60)

    for key in (
        "task_type",
        "accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1_score",
        "roc_auc",
        "auprc",
        "micro_roc_auc",
        "micro_auprc",
        "false_negative_rate",
    ):
        if metrics.get(key) is not None:
            logger.info("%s: %s", key, metrics[key])

    if "per_class" in metrics:
        for name, values in metrics["per_class"].items():
            logger.info("%s: %s", name, values)

    logger.info("\n%s", metrics["classification_report"])
    logger.info("=" * 60)
