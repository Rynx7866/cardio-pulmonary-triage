import numpy as np
import pytest

from src.evaluation.metrics import compute_classification_metrics


class TestBinaryMetrics:

    def test_binary_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.9, 0.4])

        result = compute_classification_metrics(y_true, y_pred, y_prob)

        assert result["task_type"] == "binary"
        assert result["accuracy"] == pytest.approx(0.75)
        assert result["sensitivity"] == pytest.approx(0.5)
        assert result["specificity"] == pytest.approx(1.0)
        assert result["roc_auc"] == pytest.approx(1.0)
        assert result["auprc"] is not None

    def test_binary_probability_column_shape(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.1], [0.2], [0.8], [0.9]])

        result = compute_classification_metrics(y_true, y_pred, y_prob)

        assert result["roc_auc"] is not None


class TestMulticlassMetrics:

    def test_multiclass_metrics(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 0, 2])
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.2, 0.7],
        ])

        result = compute_classification_metrics(
            y_true,
            y_pred,
            y_prob,
            class_names=["A", "B", "C"],
        )

        assert result["task_type"] == "multiclass"
        assert result["accuracy"] == pytest.approx(5 / 6)
        assert result["roc_auc"] is not None
        assert result["auprc"] is not None


class TestMultilabelMetrics:

    def test_five_label_metrics(self):
        y_true = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ])

        y_pred = y_true.copy()

        y_prob = np.where(y_true == 1, 0.9, 0.1)

        names = ["NORM", "MI", "STTC", "CD", "HYP"]

        result = compute_classification_metrics(
            y_true, y_pred, y_prob, class_names=names
        )

        assert result["task_type"] == "multilabel"
        assert result["accuracy"] == pytest.approx(1.0)
        assert set(result["per_class"]) == set(names)

        for name in names:
            assert result["per_class"][name]["sensitivity"] == pytest.approx(1.0)
            assert result["per_class"][name]["specificity"] == pytest.approx(1.0)
            assert result["per_class"][name]["roc_auc"] == pytest.approx(1.0)
            assert result["per_class"][name]["auprc"] == pytest.approx(1.0)

    def test_multilabel_false_negative_rate(self):
        y_true = np.array([[1, 0], [1, 1], [0, 1]])
        y_pred = np.array([[0, 0], [1, 1], [0, 1]])

        result = compute_classification_metrics(y_true, y_pred)

        assert "false_negative_rate" in result["per_class"]["Class_0"]
        assert result["per_class"]["Class_0"]["false_negative_rate"] == pytest.approx(0.5)


class TestInputValidation:

    def test_mismatched_shapes_raise_error(self):
        with pytest.raises(ValueError):
            compute_classification_metrics(
                np.array([0, 1]), np.array([0, 1, 0])
            )

    def test_invalid_dimensions_raise_error(self):
        with pytest.raises(ValueError):
            compute_classification_metrics(
                np.zeros((2, 2, 2)), np.zeros((2, 2, 2))
            )

    def test_class_name_count_validation(self):
        with pytest.raises(ValueError):
            compute_classification_metrics(
                np.array([[1, 0], [0, 1]]),
                np.array([[1, 0], [0, 1]]),
                class_names=["OnlyOne"],
            )
