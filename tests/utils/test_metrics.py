import numpy as np
import pytest

from src.utils.metrics import MetricsCalculator


def test_calculate_accuracy_basic():
    calc = MetricsCalculator(num_classes=5)
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 0])
    assert calc.calculate_accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_directional_accuracy_mapping():
    calc = MetricsCalculator(num_classes=5)
    # Down (0/1), Neutral (2), Up (3/4)
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([1, 0, 2, 4, 3])
    assert calc.calculate_directional_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_per_class_metrics_shape_and_zeros_for_missing_classes():
    calc = MetricsCalculator(num_classes=5)
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 1, 2, 2])
    metrics = calc.calculate_per_class_metrics(y_true, y_pred)
    for key in ["precision", "recall", "f1_score", "support"]:
        assert key in metrics
        assert len(metrics[key]) == 5
    # Classes 3 and 4 are missing -> zero_division=0 ensures zeros
    assert metrics["precision"][3] == 0
    assert metrics["recall"][4] == 0


def test_per_coin_accuracy_1d_and_2d():
    calc = MetricsCalculator(num_classes=5)
    # 1D input -> single accuracy value array
    y_true_1d = np.array([0, 1, 1, 0])
    y_pred_1d = np.array([0, 1, 0, 0])
    accs_1d = calc.calculate_per_coin_accuracy(y_true_1d, y_pred_1d)
    assert accs_1d.shape == (1,)
    assert accs_1d[0] == pytest.approx(0.75)

    # 2D input -> per-column accuracies
    y_true = np.array([[0, 1], [1, 1]])
    y_pred = np.array([[0, 0], [1, 1]])
    accs = calc.calculate_per_coin_accuracy(y_true, y_pred)
    assert accs.shape == (2,)
    assert accs[0] == pytest.approx(1.0)
    assert accs[1] == pytest.approx(0.5)


def test_confusion_matrix_shape():
    calc = MetricsCalculator(num_classes=5)
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 1, 2, 2])
    cm = calc.calculate_confusion_matrix(y_true, y_pred)
    assert cm.shape == (5, 5)


def test_calculate_all_metrics_includes_expected_keys():
    calc = MetricsCalculator(num_classes=5)
    # 2D to trigger per_coin keys
    y_true = np.array([[0, 1], [1, 1], [2, 2]])
    y_pred = np.array([[0, 1], [0, 1], [2, 2]])
    metrics = calc.calculate_all_metrics(y_true, y_pred)

    # Core keys
    assert "accuracy" in metrics
    assert "directional_accuracy" in metrics
    assert "confusion_matrix" in metrics

    # Per-class keys for each class
    for i in range(5):
        assert f"precision_class_{i}" in metrics
        assert f"recall_class_{i}" in metrics
        assert f"f1_class_{i}" in metrics

    # Per-coin keys present
    assert "per_coin_accuracy" in metrics
    assert "mean_per_coin_accuracy" in metrics



