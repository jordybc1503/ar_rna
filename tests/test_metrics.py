"""Tests para evaluación y métricas."""

import pytest
import numpy as np
from mantaro_gf.evaluation.metrics import calculate_r, calculate_mae, calculate_rmse, calculate_bias


def test_metrics_perfect_prediction():
    """Test métricas con predicción perfecta."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    assert abs(calculate_r(y_true, y_pred) - 1.0) < 0.001
    assert calculate_mae(y_true, y_pred) == 0.0
    assert calculate_rmse(y_true, y_pred) == 0.0
    assert calculate_bias(y_true, y_pred) == 0.0


def test_metrics_with_nans():
    """Test métricas manejan NaNs."""
    y_true = np.array([1, 2, np.nan, 4, 5])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, np.nan])

    r = calculate_r(y_true, y_pred)
    assert not np.isnan(r)


if __name__ == "__main__":
    pytest.main([__file__])
