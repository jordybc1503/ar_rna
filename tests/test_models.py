"""Tests básicos para modelos."""

import pytest
import pandas as pd
import numpy as np
from mantaro_gf.models import RegressionTreeModel, MLPModel, HybridARMLPModel


def generate_synthetic_data(n_samples=100, n_features=5):
    """Genera datos sintéticos para tests."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)]
    )
    y = pd.Series(X["feat_0"] * 2 + X["feat_1"] + np.random.randn(n_samples) * 0.1)
    return X, y


def test_ar_model_fit_predict():
    """Test AR entrena y predice."""
    X, y = generate_synthetic_data()

    model = RegressionTreeModel(max_depth=3)
    model.fit(X, y)

    y_pred = model.predict(X)

    assert len(y_pred) == len(y)
    assert model.is_fitted


def test_mlp_model_fit_predict():
    """Test MLP entrena y predice."""
    X, y = generate_synthetic_data()

    model = MLPModel(hidden_layer_sizes=(16, 8), max_iter=50)
    model.fit(X, y)

    y_pred = model.predict(X)

    assert len(y_pred) == len(y)
    assert model.is_fitted


def test_hybrid_model():
    """Test modelo híbrido."""
    X, y = generate_synthetic_data(n_samples=200, n_features=10)

    model = HybridARMLPModel(
        top_k_features=5,
        mlp_hidden_layers=(16,),
        mlp_max_iter=50
    )
    model.fit(X, y)

    assert model.is_fitted
    assert len(model.selected_features) <= 5

    y_pred = model.predict(X)
    assert len(y_pred) == len(y)


if __name__ == "__main__":
    pytest.main([__file__])
