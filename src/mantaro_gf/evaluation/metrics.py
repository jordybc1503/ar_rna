"""Métricas de evaluación para gap-filling."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def calculate_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula coeficiente de correlación de Pearson."""
    # Remover NaNs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return np.nan
    r, _ = pearsonr(y_true[mask], y_pred[mask])
    return r


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Mean Absolute Error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Root Mean Squared Error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))


def calculate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula BIAS (Mean Error)."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return np.mean(y_pred[mask] - y_true[mask])


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula todas las métricas de evaluación.

    Args:
        y_true: Valores reales
        y_pred: Valores predichos

    Returns:
        Dict con r, mae, rmse, bias
    """
    return {
        "r": calculate_r(y_true, y_pred),
        "mae": calculate_mae(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "bias": calculate_bias(y_true, y_pred),
    }


class MetricsCalculator:
    """Calculador de métricas con soporte para múltiples estaciones."""

    def __init__(self):
        self.results = []

    def add_result(
        self,
        station_id: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metadata: Dict = None,
    ):
        """Agrega resultado para cálculo de métricas."""
        metrics = calculate_all_metrics(y_true, y_pred)
        metrics["station_id"] = station_id
        metrics["n_samples"] = (~np.isnan(y_true) & ~np.isnan(y_pred)).sum()

        if metadata:
            metrics.update(metadata)

        self.results.append(metrics)

    def get_summary(self) -> pd.DataFrame:
        """Retorna DataFrame con todas las métricas."""
        return pd.DataFrame(self.results)

    def get_aggregated_metrics(self) -> Dict:
        """Retorna métricas agregadas (promedio)."""
        df = self.get_summary()
        return {
            "r_mean": df["r"].mean(),
            "mae_mean": df["mae"].mean(),
            "rmse_mean": df["rmse"].mean(),
            "bias_mean": df["bias"].mean(),
            "r_std": df["r"].std(),
            "mae_std": df["mae"].std(),
            "rmse_std": df["rmse"].std(),
            "bias_std": df["bias"].std(),
        }
