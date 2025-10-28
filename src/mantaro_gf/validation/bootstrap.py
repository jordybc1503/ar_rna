"""Bootstrap para estimación de incertidumbre."""

import numpy as np
from typing import Callable, Dict
import logging

logger = logging.getLogger(__name__)


class Bootstrap:
    """Bootstrap para intervalos de confianza."""

    def __init__(self, n_iterations: int = 100, confidence: float = 0.95):
        self.n_iterations = n_iterations
        self.confidence = confidence

    def estimate_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn: Callable,
    ) -> Dict:
        """
        Calcula intervalos de confianza por bootstrap.

        Args:
            y_true: Valores reales
            y_pred: Predicciones
            metric_fn: Función de métrica

        Returns:
            Dict con mean, ci_lower, ci_upper
        """
        n = len(y_true)
        metrics = []

        for _ in range(self.n_iterations):
            # Resample con reemplazo
            indices = np.random.choice(n, size=n, replace=True)
            metric_value = metric_fn(y_true[indices], y_pred[indices])
            metrics.append(metric_value)

        metrics = np.array(metrics)
        alpha = 1 - self.confidence

        return {
            "mean": metrics.mean(),
            "ci_lower": np.percentile(metrics, alpha/2 * 100),
            "ci_upper": np.percentile(metrics, (1 - alpha/2) * 100),
        }
