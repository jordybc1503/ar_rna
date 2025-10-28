"""Modelo de árbol de regresión (Decision Tree) para gap-filling."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from typing import Optional, Dict, Tuple
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RegressionTreeModel:
    """
    Árbol de regresión para completar huecos en precipitación.

    Ventajas:
    - No requiere normalización
    - Interpreta features importantes (importances)
    - Robusto a outliers
    """

    def __init__(
        self,
        max_depth: int = 8,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        criterion: str = "squared_error",
        random_state: int = 42,
    ):
        """
        Args:
            max_depth: Profundidad máxima del árbol
            min_samples_split: Mínimo de muestras para split
            min_samples_leaf: Mínimo de muestras por hoja
            criterion: Criterio de división ("squared_error", "friedman_mse", "absolute_error")
            random_state: Semilla aleatoria
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state

        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state,
        )

        self.feature_names: Optional[list] = None
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "RegressionTreeModel":
        """
        Entrena árbol de regresión.

        Args:
            X: Features (DataFrame o array)
            y: Target (precipitación)

        Returns:
            Self para chaining
        """
        logger.info(f"Entrenando árbol de regresión: {X.shape[0]} muestras, {X.shape[1]} features")

        # Guardar nombres de features
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        # Entrenar
        self.model.fit(X, y)
        self.is_fitted = True

        logger.info(
            f"Árbol entrenado: {self.model.get_n_leaves()} hojas, "
            f"profundidad {self.model.get_depth()}"
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice precipitación.

        Args:
            X: Features

        Returns:
            Array de predicciones
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Retorna importancias de features (Gini importance).

        Returns:
            DataFrame con feature y importance ordenado
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")

        importances = self.model.feature_importances_

        if self.feature_names:
            df = pd.DataFrame({
                "feature": self.feature_names,
                "importance": importances,
            })
        else:
            df = pd.DataFrame({
                "feature": [f"feature_{i}" for i in range(len(importances))],
                "importance": importances,
            })

        return df.sort_values("importance", ascending=False)

    def get_top_features(self, k: int = 10, min_importance: float = 0.01) -> list:
        """
        Retorna top-k features más importantes.

        Args:
            k: Número de features a retornar
            min_importance: Umbral mínimo de importancia

        Returns:
            Lista de nombres de features
        """
        importances_df = self.get_feature_importances()

        # Filtrar por umbral y tomar top-k
        filtered = importances_df[importances_df["importance"] >= min_importance]
        top_k = filtered.head(k)

        logger.info(f"Top-{k} features: {top_k['feature'].tolist()}")

        return top_k["feature"].tolist()

    def get_model_complexity(self) -> Dict:
        """
        Retorna métricas de complejidad del árbol.

        Returns:
            Dict con n_leaves, depth, n_features
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")

        return {
            "n_leaves": self.model.get_n_leaves(),
            "depth": self.model.get_depth(),
            "n_features": self.model.n_features_in_,
        }

    def save(self, path: Path) -> None:
        """Guarda modelo a disco."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "params": {
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "criterion": self.criterion,
            }
        }, path)

        logger.info(f"Modelo guardado en {path}")

    def load(self, path: Path) -> "RegressionTreeModel":
        """Carga modelo desde disco."""
        data = joblib.load(path)

        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.is_fitted = True

        # Restaurar params
        params = data["params"]
        self.max_depth = params["max_depth"]
        self.min_samples_split = params["min_samples_split"]
        self.min_samples_leaf = params["min_samples_leaf"]
        self.criterion = params["criterion"]

        logger.info(f"Modelo cargado desde {path}")
        return self
