"""Modelo híbrido AR+MLP: AR selecciona features, MLP predice."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import joblib
from pathlib import Path
import logging
import time

from mantaro_gf.models.ar import RegressionTreeModel
from mantaro_gf.models.mlp import MLPModel

logger = logging.getLogger(__name__)


class HybridARMLPModel:
    """
    Modelo híbrido de dos pasos:
    1. Árbol de regresión (AR) selecciona top-k features más influyentes
    2. MLP entrena solo con features seleccionadas para predicción final

    Ventajas:
    - AR filtra ruido y reduce dimensionalidad
    - MLP captura relaciones no lineales en espacio reducido
    - Más interpretable que MLP puro
    - Potencialmente más preciso que AR puro
    """

    def __init__(
        self,
        # Parámetros AR (selector)
        ar_max_depth: int = 6,
        ar_min_samples_split: int = 20,
        ar_min_samples_leaf: int = 10,
        top_k_features: int = 15,
        min_importance: float = 0.01,
        # Parámetros MLP (predictor)
        mlp_hidden_layers: tuple = (64, 32),
        mlp_activation: str = "relu",
        mlp_alpha: float = 0.0005,
        mlp_batch_size: int = 128,
        mlp_learning_rate_init: float = 0.001,
        mlp_max_iter: int = 300,
        # General
        random_state: int = 42,
    ):
        """
        Args:
            ar_max_depth: Profundidad máxima del árbol selector
            ar_min_samples_split: Min samples split para AR
            ar_min_samples_leaf: Min samples leaf para AR
            top_k_features: Número de features a seleccionar
            min_importance: Umbral mínimo de importancia
            mlp_hidden_layers: Arquitectura del MLP
            mlp_activation: Activación del MLP
            mlp_alpha: Regularización L2 del MLP
            mlp_batch_size: Batch size del MLP
            mlp_learning_rate_init: Learning rate del MLP
            mlp_max_iter: Máx iteraciones del MLP
            random_state: Semilla aleatoria
        """
        # Componente AR (selector)
        self.ar_selector = RegressionTreeModel(
            max_depth=ar_max_depth,
            min_samples_split=ar_min_samples_split,
            min_samples_leaf=ar_min_samples_leaf,
            random_state=random_state,
        )

        # Componente MLP (predictor)
        self.mlp_predictor = MLPModel(
            hidden_layer_sizes=mlp_hidden_layers,
            activation=mlp_activation,
            alpha=mlp_alpha,
            batch_size=mlp_batch_size,
            learning_rate_init=mlp_learning_rate_init,
            max_iter=mlp_max_iter,
            early_stopping=True,
            random_state=random_state,
        )

        # Parámetros de selección
        self.top_k_features = top_k_features
        self.min_importance = min_importance

        # Features seleccionadas
        self.selected_features: Optional[List[str]] = None
        self.is_fitted = False
        self.training_time: Optional[float] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "HybridARMLPModel":
        """
        Entrena modelo híbrido en dos pasos.

        Args:
            X: Features completas (DataFrame)
            y: Target (precipitación)

        Returns:
            Self para chaining
        """
        logger.info(
            f"Entrenando modelo híbrido AR+MLP: {X.shape[0]} muestras, "
            f"{X.shape[1]} features iniciales"
        )

        start_time = time.time()

        # PASO 1: Entrenar AR y seleccionar features
        logger.info("Paso 1/2: Entrenando AR para selección de features")
        self.ar_selector.fit(X, y)

        # Extraer top-k features importantes
        self.selected_features = self.ar_selector.get_top_features(
            k=self.top_k_features,
            min_importance=self.min_importance,
        )

        if len(self.selected_features) == 0:
            raise ValueError(
                f"No se encontraron features con importancia >= {self.min_importance}. "
                "Considera reducir min_importance."
            )

        logger.info(f"Features seleccionadas ({len(self.selected_features)}): {self.selected_features}")

        # PASO 2: Entrenar MLP con features seleccionadas
        logger.info("Paso 2/2: Entrenando MLP con features seleccionadas")
        X_selected = X[self.selected_features]
        self.mlp_predictor.fit(X_selected, y, fit_scaler=True)

        self.training_time = time.time() - start_time
        self.is_fitted = True

        logger.info(
            f"Modelo híbrido entrenado: {len(self.selected_features)} features finales, "
            f"tiempo total {self.training_time:.2f}s"
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice usando MLP con features seleccionadas por AR.

        Args:
            X: Features completas

        Returns:
            Array de predicciones
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")

        # Seleccionar features
        X_selected = X[self.selected_features]

        # Predecir con MLP
        return self.mlp_predictor.predict(X_selected)

    def get_selected_features(self) -> List[str]:
        """
        Retorna lista de features seleccionadas.

        Returns:
            Lista de nombres de features
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")

        return self.selected_features

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Retorna importancias del AR (para interpretabilidad).

        Returns:
            DataFrame con features e importancias
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")

        return self.ar_selector.get_feature_importances()

    def get_model_complexity(self) -> Dict:
        """
        Retorna métricas de complejidad de ambos componentes.

        Returns:
            Dict con complejidad AR y MLP
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")

        ar_complexity = self.ar_selector.get_model_complexity()
        mlp_complexity = self.mlp_predictor.get_model_complexity()

        return {
            "ar_selector": ar_complexity,
            "mlp_predictor": mlp_complexity,
            "selected_features_count": len(self.selected_features),
            "feature_reduction_ratio": len(self.selected_features) / ar_complexity["n_features"],
            "total_training_time_s": self.training_time,
        }

    def save(self, path: Path) -> None:
        """
        Guarda modelo híbrido (AR + MLP + metadata).

        Args:
            path: Path base (se guardarán múltiples archivos)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Guardar componentes
        ar_path = path.parent / f"{path.stem}_ar_selector.joblib"
        mlp_path = path.parent / f"{path.stem}_mlp_predictor.joblib"

        self.ar_selector.save(ar_path)
        self.mlp_predictor.save(mlp_path)

        # Guardar metadata del híbrido
        joblib.dump({
            "selected_features": self.selected_features,
            "top_k_features": self.top_k_features,
            "min_importance": self.min_importance,
            "training_time": self.training_time,
        }, path)

        logger.info(f"Modelo híbrido guardado en {path}")

    def load(self, path: Path) -> "HybridARMLPModel":
        """
        Carga modelo híbrido desde disco.

        Args:
            path: Path base

        Returns:
            Self
        """
        path = Path(path)

        # Cargar componentes
        ar_path = path.parent / f"{path.stem}_ar_selector.joblib"
        mlp_path = path.parent / f"{path.stem}_mlp_predictor.joblib"

        self.ar_selector.load(ar_path)
        self.mlp_predictor.load(mlp_path)

        # Cargar metadata
        data = joblib.load(path)
        self.selected_features = data["selected_features"]
        self.top_k_features = data["top_k_features"]
        self.min_importance = data["min_importance"]
        self.training_time = data.get("training_time")

        self.is_fitted = True

        logger.info(f"Modelo híbrido cargado desde {path}")
        return self
