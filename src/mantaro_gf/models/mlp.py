"""Modelo MLP (Multi-Layer Perceptron) para gap-filling."""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Tuple
import joblib
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


class MLPModel:
    """
    Red neuronal (MLP) para completar huecos en precipitación.

    Ventajas:
    - Aprende relaciones no lineales complejas
    - Generaliza bien con datos suficientes

    Requiere:
    - Normalización de features
    - Tuning de hiperparámetros
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: int = 256,
        learning_rate_init: float = 0.001,
        learning_rate: str = "adaptive",
        max_iter: int = 300,
        early_stopping: bool = True,
        validation_fraction: float = 0.15,
        n_iter_no_change: int = 20,
        random_state: int = 42,
        verbose: bool = False,
    ):
        """
        Args:
            hidden_layer_sizes: Tupla con número de neuronas por capa (ej: (128, 64, 32))
            activation: Función de activación ("relu", "tanh", "logistic")
            solver: Optimizador ("adam", "sgd", "lbfgs")
            alpha: Regularización L2
            batch_size: Tamaño de mini-batch
            learning_rate_init: Learning rate inicial
            learning_rate: Esquema de learning rate ("constant", "invscaling", "adaptive")
            max_iter: Máximo de épocas
            early_stopping: Si usar early stopping
            validation_fraction: Fracción de datos para validación (early stopping)
            n_iter_no_change: Épocas sin mejora para early stopping
            random_state: Semilla aleatoria
            verbose: Si imprimir progreso
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.verbose = verbose

        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            learning_rate=learning_rate,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
            verbose=verbose,
        )

        self.scaler = StandardScaler()
        self.feature_names: Optional[list] = None
        self.is_fitted = False
        self.training_time: Optional[float] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit_scaler: bool = True,
    ) -> "MLPModel":
        """
        Entrena MLP.

        Args:
            X: Features (DataFrame o array)
            y: Target (precipitación)
            fit_scaler: Si ajustar scaler (False si ya está ajustado)

        Returns:
            Self para chaining
        """
        logger.info(
            f"Entrenando MLP: {X.shape[0]} muestras, {X.shape[1]} features, "
            f"arquitectura {self.hidden_layer_sizes}"
        )

        # Guardar nombres de features
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        # Normalizar features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Entrenar
        start_time = time.time()
        self.model.fit(X_scaled, y)
        self.training_time = time.time() - start_time

        self.is_fitted = True

        logger.info(
            f"MLP entrenado: {len(self.hidden_layer_sizes)} capas, "
            f"{sum(self.hidden_layer_sizes)} neuronas, "
            f"{self.model.n_iter_} iteraciones, "
            f"tiempo {self.training_time:.2f}s"
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

        # Normalizar
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)

    def get_model_complexity(self) -> Dict:
        """
        Retorna métricas de complejidad del MLP.

        Returns:
            Dict con n_layers, n_neurons, n_parameters
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")

        # Contar parámetros totales
        n_params = 0
        for coef in self.model.coefs_:
            n_params += coef.size
        for intercept in self.model.intercepts_:
            n_params += intercept.size

        return {
            "n_layers": len(self.hidden_layer_sizes) + 1,  # +1 output layer
            "n_neurons": sum(self.hidden_layer_sizes),
            "n_parameters": n_params,
            "n_iterations": self.model.n_iter_,
            "training_time_s": self.training_time,
        }

    def get_training_history(self) -> Dict:
        """
        Retorna histórico de entrenamiento (loss curve).

        Returns:
            Dict con loss_curve
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado.")

        return {
            "loss_curve": self.model.loss_curve_,
            "best_loss": self.model.best_loss_ if self.early_stopping else None,
        }

    def save(self, path: Path) -> None:
        """Guarda modelo y scaler a disco."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "params": {
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "activation": self.activation,
                "solver": self.solver,
                "alpha": self.alpha,
                "batch_size": self.batch_size,
                "learning_rate_init": self.learning_rate_init,
            },
            "training_time": self.training_time,
        }, path)

        logger.info(f"Modelo MLP guardado en {path}")

    def load(self, path: Path) -> "MLPModel":
        """Carga modelo y scaler desde disco."""
        data = joblib.load(path)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.training_time = data.get("training_time")
        self.is_fitted = True

        # Restaurar params
        params = data["params"]
        self.hidden_layer_sizes = params["hidden_layer_sizes"]
        self.activation = params["activation"]
        self.solver = params["solver"]
        self.alpha = params["alpha"]
        self.batch_size = params["batch_size"]
        self.learning_rate_init = params["learning_rate_init"]

        logger.info(f"Modelo MLP cargado desde {path}")
        return self
