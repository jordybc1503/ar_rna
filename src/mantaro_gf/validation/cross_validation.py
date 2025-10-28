"""Cross-validation temporal y espacial."""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Dict, Callable
import logging

logger = logging.getLogger(__name__)


class CrossValidator:
    """Cross-validation para series temporales."""

    def __init__(self, n_splits: int = 5, test_size: int = 365):
        self.n_splits = n_splits
        self.test_size = test_size

    def time_series_split(self, X: pd.DataFrame, y: pd.Series):
        """CV temporal por bloques."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        for train_idx, test_idx in tscv.split(X):
            yield X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    def leave_one_station_out(
        self,
        feature_dfs: Dict[str, pd.DataFrame],
        fit_predict_fn: Callable,
    ) -> pd.DataFrame:
        """
        Leave-One-Station-Out CV.

        Args:
            feature_dfs: Dict {station_id: feature_df}
            fit_predict_fn: Función que entrena y predice

        Returns:
            DataFrame con resultados por estación
        """
        results = []
        station_ids = list(feature_dfs.keys())

        for test_station in station_ids:
            logger.info(f"LOSO: Test en {test_station}")

            # Train en todas menos test_station
            train_stations = [s for s in station_ids if s != test_station]

            # Consolidar datos de entrenamiento
            X_train_list, y_train_list = [], []
            for st in train_stations:
                df = feature_dfs[st].dropna()
                X_train_list.append(df.drop(columns=["y"]))
                y_train_list.append(df["y"])

            X_train = pd.concat(X_train_list, ignore_index=True)
            y_train = pd.concat(y_train_list, ignore_index=True)

            # Test
            test_df = feature_dfs[test_station].dropna()
            X_test = test_df.drop(columns=["y"])
            y_test = test_df["y"]

            # Entrenar y predecir
            y_pred = fit_predict_fn(X_train, y_train, X_test)

            results.append({
                "station_id": test_station,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "y_true": y_test.values,
                "y_pred": y_pred,
            })

        return pd.DataFrame(results)
