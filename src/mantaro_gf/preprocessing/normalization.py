"""Normalización de datos de precipitación."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from typing import Optional, Dict, Literal
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Normalizer:
    """
    Clase para normalizar datos de precipitación.

    Métodos:
    - zscore: StandardScaler (media=0, std=1)
    - robust: RobustScaler (mediana=0, IQR=1)
    - minmax: MinMaxScaler [0, 1]
    """

    def __init__(
        self,
        method: Literal["zscore", "robust", "minmax"] = "robust",
        scale_by_station: bool = True,
    ):
        """
        Args:
            method: Método de normalización
            scale_by_station: Si True, normaliza cada estación independientemente
        """
        self.method = method
        self.scale_by_station = scale_by_station
        self.scalers: Dict[str, object] = {}

    def fit(self, df: pd.DataFrame) -> "Normalizer":
        """
        Ajusta escaladores a los datos.

        Args:
            df: DataFrame con columnas: station_id, precip_mm

        Returns:
            Self para chaining
        """
        logger.info(f"Ajustando normalizadores ({self.method})")

        if self.scale_by_station:
            # Un scaler por estación
            for station_id in df["station_id"].unique():
                station_data = df[df["station_id"] == station_id]["precip_mm"].values.reshape(-1, 1)
                # Remover NaNs para fit
                station_data = station_data[~np.isnan(station_data).flatten()]

                if len(station_data) == 0:
                    logger.warning(f"Estación {station_id} sin datos válidos, skip")
                    continue

                scaler = self._create_scaler()
                scaler.fit(station_data.reshape(-1, 1))
                self.scalers[station_id] = scaler
        else:
            # Un scaler global
            all_data = df["precip_mm"].dropna().values.reshape(-1, 1)
            scaler = self._create_scaler()
            scaler.fit(all_data)
            self.scalers["global"] = scaler

        logger.info(f"Normalizadores ajustados para {len(self.scalers)} {'estaciones' if self.scale_by_station else 'global'}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma datos usando escaladores ajustados.

        Args:
            df: DataFrame con station_id, precip_mm

        Returns:
            DataFrame con columna precip_mm_norm
        """
        df = df.copy()
        df["precip_mm_norm"] = np.nan

        if self.scale_by_station:
            for station_id in df["station_id"].unique():
                if station_id not in self.scalers:
                    logger.warning(f"Scaler no encontrado para estación {station_id}, skip")
                    continue

                mask = df["station_id"] == station_id
                values = df.loc[mask, "precip_mm"].values.reshape(-1, 1)

                # Transform solo valores válidos
                valid_mask = ~np.isnan(values).flatten()
                if valid_mask.sum() > 0:
                    transformed = self.scalers[station_id].transform(values[valid_mask].reshape(-1, 1))
                    df.loc[mask & df["precip_mm"].notna(), "precip_mm_norm"] = transformed.flatten()
        else:
            scaler = self.scalers["global"]
            values = df["precip_mm"].values.reshape(-1, 1)
            valid_mask = ~np.isnan(values).flatten()
            transformed = scaler.transform(values[valid_mask].reshape(-1, 1))
            df.loc[df["precip_mm"].notna(), "precip_mm_norm"] = transformed.flatten()

        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Revierte normalización.

        Args:
            df: DataFrame con precip_mm_norm

        Returns:
            DataFrame con columna precip_mm_denorm
        """
        df = df.copy()
        df["precip_mm_denorm"] = np.nan

        if self.scale_by_station:
            for station_id in df["station_id"].unique():
                if station_id not in self.scalers:
                    continue

                mask = df["station_id"] == station_id
                values = df.loc[mask, "precip_mm_norm"].values.reshape(-1, 1)
                valid_mask = ~np.isnan(values).flatten()

                if valid_mask.sum() > 0:
                    denormalized = self.scalers[station_id].inverse_transform(
                        values[valid_mask].reshape(-1, 1)
                    )
                    df.loc[mask & df["precip_mm_norm"].notna(), "precip_mm_denorm"] = denormalized.flatten()
        else:
            scaler = self.scalers["global"]
            values = df["precip_mm_norm"].values.reshape(-1, 1)
            valid_mask = ~np.isnan(values).flatten()
            denormalized = scaler.inverse_transform(values[valid_mask].reshape(-1, 1))
            df.loc[df["precip_mm_norm"].notna(), "precip_mm_denorm"] = denormalized.flatten()

        return df

    def _create_scaler(self):
        """Crea scaler según método configurado."""
        if self.method == "zscore":
            return StandardScaler()
        elif self.method == "robust":
            return RobustScaler()
        elif self.method == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Método desconocido: {self.method}")

    def save(self, path: Path) -> None:
        """Guarda scalers a disco."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scalers, path)
        logger.info(f"Normalizadores guardados en {path}")

    def load(self, path: Path) -> "Normalizer":
        """Carga scalers desde disco."""
        self.scalers = joblib.load(path)
        logger.info(f"Normalizadores cargados desde {path}")
        return self
