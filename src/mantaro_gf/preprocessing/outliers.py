"""Detección de outliers en series de precipitación."""

import pandas as pd
import numpy as np
from typing import Literal, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class OutlierDetector:
    """
    Clase para detectar y manejar outliers en datos de precipitación.

    Métodos:
    - zscore: Z-score threshold
    - iqr: Interquartile range
    - modified_zscore: MAD-based (robusto)
    """

    def __init__(
        self,
        method: Literal["zscore", "iqr", "modified_zscore"] = "zscore",
        threshold: float = 5.0,
        action: Literal["flag", "remove", "cap"] = "flag",
    ):
        """
        Args:
            method: Método de detección
            threshold: Umbral (z-score o IQR multiplier)
            action: Qué hacer con outliers ("flag", "remove", "cap")
        """
        self.method = method
        self.threshold = threshold
        self.action = action

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta outliers en precipitación.

        Args:
            df: DataFrame con station_id, precip_mm

        Returns:
            DataFrame con columna adicional "is_outlier" (bool)
        """
        logger.info(f"Detectando outliers con método {self.method}")

        df = df.copy()
        df["is_outlier"] = False

        # Detectar por estación (variabilidad local)
        for station_id in df["station_id"].unique():
            mask = df["station_id"] == station_id
            values = df.loc[mask, "precip_mm"].dropna()

            if len(values) < 10:  # Mínimo de datos
                continue

            if self.method == "zscore":
                outlier_mask = self._zscore_method(values)
            elif self.method == "iqr":
                outlier_mask = self._iqr_method(values)
            elif self.method == "modified_zscore":
                outlier_mask = self._modified_zscore_method(values)
            else:
                raise ValueError(f"Método desconocido: {self.method}")

            # Mapear a índices originales
            df.loc[mask & df["precip_mm"].notna(), "is_outlier"] = outlier_mask

        n_outliers = df["is_outlier"].sum()
        logger.info(f"Detectados {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")

        return df

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maneja outliers según acción configurada.

        Args:
            df: DataFrame con columna "is_outlier"

        Returns:
            DataFrame procesado
        """
        if "is_outlier" not in df.columns:
            df = self.detect(df)

        df = df.copy()

        if self.action == "flag":
            # Solo marcar en qc_flag
            df.loc[df["is_outlier"], "qc_flag"] = "outlier"

        elif self.action == "remove":
            # Convertir a NaN
            df.loc[df["is_outlier"], "precip_mm"] = np.nan
            df.loc[df["is_outlier"], "qc_flag"] = "outlier_removed"

        elif self.action == "cap":
            # Cap a percentil 99
            for station_id in df["station_id"].unique():
                mask = (df["station_id"] == station_id) & df["is_outlier"]
                if mask.sum() == 0:
                    continue

                p99 = df.loc[df["station_id"] == station_id, "precip_mm"].quantile(0.99)
                df.loc[mask, "precip_mm"] = p99
                df.loc[mask, "qc_flag"] = "outlier_capped"

        return df

    def _zscore_method(self, values: pd.Series) -> np.ndarray:
        """Detección por z-score estándar."""
        z_scores = np.abs(stats.zscore(values, nan_policy="omit"))
        return z_scores > self.threshold

    def _iqr_method(self, values: pd.Series) -> np.ndarray:
        """Detección por IQR (Tukey's fences)."""
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR

        return (values < lower_bound) | (values > upper_bound)

    def _modified_zscore_method(self, values: pd.Series) -> np.ndarray:
        """
        Detección por modified z-score (MAD - Median Absolute Deviation).
        Más robusto a outliers extremos.
        """
        median = values.median()
        mad = np.median(np.abs(values - median))

        # Evitar división por 0
        if mad == 0:
            mad = np.mean(np.abs(values - median))

        modified_z_scores = 0.6745 * (values - median) / mad
        return np.abs(modified_z_scores) > self.threshold

    def get_outlier_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera resumen de outliers por estación.

        Args:
            df: DataFrame con "is_outlier"

        Returns:
            DataFrame con estadísticas de outliers por estación
        """
        if "is_outlier" not in df.columns:
            df = self.detect(df)

        summary = df.groupby("station_id").agg({
            "is_outlier": ["sum", "mean"],
            "precip_mm": ["count", "max"],
        })
        summary.columns = ["n_outliers", "outlier_ratio", "n_total", "max_precip"]

        return summary.sort_values("outlier_ratio", ascending=False)
