"""Control de calidad (QC) de datos de precipitación."""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class QualityControl:
    """
    Clase para aplicar control de calidad a series de precipitación diaria.

    Incluye:
    - Detección de duplicados
    - Remoción de valores físicamente imposibles
    - Cálculo de % de datos válidos
    - Flags de calidad
    """

    def __init__(
        self,
        min_valid_ratio: float = 0.7,
        max_precip_mm: float = 500.0,
        min_precip_mm: float = 0.0,
    ):
        """
        Args:
            min_valid_ratio: Ratio mínimo de datos válidos para mantener estación
            max_precip_mm: Valor máximo físicamente plausible (mm/día)
            min_precip_mm: Valor mínimo (generalmente 0)
        """
        self.min_valid_ratio = min_valid_ratio
        self.max_precip_mm = max_precip_mm
        self.min_precip_mm = min_precip_mm

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica control de calidad a DataFrame de precipitación.

        Args:
            df: DataFrame con columnas: station_id, date, precip_mm, qc_flag

        Returns:
            DataFrame limpio con flags QC actualizados
        """
        logger.info(f"Aplicando QC a {len(df)} registros")

        df = df.copy()

        # 1. Remover duplicados
        df = self._remove_duplicates(df)

        # 2. Validar rango físico
        df = self._validate_physical_range(df)

        # 3. Calcular completitud por estación
        df = self._calculate_completeness(df)

        # 4. Filtrar estaciones con baja calidad
        df = self._filter_low_quality_stations(df)

        logger.info(f"QC completado: {len(df)} registros válidos")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remueve registros duplicados (station_id, date)."""
        n_before = len(df)
        df = df.drop_duplicates(subset=["station_id", "date"], keep="first")
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.warning(f"Removidos {n_removed} duplicados")
        return df

    def _validate_physical_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida que valores estén en rango físico plausible."""
        # Identificar valores fuera de rango
        invalid_mask = (
            (df["precip_mm"] < self.min_precip_mm) |
            (df["precip_mm"] > self.max_precip_mm)
        )

        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            logger.warning(f"Encontrados {n_invalid} valores fuera de rango físico")
            df.loc[invalid_mask, "qc_flag"] = "invalid_range"
            df.loc[invalid_mask, "precip_mm"] = np.nan

        return df

    def _calculate_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula % de datos válidos por estación."""
        # Agrupar por estación
        station_stats = df.groupby("station_id").agg({
            "precip_mm": ["count", lambda x: x.notna().sum()]
        })
        station_stats.columns = ["total", "valid"]
        station_stats["valid_ratio"] = station_stats["valid"] / station_stats["total"]

        # Agregar al DataFrame
        df = df.merge(
            station_stats[["valid_ratio"]],
            left_on="station_id",
            right_index=True,
            how="left"
        )

        return df

    def _filter_low_quality_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra estaciones con ratio de datos válidos bajo threshold."""
        low_quality = df["valid_ratio"] < self.min_valid_ratio
        n_low_quality = df[low_quality]["station_id"].nunique()

        if n_low_quality > 0:
            logger.warning(
                f"Removiendo {n_low_quality} estaciones con <{self.min_valid_ratio*100}% datos válidos"
            )
            df = df[~low_quality]

        # Remover columna auxiliar
        df = df.drop(columns=["valid_ratio"], errors="ignore")

        return df

    def get_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Genera reporte de calidad de datos.

        Args:
            df: DataFrame con QC aplicado

        Returns:
            Diccionario con estadísticas de calidad
        """
        report = {
            "n_stations": df["station_id"].nunique(),
            "n_records": len(df),
            "date_range": (df["date"].min(), df["date"].max()),
            "missing_ratio": df["precip_mm"].isna().mean(),
            "qc_flags": df["qc_flag"].value_counts().to_dict(),
        }

        # Por estación
        station_stats = df.groupby("station_id").agg({
            "precip_mm": lambda x: x.notna().mean(),
            "date": ["min", "max"],
        })
        station_stats.columns = ["valid_ratio", "start_date", "end_date"]
        report["station_stats"] = station_stats.to_dict()

        return report
