"""Constructor de features desfasadas (lags) para modelado temporal."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging
from mantaro_gf.utils.time_windows import (
    create_lagged_features,
    create_rolling_features,
    create_temporal_features,
)

logger = logging.getLogger(__name__)


class LaggedFeatureBuilder:
    """
    Construye features desfasadas para cada estación objetivo.

    Incluye:
    - Lags de la propia estación
    - Lags de estaciones vecinas/influyentes
    - Rolling features (medias móviles, std, percentiles)
    - Features temporales cíclicas
    """

    def __init__(
        self,
        lags: List[int] = [1, 2, 3, 7, 14, 30],
        rolling_windows: List[int] = [7, 14, 30],
        rolling_stats: List[str] = ["mean", "max", "std"],
        include_temporal: bool = True,
        neighbor_stations: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Args:
            lags: Lista de desfases temporales
            rolling_windows: Ventanas para agregación móvil
            rolling_stats: Estadísticas rolling
            include_temporal: Si incluir features temporales cíclicas
            neighbor_stations: Dict {station_id: [neighbor_ids]} para incluir sus lags
        """
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.rolling_stats = rolling_stats
        self.include_temporal = include_temporal
        self.neighbor_stations = neighbor_stations or {}

    def build(
        self,
        df: pd.DataFrame,
        target_station: str,
    ) -> pd.DataFrame:
        """
        Construye matriz de features para una estación objetivo.

        Args:
            df: DataFrame con columnas: station_id, date, precip_mm
            target_station: ID de estación objetivo

        Returns:
            DataFrame con features (índice = date)
        """
        logger.info(f"Construyendo features para estación {target_station}")

        # Serie objetivo
        target_series = df[df["station_id"] == target_station].set_index("date")["precip_mm"]
        target_series = target_series.sort_index()

        # Inicializar DataFrame de features
        feature_df = pd.DataFrame(index=target_series.index)

        # 1. Lags de estación objetivo
        lag_features = create_lagged_features(target_series, self.lags, prefix="target_lag")
        feature_df = feature_df.join(lag_features)

        # 2. Rolling features de estación objetivo
        if self.rolling_windows:
            rolling_features = create_rolling_features(
                target_series,
                self.rolling_windows,
                self.rolling_stats,
                prefix="target_roll",
            )
            feature_df = feature_df.join(rolling_features)

        # 3. Lags de estaciones vecinas
        if target_station in self.neighbor_stations:
            neighbors = self.neighbor_stations[target_station]
            for neighbor_id in neighbors:
                neighbor_series = df[df["station_id"] == neighbor_id].set_index("date")["precip_mm"]
                neighbor_series = neighbor_series.sort_index()

                # Alinear temporalmente
                neighbor_series = neighbor_series.reindex(target_series.index)

                # Lags del vecino
                neighbor_lags = create_lagged_features(
                    neighbor_series,
                    self.lags,
                    prefix=f"neighbor_{neighbor_id}_lag",
                )
                feature_df = feature_df.join(neighbor_lags)

        # 4. Features temporales
        if self.include_temporal:
            temporal_features = create_temporal_features(feature_df.index)
            feature_df = feature_df.join(temporal_features)

        # 5. Target (y)
        feature_df["y"] = target_series

        logger.info(
            f"Features construidas para {target_station}: "
            f"{feature_df.shape[1]-1} features, {len(feature_df)} muestras"
        )

        return feature_df

    def build_all_stations(
        self,
        df: pd.DataFrame,
        stations: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Construye features para múltiples estaciones.

        Args:
            df: DataFrame completo
            stations: Lista de estaciones objetivo (si None, usa todas)

        Returns:
            Dict {station_id: feature_df}
        """
        if stations is None:
            stations = df["station_id"].unique()

        logger.info(f"Construyendo features para {len(stations)} estaciones")

        feature_dfs = {}
        for station_id in stations:
            try:
                feature_df = self.build(df, station_id)
                feature_dfs[station_id] = feature_df
            except Exception as e:
                logger.warning(f"Error construyendo features para {station_id}: {e}")

        return feature_dfs

    def save_features(
        self,
        feature_dfs: Dict[str, pd.DataFrame],
        output_dir: str,
    ) -> None:
        """
        Guarda features a disco (Parquet).

        Args:
            feature_dfs: Dict de DataFrames de features
            output_dir: Directorio de salida
        """
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for station_id, feature_df in feature_dfs.items():
            file_path = output_path / f"features_{station_id}.parquet"
            feature_df.to_parquet(file_path)

        logger.info(f"Features guardadas en {output_dir}")

    def load_features(
        self,
        input_dir: str,
        stations: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Carga features desde disco.

        Args:
            input_dir: Directorio con archivos Parquet
            stations: Lista de estaciones (si None, carga todas)

        Returns:
            Dict {station_id: feature_df}
        """
        from pathlib import Path
        input_path = Path(input_dir)

        if stations is None:
            # Buscar todos los archivos
            files = list(input_path.glob("features_*.parquet"))
            stations = [f.stem.replace("features_", "") for f in files]

        feature_dfs = {}
        for station_id in stations:
            file_path = input_path / f"features_{station_id}.parquet"
            if file_path.exists():
                feature_dfs[station_id] = pd.read_parquet(file_path)

        logger.info(f"Features cargadas para {len(feature_dfs)} estaciones desde {input_dir}")
        return feature_dfs
