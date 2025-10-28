"""Cliente para descargar datos de SENAMHI (Servicio Nacional de Meteorología e Hidrología del Perú)."""

import pandas as pd
import requests
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SenamhiClient:
    """
    Cliente para interactuar con datos de SENAMHI.

    Nota: SENAMHI no tiene API pública oficial robusta (a Oct 2025).
    Este adaptador lee archivos TXT/CSV descargados manualmente o vía scraping.
    """

    def __init__(
        self,
        data_dir: Path,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Args:
            data_dir: Directorio donde están/estarán los archivos descargados
            api_key: Clave API (si aplica en el futuro)
            base_url: URL base del servicio
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.base_url = base_url

    def read_station_file(
        self,
        station_name: str,
        file_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Lee archivo TXT/CSV de una estación SENAMHI.

        Args:
            station_name: Nombre de la estación (ej: "Huayao")
            file_path: Path al archivo; si None, busca en data_dir

        Returns:
            DataFrame con columnas: date, precip_mm, qc_flag
        """
        if file_path is None:
            # Buscar en data_dir
            pattern = f"Estación-{station_name}.*"
            files = list(self.data_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No se encontró archivo para estación {station_name}")
            file_path = files[0]

        logger.info(f"Leyendo estación SENAMHI: {station_name} desde {file_path}")

        # Parser flexible (ajustar según formato real)
        # Ejemplo: archivos con columnas separadas por tabs/espacios
        df = pd.read_csv(
            file_path,
            sep=r"\s+|,",
            engine="python",
            parse_dates=["Fecha"] if "Fecha" in pd.read_csv(file_path, nrows=1, sep=r"\s+|,", engine="python").columns else False,
            na_values=["-", "S/D", "NaN", ""],
        )

        # Normalizar nombres de columnas (adaptar a estructura real)
        df = self._normalize_columns(df, station_name)

        return df

    def _normalize_columns(self, df: pd.DataFrame, station_name: str) -> pd.DataFrame:
        """
        Normaliza columnas a schema estándar: date, precip_mm, qc_flag.
        """
        # Ejemplo de mapeo (ajustar según archivos reales)
        column_mapping = {
            "Fecha": "date",
            "Precipitación": "precip_mm",
            "Precipitacion": "precip_mm",
            "Prec": "precip_mm",
            "PP": "precip_mm",
        }

        df = df.rename(columns=column_mapping)

        # Asegurar columna date
        if "date" not in df.columns:
            # Intentar construir desde Año/Mes/Día
            if {"Año", "Mes", "Día"}.issubset(df.columns):
                df["date"] = pd.to_datetime(df[["Año", "Mes", "Día"]].rename(
                    columns={"Año": "year", "Mes": "month", "Día": "day"}
                ))
            else:
                raise ValueError(f"No se pudo identificar columna de fecha en {station_name}")

        # Asegurar precip_mm
        if "precip_mm" not in df.columns:
            raise ValueError(f"No se pudo identificar columna de precipitación en {station_name}")

        # QC flag inicial
        df["qc_flag"] = "orig"
        df["station_id"] = station_name

        return df[["station_id", "date", "precip_mm", "qc_flag"]]

    def fetch_all_stations(
        self,
        bbox: Optional[List[float]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Lee todas las estaciones disponibles en data_dir.

        Args:
            bbox: [lon_min, lat_min, lon_max, lat_max] (filtrado futuro)
            start_date: Fecha inicio
            end_date: Fecha fin

        Returns:
            DataFrame consolidado de todas las estaciones
        """
        all_files = list(self.data_dir.glob("Estación-*.txt")) + list(self.data_dir.glob("Estación-*.csv"))
        logger.info(f"Encontrados {len(all_files)} archivos de estaciones SENAMHI")

        dfs = []
        for file_path in all_files:
            station_name = file_path.stem.replace("Estación-", "")
            try:
                df = self.read_station_file(station_name, file_path)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error leyendo {station_name}: {e}")

        if not dfs:
            raise ValueError("No se pudieron leer estaciones SENAMHI")

        combined = pd.concat(dfs, ignore_index=True)

        # Filtrar por fechas
        if start_date:
            combined = combined[combined["date"] >= start_date]
        if end_date:
            combined = combined[combined["date"] <= end_date]

        logger.info(f"Datos SENAMHI cargados: {len(combined)} registros")
        return combined
