"""Cliente para descargar datos de ANA (Autoridad Nacional del Agua) - SNIRH."""

import pandas as pd
import requests
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ANAClient:
    """
    Cliente para interactuar con datos de ANA (Sistema Nacional de Información de Recursos Hídricos).

    SNIRH proporciona acceso a datos hidrológicos y meteorológicos.
    Este adaptador lee archivos CSV descargados desde el portal SNIRH.
    """

    def __init__(
        self,
        data_dir: Path,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Args:
            data_dir: Directorio con archivos CSV de SNIRH
            username: Usuario (si se requiere autenticación futura)
            password: Contraseña
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.username = username
        self.password = password

    def read_station_file(
        self,
        station_code: str,
        file_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Lee archivo CSV de estación descargado de SNIRH.

        Args:
            station_code: Código de estación (ej: "000635", "155115")
            file_path: Path al archivo; si None, busca en data_dir

        Returns:
            DataFrame con: station_id, date, precip_mm, qc_flag
        """
        if file_path is None:
            # Buscar por código
            pattern = f"*{station_code}*.csv"
            files = list(self.data_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No se encontró archivo para estación ANA {station_code}")
            file_path = files[0]

        logger.info(f"Leyendo estación ANA: {station_code} desde {file_path}")

        # Leer CSV (formato SNIRH típico)
        df = pd.read_csv(
            file_path,
            parse_dates=["Fecha"] if "Fecha" in pd.read_csv(file_path, nrows=1).columns else False,
            na_values=["-", "S/D", "NaN", "", "---"],
        )

        # Normalizar
        df = self._normalize_columns(df, station_code)

        return df

    def _normalize_columns(self, df: pd.DataFrame, station_code: str) -> pd.DataFrame:
        """
        Normaliza columnas a schema estándar.
        """
        # Mapeo típico de SNIRH
        column_mapping = {
            "Fecha": "date",
            "Precipitación Total Diario(mm)": "precip_mm",
            "Precipitación(mm)": "precip_mm",
            "Precipitacion": "precip_mm",
            "Valor": "precip_mm",
        }

        df = df.rename(columns=column_mapping)

        # Asegurar date
        if "date" not in df.columns:
            # Intentar otras variantes
            date_cols = [c for c in df.columns if "fecha" in c.lower()]
            if date_cols:
                df["date"] = pd.to_datetime(df[date_cols[0]])
            else:
                raise ValueError(f"No se encontró columna de fecha en {station_code}")

        # Asegurar precip_mm
        if "precip_mm" not in df.columns:
            precip_cols = [c for c in df.columns if "precip" in c.lower() or "valor" in c.lower()]
            if precip_cols:
                df["precip_mm"] = pd.to_numeric(df[precip_cols[0]], errors="coerce")
            else:
                raise ValueError(f"No se encontró columna de precipitación en {station_code}")

        df["qc_flag"] = "orig"
        df["station_id"] = station_code

        return df[["station_id", "date", "precip_mm", "qc_flag"]]

    def fetch_all_stations(
        self,
        bbox: Optional[List[float]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Lee todas las estaciones ANA disponibles en data_dir.

        Args:
            bbox: [lon_min, lat_min, lon_max, lat_max]
            start_date: Fecha inicio
            end_date: Fecha fin

        Returns:
            DataFrame consolidado
        """
        all_files = list(self.data_dir.glob("Estación-*.csv"))
        logger.info(f"Encontrados {len(all_files)} archivos de estaciones ANA/SNIRH")

        dfs = []
        for file_path in all_files:
            # Extraer código de estación del nombre de archivo
            # Formato típico: "Estación-Nombre - CODIGO.csv"
            parts = file_path.stem.split(" - ")
            if len(parts) > 1:
                station_code = parts[-1]
            else:
                station_code = file_path.stem.replace("Estación-", "")

            try:
                df = self.read_station_file(station_code, file_path)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error leyendo ANA {station_code}: {e}")

        if not dfs:
            raise ValueError("No se pudieron leer estaciones ANA")

        combined = pd.concat(dfs, ignore_index=True)

        # Filtrar por fechas
        if start_date:
            combined = combined[combined["date"] >= pd.Timestamp(start_date)]
        if end_date:
            combined = combined[combined["date"] <= pd.Timestamp(end_date)]

        logger.info(f"Datos ANA cargados: {len(combined)} registros")
        return combined

    def get_station_metadata(self) -> pd.DataFrame:
        """
        Extrae metadata de estaciones (nombre, código, coordenadas si están en archivos).

        Returns:
            DataFrame con: station_id, name, lat, lon, elev_m, provider
        """
        # Placeholder: en producción, leer de archivo metadata o scraping
        stations = []
        for file_path in self.data_dir.glob("Estación-*.csv"):
            parts = file_path.stem.split(" - ")
            name = parts[0].replace("Estación-", "")
            code = parts[-1] if len(parts) > 1 else name

            stations.append({
                "station_id": code,
                "name": name,
                "lat": None,  # TODO: extraer de metadata
                "lon": None,
                "elev_m": None,
                "provider": "ANA",
            })

        return pd.DataFrame(stations)
