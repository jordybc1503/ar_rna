"""Cliente para descargar datos DEM (Digital Elevation Model) desde Google Earth Engine."""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

logger = logging.getLogger(__name__)


class DEMClient:
    """
    Cliente para obtener datos de elevación (DEM) desde Google Earth Engine.

    Soporta:
    - ALOS PALSAR
    - SRTM
    - Copernicus DEM

    Requiere: earthengine-api instalado y autenticado
    """

    def __init__(
        self,
        output_dir: Path,
        project_id: Optional[str] = None,
        service_account_json: Optional[Path] = None,
    ):
        """
        Args:
            output_dir: Directorio para guardar DEMs descargados
            project_id: ID del proyecto GEE
            service_account_json: Path al JSON de service account
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.project_id = project_id
        self.service_account_json = service_account_json

        # Lazy import de ee (solo si se usa)
        self.ee = None

    def _init_ee(self):
        """Inicializa Earth Engine (lazy loading)."""
        if self.ee is not None:
            return

        try:
            import ee
            self.ee = ee

            if self.service_account_json:
                credentials = ee.ServiceAccountCredentials(
                    email=None,  # Se lee del JSON
                    key_file=str(self.service_account_json)
                )
                self.ee.Initialize(credentials, project=self.project_id)
            else:
                self.ee.Initialize(project=self.project_id)

            logger.info("Google Earth Engine inicializado")
        except ImportError:
            logger.error("earthengine-api no está instalado. Instala con: pip install earthengine-api")
            raise
        except Exception as e:
            logger.error(f"Error inicializando Earth Engine: {e}")
            raise

    def get_elevation_at_points(
        self,
        points: pd.DataFrame,
        dem_source: str = "ALOS",
    ) -> pd.DataFrame:
        """
        Obtiene elevación para una lista de puntos.

        Args:
            points: DataFrame con columnas 'lat', 'lon'
            dem_source: Fuente DEM ("ALOS", "SRTM", "Copernicus")

        Returns:
            DataFrame original con columna adicional 'elev_m'
        """
        self._init_ee()

        # Seleccionar dataset
        if dem_source == "ALOS":
            dem = self.ee.Image("JAXA/ALOS/AW3D30/V2_2").select("AVE_DSM")
        elif dem_source == "SRTM":
            dem = self.ee.Image("USGS/SRTMGL1_003").select("elevation")
        elif dem_source == "Copernicus":
            dem = self.ee.Image("COPERNICUS/DEM/GLO30").select("DEM")
        else:
            raise ValueError(f"DEM source desconocido: {dem_source}")

        logger.info(f"Extrayendo elevación de {dem_source} para {len(points)} puntos")

        elevations = []
        for _, row in points.iterrows():
            point = self.ee.Geometry.Point([row["lon"], row["lat"]])
            elev = dem.reduceRegion(
                reducer=self.ee.Reducer.first(),
                geometry=point,
                scale=30,  # metros
            ).getInfo()

            # Extraer valor
            key = "AVE_DSM" if dem_source == "ALOS" else "elevation" if dem_source == "SRTM" else "DEM"
            elevations.append(elev.get(key, None))

        points["elev_m"] = elevations
        return points

    def download_dem_bbox(
        self,
        bbox: List[float],
        output_file: Path,
        dem_source: str = "ALOS",
        scale: int = 30,
    ) -> Path:
        """
        Descarga DEM para un bounding box.

        Args:
            bbox: [lon_min, lat_min, lon_max, lat_max]
            output_file: Path para guardar GeoTIFF
            dem_source: Fuente DEM
            scale: Resolución en metros

        Returns:
            Path al archivo descargado
        """
        self._init_ee()

        # Seleccionar dataset
        if dem_source == "ALOS":
            dem = self.ee.Image("JAXA/ALOS/AW3D30/V2_2").select("AVE_DSM")
        elif dem_source == "SRTM":
            dem = self.ee.Image("USGS/SRTMGL1_003").select("elevation")
        elif dem_source == "Copernicus":
            dem = self.ee.Image("COPERNICUS/DEM/GLO30").select("DEM")
        else:
            raise ValueError(f"DEM source desconocido: {dem_source}")

        # Definir región
        region = self.ee.Geometry.Rectangle(bbox)

        logger.info(f"Descargando DEM {dem_source} para bbox {bbox}")

        # Exportar (esto inicia un task en GEE; para descarga directa usar geemap)
        task = self.ee.batch.Export.image.toDrive(
            image=dem,
            description=f"DEM_{dem_source}",
            folder="mantaro_gf",
            fileNamePrefix=output_file.stem,
            region=region,
            scale=scale,
            crs="EPSG:4326",
        )
        task.start()

        logger.info(f"Tarea de descarga iniciada: {task.id}")
        logger.warning("Revisar Google Drive para archivo descargado (método asíncrono)")

        return output_file
