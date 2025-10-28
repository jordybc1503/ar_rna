"""Grafo de estaciones para vecindad e influencia espacial."""

import pandas as pd
import numpy as np
from typing import List, Dict, Literal, Optional
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


class StationGraph:
    """
    Construye grafo de estaciones para identificar vecinos influyentes.

    Métodos:
    - distance: Por distancia geográfica (haversine)
    - correlation: Por correlación de series temporales
    - elevation: Por proximidad en elevación
    - hybrid: Combinación de múltiples criterios
    """

    def __init__(
        self,
        method: Literal["distance", "correlation", "elevation", "hybrid"] = "distance",
        n_neighbors: int = 5,
    ):
        """
        Args:
            method: Método para determinar vecindad
            n_neighbors: Número de vecinos más cercanos/influyentes
        """
        self.method = method
        self.n_neighbors = n_neighbors
        self.neighbors: Dict[str, List[str]] = {}

    def build(
        self,
        stations_metadata: pd.DataFrame,
        precip_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, List[str]]:
        """
        Construye grafo de vecindad.

        Args:
            stations_metadata: DataFrame con station_id, lat, lon, elev_m
            precip_data: DataFrame con station_id, date, precip_mm (requerido para correlation)

        Returns:
            Dict {station_id: [neighbor_ids]}
        """
        logger.info(f"Construyendo grafo de estaciones con método {self.method}")

        if self.method == "distance":
            self.neighbors = self._build_distance_graph(stations_metadata)

        elif self.method == "correlation":
            if precip_data is None:
                raise ValueError("precip_data requerido para método correlation")
            self.neighbors = self._build_correlation_graph(stations_metadata, precip_data)

        elif self.method == "elevation":
            self.neighbors = self._build_elevation_graph(stations_metadata)

        elif self.method == "hybrid":
            if precip_data is None:
                raise ValueError("precip_data requerido para método hybrid")
            self.neighbors = self._build_hybrid_graph(stations_metadata, precip_data)

        else:
            raise ValueError(f"Método desconocido: {self.method}")

        logger.info(f"Grafo construido: {len(self.neighbors)} estaciones")
        return self.neighbors

    def _build_distance_graph(self, stations: pd.DataFrame) -> Dict[str, List[str]]:
        """Construye grafo por distancia geográfica (Haversine)."""
        # Extraer coordenadas
        coords = stations[["lat", "lon"]].values
        station_ids = stations["station_id"].values

        # Matriz de distancias (Haversine)
        distances = self._haversine_distance_matrix(coords)

        # Top-k vecinos más cercanos
        neighbors = {}
        for i, station_id in enumerate(station_ids):
            # Ordenar por distancia (excluyendo a sí misma)
            distances_i = distances[i].copy()
            distances_i[i] = np.inf  # Excluir estación misma

            nearest_indices = np.argsort(distances_i)[:self.n_neighbors]
            neighbors[station_id] = station_ids[nearest_indices].tolist()

        return neighbors

    def _build_correlation_graph(
        self,
        stations: pd.DataFrame,
        precip_data: pd.DataFrame,
    ) -> Dict[str, List[str]]:
        """Construye grafo por correlación de series temporales."""
        # Pivot para matriz estación x fecha
        pivot = precip_data.pivot(index="date", columns="station_id", values="precip_mm")

        # Matriz de correlación
        corr_matrix = pivot.corr()

        # Top-k estaciones más correlacionadas
        neighbors = {}
        for station_id in corr_matrix.index:
            # Ordenar por correlación (excluyendo a sí misma)
            corr_values = corr_matrix[station_id].copy()
            corr_values[station_id] = -np.inf

            top_indices = corr_values.nlargest(self.n_neighbors).index.tolist()
            neighbors[station_id] = top_indices

        return neighbors

    def _build_elevation_graph(self, stations: pd.DataFrame) -> Dict[str, List[str]]:
        """Construye grafo por proximidad en elevación."""
        elevations = stations["elev_m"].values
        station_ids = stations["station_id"].values

        # Matriz de diferencias de elevación
        elev_diff_matrix = np.abs(elevations[:, None] - elevations[None, :])

        # Top-k con elevación más similar
        neighbors = {}
        for i, station_id in enumerate(station_ids):
            elev_diffs = elev_diff_matrix[i].copy()
            elev_diffs[i] = np.inf

            nearest_indices = np.argsort(elev_diffs)[:self.n_neighbors]
            neighbors[station_id] = station_ids[nearest_indices].tolist()

        return neighbors

    def _build_hybrid_graph(
        self,
        stations: pd.DataFrame,
        precip_data: pd.DataFrame,
        weights: Dict[str, float] = {"distance": 0.4, "correlation": 0.6},
    ) -> Dict[str, List[str]]:
        """
        Construye grafo híbrido combinando distancia y correlación.

        Args:
            stations: Metadata de estaciones
            precip_data: Datos de precipitación
            weights: Pesos para combinar métricas

        Returns:
            Dict de vecinos
        """
        # Calcular ambos grafos
        distance_graph = self._build_distance_graph(stations)
        corr_graph = self._build_correlation_graph(stations, precip_data)

        # Combinar scores (placeholder: simplificado)
        # En producción: normalizar distancia/correlación y ponderar
        neighbors = {}
        for station_id in stations["station_id"]:
            # Union de vecinos de ambos métodos
            distance_neighbors = set(distance_graph.get(station_id, []))
            corr_neighbors = set(corr_graph.get(station_id, []))

            # Prioritizar correlación > distancia
            combined = list(corr_neighbors) + [n for n in distance_neighbors if n not in corr_neighbors]
            neighbors[station_id] = combined[:self.n_neighbors]

        return neighbors

    def _haversine_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de distancias Haversine (km).

        Args:
            coords: Array Nx2 de [lat, lon] en grados

        Returns:
            Matriz NxN de distancias en km
        """
        # Convertir a radianes
        coords_rad = np.radians(coords)

        # Radio de la Tierra en km
        R = 6371.0

        # Haversine formula (vectorizado)
        lat1 = coords_rad[:, 0:1]
        lon1 = coords_rad[:, 1:2]
        lat2 = coords_rad[:, 0:1].T
        lon2 = coords_rad[:, 1:2].T

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def get_neighbors(self, station_id: str) -> List[str]:
        """Retorna lista de vecinos para una estación."""
        return self.neighbors.get(station_id, [])

    def save(self, path: str) -> None:
        """Guarda grafo a archivo JSON."""
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.neighbors, f, indent=2)

        logger.info(f"Grafo guardado en {path}")

    def load(self, path: str) -> "StationGraph":
        """Carga grafo desde archivo JSON."""
        import json

        with open(path, "r") as f:
            self.neighbors = json.load(f)

        logger.info(f"Grafo cargado desde {path}")
        return self
