"""Utilidades para ventanas temporales, lags y máscaras."""

import numpy as np
import pandas as pd
from typing import List, Optional


def create_lagged_features(
    series: pd.Series,
    lags: List[int],
    prefix: str = "lag",
) -> pd.DataFrame:
    """
    Crea features desfasadas (lags) a partir de una serie temporal.

    Args:
        series: Serie temporal (pd.Series con DatetimeIndex)
        lags: Lista de desfases (ej: [1, 3, 7, 14, 30])
        prefix: Prefijo para columnas generadas

    Returns:
        DataFrame con columnas lag_1, lag_3, etc.
    """
    df = pd.DataFrame(index=series.index)
    for lag in lags:
        df[f"{prefix}_{lag}"] = series.shift(lag)
    return df


def create_rolling_features(
    series: pd.Series,
    windows: List[int],
    stats: List[str] = ["mean", "max", "std"],
    prefix: str = "roll",
) -> pd.DataFrame:
    """
    Crea features de agregación móvil (rolling window).

    Args:
        series: Serie temporal
        windows: Tamaños de ventana (ej: [7, 14, 30])
        stats: Estadísticas a calcular (mean, max, std, quantile)
        prefix: Prefijo para columnas

    Returns:
        DataFrame con rolling features
    """
    df = pd.DataFrame(index=series.index)
    for window in windows:
        for stat in stats:
            if stat == "quantile":
                # Percentiles específicos
                for q in [0.95, 0.99]:
                    col_name = f"{prefix}_{window}_{stat}{int(q*100)}"
                    df[col_name] = series.rolling(window).quantile(q)
            else:
                col_name = f"{prefix}_{window}_{stat}"
                df[col_name] = series.rolling(window).agg(stat)
    return df


def create_temporal_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Crea features temporales cíclicas (día del año, mes, estación).

    Args:
        dates: Índice de fechas

    Returns:
        DataFrame con features temporales
    """
    df = pd.DataFrame(index=dates)

    # Día del año (cíclico)
    day_of_year = dates.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Mes (cíclico)
    month = dates.month
    df["sin_month"] = np.sin(2 * np.pi * month / 12)
    df["cos_month"] = np.cos(2 * np.pi * month / 12)

    # Estación húmeda/seca (Mantaro: Nov-Mar húmedo)
    df["is_wet_season"] = dates.month.isin([11, 12, 1, 2, 3]).astype(int)

    return df


def create_missing_mask(
    series: pd.Series,
    missing_rate: float = 0.2,
    method: str = "random",
    max_length: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Crea máscara booleana de valores faltantes (para evaluación).

    Args:
        series: Serie temporal original
        missing_rate: Proporción de datos a marcar como faltantes
        method: "random" o "sequential"
        max_length: Para sequential, longitud máxima de rachas
        seed: Semilla aleatoria

    Returns:
        Array booleano (True = missing)
    """
    np.random.seed(seed)
    n = len(series)
    n_missing = int(n * missing_rate)
    mask = np.zeros(n, dtype=bool)

    if method == "random":
        # Aleatorio uniforme
        missing_indices = np.random.choice(n, size=n_missing, replace=False)
        mask[missing_indices] = True

    elif method == "sequential":
        # Rachas consecutivas
        if max_length is None:
            max_length = 7

        remaining = n_missing
        attempts = 0
        max_attempts = n_missing * 10

        while remaining > 0 and attempts < max_attempts:
            # Elegir longitud de racha
            gap_length = np.random.randint(1, min(max_length, remaining) + 1)
            # Elegir posición de inicio
            start_idx = np.random.randint(0, n - gap_length)

            # Verificar que no se solape con gaps existentes
            if not mask[start_idx:start_idx + gap_length].any():
                mask[start_idx:start_idx + gap_length] = True
                remaining -= gap_length

            attempts += 1

    return mask
