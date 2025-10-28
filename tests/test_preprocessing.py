"""Tests básicos para módulo de preprocessing."""

import pytest
import pandas as pd
import numpy as np
from mantaro_gf.preprocessing import QualityControl, Normalizer, OutlierDetector


def test_qc_removes_duplicates():
    """Test que QC remueve duplicados."""
    df = pd.DataFrame({
        "station_id": ["A", "A", "B"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01"]),
        "precip_mm": [10.0, 10.0, 5.0],
        "qc_flag": ["orig", "orig", "orig"],
    })

    qc = QualityControl()
    result = qc.apply(df)

    assert len(result) == 2  # Un duplicado removido


def test_normalizer_zscore():
    """Test normalización z-score."""
    df = pd.DataFrame({
        "station_id": ["A"] * 10,
        "precip_mm": np.random.randn(10) * 10 + 50,
    })

    normalizer = Normalizer(method="zscore", scale_by_station=False)
    normalizer.fit(df)
    result = normalizer.transform(df)

    assert "precip_mm_norm" in result.columns
    assert abs(result["precip_mm_norm"].mean()) < 0.1  # ~0


def test_outlier_detection():
    """Test detección de outliers."""
    df = pd.DataFrame({
        "station_id": ["A"] * 10,
        "precip_mm": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],  # 1000 es outlier
    })

    detector = OutlierDetector(method="zscore", threshold=3.0)
    result = detector.detect(df)

    assert result["is_outlier"].sum() >= 1  # Al menos 1 outlier


if __name__ == "__main__":
    pytest.main([__file__])
