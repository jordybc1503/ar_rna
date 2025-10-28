"""Generación de huecos secuenciales (rachas consecutivas)."""

import numpy as np
import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)


class SequentialGapGenerator:
    """Genera huecos secuenciales con longitudes máximas."""

    def __init__(
        self,
        max_lengths: List[int],
        n_gaps_per_length: int = 20,
        min_separation: int = 7,
        seed: int = 42,
    ):
        self.max_lengths = max_lengths
        self.n_gaps_per_length = n_gaps_per_length
        self.min_separation = min_separation
        self.seed = seed

    def generate_masks(self, series: pd.Series) -> dict:
        """Genera máscaras de huecos secuenciales."""
        np.random.seed(self.seed)
        masks = {}

        for max_len in self.max_lengths:
            for gap_id in range(self.n_gaps_per_length):
                mask = np.zeros(len(series), dtype=bool)

                # Elegir longitud de racha (1 a max_len)
                gap_length = np.random.randint(1, max_len + 1)

                # Elegir posición de inicio
                max_start = len(series) - gap_length
                if max_start <= 0:
                    continue

                start_idx = np.random.randint(0, max_start)
                mask[start_idx:start_idx + gap_length] = True

                key = f"maxlen_{max_len}_gap_{gap_id}"
                masks[key] = mask

        return masks
