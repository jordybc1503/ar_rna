"""Generación de huecos aleatorios."""

import numpy as np
import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)


class RandomGapGenerator:
    """Genera huecos aleatorios en series temporales."""

    def __init__(self, missing_rates: List[float], n_trials: int = 10, seed: int = 42):
        self.missing_rates = missing_rates
        self.n_trials = n_trials
        self.seed = seed

    def generate_masks(self, series: pd.Series) -> dict:
        """Genera máscaras de huecos aleatorios."""
        np.random.seed(self.seed)
        masks = {}

        for rate in self.missing_rates:
            for trial in range(self.n_trials):
                n_missing = int(len(series) * rate)
                mask = np.zeros(len(series), dtype=bool)
                indices = np.random.choice(len(series), size=n_missing, replace=False)
                mask[indices] = True

                key = f"rate_{rate}_trial_{trial}"
                masks[key] = mask

        return masks
