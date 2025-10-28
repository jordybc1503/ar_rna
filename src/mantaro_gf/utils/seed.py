"""Utilidades para configurar semillas y reproducibilidad."""

import random
import numpy as np
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Fija semillas para reproducibilidad en random, numpy y sklearn.

    Args:
        seed: Semilla aleatoria
    """
    random.seed(seed)
    np.random.seed(seed)
    # Si usas PyTorch:
    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def get_seed_from_env(default: int = 42) -> int:
    """
    Lee RANDOM_SEED de variables de entorno o usa default.

    Args:
        default: Valor por defecto si no existe variable

    Returns:
        Semilla como entero
    """
    import os
    return int(os.getenv("RANDOM_SEED", default))
