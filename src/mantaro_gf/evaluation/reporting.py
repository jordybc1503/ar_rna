"""Generación de reportes y visualizaciones."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Reporter:
    """Generador de reportes de evaluación."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_comparison_table(
        self,
        results: dict,
        filename: str = "model_comparison.csv",
    ):
        """Genera tabla comparativa de modelos."""
        # Consolidar resultados de múltiples modelos
        rows = []
        for model_name, metrics in results.items():
            metrics["model"] = model_name
            rows.append(metrics)

        df = pd.DataFrame(rows)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Tabla comparativa guardada en {output_path}")
        return df

    def plot_metrics_comparison(
        self,
        results_df: pd.DataFrame,
        filename: str = "metrics_comparison.png",
    ):
        """Genera gráfico de barras comparativo."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        metrics = ["r", "mae", "rmse", "bias"]

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            results_df.plot(x="model", y=metric, kind="bar", ax=ax, legend=False)
            ax.set_title(metric.upper())
            ax.set_ylabel(metric)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Gráfico comparativo guardado en {output_path}")
