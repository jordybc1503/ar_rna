"""CLI principal con Typer para comandos del pipeline."""

import typer
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime

from mantaro_gf.utils.logging import setup_logger
from mantaro_gf.utils.seed import set_seed

app = typer.Typer(help="Mantaro Gap-Filling: AR, RNA y Hybrid AR+RNA")
logger = setup_logger()


@app.command()
def fetch(
    source: str = typer.Option(..., help="Fuente de datos: senamhi, ana"),
    bbox: Optional[str] = typer.Option(None, help="Bounding box: 'lon_min,lat_min,lon_max,lat_max'"),
    start: str = typer.Option("2000-01-01", help="Fecha inicio (YYYY-MM-DD)"),
    end: str = typer.Option("2023-12-31", help="Fecha fin (YYYY-MM-DD)"),
    output_dir: Path = typer.Option("data/raw", help="Directorio de salida"),
):
    """Descarga datos de SENAMHI/ANA."""
    logger.info(f"Descargando datos desde {source}")

    from mantaro_gf.io import SenamhiClient, ANAClient

    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    if source.lower() == "senamhi":
        client = SenamhiClient(output_dir / "SENAMHI")
        df = client.fetch_all_stations(start_date=start_date, end_date=end_date)
    elif source.lower() == "ana":
        client = ANAClient(output_dir / "SNIRH")
        df = client.fetch_all_stations(start_date=start_date, end_date=end_date)
    else:
        raise ValueError(f"Fuente desconocida: {source}")

    # Guardar
    output_file = output_dir / f"{source.lower()}_precip_daily.parquet"
    df.to_parquet(output_file)
    logger.info(f"Datos guardados: {len(df)} registros en {output_file}")


@app.command()
def preprocess(
    config: Path = typer.Option("configs/data.yaml", help="Config YAML"),
    input_dir: Path = typer.Option("data/raw", help="Directorio de entrada"),
    output_dir: Path = typer.Option("data/interim", help="Directorio de salida"),
):
    """Preprocesa datos (QC, normalización, outliers)."""
    logger.info("Iniciando preprocesamiento")

    # TODO: Implementar pipeline completo
    typer.echo(f"Preprocesando con config {config}")


@app.command()
def featurize(
    lags: List[int] = typer.Option([1, 3, 7, 14, 30], help="Lags temporales"),
    neighbors: int = typer.Option(5, help="Número de vecinos"),
    input_dir: Path = typer.Option("data/interim", help="Directorio de entrada"),
    output_dir: Path = typer.Option("data/processed", help="Directorio de salida"),
):
    """Construye features (lags, vecindad)."""
    logger.info(f"Construyendo features con lags {lags}")

    # TODO: Implementar
    typer.echo(f"Features con {neighbors} vecinos")


@app.command()
def train(
    model: str = typer.Option(..., help="Modelo: ar, mlp, hybrid_ar_mlp"),
    config: Path = typer.Option(..., help="Config del modelo (YAML)"),
    cv_config: Optional[Path] = typer.Option(None, help="Config de CV (YAML)"),
    input_dir: Path = typer.Option("data/processed", help="Features procesadas"),
    output_dir: Path = typer.Option("experiments/models", help="Modelos entrenados"),
):
    """Entrena modelo."""
    logger.info(f"Entrenando modelo {model} con config {config}")

    # TODO: Implementar entrenamiento
    typer.echo(f"Modelo {model} entrenado")


@app.command()
def evaluate(
    suite: str = typer.Option("core", help="Suite de evaluación"),
    models_dir: Path = typer.Option("experiments/models", help="Modelos entrenados"),
    output_dir: Path = typer.Option("experiments/results", help="Resultados"),
):
    """Evalúa modelos y genera reportes."""
    logger.info(f"Evaluando suite {suite}")

    # TODO: Implementar evaluación
    typer.echo(f"Resultados en {output_dir}")


@app.command()
def gaps(
    scenario: str = typer.Option("random", help="Escenario: random, sequential"),
    config: Path = typer.Option(..., help="Config del escenario (YAML)"),
    output_dir: Path = typer.Option("data/processed/gaps", help="Máscaras de gaps"),
):
    """Genera escenarios de huecos."""
    logger.info(f"Generando escenario {scenario}")

    # TODO: Implementar
    typer.echo(f"Escenario {scenario} generado")


if __name__ == "__main__":
    app()
