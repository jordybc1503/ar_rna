.PHONY: help install data preprocess features train evaluate clean test format lint docker-build docker-run

help:
	@echo "Comandos disponibles:"
	@echo "  make install         - Instalar dependencias (pip o poetry)"
	@echo "  make data            - Descargar datos de SENAMHI/ANA"
	@echo "  make preprocess      - Preprocesar y limpiar datos"
	@echo "  make features        - Construir features (lags, vecindad)"
	@echo "  make train           - Entrenar modelos (AR, MLP, Hybrid)"
	@echo "  make evaluate        - Evaluar modelos y generar reportes"
	@echo "  make test            - Ejecutar tests con pytest"
	@echo "  make format          - Formatear código con black"
	@echo "  make lint            - Linting con flake8"
	@echo "  make clean           - Limpiar archivos temporales"
	@echo "  make docker-build    - Construir imagen Docker"
	@echo "  make docker-run      - Ejecutar contenedor Docker"

install:
	pip install -r requirements.txt
	# O con Poetry: poetry install

data:
	python -m src.main fetch --source senamhi --bbox "-75.8,-13.7,-73.6,-10.8" --start 2000-01-01 --end 2023-12-31
	python -m src.main fetch --source ana --stations configs/stations.csv

preprocess:
	python -m src.main preprocess --config configs/data.yaml

features:
	python -m src.main featurize --lags 1 3 7 14 30 --neighbors 5

train:
	python -m src.main train --model ar --config configs/model/ar.yaml
	python -m src.main train --model mlp --config configs/model/mlp.yaml
	python -m src.main train --model hybrid_ar_mlp --config configs/model/hybrid_ar_mlp.yaml

evaluate:
	python -m src.main evaluate --suite core --out experiments/results/

test:
	pytest tests/ -v --cov=src/mantaro_gf --cov-report=html

format:
	black src/ tests/

lint:
	flake8 src/ tests/ --max-line-length=100

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name "*.egg-info" -delete
	rm -rf htmlcov/ .coverage

docker-build:
	docker build -t mantaro-gapfilling:latest -f docker/Dockerfile .

docker-run:
	docker run -it --rm -v $(PWD)/data:/app/data -v $(PWD)/experiments:/app/experiments mantaro-gapfilling:latest
