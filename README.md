# Mantaro Gap-Filling: AR, RNA y Hybrid AR+RNA

Proyecto de gap-filling para precipitación diaria en la cuenca del Mantaro (Perú) usando tres enfoques:
- **AR**: Árbol de Regresión (Decision Tree)
- **RNA**: Red Neuronal Artificial (MLP)
- **Híbrido AR+RNA**: AR selecciona features → MLP predice

---

## 🎯 Objetivos

1. Completar series de precipitación diaria con huecos
2. Comparar precisión y eficacia de AR, MLP y modelo híbrido
3. Evaluar robustez con validación cruzada temporal y espacial
4. Análisis de escenarios con huecos aleatorios y secuenciales
5. Métricas: **r (Pearson), MAE, RMSE, BIAS**

---

## 📁 Estructura del Proyecto

```
mantaro-gapfilling/
├── pyproject.toml           # Gestión de dependencias (Poetry)
├── requirements.txt         # Dependencias (pip)
├── Makefile                 # Tareas automatizadas
├── .env.example             # Variables de entorno
├── docker/
│   └── Dockerfile           # Contenedor Docker
├── data/
│   ├── raw/                 # Datos originales (SENAMHI/ANA)
│   ├── interim/             # Post-QC
│   └── processed/           # Features listas para modelado
├── configs/                 # Configuraciones YAML (Hydra)
│   ├── data.yaml
│   ├── model/
│   │   ├── ar.yaml
│   │   ├── mlp.yaml
│   │   └── hybrid_ar_mlp.yaml
│   ├── cv/
│   │   ├── time_block.yaml
│   │   └── loso.yaml
│   └── gaps/
│       ├── random.yaml
│       └── sequential.yaml
├── src/mantaro_gf/
│   ├── io/                  # Descarga de datos (SENAMHI/ANA/DEM)
│   ├── preprocessing/       # QC, normalización, outliers
│   ├── features/            # Lags, rolling, station graph
│   ├── models/              # AR, MLP, Hybrid
│   ├── evaluation/          # Métricas y reportes
│   ├── validation/          # CV, bootstrap
│   ├── scenarios/           # Generación de gaps
│   └── cli.py               # CLI con Typer
├── experiments/
│   ├── mlruns/              # Tracking con MLflow
│   └── results/             # Tablas y gráficos
├── notebooks/
│   └── 1_Preprocessing_&EDA.ipynb
└── tests/                   # Tests con pytest
```

---

## 🚀 Instalación

### Con pip
```bash
pip install -r requirements.txt
```

### Con Poetry
```bash
poetry install
```

### Con Docker
```bash
make docker-build
make docker-run
```

---

## 📊 Pipeline de Trabajo

### 1. Descargar datos
```bash
# SENAMHI
python -m src.main fetch --source senamhi --bbox "-75.8,-13.7,-73.6,-10.8" --start 2000-01-01 --end 2023-12-31

# ANA/SNIRH
python -m src.main fetch --source ana --stations configs/stations.csv
```

### 2. Preprocesar (QC + Normalización)
```bash
python -m src.main preprocess --config configs/data.yaml
```

### 3. Construir Features
```bash
python -m src.main featurize --lags 1 3 7 14 30 --neighbors 5
```

### 4. Entrenar Modelos
```bash
# Árbol de Regresión
python -m src.main train --model ar --config configs/model/ar.yaml

# MLP
python -m src.main train --model mlp --config configs/model/mlp.yaml

# Híbrido AR+MLP
python -m src.main train --model hybrid_ar_mlp --config configs/model/hybrid_ar_mlp.yaml
```

### 5. Evaluar y Comparar
```bash
python -m src.main evaluate --suite core --out experiments/results/
```

---

## 🔬 Validación

### Cross-Validation Temporal
```yaml
# configs/cv/time_block.yaml
cv:
  method: time_series_split
  n_splits: 5
  test_size: 365  # 1 año
```

### Cross-Validation Espacial (LOSO)
```yaml
# configs/cv/loso.yaml
cv:
  method: leave_one_station_out
  min_train_stations: 3
```

### Bootstrap
```python
from mantaro_gf.validation.bootstrap import Bootstrap

bs = Bootstrap(n_iterations=100, confidence=0.95)
ci = bs.estimate_confidence_intervals(y_true, y_pred, metric_fn=calculate_mae)
```

---

## 📈 Escenarios de Gaps

### Aleatorios
```bash
python -m src.main gaps --scenario random --config configs/gaps/random.yaml
```

### Secuenciales
```bash
python -m src.main gaps --scenario sequential --config configs/gaps/sequential.yaml
```

---

## 📏 Métricas de Evaluación

| Métrica | Descripción |
|---------|-------------|
| **r**   | Coeficiente de correlación de Pearson |
| **MAE** | Mean Absolute Error |
| **RMSE**| Root Mean Squared Error |
| **BIAS**| Mean Error (sesgo) |

Además: **tiempo de entrenamiento**, **complejidad del modelo** (hojas del árbol, neuronas del MLP).

---

## 🧪 Testing

```bash
# Ejecutar tests
make test

# Con cobertura
pytest tests/ --cov=src/mantaro_gf --cov-report=html
```

---

## 🐳 Docker

```bash
# Construir imagen
docker build -t mantaro-gapfilling:latest -f docker/Dockerfile .

# Ejecutar
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/experiments:/app/experiments \
  mantaro-gapfilling:latest python -m src.main --help
```

---

## 📚 Referencias Científicas

- **AR (Decision Trees)**: Breiman et al. (1984), Classification and Regression Trees
- **MLP**: Rumelhart et al. (1986), Backpropagation
- **Gap-filling hidrológico**: Teegavarapu & Chandramouli (2005), López et al. (2020)

---

## 🤝 Contribuciones

1. Fork el proyecto
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m "Descripción"`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Pull Request

---

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE)

---

## ✉️ Contacto

**Autor**: Jordy BC
**Email**: jordybc1503@example.com
**GitHub**: [@jordybc1503](https://github.com/jordybc1503)

---

## 🎓 Citación

Si usas este código en investigación, por favor cita:

```bibtex
@software{mantaro_gapfilling_2025,
  author = {BC, Jordy},
  title = {Mantaro Gap-Filling: AR, RNA y Hybrid AR+RNA},
  year = {2025},
  url = {https://github.com/jordybc1503/mantaro-gapfilling}
}
```

---

**Última actualización**: Octubre 2025
