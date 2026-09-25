# Kepler Exoplanet Classification & Astrophysical Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)

Pipeline completo de **Data Science y Machine Learning** sobre el conjunto de datos de la misión Kepler de la NASA. Clasificación automática de señales astronómicas en tres categorías: `CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`, integrando análisis astrofísico, ingeniería de características, modelos supervisados, explicabilidad con **SHAP** y balanceo de datos mediante **SMOTE**.

---

## Principales Hallazgos Astrofísicos

- **Mundos en la Zona de Habitabilidad:** Se identificaron **86 exoplanetas y candidatos** que cumplen condiciones teóricas de habitabilidad (temperatura equivalente entre 0°C y 100°C, radio planetario rocoso ≤ 2.0 R⊕).

- **Tratamiento de Valores Atípicos:** La detección por IQR reveló más de 1,500 datos extremos en variables clave como radio planetario (`koi_prad`) y período orbital (`koi_period`), justificando el uso prioritario de algoritmos de árboles de decisión.

- **Reducción de Dimensionalidad (PCA):** La proyección en 2 componentes principales demostró visualmente la alta superposición entre candidatos y confirmados, explicando el límite teórico de precisión del problema.

---

## Pipeline del Proyecto

1. **Análisis Exploratorio y Limpieza de Datos (EDA)**
   - Tratamiento de valores nulos
   - Eliminación de columnas de sesgo de fuga (data leakage)
   - Conversión de unidades físicas (Kelvin a Celsius)

2. **Ingeniería de Características**
   - `volumen_estimado`: Estimación geométrica del volumen planetario
   - `ratio_radio_estrella`: Proporción entre radio del planeta y estrella anfitriona

3. **Preprocesamiento y Escalado**
   - Estandarización de variables con `StandardScaler`
   - Aplicación para algoritmos sensibles (Redes Neuronales y PCA)

4. **Manejo de Desequilibrio de Clases**
   - Aplicación de **SMOTE** para sintetizar ejemplos de la clase minoritaria (`CANDIDATE`)

5. **Explicabilidad de IA (XAI)**
   - Evaluación global mediante Permutation Importance
   - Auditoría de predicciones individuales con Valores SHAP
   - Summary plots y Waterfall plots

---

## Comparativa de Modelos y Resultados

Evaluación realizada sobre el conjunto de prueba independiente (N = 1913):

| Modelo / Estrategia | Accuracy | Precision | Recall | F1-Score | Notas |
|:---|:---:|:---:|:---:|:---:|:---|
| Random Forest (Base) | 73.50% | 0.71 | 0.73 | 0.71 | Alta robustez frente a valores atípicos |
| Gradient Boosting | 72.61% | 0.70 | 0.73 | 0.70 | Ligeramente inferior en clase CANDIDATE |
| RF + GridSearchCV | 73.65% | 0.72 | 0.74 | 0.72 | Optimización: n_estimators=200, min_samples_split=5 |
| Red Neuronal (MLP) | 70.83% | 0.69 | 0.71 | 0.69 | Perceptrón Multicapa (100, 50) sensible a ruido |
| **Soft Voting Classifier** | **73.86%** | **0.72** | **0.74** | **0.72** | **Mejor modelo: Ensamblaje ponderado** |
| Random Forest + SMOTE | 71.88% | 0.73 | 0.72 | 0.72 | Elevó Recall de CANDIDATE del 35% al 50% |

---

## Explicabilidad del Modelo (XAI)

- **Variables Más Críticas (Permutación):** `koi_duration` (duración del tránsito) y `koi_period` (período orbital) tienen mayor impacto en precisión global.

- **Auditoría de Predicción Individual (SHAP):** La descomposición SHAP demostró que radios planetarios desproporcionados (koi_prad > 300 R⊕) reducen drásticamente la probabilidad de confirmación, permitiendo descartar falsos positivos astronómicos de forma interpretable.

---

## Estructura del Repositorio

```
kepler-exoplanet-classification/
├── data/
│   └── kepler_dataset.csv                    # Datos de la misión Kepler (NASA)
├── notebooks/
│   └── kepler_analysis.ipynb                 # Análisis completo (Jupyter)
├── models/
│   └── voting_classifier.pkl                 # Modelo entrenado (Soft Voting)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                      # Limpieza y preparación de datos
│   ├── feature_engineering.py                # Ingeniería de características
│   ├── model_training.py                     # Entrenamiento de modelos
│   └── explainability.py                     # SHAP y análisis de interpretabilidad
├── .gitignore
├── requirements.txt                          # Dependencias del proyecto
├── README.md                                 # Este archivo
└── LICENSE                                   # Licencia del proyecto
```

---

## Instalación y Configuración

### Requisitos Previos
- Python 3.10 o superior
- pip (gestor de paquetes de Python)
- Git

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/MiguelCollado92/analisis_kepler.py.git
cd kepler-exoplanet-classification
```

2. **Crear un entorno virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

---

## Uso Rápido

### Opción 1: Ejecutar el análisis completo en Jupyter

```bash
jupyter notebook notebooks/kepler_analysis.ipynb
```

### Opción 2: Entrenar el modelo desde cero

```python
from src.preprocessing import load_and_clean_data
from src.feature_engineering import engineer_features
from src.model_training import train_voting_classifier
import pickle

# 1. Cargar y limpiar datos
X, y = load_and_clean_data('data/kepler_dataset.csv')

# 2. Ingeniería de características
X_engineered = engineer_features(X)

# 3. Entrenar modelo
model = train_voting_classifier(X_engineered, y)

# 4. Guardar modelo
with open('models/voting_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo entrenado y guardado exitosamente!")
```

### Opción 3: Hacer predicciones con modelo guardado

```python
import pickle
import pandas as pd

# Cargar modelo entrenado
with open('models/voting_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargar nuevos datos
X_new = pd.read_csv('data/kepler_dataset.csv')

# Predicciones
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

print(f"Predicción: {predictions}")
print(f"Confianza: {probabilities.max(axis=1)}")
```

### Opción 4: Explicabilidad SHAP (análisis de importancia)

```python
import shap
import pickle

# Cargar modelo y datos
with open('models/voting_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

X_test = pd.read_csv('data/kepler_dataset.csv').head(100)

# Calcular valores SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualizaciones
shap.summary_plot(shap_values, X_test, show=True)
shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                     base_values=explainer.expected_value, 
                                     data=X_test.iloc[0]))
```

---

## Dependencias

Todas las librerías necesarias están listadas en `requirements.txt`:

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
shap>=0.41.0
imbalanced-learn>=0.9.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
```

Para instalar: `pip install -r requirements.txt`

---

## Resultados Clave

✓ **Mejor accuracy global:** 73.86% (Soft Voting Classifier)  
✓ **Identificadas 86 exoplanetas** en zona de habitabilidad  
✓ **Variables críticas identificadas** mediante SHAP  
✓ **Balanceo de clases mejorado** con SMOTE (Recall CANDIDATE: 35% → 50%)  
✓ **Modelo interpretable y robusto** frente a outliers  
✓ **Pipeline reproducible** con código limpio y documentado  

---

## Próximas Mejoras

- [ ] Implementar Ensemble con XGBoost y LightGBM
- [ ] Optimización bayesiana de hiperparámetros
- [ ] Análisis detallado de curvas ROC por clase
- [ ] Dashboard interactivo con Plotly/Streamlit
- [ ] API REST con FastAPI para predicciones en tiempo real
- [ ] Deployment en Docker y cloud (AWS/GCP)

---

## Métricas de Calidad

- **Test Set Size:** 1,913 muestras independientes
- **Imbalance Ratio:** 1:5 (CANDIDATE:CONFIRMED)
- **SMOTE Ratio:** 1:3 (tras balanceo)
- **Cross-Validation:** 5-fold estratificado
- **Interpretabilidad:** SHAP values para cada predicción

---

## Notas Técnicas

### Data Leakage Prevention
Se eliminaron variables que contenían información del resultado (ej: `koi_disposition`, `koi_score_ph`) para evitar overfitting.

### Manejo de Outliers
Se empleó IQR (Interquartile Range) en lugar de z-score, más robusto para distribuciones sesgadas.

### Escalado Selectivo
Solo se escalaron variables para MLP y PCA; Random Forest y Gradient Boosting son agnósticos al escalado.

---

## Fuentes de Datos

- **Dataset Principal:** [NASA Kepler Exoplanet Archive](https://exoplanetarchive.ipcc.caltech.edu/)
- **Descripción Variables:** Archivo README del dataset oficial
- **Zona Habitable:** Criterios de [Kasting et al. (1993)](https://science.sciencemag.org/content/259/5102/915)

---

## Referencias Bibliográficas

- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281-305.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *arXiv preprint arXiv:1705.07874*.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

---

## Contacto y Redes

**Miguel Collado**  
- LinkedIn: [miguel-collado-moreno-832924b5](https://www.linkedin.com/in/miguel-collado-moreno-832924b5/)
- GitHub: [MiguelCollado92](https://github.com/MiguelCollado92/)
- Email: miguel.colladomoreno@hotmail.com

Proyecto desarrollado como parte del **Máster en Data Science** (2026).

---
