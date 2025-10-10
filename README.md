# AI Insurance Models: Regresión de Costos y Predicción de Diabetes

Este repositorio contiene un proyecto completo para: (1) predecir **costos de seguro médico** mediante regresión lineal y (2) predecir **diabetes** (clasificación). Además incluye comparación con RandomForest, análisis de umbral óptimo, técnicas de optimización, análisis de sesgo y una pequeña aplicación web para desplegar los modelos.

---

## Índice

1. Descripción del proyecto
2. Contexto de los datos
3. Estructura del repositorio
4. Requisitos e instalación
5. Cómo ejecutar (entorno local)
6. Respuestas a las preguntas exigidas
7. Detalles técnicos: metodología, evaluación y optimización
8. Consideraciones de sesgo y mitigación
9. Deploy en la nube (opciones)
10. Créditos y referencias

---

## 1. Descripción del proyecto

Se desarrollan dos modelos principales:

* **Regresión lineal** para predecir el *coste del seguro médico* (variable continua: `charges`).
* **Clasificador** para predecir *diabetes* (variable binaria: `1` = diabetes, `0` = no diabetes).

Además se realizará:

* Comparación con **RandomForest** (importancias de características, métricas) para ambos problemas.
* Selección del umbral óptimo para el clasificador de diabetes.
* Análisis de sesgo.
* Interfaz web (Streamlit + Flask minimal) para subir datos y obtener predicciones.

---

## 2. Contexto de los datos

Se recomienda partir con dos datasets públicos para desarrollar y probar el pipeline:

* **Insurance costs**: `insurance.csv` (columnas típicas: `age, sex, bmi, children, smoker, region, charges`). Este dataset es muy usado para modelar costos de seguros y explica la influencia de edad, IMC (bmi), tabaquismo, etc.
* **Diabetes**: `pima-indians-diabetes.csv` (o dataset de Pima Indians - columnas: `pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age, outcome`).

> Si ya tienes tus propios datos, sustituye los archivos `data/insurance.csv` y `data/diabetes.csv` por los tuyos. Las instrucciones en `notebooks/data_prep.ipynb` muestran cómo adaptar nombres de columnas.

---

## 3. Estructura del repositorio

```
AI-Insurance-Models/
├─ data/
│  ├─ insurance.csv
│  └─ diabetes.csv
├─ notebooks/
│  ├─ 01_data_exploration.ipynb
│  ├─ 02_modeling_regression.ipynb
│  └─ 03_modeling_classification.ipynb
├─ src/
│  ├─ data_pipeline.py        # carga, limpieza y features
│  ├─ models.py               # clases para entrenar y guardar modelos
│  ├─ train_regression.py
│  ├─ train_classification.py
│  ├─ evaluate.py
│  └─ utils.py
├─ app/
│  ├─ streamlit_app.py
│  └─ flask_app.py
├─ requirements.txt
├─ README.md                  # este archivo
└─ LICENSE
```

---

## 4. Requisitos e instalación

Se recomienda crear un entorno virtual (venv / conda).

```
python -m venv venv
source venv/bin/activate    # unix
venv\Scripts\activate      # windows
pip install -r requirements.txt
```

`requirements.txt` contiene librerías clave: `pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, streamlit, flask, xgboost, shap`.

---

## 5. Cómo ejecutar (entorno local)

### Notebooks

Abrir los notebooks en `notebooks/` con JupyterLab/Notebook.

### Entrenar modelos (línea de comandos)

* Regresión de costos:

```
python src/train_regression.py --data data/insurance.csv --out models/regression.joblib
```

* Clasificación diabetes:

```
python src/train_classification.py --data data/diabetes.csv --out models/classifier.joblib
```

### Aplicación web (Streamlit)

```
streamlit run app/streamlit_app.py
```

La app permite subir CSV con observaciones nuevas y devuelve predicción de `charges` y probabilidad / clase de diabetes (con opción de ajustar umbral).

---

## 6. Respuestas a las preguntas exigidas

### 1) ¿Cuál es el umbral ideal para el modelo de predicción de diabetes?

* **Método recomendado:** seleccionar el umbral que optimiza una métrica de interés. Las opciones comunes:

  * **Youden's J = TPR - FPR** (maximiza la distancia a la diagonal en la curva ROC).
  * **Max F1-score** (balance entre precisión y recall).
  * **Cost-sensitive thresholding** (definir un costo mayor para falsos negativos si la prioridad es no dejar sin diagnosticar a alguien con diabetes).

* **Implementación práctica:** calcular probabilidades de validación, luego barrer umbrales entre 0 y 1 y elegir el que maximiza la métrica elegida (F1 o Youden). En el README hay un snippet para esto (notebooks/03_modeling_classification.ipynb).

* **Recomendación estándar:** si el objetivo es detección médica (evitar falsos negativos), usar un umbral que priorice **recall**, por ejemplo el que maximiza la sensibilidad manteniendo una precisión razonable — se sugiere reportar varios umbrales (por ejemplo: umbral que maximiza F1, umbral que da recall>=0.9).

### 2) ¿Cuáles son los factores que más influyen en el precio de los costos asociados al seguro médico?

Basado en el dataset típico `insurance.csv`, los factores clave suelen ser:

* **Smoker (tabaquismo):** incrementa fuertemente los `charges`.
* **BMI (IMC):** IMC alto se asocia con mayor coste.
* **Age (edad):** mayores edades incrementan siniestros/costes.
* **Sex (en algunos datasets puede tener influencia moderada)**
* **Children, region:** efecto menor, pero puede aparecer.

**Cómo obtenerlo en el proyecto:**

* Regresión lineal: coeficientes estandarizados (Beta) para ver efecto por desviación estándar.
* RandomForest: `feature_importances_` y SHAP values para explicar efectos no lineales.

### 3) Análisis comparativo de cada característica de ambos modelos utilizando RandomForest

* Entrenar un **RandomForestRegressor** (para costos) y un **RandomForestClassifier** (para diabetes).
* Para cada variable calcular:

  * Importancia base por Gini (o reducción de varianza para regresión).
  * Importancia permutacional (más robusta) — `sklearn.inspection.permutation_importance`.
  * Valores SHAP para entender dirección y no-linealidad.

**Comparativa:** crear una tabla con columnas: `feature`, `importance_regression`, `importance_classifier`, `corr_with_target_regression`, `mean_effect_shap_regression`, `mean_effect_shap_classifier`.

### 4) ¿Qué técnica de optimización mejora el rendimiento de ambos modelos?

* **Hyperparameter tuning**: `RandomizedSearchCV` o `GridSearchCV` sobre `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.
* **Optimización Bayesiana** (Optuna / scikit-optimize) suele converger más rápido a buenos hiperparámetros.
* **Feature engineering**: crear interacciones (e.g., `age*bmi`, `smoker*bmi`), transformar variables (log, bins), y selección de features.
* **Ensamblados**: combinar regresor lineal + RandomForest (stacking) para mejorar precisión.
* **Regularización**: para la regresión lineal usar **Ridge/Lasso** y validación por CV.

Recomendación práctica: usar `RandomizedSearchCV` para un primer barrido y luego `Optuna` para pulir resultados.

### 5) Explicar contexto de los datos

* Los datasets propuestos provienen de muestras poblacionales y están pensados para ejemplificar relaciones entre características demográficas y de salud con costos y riesgo de diabetes.
* Limitaciones frecuentes: no representan todas las poblaciones (problemas de representatividad), variables autopreportadas (IMC) o mediciones incompletas (insulin, skinthickness).

### 6) Analizar el sesgo que presentan los modelos y explicar por qué

* **Sesgo por muestreo:** si los datos provienen de una población específica (p. ej. una región o clínica), el modelo no generaliza.
* **Sesgo por variables ausentes:** si faltan variables socioeconómicas o accesibilidad, el modelo puede confundir correlación con causalidad.
* **Desequilibrio en clases:** para diabetes, si `outcome` está desbalanceado, el modelo favorece la clase mayoritaria.
* **Fairness:** características como `age`, `sex`, `race` (si existiera) pueden llevar a decisiones discriminatorias.

**Mitigación:** re-muestreo (SMOTE), reponderación por costo, fairness-aware learning, reporte de métricas por subgrupos.

---

## 7. Detalles técnicos: metodología, código y snippets

Aquí se incluyen fragmentos de ejemplo (los notebooks contienen la versión ejecutable):

### A) Pipeline de datos (`src/data_pipeline.py`) - idea

```python
import pandas as pd
from sklearn.model_selection import train_test_split

def load_insurance(path):
    df = pd.read_csv(path)
    # tratamiento de categóricas
    df = pd.get_dummies(df, drop_first=True)
    return df

# split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### B) Entrenamiento rápido RandomForest para regresión

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestRegressor(random_state=42)
param_dist = {
 'n_estimators': [100,300,500],
 'max_depth': [None, 5,10,20],
 'min_samples_leaf': [1,2,4],
 'max_features': ['auto','sqrt']
}
rs = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
rs.fit(X_train, y_train)
```

### C) Umbral óptimo para clasificación

```python
from sklearn.metrics import precision_recall_curve, roc_curve
probs = clf.predict_proba(X_val)[:,1]
# obtener umbral max F1
from sklearn.metrics import f1_score
best_thr, best_f1 = 0.5, 0
for t in np.linspace(0,1,101):
    preds = (probs>=t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1>best_f1:
        best_f1 = f1; best_thr = t
```

### D) Importancias con SHAP

```python
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample)
```

---

## 8. Consideraciones de sesgo y evaluación por subgrupos

* Siempre evaluar métricas por subgrupos (edad, sexo, región) y reportar diferencia en TPR/FPR.
* Documentar limitaciones en README y proponer pruebas externas (validación en dataset diferente).

---

## 9. Deploy: opciones y pasos rápidos

* **Streamlit + Render:** subir repo, crear `requirements.txt`, y desplegar con Render (o Streamlit Cloud).
* **Flask + Gunicorn + Docker:** si se prefiere contenerizar y desplegar en Heroku/Render/Azure.

Archivo de ejemplo para Streamlit en `app/streamlit_app.py` permite subir CSV y ver predicciones.

---

## 10. Créditos y referencias

* Dataset ejemplo: `insurance.csv` (disponible en repositorios públicos) y `Pima Indians Diabetes`.
* Bibliotecas: `scikit-learn`, `pandas`, `shap`, `streamlit`.

---

## Próximos pasos (sugeridos)

1. Indicar si tienes datos propios: colocarlos en `data/` con los nombres sugeridos.
2. Ejecutar notebooks y revisar resultados iniciales.
3. Generar los scripts `src/*` y la app en `app/` (puedo generarlos para ti).
4. Entrenar modelos finales y subir al repositorio.

---

Si quieres, continúo y genero ahora mismo:

* Notebooks ejecutables y scripts `src/` completos, o
* La app Streamlit lista para desplegar (con modelos de ejemplo entrenados), o
* Un repositorio GitHub inicial con commits y `README.md` (este archivo) listo.

Dime qué prefieres y lo genero en la siguiente iteración.
