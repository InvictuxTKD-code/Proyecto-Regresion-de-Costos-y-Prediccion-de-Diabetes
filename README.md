# 🧠 Proyecto: Predicción de Diabetes y Costos de Seguro Médico con IA

### 👨‍💻 Autor: **Andrés Jaramillo**  
📅 Octubre 2025  
🚀 Desplegado en Render: [🌐 Ver Aplicación en Producción](https://proyecto-regresion-de-costos-y-6m7d.onrender.com)

---

## 🧩 Descripción General

Este proyecto integra **Inteligencia Artificial (Machine Learning)**, **Streamlit** y **PostgreSQL** para desarrollar una aplicación web que permite:

1. 🏥 **Predecir la probabilidad de diabetes** a partir de datos biomédicos.  
2. 💰 **Estimar el costo de un seguro médico** considerando características demográficas y de salud.

La aplicación fue diseñada, entrenada y desplegada completamente en la nube con **Render**, incluyendo la conexión a una base de datos **PostgreSQL** para registrar y consultar el historial de predicciones de ambos modelos.

---

## ⚙️ Tecnologías Utilizadas

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?logo=streamlit)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue?logo=postgresql)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?logo=scikit-learn)
![Render](https://img.shields.io/badge/Render-Deployment-black?logo=render)
![Poetry](https://img.shields.io/badge/Poetry-Dependency_Manager-purple?logo=poetry)

---

## 🧠 Arquitectura del Proyecto
```
📁 Proyecto-Regresion-de-Costos-y-Prediccion-de-Diabetes
│
├── 📂 models # Modelos entrenados (joblib)
├── 📂 data # Datasets originales
├── 📂 predicciones # Resultados generados
├── app.py # Aplicación principal (Streamlit)
├── data_pipeline.py # Pipeline de procesamiento
├── train_diabetes.py # Entrenamiento del modelo de diabetes
├── train_regression.py # Entrenamiento del modelo de seguros
├── pyproject.toml # Configuración Poetry
├── requirements.txt # Dependencias del entorno
└── runtime.txt # Versión de Python para Render
```
---

## ⚙️ Instrucciones de Ejecución Local

### 🔹 1. Clonar el repositorio
```bash
git clone https://github.com/InvictuxTKD-code/Proyecto-Regresion-de-Costos-y-Prediccion-de-Diabetes
cd Proyecto-Regresion-de-Costos-y-Prediccion-de-Diabetes
```

---

### 🔹 2. Crear entorno virtual e instalar dependencias
```
poetry install
```
---

### 🔹 3. Configurar variables de entorno
Crear un archivo ```.env``` con la siguiente línea:
```DATABASE_URL=postgresql://usuario:contraseña@host:puerto/nombre_base```

---

### 🔹 4. Ejecutar la aplicación
```streamlit run app.py```

---

### 🔹 5. Abrir en navegador
```http://localhost:8501```

---

## ☁️ Despliegue en Render
### Start Command
```
poetry run streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### Variables de Entorno
```
DATABASE_URL = <cadena de conexión PostgreSQL>
PYTHON_VERSION = 3.11.9
```
---
## 📊 Análisis de Modelos y Respuestas del Informe
### 🧩 1️⃣ ¿Cuál es el umbral ideal para el modelo de predicción de diabetes?
El modelo de Regresión Logística utiliza un umbral óptimo de 0.287, determinado mediante análisis ROC y F1-score.
Este valor equilibra precisión y sensibilidad, reduciendo falsos negativos sin aumentar falsos positivos.

✅ Conclusión: El umbral de 0.287 maximiza la detección de casos potenciales de diabetes.

---

### 💰 2️⃣ Factores que más influyen en el precio del seguro médico
Variable	Influencia	Descripción
smoker_yes	🔥 Muy alta	Fumar eleva drásticamente el costo
bmi	Alta	Refleja el riesgo por sobrepeso
age	Alta	Los costos suben con la edad
children	Media	Incrementa gastos asociados
region / sex	Baja	Afectan marginalmente

✅ Conclusión: Tabaquismo, IMC y edad son los factores determinantes del costo del seguro.

---

### 🌲 3️⃣ Comparativa de características usando RandomForest
```
🔹 Modelo de Diabetes

Variables más relevantes: Glucosa, Edad, IMC.

Menor influencia: Espesor de piel, Presión arterial.

🔹 Modelo de Seguro Médico

Variables más relevantes: Fumar, Edad, BMI.

Menor influencia: Región, Sexo.
```
✅ Conclusión: En ambos casos, los factores fisiológicos y de riesgo predominan sobre los demográficos.

---

### ⚙️ 4️⃣ Técnicas de optimización aplicadas
```
Técnica	Propósito	Resultado
StandardScaler	Escalar variables	Aumenta estabilidad del modelo
GridSearchCV	Ajuste de hiperparámetros	Mejora F1 y R²
Ajuste de Umbral	Balancear precisión y recall	Reduce sesgo de clase
```
✅ Conclusión: La combinación de escalado + búsqueda de hiperparámetros + ajuste de umbral optimizó ambos modelos.

---

### 📚 5️⃣ Contexto de los datos

Diabetes: Conjunto Pima Indians Diabetes Database (EE.UU.), mujeres indígenas Pima.
Variables clínicas: glucosa, IMC, insulina, edad, etc.

Seguro Médico: Dataset público de Kaggle con datos de edad, IMC, hijos, región y tabaquismo.

✅ Ambos datasets fueron preprocesados, limpios y estandarizados antes del entrenamiento.

---

⚖️ 6️⃣ Análisis de sesgo de los modelos
🩺 Modelo de Diabetes

Sesgo leve hacia “no diabético” por desbalance de clases.

Mitigado ajustando el umbral y aplicando balanceo parcial.

💵 Modelo de Seguro Médico

Tiende a sobreestimar costos para fumadores.

Ajuste mediante normalización de valores extremos.

✅ Conclusión: Ambos modelos reflejan sesgos inherentes al dataset, pero se mitigaron mediante normalización y ajuste de umbrales.

---

🧠 Conclusiones Finales

Se integraron con éxito modelos de IA dentro de una aplicación web interactiva.

Se logró una conexión estable a PostgreSQL en la nube y almacenamiento persistente.

El flujo completo de entrenamiento, evaluación y despliegue sigue principios de MLOps.

El proyecto demuestra un nivel técnico profesional, combinando análisis de datos, desarrollo backend, interfaz web y despliegue productivo.

---

📎 Licencia

Este proyecto fue desarrollado con fines educativos y de demostración de un flujo completo de IA aplicada.
© 2025 Andrés Jaramillo. Todos los derechos reservados.


