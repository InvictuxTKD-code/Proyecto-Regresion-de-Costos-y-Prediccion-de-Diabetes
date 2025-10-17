import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_pipeline import load_insurance

# ==============================
# 1. CARGA Y PREPROCESAMIENTO
# ==============================
print("Cargando datos...")
X, y = load_insurance("data/insurance.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado para regresión lineal
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 2. ENTRENAMIENTO DE MODELOS
# ==============================
print("Entrenando modelos...")
# RidgeCV (regresión lineal regularizada)
lr = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5).fit(X_train_scaled, y_train)

# RandomForest (comparativo)
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(
    X_train, y_train
)

# ==============================
# 3. EVALUACIÓN DE MODELOS
# ==============================

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)   # usar MSE estándar
    rmse = float(np.sqrt(mse))                 # calcular RMSE manualmente
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3),
    }

lr_metrics = metrics(y_test, lr.predict(X_test_scaled))
rf_metrics = metrics(y_test, rf.predict(X_test))

print("\n--- Evaluación Modelos ---")
print(f"RidgeCV -> MAE: {lr_metrics['MAE']}, RMSE: {lr_metrics['RMSE']}, R2: {lr_metrics['R2']}")
print(f"RandomForest -> MAE: {rf_metrics['MAE']}, RMSE: {rf_metrics['RMSE']}, R2: {rf_metrics['R2']}")

# ==============================
# 4. COEFICIENTES Y FACTORES
# ==============================
coef = pd.Series(lr.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print("\nTop 10 coeficientes (Ridge):")
print(coef.head(10))

# ==============================
# 5. GUARDAR RESULTADOS
# ==============================
os.makedirs("models", exist_ok=True)

# Guardar nombres de columnas
feature_columns = X_train.columns.tolist()

# Guardar modelos
joblib.dump({"lr": lr, "scaler": scaler, "columns": feature_columns, "rf": rf}, "models/insurance_models.joblib")

# Guardar métricas en JSON
results = {"RidgeCV": lr_metrics, "RandomForest": rf_metrics}
with open("models/metrics_insurance.json", "w") as f:
    json.dump(results, f, indent=4)

# Guardar coeficientes en CSV
coef.to_csv("models/ridge_coefficients.csv", header=["coeficiente"])

print("\nModelos guardados en: models/insurance_models.joblib")
print("Métricas guardadas en: models/metrics_insurance.json")
print("Coeficientes guardados en: models/ridge_coefficients.csv")

print("\n✅ Entrenamiento completado con éxito.")
