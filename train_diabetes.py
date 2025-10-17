import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from data_pipeline import load_diabetes

# ==============================
# 1. CARGA Y PREPROCESAMIENTO
# ==============================
print("Cargando datos de diabetes...")
X, y = load_diabetes("data/diabetes.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 2. ENTRENAMIENTO DE MODELOS
# ==============================
print("Entrenando modelos...")

# Regresión logística
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ==============================
# 3. EVALUACIÓN DE MODELOS
# ==============================
def metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred), 3),
        "recall": round(recall_score(y_true, y_pred), 3),
        "f1": round(f1_score(y_true, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 3)
    }

y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

lr_metrics = metrics(y_test, y_pred_lr, y_proba_lr)
rf_metrics = metrics(y_test, y_pred_rf, y_proba_rf)

print("\n--- Evaluación Modelos ---")
print(f"LogisticRegression -> {lr_metrics}")
print(f"RandomForest -> {rf_metrics}")

# ==============================
# 4. UMBRAL ÓPTIMO (para LogisticRegression)
# ==============================
fpr, tpr, thresholds = roc_curve(y_test, y_proba_lr)
J = tpr - fpr
ix = np.argmax(J)
best_threshold = thresholds[ix]

print(f"\n✅ Umbral óptimo según índice de Youden: {best_threshold:.3f}")

# ==============================
# 5. FACTORES IMPORTANTES (RandomForest)
# ==============================
importances = pd.Series(
    rf.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\nTop 10 variables más influyentes:")
print(importances.head(10))

# ==============================
# 6. GUARDAR RESULTADOS
# ==============================
os.makedirs("models", exist_ok=True)

# Guardar modelos y columnas
joblib.dump(
    {"lr": lr, "rf": rf, "scaler": scaler, "columns": X.columns.tolist()},
    "models/diabetes_models.joblib"
)

# Guardar métricas y umbral
results = {
    "LogisticRegression": lr_metrics,
    "RandomForest": rf_metrics,
    "best_threshold": round(float(best_threshold), 3)
}

with open("models/metrics_diabetes.json", "w") as f:
    json.dump(results, f, indent=4)

# Guardar importancias
importances.to_csv("models/diabetes_feature_importances.csv", header=["importance"])

print("\nModelos guardados en: models/diabetes_models.joblib")
print("Métricas guardadas en: models/metrics_diabetes.json")
print("Importancias guardadas en: models/diabetes_feature_importances.csv")

print("\n✅ Entrenamiento de modelo de diabetes completado con éxito.")

