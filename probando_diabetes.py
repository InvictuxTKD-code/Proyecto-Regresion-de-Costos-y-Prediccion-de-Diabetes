import joblib
import pandas as pd
import numpy as np

# ==============================
# 1. CARGAR MODELOS
# ==============================
models = joblib.load("models/diabetes_models.joblib")
lr = models["lr"]
rf = models["rf"]
scaler = models["scaler"]
columns = models["columns"]

# ==============================
# 2. FUNCIÓN DE PREDICCIÓN
# ==============================
def predecir_diabetes(
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, diabetes_pedigree, age, model_type="lr",
    threshold=0.287
):
    """
    Predice si una persona tiene o no diabetes (1 = sí, 0 = no).
    """

    # Crear DataFrame con los datos del usuario
    new_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    # Asegurar columnas en el mismo orden
    new_data = new_data.reindex(columns=columns)

    # Seleccionar modelo
    if model_type == "lr":
        X_scaled = scaler.transform(new_data)
        prob = lr.predict_proba(X_scaled)[0, 1]
    else:
        prob = rf.predict_proba(new_data)[0, 1]

    prediccion = int(prob >= threshold)
    return prob, prediccion

# ==============================
# 3. INTERFAZ DE CONSOLA
# ==============================
print("\n=== PREDICCIÓN DE DIABETES ===")

pregnancies = int(input("Número de embarazos (0 si es hombre): "))
glucose = float(input("Nivel de glucosa: "))
blood_pressure = float(input("Presión arterial: "))
skin_thickness = float(input("Espesor de piel: "))
insulin = float(input("Nivel de insulina: "))
bmi = float(input("Índice de masa corporal (BMI): "))
diabetes_pedigree = float(input("Función de pedigrí de diabetes: "))
age = int(input("Edad: "))

# Elegir modelo
modelo = input("Modelo a usar [lr / rf]: ").strip().lower()
if modelo not in ["lr", "rf"]:
    modelo = "lr"

# ==============================
# 4. EJECUTAR PREDICCIÓN
# ==============================
prob, pred = predecir_diabetes(
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, diabetes_pedigree, age, model_type=modelo
)

print("\n--- RESULTADO ---")
print(f"Probabilidad estimada de diabetes: {prob:.2%}")

if pred == 1:
    print("⚠️ Alta probabilidad de tener diabetes.")
else:
    print("✅ Baja probabilidad de diabetes.")

# ==============================
# 5. GUARDAR RESULTADO
# ==============================
import os

os.makedirs("predicciones", exist_ok=True)
file_path = "predicciones/diabetes_predicciones.csv"

resultado = pd.DataFrame([{
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": diabetes_pedigree,
    "Age": age,
    "Modelo": modelo,
    "Probabilidad": prob,
    "Predicción": pred
}])

# Escribir encabezado solo si el archivo no existe
write_header = not os.path.exists(file_path)
resultado.to_csv(file_path, mode="a", index=False, header=write_header)

print(f"\n✅ Resultado guardado en '{file_path}'")