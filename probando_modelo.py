import joblib
import pandas as pd
import os

# ==============================
# 1. CARGAR MODELO
# ==============================
models = joblib.load("models/insurance_models.joblib")
lr_model = models["lr"]
scaler = models["scaler"]
feature_columns = models["columns"]

# ==============================
# 2. FUNCIÓN PARA PREDECIR
# ==============================
def predecir_seguro(age, sex, bmi, children, smoker, region):
    df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region
    }])

    # Aplicar get_dummies igual que en entrenamiento
    df_dummies = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

    # Asegurar que todas las columnas que el modelo espera estén presentes
    for col in feature_columns:
        if col not in df_dummies.columns:
            df_dummies[col] = 0

    df_dummies = df_dummies[feature_columns]

    df_scaled = scaler.transform(df_dummies)
    prediccion = lr_model.predict(df_scaled)
    return prediccion[0]

# ==============================
# 3. FUNCIONES DE VALIDACIÓN
# ==============================
def input_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Por favor, ingresa un número entero válido.")

def input_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Por favor, ingresa un número válido (decimal permitido).")

def input_option(prompt, options):
    options = [opt.lower() for opt in options]
    while True:
        val = input(prompt).lower()
        if val in options:
            return val
        else:
            print(f"Entrada inválida. Debe ser una de: {', '.join(options)}")

# ==============================
# 4. BUCLE INTERACTIVO
# ==============================
print("=== Predicción de Costo de Seguro Médico ===")
pacientes = []

while True:
    print("\nIngrese los datos del paciente:")

    age = input_int("Edad: ")
    sex = input_option("Sexo (male/female): ", ["male", "female"])
    bmi = input_float("BMI: ")
    children = input_int("Número de hijos: ")
    smoker = input_option("Fumador (yes/no): ", ["yes", "no"])
    region = input_option("Región (southwest/southeast/northwest/northeast): ",
                          ["southwest", "southeast", "northwest", "northeast"])

    costo = predecir_seguro(age, sex, bmi, children, smoker, region)
    print(f"Predicción de costo de seguro médico: ${costo:,.2f}")

    pacientes.append({
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "predicted_cost": costo
    })

    continuar = input_option("¿Desea ingresar otro paciente? (yes/no): ", ["yes", "no"])
    if continuar == "no":
        break

# ==============================
# 5. RESUMEN FINAL
# ==============================
print("\n=== Resumen de Predicciones ===")
for i, p in enumerate(pacientes, start=1):
    print(f"{i}. Edad: {p['age']}, Sexo: {p['sex']}, BMI: {p['bmi']}, Hijos: {p['children']}, "
          f"Fumador: {p['smoker']}, Región: {p['region']} -> Predicción: ${p['predicted_cost']:,.2f}")

# ==============================
# 6. GUARDAR RESULTADOS EN CSV
# ==============================
os.makedirs("predicciones", exist_ok=True)
df_resultados = pd.DataFrame(pacientes)
csv_path = "predicciones/predicciones_seguro.csv"
df_resultados.to_csv(csv_path, index=False)
print(f"\n✅ Todas las predicciones han sido guardadas en: {csv_path}")
