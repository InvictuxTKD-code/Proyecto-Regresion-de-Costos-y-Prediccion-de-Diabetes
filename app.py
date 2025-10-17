import streamlit as st
import pandas as pd
import joblib
import os

# ==============================
# CONFIGURACIÓN Y CARGA DE MODELOS
# ==============================
st.set_page_config(page_title="Modelos IA - Predicciones", page_icon="🧠", layout="centered")

# Cargar modelos
diabetes_models = joblib.load("models/diabetes_models.joblib")
lr_diabetes = diabetes_models["lr"]
rf_diabetes = diabetes_models["rf"]
scaler_diabetes = diabetes_models["scaler"]
cols_diabetes = diabetes_models["columns"]
threshold_diabetes = 0.287

insurance_models = joblib.load("models/insurance_models.joblib")
lr_insurance = insurance_models["lr"]
scaler_insurance = insurance_models["scaler"]
columns_insurance = insurance_models["columns"]

# Crear carpeta de predicciones si no existe
os.makedirs("predicciones", exist_ok=True)

# ==============================
# MENÚ LATERAL
# ==============================
st.sidebar.title("Menú Principal")
option = st.sidebar.radio(
    "Selecciona una opción:",
    ["🏥 Predicción de Diabetes", "💰 Predicción de Costo de Seguro Médico", "📜 Historial de Predicciones"]
)

# ==============================
# PREDICCIÓN DE DIABETES
# ==============================
if option == "🏥 Predicción de Diabetes":
    st.header("🏥 Predicción de Diabetes")

    pregnancies = st.number_input("Número de embarazos (0 si es hombre)", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Nivel de glucosa", min_value=0.0, max_value=300.0, value=120.0)
    blood_pressure = st.number_input("Presión arterial", min_value=0.0, max_value=200.0, value=70.0)
    skin_thickness = st.number_input("Espesor de piel", min_value=0.0, max_value=100.0, value=25.0)
    insulin = st.number_input("Nivel de insulina", min_value=0.0, max_value=900.0, value=80.0)
    bmi = st.number_input("Índice de masa corporal (BMI)", min_value=0.0, max_value=100.0, value=28.4)
    diabetes_pedigree = st.number_input("Función de pedigrí de diabetes", min_value=0.0, max_value=5.0, value=0.45)
    age = st.number_input("Edad", min_value=0, max_value=120, value=32)
    model_type = st.selectbox("Modelo a usar", ["Logistic Regression", "Random Forest"])

    if st.button("Predecir Diabetes"):
        df = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree,
            "Age": age
        }])
        df = df.reindex(columns=cols_diabetes)

        # Seleccionar modelo
        if model_type == "Logistic Regression":
            X_scaled = scaler_diabetes.transform(df)
            prob = lr_diabetes.predict_proba(X_scaled)[0, 1]
        else:
            prob = rf_diabetes.predict_proba(df)[0, 1]

        pred = int(prob >= threshold_diabetes)
        st.write(f"Probabilidad estimada de diabetes: **{prob:.2%}**")

        if pred == 1:
            st.warning("⚠️ Alta probabilidad de tener diabetes")
        else:
            st.success("✅ Baja probabilidad de diabetes")

        # Guardar predicción
        df["Modelo"] = model_type
        df["Probabilidad"] = prob
        df["Predicción"] = pred
        file_path = "predicciones/diabetes_predicciones.csv"
        df.to_csv(file_path, mode="a", index=False, header=not os.path.exists(file_path))
        st.write(f"✅ Resultado guardado en '{file_path}'")

# ==============================
# PREDICCIÓN DE COSTO DE SEGURO MÉDICO
# ==============================
elif option == "💰 Predicción de Costo de Seguro Médico":
    st.header("💰 Predicción de Costo de Seguro Médico")

    age = st.number_input("Edad", min_value=0, max_value=120, value=35)
    sex = st.selectbox("Sexo", ["male", "female"])
    bmi = st.number_input("Índice de masa corporal (BMI)", min_value=0.0, max_value=100.0, value=28.0)
    children = st.number_input("Número de hijos", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Fumador", ["yes", "no"])
    region = st.selectbox("Región", ["southwest", "southeast", "northwest", "northeast"])

    if st.button("Predecir Costo"):
        df = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "children": children,
            "sex_female": 1 if sex == "female" else 0,
            "smoker_yes": 1 if smoker == "yes" else 0,
            "region_northwest": 1 if region == "northwest" else 0,
            "region_southeast": 1 if region == "southeast" else 0,
            "region_southwest": 1 if region == "southwest" else 0
        }])
        df = df.reindex(columns=columns_insurance, fill_value=0)

        X_scaled = scaler_insurance.transform(df)
        cost = lr_insurance.predict(X_scaled)[0]
        st.write(f"Costo estimado del seguro médico: **${cost:,.2f}**")

        # Guardar predicción
        df["predicted_cost"] = cost
        file_path = "predicciones/predicciones_seguro.csv"
        df.to_csv(file_path, mode="a", index=False, header=not os.path.exists(file_path))
        st.write(f"✅ Resultado guardado en '{file_path}'")

# ==============================
# HISTORIAL DE PREDICCIONES
# ==============================
elif option == "📜 Historial de Predicciones":
    st.header("📜 Historial de Predicciones")
    st.write("Consulta aquí los registros históricos generados por los modelos IA.")

    # --- Mostrar predicciones de Diabetes ---
    diabetes_path = "predicciones/diabetes_predicciones.csv"
    if os.path.exists(diabetes_path):
        st.subheader("🔹 Historial de Predicciones de Diabetes")
        df_diabetes = pd.read_csv(diabetes_path)
        st.dataframe(df_diabetes)
        st.download_button(
            label="⬇️ Descargar historial de Diabetes",
            data=df_diabetes.to_csv(index=False).encode("utf-8"),
            file_name="diabetes_predicciones.csv",
            mime="text/csv"
        )
    else:
        st.info("No hay predicciones de diabetes registradas aún.")

    # --- Mostrar predicciones del Seguro Médico ---
    seguro_path = "predicciones/predicciones_seguro.csv"
    if os.path.exists(seguro_path):
        st.subheader("🔹 Historial de Predicciones de Seguro Médico")
        df_seguro = pd.read_csv(seguro_path)
        st.dataframe(df_seguro)
        st.download_button(
            label="⬇️ Descargar historial de Seguros Médicos",
            data=df_seguro.to_csv(index=False).encode("utf-8"),
            file_name="predicciones_seguro.csv",
            mime="text/csv"
        )
    else:
        st.info("No hay predicciones de seguro médico registradas aún.")


