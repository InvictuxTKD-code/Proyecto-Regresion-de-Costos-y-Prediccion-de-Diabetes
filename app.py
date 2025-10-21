import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import psycopg2
import os
from sqlalchemy import create_engine

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

# ==============================
# CONEXIÓN A BASE DE DATOS
# ==============================

# Leer la variable de entorno (Render la proveerá automáticamente)
DATABASE_URL = os.getenv("DATABASE_URL")

# Crear motor SQLAlchemy (para consultas con pandas)
engine = create_engine(
    DATABASE_URL,
    connect_args={"sslmode": "require"}  # 🔒 SSL obligatorio para Render PostgreSQL
)

# Función auxiliar para conexión con psycopg2
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# --------------------------
# FUNCIONES CRUD
# --------------------------

# Guardar predicción de diabetes
def guardar_prediccion_diabetes(resultado_dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO historial_diabetes (fecha, resultado)
        VALUES (%s, %s);
    """, (datetime.now(), str(resultado_dict)))
    conn.commit()
    cur.close()
    conn.close()

# Guardar predicción de seguro médico
def guardar_prediccion_seguro(resultado_dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO historial_seguro (fecha, resultado)
        VALUES (%s, %s);
    """, (datetime.now(), str(resultado_dict)))
    conn.commit()
    cur.close()
    conn.close()

# Leer historial desde cualquier tabla
def leer_historial(tabla):
    df = pd.read_sql(f"SELECT fecha, resultado FROM {tabla} ORDER BY fecha DESC;", engine)
    return df

# Limpiar historial
def limpiar_historial(tabla):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {tabla};")
    conn.commit()
    cur.close()
    conn.close()

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
    bmi = st.number_input("Índice de masa corporal (IMC)", min_value=0.0, max_value=100.0, value=28.4)
    diabetes_pedigree = st.number_input("Función de pedigrí de diabetes", min_value=0.0, max_value=5.0, value=0.45)
    age = st.number_input("Edad", min_value=0, max_value=120, value=32)
    model_type = st.selectbox("Modelo a usar", ["Logistic Regression", "Random Forest"])

    if st.button("Predecir Diabetes"):
        df_input = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree,
            "Age": age
        }])
        df_input = df_input.reindex(columns=cols_diabetes)

        if model_type == "Logistic Regression":
            X_scaled = scaler_diabetes.transform(df_input)
            prob = lr_diabetes.predict_proba(X_scaled)[0, 1]
        else:
            prob = rf_diabetes.predict_proba(df_input)[0, 1]

        pred = int(prob >= threshold_diabetes)
        st.write(f"Probabilidad estimada de diabetes: **{prob:.2%}**")

        # Mostrar alerta correctamente
        if pred == 1:
            st.warning("⚠️ Alta probabilidad de tener diabetes")
        else:
            st.success("✅ Baja probabilidad de diabetes")

        resultado_dict = {
            "Embarazos": pregnancies,
            "Glucosa": glucose,
            "Presión Arterial": blood_pressure,
            "Espesor de Piel": skin_thickness,
            "Insulina": insulin,
            "IMC": bmi,
            "Función Pedigrí": diabetes_pedigree,
            "Edad": age,
            "Modelo": model_type,
            "Probabilidad": prob,
            "Predicción": pred
        }
        guardar_prediccion_diabetes(resultado_dict)
        st.success("✅ Resultado guardado en la base de datos.")

# ==============================
# PREDICCIÓN DE SEGURO MÉDICO
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
        df_encoded = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "children": children,
            "sex_female": 1 if sex == "female" else 0,
            "smoker_yes": 1 if smoker == "yes" else 0,
            "region_northwest": 1 if region == "northwest" else 0,
            "region_southeast": 1 if region == "southeast" else 0,
            "region_southwest": 1 if region == "southwest" else 0
        }])
        df_encoded = df_encoded.reindex(columns=columns_insurance, fill_value=0)

        X_scaled = scaler_insurance.transform(df_encoded)
        cost = lr_insurance.predict(X_scaled)[0]
        st.write(f"Costo estimado del seguro médico: **${cost:,.2f}**")

        resultado_dict = {
            "Edad": age,
            "Sexo": sex,
            "IMC": bmi,
            "Hijos": children,
            "Fumador": smoker,
            "Región": region,
            "Costo_Predicho": round(cost, 2)
        }
        guardar_prediccion_seguro(resultado_dict)
        st.success("✅ Resultado guardado en la base de datos.")

# ==============================
# HISTORIAL DE PREDICCIONES
# ==============================
elif option == "📜 Historial de Predicciones":
    st.header("📜 Historial de Predicciones")
    st.write("Consulta aquí los registros históricos generados por los modelos IA.")

    # Historial Diabetes
    df_diabetes = leer_historial("historial_diabetes")
    if not df_diabetes.empty:
        st.subheader("🔹 Historial de Predicciones de Diabetes")
        df_diabetes_display = pd.json_normalize(df_diabetes["resultado"].apply(eval))  # convertir dict a tabla
        df_diabetes_display["Fecha"] = df_diabetes["fecha"]
        st.dataframe(df_diabetes_display)
    else:
        st.info("No hay predicciones de diabetes registradas aún.")

    # Historial Seguro Médico
    df_seguro = leer_historial("historial_seguro")
    if not df_seguro.empty:
        st.subheader("🔹 Historial de Predicciones de Seguro Médico")
        df_seguro_display = pd.json_normalize(df_seguro["resultado"].apply(eval))
        df_seguro_display["Fecha"] = df_seguro["fecha"]
        st.dataframe(df_seguro_display)
    else:
        st.info("No hay predicciones de seguro médico registradas aún.")

    st.divider()

    # Limpieza de historial
    st.subheader("🧹 Administración del Historial")
    limpiar = st.radio(
        "Selecciona qué historial deseas limpiar:",
        ["Ninguno", "Historial de Diabetes", "Historial de Seguro Médico", "Ambos"]
    )
    confirmar = st.checkbox("✅ Confirmo que deseo eliminar el historial seleccionado permanentemente.")

    if st.button("🗑️ Limpiar Historial Seleccionado"):
        if not confirmar:
            st.warning("⚠️ Debes confirmar la eliminación antes de continuar.")
        else:
            if limpiar == "Historial de Diabetes":
                limpiar_historial("historial_diabetes")
                st.success("✅ Historial de Diabetes eliminado.")
            elif limpiar == "Historial de Seguro Médico":
                limpiar_historial("historial_seguro")
                st.success("✅ Historial de Seguro Médico eliminado.")
            elif limpiar == "Ambos":
                limpiar_historial("historial_diabetes")
                limpiar_historial("historial_seguro")
                st.success("✅ Ambos historiales eliminados.")
