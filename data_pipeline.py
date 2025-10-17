import pandas as pd
from sklearn.model_selection import train_test_split

def load_insurance(path="data/insurance.csv"):
    df = pd.read_csv(path)
    # limpieza básica (ya verificada)
    df = df.dropna()  # si existe NA
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop(columns=["charges"])
    y = df["charges"]
    return X, y

def load_diabetes(path="data/diabetes.csv"):
    df = pd.read_csv(path)
    # asegurarse de que la columna objetivo se llame 'outcome' o 'target'
    if "Outcome" in df.columns:
        df = df.rename(columns={"Outcome":"outcome"})
    # transformaciones mínimas
    df = df.dropna()
    X = df.drop(columns=["outcome"])
    y = df["outcome"]
    return X, y
