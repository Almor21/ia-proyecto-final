"""
Script para generar archivos CSV reducidos para la implementación en C
Toma el dataset completo, selecciona las variables más importantes y crea
archivos train_reducido.csv y test_reducido.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar dataset completo
print("Cargando dataset completo...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Limpiar nombres de columnas
df.columns = df.columns.str.replace(' ', '_')

# Convertir TotalCharges a numérico
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# Eliminar customerID
df = df.drop(columns=['customerID'])

# Variable objetivo
y = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop(columns=['Churn'])

# Seleccionar solo las variables más importantes (reducir dimensionalidad)
# Usaremos las variables numéricas principales y algunas categóricas clave
variables_seleccionadas = [
    'tenure',
    'MonthlyCharges', 
    'TotalCharges',
    'SeniorCitizen',
    'Contract',
    'InternetService',
    'OnlineSecurity',
    'TechSupport',
    'PaymentMethod'
]

# Filtrar solo las que existen
variables_seleccionadas = [v for v in variables_seleccionadas if v in X.columns]
X_reducido = X[variables_seleccionadas].copy()

# One-hot encoding solo de las categóricas seleccionadas
categoricas = ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport', 'PaymentMethod']
categoricas = [c for c in categoricas if c in X_reducido.columns]

X_reducido = pd.get_dummies(X_reducido, columns=categoricas, drop_first=True)

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reducido)

# Convertir a DataFrame para facilitar exportación
X_scaled_df = pd.DataFrame(X_scaled, columns=X_reducido.columns)

# Dividir en train/test (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.30, random_state=42, stratify=y
)

# Combinar características con etiquetas
train_data = X_train.copy()
train_data['label'] = y_train.values

test_data = X_test.copy()
test_data['label'] = y_test.values

# Guardar como CSV (sin índices, sin headers)
print(f"Guardando train_reducido.csv ({train_data.shape[0]} muestras, {train_data.shape[1]-1} características)...")
train_data.to_csv('train_reducido.csv', index=False, header=False)

print(f"Guardando test_reducido.csv ({test_data.shape[0]} muestras, {test_data.shape[1]-1} características)...")
test_data.to_csv('test_reducido.csv', index=False, header=False)

print("\nArchivos generados exitosamente:")
print(f"  - train_reducido.csv: {train_data.shape[0]} filas")
print(f"  - test_reducido.csv: {test_data.shape[0]} filas")
print(f"  - Características: {train_data.shape[1]-1}")
print(f"\nFormato: {train_data.shape[1]-1} columnas numéricas + 1 columna de etiqueta (0 o 1)")

