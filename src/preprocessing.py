# src/preprocessing.py
"""
Módulo de preprocesamiento de datos para el análisis de exoplanetas Kepler.
Contiene funciones para limpieza, transformación y preparación de datos.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_and_clean_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carga y realiza limpieza inicial del dataset de Kepler.
    
    Args:
        filepath (str): Ruta al archivo CSV del dataset
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) y Target (y)
    """
    # Cargar datos
    df = pd.read_csv(filepath)
    
    # Eliminar columnas de data leakage
    columns_to_drop = ['koi_disposition', 'koi_score_ph', 'koi_score', 
                       'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 
                       'koi_fpflag_ec']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Variable objetivo
    y = df['koi_disposition']
    
    # Features (eliminar ID y target)
    X = df.drop(columns=['koi_disposition', 'koi_id', 'kepid'])
    
    # Manejo de valores nulos
    X = X.fillna(X.median(numeric_only=True))
    
    # Conversión de temperatura: Kelvin a Celsius
    if 'koi_teq' in X.columns:
        X['koi_teq'] = X['koi_teq'] - 273.15
    
    print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Distribución de clases:\n{y.value_counts()}")
    
    return X, y


def remove_outliers_iqr(X: pd.DataFrame, y: pd.Series, threshold: float = 1.5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Elimina outliers usando el método IQR (Interquartile Range).
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        threshold (float): Factor IQR para determinar límites (1.5 es estándar)
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Datos sin outliers
    """
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    
    # Aplicar máscara
    mask = ~((X < (Q1 - threshold * IQR)) | (X > (Q3 + threshold * IQR))).any(axis=1)
    
    X_clean = X[mask]
    y_clean = y[mask]
    
    n_removed = len(X) - len(X_clean)
    print(f"Outliers removidos: {n_removed} ({100*n_removed/len(X):.2f}%)")
    
    return X_clean, y_clean


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Escala las características usando StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Datos de entrenamiento
        X_test (pd.DataFrame): Datos de prueba (opcional)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Datos escalados
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    return X_train_scaled, None


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos
    X, y = load_and_clean_data('data/kepler_dataset.csv')
    
    # Remover outliers
    X_clean, y_clean = remove_outliers_iqr(X, y)
    
    print("\nPreprocesamiento completado!")
    print(f"Forma final: {X_clean.shape}")
