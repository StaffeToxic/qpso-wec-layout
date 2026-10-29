# Copyright (c) 2026 Anthony Oliveira. All rights reserved.
# See LICENSE for terms.

"""
Verifica R² e MAE para as 4 combinações:
  2 modelos (Layout=150 árvores, ML/Discreto=100 árvores) × 2 datasets (Sydney, Perth)
"""
import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

def _make_coord_cols(num_wecs):
    cols = []
    for i in range(1, num_wecs + 1):
        cols.append(f'X{i}')
        cols.append(f'Y{i}')
    return cols

def treinar_e_avaliar(db_path, num_wecs, n_estimators):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM wec_data", conn)
    conn.close()

    coord_cols = _make_coord_cols(num_wecs)
    X = df[coord_cols].values
    y = df['Total_Power'].values

    best_idx = np.argmax(y)
    best_power = y[best_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=n_estimators, n_jobs=1, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return r2, mae, best_power, len(df)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    configs = [
        {"label": "QPSO Layout Standalone (contínuo)", "n_estimators": 150},
        {"label": "QPSO ML Standalone (discreto/SHAP)", "n_estimators": 100},
    ]

    datasets = [
        {"db": os.path.join(script_dir, "WEC_Sydney_49.db"), "name": "Sydney 49"},
        {"db": os.path.join(script_dir, "WEC_Perth_49.db"),  "name": "Perth 49"},
    ]

    print("=" * 90)
    print(f"  {'Modelo':<40} {'Dataset':<12} {'Trees':>6} {'R²':>8} {'MAE (W)':>12} {'Best (W)':>14}")
    print("-" * 90)

    for cfg in configs:
        for ds in datasets:
            if not os.path.exists(ds["db"]):
                print(f"  {cfg['label']:<40} {ds['name']:<12} --- ARQUIVO NÃO ENCONTRADO ---")
                continue
            r2, mae, best_power, n_samples = treinar_e_avaliar(
                ds["db"], 49, cfg["n_estimators"]
            )
            print(f"  {cfg['label']:<40} {ds['name']:<12} {cfg['n_estimators']:>6} {r2:>8.4f} {mae:>12,.0f} {best_power:>14,.2f}")

    print("=" * 90)
