"""
customerC_loader.py
===================
Loads and cleans Customer C dataset (Residential).
Prepares structural columns for the global fusion.
CRITICAL: Preserves granular 'sub_category' (Flush, Sink, Tap).
Extracts hardware errors (NTP Sync Failure / Unix Epoch Reset).

Produces: 
- data/customerC_base.parquet
- data/customerC_errors.parquet
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    # Customer C utilise un point-virgule comme séparateur
    df = pd.read_csv(path, sep=';')
    
    # Renommage immédiat pour correspondre au Schéma Commun
    # ATTENTION : On mappe 'sub_category_name' vers 'sub_category'
    col_mapping = {
        'device': 'device_id',
        'data_consumption': 'consumption',
        'data_time': 'timestamp_raw',
        'data_period': 'period_s',
        'main_category_name': 'category',
        'sub_category_name': 'sub_category' 
    }
    df = df.rename(columns=col_mapping)
    return df

# ─────────────────────────────────────────────
# 2. CLEAN & SPLIT (Extraction des erreurs)
# ─────────────────────────────────────────────
def clean_and_prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    report = {"raw_count": len(df)}

    # 2a. Parser les dates
    df["timestamp"] = pd.to_datetime(df["timestamp_raw"], errors="coerce", utc=True)
    
    # 2b. Ajouter les métadonnées manquantes (Schéma Commun)
    df["customer"] = "CustomerC"
    df["cabin"] = np.nan # N'existe que pour le Gym
    # CustomerC a déjà sa colonne 'tag' et on vient de renommer sa précieuse 'sub_category'

    # 2c. LE SPLIT : Masques de filtrage temporel
    mask_good_dates = (df["timestamp"].dt.year >= 2024) & (df["timestamp"] <= pd.Timestamp("2026-05-09", tz="UTC"))
    
    df_clean = df[mask_good_dates].copy()
    df_error = df[~mask_good_dates].copy()

    # 2d. Nettoyage final des bonnes données (zéro consommation/durée)
    before_zero = len(df_clean)
    df_clean = df_clean[(df_clean["consumption"] > 0) & (df_clean["period_s"] > 0)].copy()
    report["dropped_zero_values"] = before_zero - len(df_clean)

    # 2e. Qualification des erreurs matérielles pour l'Agent IA
    def classify_error(row):
        if pd.isna(row['timestamp']):
            return "Invalid Timestamp"
        elif row['timestamp'].year <= 1970:
            return "Unix Epoch Reset (NTP Failure)"
        elif row['timestamp'].year >= 2036:
            return "NTP Rollover Error"
        return "Unknown Date Error"

    if not df_error.empty:
        df_error["error_type"] = df_error.apply(classify_error, axis=1)
    else:
        df_error["error_type"] = pd.Series(dtype='str')

    # Colonnes finales du Schéma (9 colonnes exactes)
    schema_cols = [
        "customer", "device_id", "cabin", "tag",
        "category", "sub_category",
        "timestamp", "consumption", "period_s"
    ]
    
    df_clean = df_clean[schema_cols]
    df_error = df_error[schema_cols + ["error_type"]]

    report["clean_count"] = len(df_clean)
    report["error_count"] = len(df_error)
    
    return df_clean, df_error, report

# ─────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────
def run(input_path: str, output_dir: str = "data") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    print("▶ Traitement des données Customer C (Résidentiel)...")
    raw = load_raw(input_path)

    print("▶ Nettoyage et Extraction des erreurs (et conservation de la granularité)...")
    clean_df, error_df, report = clean_and_prepare(raw)

    out_clean = os.path.join(output_dir, "customerC_base.parquet")
    out_error = os.path.join(output_dir, "customerC_errors.parquet")
    
    clean_df.to_parquet(out_clean, index=False)
    error_df.to_parquet(out_error, index=False)

    print("\n" + "="*50)
    print("  CUSTOMER C PRE-FUSION REPORT")
    print("="*50)
    print(f"  Lignes brutes          : {report['raw_count']:,}")
    print(f"  Erreurs matérielles    : {report['error_count']:,} (Dates 1970 isolées)")
    print(f"  Valeurs zéros (jetées) : {report['dropped_zero_values']:,}")
    print(f"  ─────────────────────────────────")
    print(f"  Données saines         : {report['clean_count']:,}")
    print(f"  Sous-catégories        : Preservées (Flush, Sink, Tap)")
    print(f"  Fichier propre         : {out_clean}")
    print(f"  Fichier journal santé  : {out_error}")
    print("="*50)

    return clean_df

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) 
    
    input_file_path = os.path.join(project_root, "data", "customerC_consumption.csv")
    output_dir_path = os.path.join(project_root, "data")
    
    run(input_file_path, output_dir=output_dir_path)