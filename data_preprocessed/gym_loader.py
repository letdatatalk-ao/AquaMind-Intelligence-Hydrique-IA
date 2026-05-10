"""
gym_loader.py
=============
Loads and cleans the gym dataset.
Prepares structural columns for the upcoming global fusion.
Extracts hardware errors (1970/2036) for the Agent's health diagnostics.

Produces: 
- data/gym_base.parquet (Clean data, 9 standard columns)
- data/gym_errors.parquet (Hardware errors log)
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# 1. DEVICE MAP  (Validé par l'analyse hivernale et la durée médiane)
# ─────────────────────────────────────────────
DEVICE_MAP = {
    "8161ea40-4a9c-11ef-82d3-2ffa8384e699": {"cabin": 1, "tag": "hot"},
    "8165a900-4a9c-11ef-8393-69e55944ae1e": {"cabin": 1, "tag": "cold"},
    
    # Cabin 2 (Validation absolue)
    "8164cb10-4a9c-11ef-b24b-055224f3eeba": {"cabin": 2, "tag": "cold"},
    "816932d0-4a9c-11ef-b3c5-85b5ce342d21": {"cabin": 2, "tag": "hot"},
    
    # Cabin 3 (Validation absolue)
    "81643740-4a9c-11ef-a595-f9c8dc6ae4ad": {"cabin": 3, "tag": "cold"},
    "81689b30-4a9c-11ef-ad50-1d2daba32da0": {"cabin": 3, "tag": "hot"},
    
    # Cabin 4 (Corrigé grâce au test de volume hivernal)
    "8170c860-4a9c-11ef-a57d-1978fe7d9161": {"cabin": 4, "tag": "hot"}, 
    "81653450-4a9c-11ef-a236-3bbfdd38639d": {"cabin": 4, "tag": "cold"},
}

# ─────────────────────────────────────────────
# 2. LOAD
# ─────────────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Renommage immédiat pour correspondre au Schéma Commun
    df.columns = ["consumption", "timestamp_raw", "period_s", "device_id"]
    return df

# ─────────────────────────────────────────────
# 3. CLEAN & SPLIT (Extraction des erreurs)
# ─────────────────────────────────────────────
def clean_and_prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    report = {"raw_count": len(df)}

    # 3a. Parser les dates
    df["timestamp"] = pd.to_datetime(df["timestamp_raw"], errors="coerce", utc=True)
    
    # 3b. Ajouter les métadonnées (Schéma Commun) avant le split pour garder le contexte des erreurs
    df["cabin"] = df["device_id"].map(lambda d: DEVICE_MAP.get(d, {}).get("cabin", np.nan))
    df["tag"] = df["device_id"].map(lambda d: DEVICE_MAP.get(d, {}).get("tag", "unknown"))
    df["customer"]      = "Gym"
    df["category"]      = "Shower"
    df["sub_category"]  = df["tag"].str.capitalize() + " shower"

    # 3c. LE SPLIT : Séparer les bonnes données des erreurs matérielles IoT
    mask_good_dates = (df["timestamp"].dt.year >= 2024) & (df["timestamp"] <= pd.Timestamp("2026-05-09", tz="UTC"))
    
    df_clean = df[mask_good_dates].copy()
    df_error = df[~mask_good_dates].copy()

    # 3d. Nettoyage final des bonnes données (zéro consommation)
    before_zero = len(df_clean)
    df_clean = df_clean[(df_clean["consumption"] > 0) & (df_clean["period_s"] > 0)].copy()
    report["dropped_zero_values"] = before_zero - len(df_clean)

    # 3e. Qualification des erreurs (Pour l'Agent IA)
    # Si l'année est <= 1970 -> Unix Epoch Reset. Sinon -> NTP Rollover (2036)
    df_error["error_type"] = np.where(df_error["timestamp"].dt.year <= 1970, 
                                      "Unix Epoch Reset", 
                                      "NTP Rollover Error")

    # Colonnes finales du Schéma (9 colonnes exactes)
    schema_cols = [
        "customer", "device_id", "cabin", "tag",
        "category", "sub_category",
        "timestamp", "consumption", "period_s"
    ]
    
    df_clean = df_clean[schema_cols]
    
    # Pour le fichier d'erreurs, on garde les mêmes + le type d'erreur
    df_error = df_error[schema_cols + ["error_type"]]

    report["clean_count"] = len(df_clean)
    report["error_count"] = len(df_error)
    
    return df_clean, df_error, report

# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────
def run(input_path: str, output_dir: str = "data") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    print("▶ Traitement des données GYM...")
    raw = load_raw(input_path)

    print("▶ Nettoyage et Extraction des erreurs matérielles...")
    clean_df, error_df, report = clean_and_prepare(raw)

    # Sauvegarde des deux fichiers
    out_clean = os.path.join(output_dir, "gym_base.parquet")
    out_error = os.path.join(output_dir, "gym_errors.parquet")
    
    clean_df.to_parquet(out_clean, index=False)
    error_df.to_parquet(out_error, index=False)

    # ── Rapport d'exécution ──
    print("\n" + "="*50)
    print("  GYM PRE-FUSION REPORT")
    print("="*50)
    print(f"  Lignes brutes          : {report['raw_count']:,}")
    print(f"  Erreurs matérielles    : {report['error_count']:,} (1970 et 2036 isolées)")
    print(f"  Valeurs zéros (jetées) : {report['dropped_zero_values']:,}")
    print(f"  ─────────────────────────────────")
    print(f"  Données saines         : {report['clean_count']:,}")
    print(f"  Fichier propre         : {out_clean}")
    print(f"  Fichier journal santé  : {out_error}")
    print("="*50)

    return clean_df

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Dossier 'data_preprocessed'
    project_root = os.path.dirname(script_dir)              # Remonte au dossier 'aquamind'
    
    input_file_path = os.path.join(project_root, "data", "gym_consumption_data.csv")
    output_dir_path = os.path.join(project_root, "data")
    
    print(f"Tentative de lecture du fichier à : {input_file_path}")
    
    df = run(input_file_path, output_dir=output_dir_path)