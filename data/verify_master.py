"""
verify_master.py
================
Audit script to verify the integrity and the features 
of the generated master datasets before Phase 3.
"""

import pandas as pd
import os

def verify_datasets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) 
    
    master_path = os.path.join(project_root, "data", "master_dataset.parquet")
    health_path = os.path.join(project_root, "data", "master_health.parquet")

    print("\n" + "="*50)
    print(" 🕵️‍♂️ AUDIT DU MASTER DATASET")
    print("="*50)
    
    # 1. Vérification du Master Dataset
    df = pd.read_parquet(master_path)
    print(f"▶ Lignes totales   : {len(df):,}")
    print(f"▶ Colonnes totales : {len(df.columns)}")
    
    expected_cols = [
        'customer', 'time_slot', 'flow_rate', 'intensity_ratio', 
        'z_score', 'anomaly_score', 'session_id', 'gap_since_last', 'transition_pair'
    ]
    
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"❌ AVERTISSEMENT - Colonnes manquantes : {missing}")
    else:
        print("✅ SUCCESS : Toutes les features Elite sont bien présentes !")

    print("\n▶ Aperçu des statistiques (Intensity Ratio) :")
    print(df[['consumption', 'period_s', 'flow_rate', 'intensity_ratio']].describe().loc[['mean', 'max']].round(2).to_string())

    print("\n▶ Top 5 des Transitions (Pour préparer le Graphe) :")
    # On regarde surtout Customer C car c'est notre référence granulaire
    df_c = df[df['customer'] == 'CustomerC']
    if not df_c.empty:
        print(df_c['transition_pair'].value_counts().head(5).to_string())
    else:
        print("Avertissement : Aucune donnée CustomerC trouvée.")

    print("\n" + "="*50)
    print(" 🏥 AUDIT DU HEALTH JOURNAL")
    print("="*50)
    
    # 2. Vérification du Health Journal
    df_health = pd.read_parquet(health_path)
    print(f"▶ Total des pannes détectées : {len(df_health):,}")
    print("\n▶ Répartition des types d'erreurs :")
    print(df_health['error_type'].value_counts().to_string())
    print("="*50)

if __name__ == "__main__":
    verify_datasets()