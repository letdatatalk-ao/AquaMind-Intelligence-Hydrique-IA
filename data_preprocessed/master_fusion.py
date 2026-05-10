"""
master_fusion.py
================
1. Fuses all cleaned datasets (Gym, A, B, C).
2. Calculates Elite Features (Temporal, Hydraulic, Statistical).
3. Prepares Behavioral Sessions (session_id, transitions).
4. Produces the final 'Master Brain' for the AI Agent.

Outputs:
- data/master_dataset.parquet
- data/master_health.parquet
"""

import pandas as pd
import numpy as np
import os

def calculate_expert_features(df):
    print("  → Calcul des features temporelles (Contextual Layer)...")
    # 1. TEMPOREL
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Time Slots Métier
    bins = [0, 5, 9, 12, 14, 18, 21, 24]
    labels = ['Night', 'Morning_Peak', 'Mid_Day', 'Afternoon', 'Evening_Peak', 'Late_Evening', 'Night_End']
    df['time_slot'] = pd.cut(df['hour'], bins=bins, labels=labels, include_lowest=True, ordered=False)

    print("  → Calcul des signatures hydrauliques (Hydraulic Layer)...")
    # 2. HYDRAULIQUE
    # flow_rate en ml/s
    df['flow_rate'] = df['consumption'] / df['period_s']
    # Intensity Ratio : (Volume / sqrt(Duration)) - La feature gagnante pour Flush vs Sink
    df['intensity_ratio'] = df['consumption'] / np.sqrt(df['period_s'])

    print("  → Calcul des anomalies statistiques (Statistical Layer)...")
    # 3. STATISTIQUE (Z-Score par device)
    df['z_score'] = df.groupby('device_id')['consumption'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    # Anomaly Score simple (0 à 1)
    df['anomaly_score'] = (df['z_score'].abs() / 3).clip(0, 1)

    print("  → Construction des sessions comportementales (Behavioral Layer)...")
    # 4. COMPORTEMENTAL (Préparation du Graphe)
    
    # Pour Customer C, on regroupe par client entier car chaque équipement a son capteur (dans la même maison).
    # Pour les autres, on garde le device_id (qui correspond à un bloc sanitaire).
    df['block_id'] = np.where(df['customer'] == 'CustomerC', df['customer'], df['device_id'])
    
    # Trier par block_id et temps est CRUCIAL
    df = df.sort_values(['block_id', 'timestamp'])
    
    # Temps depuis l'événement précédent dans le MÊME bloc/maison
    df['gap_since_last'] = df.groupby('block_id')['timestamp'].diff().dt.total_seconds()
    
    # --- CORRECTION DES NAN POUR L'AGENT IA ---
    # 1. Remplir gap_since_last par -1 (Marqueur de premier événement de la journée/bloc)
    df['gap_since_last'] = df['gap_since_last'].fillna(-1)
    
    # Création des sessions (Seuil : 120 secondes ou si c'est le tout premier événement "-1")
    new_session_mask = (df['gap_since_last'] > 120) | (df['gap_since_last'] == -1)
    df['session_id'] = new_session_mask.cumsum()

    # Transition Pair (ex: Flush -> Sink)
    # On récupère le tag précédent dans la même session
    df['prev_sub_cat'] = df.groupby('session_id')['sub_category'].shift(1)
    
    # --- CORRECTION DES NAN POUR L'AGENT IA ET PYARROW ---
    # 2. Remplir prev_sub_cat par 'START' et forcer en string pour PyArrow
    df['prev_sub_cat'] = df['prev_sub_cat'].fillna('START').astype(str)
    
    # 3. Remplir cabin par 'N/A', forcer en string, et nettoyer les 'nan' résiduels
    if 'cabin' in df.columns:
        df['cabin'] = df['cabin'].fillna('N/A').astype(str).replace('nan', 'N/A')

    # La fonction lambda devient beaucoup plus simple car il n'y a plus de NaNs
    df['transition_pair'] = df.apply(
        lambda x: f"{x['prev_sub_cat']} -> {x['sub_category']}", 
        axis=1
    )
    
    return df

def run_fusion():
    # Détection automatique du bon chemin vers le dossier data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) 
    data_dir = os.path.join(project_root, "data")
    
    # Si le dossier data n'est pas à la racine, on utilise un chemin relatif simple
    if not os.path.exists(data_dir):
        data_dir = "data"
        
    customers = ["gym", "customerA", "customerB", "customerC"]
    
    all_base = []
    all_errors = []

    print("▶ Chargement des Parquets individuels...")
    for c in customers:
        base_path = os.path.join(data_dir, f"{c}_base.parquet")
        err_path = os.path.join(data_dir, f"{c}_errors.parquet")
        
        if os.path.exists(base_path):
            all_base.append(pd.read_parquet(base_path))
        else:
            print(f"  ⚠️ Fichier non trouvé : {base_path}")
            
        if os.path.exists(err_path):
            all_errors.append(pd.read_parquet(err_path))

    if not all_base:
        print("❌ ERREUR CRITIQUE : Aucun fichier parquet trouvé. Vérifiez le dossier 'data'.")
        return

    # Fusion des données saines
    master_df = pd.concat(all_base, ignore_index=True)
    
    # Application de l'intelligence
    print("▶ Enrichissement du dataset (Intelligence Layer)...")
    master_df = calculate_expert_features(master_df)

    # Fusion des erreurs (Journal de Santé)
    print("▶ Consolidation du journal de santé matériel...")
    if all_errors:
        health_df = pd.concat(all_errors, ignore_index=True)
    else:
        # Création d'un dataframe vide avec la bonne structure si aucun fichier d'erreur n'existe
        print("  ⚠️ Aucun fichier d'erreur trouvé, création d'un journal vide.")
        health_df = pd.DataFrame(columns=['customer', 'device_id', 'timestamp', 'error_code'])

    # Sauvegarde finale
    master_output = os.path.join(data_dir, "master_dataset.parquet")
    health_output = os.path.join(data_dir, "master_health.parquet")
    
    # Création du dossier data s'il n'existe pas
    os.makedirs(data_dir, exist_ok=True)
    
    master_df.to_parquet(master_output, index=False)
    health_df.to_parquet(health_output, index=False)

    print("\n" + "="*50)
    print("  🏆 MASTER FUSION COMPLETE (AGENT READY)")
    print("="*50)
    print(f"  Lignes totales enrichies : {len(master_df):,}")
    print(f"  Sessions identifiées    : {master_df['session_id'].nunique():,}")
    print(f"  Valeurs manquantes      : {master_df[['gap_since_last', 'prev_sub_cat']].isnull().sum().sum()}")
    print(f"  Fichier Master          : {master_output}")
    print("="*50)

if __name__ == "__main__":
    run_fusion()