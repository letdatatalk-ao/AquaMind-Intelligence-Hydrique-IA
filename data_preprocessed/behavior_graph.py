"""
behavior_graph.py
=================
Phase 3 : Extraction du Graphe Comportemental (Markov Chain multi-niveaux)
Apprend les routines humaines EXCLUSIVEMENT sur le Gold Standard (Customer C)
pour générer la matrice de probabilités comportementales.

Output: data/behavioral_matrix.json
"""

import pandas as pd
import json
import os

def compute_transition_matrix(df, state_col, next_state_col):
    """
    Calcule les probabilités de transition d'un état vers le suivant.
    Retourne un dictionnaire prêt pour le JSON.
    """
    if df.empty:
        return {}
    
    # Compter les transitions (A -> B)
    counts = df.groupby([state_col, next_state_col]).size().unstack(fill_value=0)
    
    # Convertir en probabilités (ligne = 100%)
    probs = counts.div(counts.sum(axis=1), axis=0).round(3)
    
    # Nettoyer l'index et convertir en dict
    # Dropna permet de s'assurer qu'on n'a pas de clés nulles
    return probs.to_dict(orient='index')

def build_behavioral_graph():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) 
    
    input_path = os.path.join(project_root, "data", "master_dataset.parquet")
    output_path = os.path.join(project_root, "data", "behavioral_matrix.json")
    
    print("\n" + "="*50)
    print("  APPRENTISSAGE DU GRAPHE COMPORTEMENTAL")
    print("="*50)
    
    # 1. Chargement et Isolation de Customer C (Gold Standard)
    print("▶ Chargement du Master Dataset...")
    df = pd.read_parquet(input_path)
    
    df_c = df[df['customer'] == 'CustomerC'].copy()
    print(f"▶ Isolation de Customer C : {len(df_c)} événements trouvés.")
    
    if df_c.empty:
        print("❌ ERREUR : Aucune donnée CustomerC trouvée. Impossible de créer le graphe.")
        return

    # 2. Préparation des États (États Actuels et Futurs)
    print("▶ Préparation des séquences de sessions...")
    # On s'assure que tout est bien trié chronologiquement par session
    df_c = df_c.sort_values(['session_id', 'timestamp'])
    
    # Quel est l'événement SUIVANT dans la MÊME session ? (Si rien = 'END')
    df_c['next_sub_cat'] = df_c.groupby('session_id')['sub_category'].shift(-1).fillna('END')
    
    # Pour le niveau 3 (Ordre 2), on crée la routine actuelle (Precedent -> Actuel)
    df_c['prev_sub_cat_clean'] = df_c.groupby('session_id')['sub_category'].shift(1).fillna('START')
    df_c['routine_state'] = df_c['prev_sub_cat_clean'] + ' -> ' + df_c['sub_category']

    # 3. Calcul des 3 Niveaux d'Intelligence
    print("▶ Calcul du Niveau 1 : Graphe de Markov (Ordre 1)...")
    level_1 = compute_transition_matrix(df_c, 'sub_category', 'next_sub_cat')
    
    print("▶ Calcul du Niveau 2 : Graphe Temporel (Conditionnel)...")
    level_2 = {}
    for time_slot, group in df_c.groupby('time_slot', observed=True):
        if len(group) > 50: # On s'assure d'avoir assez de données statistiques
            level_2[str(time_slot)] = compute_transition_matrix(group, 'sub_category', 'next_sub_cat')
            
    print("▶ Calcul du Niveau 3 : Graphe d'Ordre 2 (Routines)...")
    level_3 = compute_transition_matrix(df_c, 'routine_state', 'next_sub_cat')

    # 4. Assemblage du Cerveau
    behavioral_matrix = {
        "metadata": {
            "source": "CustomerC_Gold_Standard",
            "description": "Matrice de probabilités de transition pour l'IA AquaMind",
            "total_events_learned": len(df_c)
        },
        "level_1_base": level_1,
        "level_2_temporal": level_2,
        "level_3_routine": level_3
    }
    
    # 5. Export JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(behavioral_matrix, f, indent=4)
        
    print(f" SUCCESS : Graphe comportemental généré avec succès ! ({output_path})")
    
    # Petit aperçu pour vérification
    print("\n▶ Aperçu des Probabilités (Niveau 1 - Flush) :")
    if 'Flush' in level_1:
        for next_evt, prob in level_1['Flush'].items():
            print(f"   Flush -> {next_evt} : {prob * 100:.1f}%")

    print("\n" + "="*50)

if __name__ == "__main__":
    build_behavioral_graph()