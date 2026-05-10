"""
test_prompts.py — Stress Test de l'Agent AquaMind
Permet de tester l'IA sur une série de requêtes métier complexes.
"""

import time
from agent.agent import run_agent

# Liste des prompts de test métier
TEST_PROMPTS = [
    "Generate a table comparing shower cabin 1 and cabin 2 between March 10 at 9:00 and April 11 at 10:00",
    "Plot the daily consumption trend for the osmosis unit over the past 2 weeks",
    "What is the average daily cold water consumption in the gym for the last 30 days?",
    "Is there any unusual behaviour on the residential toilet flush this month?",
    "Which source has the highest hot water consumption across all sites?",
    "Compare bathroom vs kitchen usage for the residential customer in March and visualise it"
]

def run_stress_test():
    print("="*80)
    print("🚀 DÉMARRAGE DU STRESS TEST AQUAMIND")
    print("="*80)

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[TEST {i}/{len(TEST_PROMPTS)}]")
        print(f"👤 PROMPT : {prompt}")
        print("-" * 40)
        
        start_time = time.time()
        
        # Appel de l'Agent IA
        result = run_agent(prompt)
        
        elapsed_time = time.time() - start_time
        
        # Affichage de la réponse
        print(f"🤖 RÉPONSE (en {elapsed_time:.1f}s) :\n{result['response']}")
        
        # Vérification si un graphique a été généré
        if result.get("chart_json"):
            print("\n📊 [SUCCÈS] Un graphique a été généré avec succès par l'Agent !")
        else:
            print("\n📉 [INFO] Aucun graphique généré pour cette requête.")
        
        print("="*80)

if __name__ == "__main__":
    run_stress_test()