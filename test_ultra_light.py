#!/usr/bin/env python3
"""
TEST ULTRA-LIGHT - Innovation Client C (Mode économie tokens)
==========================================

Version optimisée pour éviter les erreurs 413 avec tokens minimaux.
"""

from agent.agent import run_agent_with_behavior_ultra_light

def test_ultra_light_innovation():
    """Test ultra-light de l'innovation Client C"""

    print("="*60)
    print("🧠 TEST ULTRA-LIGHT - CHAÎNES DE MARKOV CLIENT C")
    print("💡 Mode économie tokens (pas d'historique, prompt minimal)")
    print("="*60)

    # Test 1: Analyse basique (court)
    print("\n🔬 Test 1: Analyse comportementale (court)")
    prompt1 = "Comportements Client C ?"
    result1 = run_agent_with_behavior_ultra_light(prompt1)
    print(f"🤖 Réponse: {result1['response'][:200]}...")
    print(f"🛠️ Modèle: {result1.get('active_model', 'N/A')}")
    print(f"🎯 Innovation: {result1.get('innovation_active', False)}")
    print(f"⚡ Ultra-light: {result1.get('ultra_light_mode', False)}")

    # Test 2: Alerte spécifique
    print("\n🔬 Test 2: Anomalie lavage mains")
    prompt2 = "Anomalie lavage mains Client C ?"
    result2 = run_agent_with_behavior_ultra_light(prompt2)
    print(f"🤖 Réponse: {result2['response'][:200]}...")
    print(f"🛠️ Modèle: {result2.get('active_model', 'N/A')}")

    print("\n" + "="*60)
    print("✅ TEST ULTRA-LIGHT TERMINÉ")
    if result1.get('innovation_active') and result2.get('innovation_active'):
        print("🎉 INNOVATION CLIENT C OPÉRATIONNELLE !")
        print("🚀 Prêt pour hackathon !")
    else:
        print("⚠️ Innovation nécessite optimisation supplémentaire")
    print("="*60)

if __name__ == "__main__":
    test_ultra_light_innovation()