from agent.tools import analyze_behavior_shift

print("🔬 TEST DIRECT OUTIL - INNOVATION CLIENT C")
result = analyze_behavior_shift('CustomerC')

print(f"Status: {result.get('verdict', 'N/A')}")
print(f"Transitions analysées: {len(result.get('behavioral_analysis', []))}")

print("\n📊 ÉCHANTILLON D'ANALYSE:")
for item in result.get('behavioral_analysis', [])[:3]:
    print(f"  🔄 {item['transition']}: {item['statut']}")
    print(f"     Gold Standard: {item['gold_standard_prob']:.3f}")
    print(f"     Current: {item['current_prob']:.3f}")
    print(f"     Deviation: {item['deviation']:.3f}")

print("\n✅ INNOVATION CHAÎNES DE MARKOV OPÉRATIONNELLE !")