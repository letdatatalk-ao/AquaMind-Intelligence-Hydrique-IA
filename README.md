# AquaMind — Intelligence Hydrique par IA Explicable

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-121212.svg)](https://langchain.com/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-00d684.svg)](https://langchain-ai.github.io/langgraph/)
[![Plotly](https://img.shields.io/badge/Charts-Plotly-3F4F75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](#)

**Depot officiel :** [https://github.com/letdatatalk-ao/-AquaMind-Intelligence-Hydrique-par-IA-Explicable](https://github.com/letdatatalk-ao/-AquaMind-Intelligence-Hydrique-par-IA-Explicable)

---

## Resume Executif

**AquaMind** est une plateforme d'IA decisionnelle concue pour transformer les flux IoT bruts en intelligence operationnelle. En combinant **modelisation probabiliste de Markov** et **agents autonomes ReAct**, AquaMind permet une gestion predictive et comportementale de la ressource en eau.

### Capacites Principales

- **Analyse comportementale :** Detection des derives d'usage et des ruptures de routines via chaines de Markov multi-niveaux.
- **Detection d'anomalies :** Identification automatique des fuites, surconsommations, et pannes capteurs via Z-Score et analyse de debit.
- **Agent conversationnel :** Interface en langage naturel permettant aux exploitants de SUEZ d'interroger le systeme sans connaissance technique.
- **Visualisation interactive :** Graphiques Plotly thematiques pour l'exploration des tendances et des anomalies.
- **Memoire persistante :** Conservation du contexte conversationnel entre sessions utilisateur.

---

## Architecture du Systeme

Le projet repose sur une architecture en **5 couches**, du nettoyage des donnees a l'interface de decision.

| Couche | Description | Fichiers cles |
|--------|-------------|---------------|
| **1. Ingestion** | Nettoyage et normalisation des CSV bruts | gym_loader.py, customerA_loader.py, customerB_loader.py, customerC_loader.py |
| **2. Fusion** | Consolidation et enrichissement (15 features) | master_fusion.py |
| **3. Apprentissage** | Graphe comportemental Markov (Gold Standard CustomerC) | ehavior_graph.py |
| **4. Agent IA** | Orchestrateur LLM ReAct + 10 outils | gent/agent.py, gent/tools.py |
| **5. Interface** | UI Streamlit avec chat, graphiques, memoire | pp.py |

---

## Couches Techniques

### 1. Ingestion & Assainissement (Data Cleaning)

Le pipeline traite les anomalies physiques courantes des capteurs IoT via des loaders dedies :

- **Correction temporelle :** Detection des Unix Epoch Resets (1970) et NTP Rollovers (2036).
- **Integrite des donnees :** Filtrage des 32-bit Overflows (4294967295).
- **Normalisation du schema :** Uniformisation en 9 colonnes.

**Clients traites :**

| Client | Type | Capteurs | Sous-categories |
|--------|------|----------|-----------------|
| **Gym** | Centre sportif | 8 capteurs (4 cabines x Hot/Cold) | Hot shower, Cold shower |
| **CustomerA** | Bureaux | 1 capteur agrege | Flush, Sink, Tap |
| **CustomerB** | Residentiel standard | 1 capteur agrege | Flush, Sink, Tap |
| **CustomerC** | Residentiel granulaire | Capteurs separes | Flush, Sink, Tap |

### 2. Feature Engineering & Hydraulique

**Couche Temporelle :** hour, day_of_week, is_weekend, 	ime_slot (7 periodes)

**Couche Hydraulique :** low_rate, intensity_ratio, gap_since_last

**Couche Statistique :** z_score, nomaly_score

**Couche Comportementale :** session_id, 	ransition_pair, prev_sub_cat

### 3. Graphe Comportemental (Markov)

AquaMind utilise **CustomerC** comme Gold Standard.

| Niveau | Description | Exemple |
|--------|-------------|---------|
| **Ordre 1** | Transition A -> B | P(Flush -> Sink) = 0.524 |
| **Ordre 2** | Conditionnee par time_slot | P(Flush -> Sink \| Morning) = 0.482 |
| **Ordre 3** | Basee sur 2 etats | P("Flush -> Sink" -> Tap) = 0.342 |

### 4. Agent IA (10 Outils)

| # | Outil | Fonction |
|---|-------|----------|
| 1 | query_water_stats | Statistiques de consommation |
| 2 | get_anomaly_report | Anomalies Z-Score et debits |
| 3 | nalyze_behavior_shift | Analyse Markov |
| 4 | generate_consumption_chart | Graphiques Plotly JSON |
| 5 | check_hardware_health | Sante capteurs IoT |
| 6 | get_hydraulic_stats | Durees, debits, intensite |
| 7 | get_trend_analysis | Tendance semaine/mois/trimestre |
| 8 | compare_customers | Comparaison multi-clients |
| 9 | get_session_analysis | Sessions |
| 10 | generate_alert_report | Rapport d'alertes consolide |

---

## Modelisation Mathematique

| Concept | Formule | Utilite |
|---------|---------|---------|
| **Debit** | Q = V / t | Fuites |
| **Intensite** | I = V / sqrt(t) | Signature appareil |
| **Z-Score** | z = (x - mu) / sigma | Anomalies |
| **Markov** | P(A->B) = count(A,B) / count(A) | Transitions |
| **Sessions** | gap > 120s | Sequences |

---

## Installation et Deploiement

git clone https://github.com/letdatatalk-ao/-AquaMind-Intelligence-Hydrique-par-IA-Explicable.git
cd -AquaMind-Intelligence-Hydrique-par-IA-Explicable
python -m venv venv
venv\\Scripts\\activate  # Windows
pip install -r requirements.txt


python data_preprocessed/gym_loader.py
python data_preprocessed/customerA_loader.py
python data_preprocessed/customerB_loader.py
python data_preprocessed/customerC_loader.py
python data_preprocessed/master_fusion.py
python data_preprocessed/behavior_graph.py
streamlit run app.py

## Exemples de Requetes

- **Stats :** "Quelle est la consommation du Gym sur 30 jours ?"
- **Anomalies :** "Y a-t-il des anomalies chez CustomerC ?"
- **Comportement :** "L'hygiene de CustomerC est-elle conforme ?"
- **Graphiques :** "Affiche la tendance des 14 derniers jours"
- **Alertes :** "Rapport de maintenance prioritaire"

---

## Licence

Proprietary - SUEZ - Tous droits reserves.
