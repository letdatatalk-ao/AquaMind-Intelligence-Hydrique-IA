# AquaMind — Intelligence Hydrique par IA Explicable

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-121212.svg)](https://langchain.com/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-00d684.svg)](https://langchain-ai.github.io/langgraph/)
[![Plotly](https://img.shields.io/badge/Charts-Plotly-3F4F75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](#)

**Depot officiel :** [https://github.com/letdatatalk-ao/-AquaMind-Intelligence-Hydrique-par-IA-Explicable](https://github.com/letdatatalk-ao/-AquaMind-Intelligence-Hydrique-par-IA-Explicable)

---

## Table des Matieres

1. [Resume Executif](#resume-executif)
2. [Architecture du Systeme](#architecture-du-systeme)
3. [Couches Techniques](#couches-techniques)
4. [Modelisation Mathematique](#modelisation-mathematique)
5. [Structure du Projet](#structure-du-projet)
6. [Installation et Deploiement](#installation-et-deploiement)
7. [Pipeline de Donnees](#pipeline-de-donnees)
8. [Exemples de Requetes](#exemples-de-requetes)
9. [Technologies et Dependances](#technologies-et-dependances)
10. [Deploiement sur GitHub](#deploiement-sur-github)
11. [Licence et Contact](#licence-et-contact)

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

Le projet repose sur une architecture en 5 couches, du nettoyage des donnees a l'interface de decision.

```mermaid
graph TD
    A[IoT Raw Data - 4 Clients] --> B[Cleaning & Normalization]
    B --> C[Feature Engineering & Fusion]
    C --> D[Behavioral Markov Graph]
    D --> E[Agent IA ReAct - 10 Outils]
    E --> F[Streamlit UI - Chat & Charts]
    
    B --> G[Health Journal - master_health.parquet]
    G --> E
    
    E --> H[SQLite Memory - SqliteSaver]
    H --> F
    
    style A fill:#0a1628,stroke:#00f0ff,color:#dff4fa
    style D fill:#0a1628,stroke:#00ffc8,color:#dff4fa
    style E fill:#0a1628,stroke:#ffb800,color:#dff4fa
    style F fill:#0a1628,stroke:#00f0ff,color:#dff4fa


 Flux de Donnees
Couche 1 — Ingestion : Les 4 loaders individuels (gym_loader.py, customerA_loader.py, customerB_loader.py, customerC_loader.py) nettoient les CSV bruts.

Couche 2 — Fusion : master_fusion.py consolide et enrichit le dataset avec 15 features expertes.

Couche 3 — Apprentissage : behavior_graph.py genere la matrice de probabilites comportementales a partir du Gold Standard (CustomerC).

Couche 4 — Agent IA : agent/agent.py orchestre le LLM ReAct, agent/tools.py expose les 10 outils specialises.

Couche 5 — Interface : app.py offre l'UI Streamlit avec chat, graphiques et memoire persistante.


Couches Techniques
1. Ingestion & Assainissement (Data Cleaning)
Le pipeline traite les anomalies physiques courantes des capteurs IoT via des loaders dedies :

Correction temporelle : Detection des Unix Epoch Resets (1970) et NTP Rollovers (2036).

Integrite des donnees : Filtrage des 32-bit Overflows (4294967295) et dedoublonnage intelligent.

Normalisation du schema : Uniformisation des colonnes heterogenes en un schema commun de 9 colonnes.

Clients traites :

Client	Type	Capteurs	Sous-categories
Gym	Centre sportif	8 capteurs (4 cabines x Hot/Cold)	Hot shower, Cold shower
CustomerA	Bureaux	1 capteur agrege	Flush, Sink, Tap
CustomerB	Residentiel standard	1 capteur agrege	Flush, Sink, Tap
CustomerC	Residentiel granulaire	Capteurs separes	Flush, Sink, Tap
2. Feature Engineering & Hydraulique
Le dataset fusionne master_dataset.parquet integre des signatures hydrauliques avancees :

Couche Temporelle :

hour, day_of_week, is_weekend, time_slot (7 periodes : Night, Morning_Peak, Mid_Day, Afternoon, Evening_Peak, Late_Evening, Night_End)

Couche Hydraulique :

Debit instantane : flow_rate = consumption / period_s

Intensite hydraulique : intensity_ratio = consumption / sqrt(period_s) (signature discriminante Flush vs Sink)

Delai inter-evenements : gap_since_last (detection de doubles-flush et frequentation)

Couche Statistique :

Z-Score par device : z_score = (x - mean) / std

Score d'anomalie normalise : anomaly_score = clamp(|z_score| / 3, 0, 1)

Couche Comportementale :

session_id : Sessions basees sur un seuil de 120 secondes d'inactivite

transition_pair : Paires de transitions explicites (ex: "Flush -> Sink")

prev_sub_cat : Etat precedent dans la session

3. Graphe Comportemental (Markov)
AquaMind utilise CustomerC comme "Gold Standard" pour modeliser les routines d'hygiene.

Niveaux du Graphe :

Niveau	Description	Exemple
Niveau 1 — Base (Ordre 1)	Transition d'un etat A a un etat B	P(Flush -> Sink) = 0.524
Niveau 2 — Temporel	Transition conditionnee par la periode	P(Flush -> Sink | Morning_Peak) = 0.482
Niveau 3 — Routine (Ordre 2)	Transition basee sur les deux derniers etats	P("Flush -> Sink" -> Tap) = 0.342
Normalisation sans END :
La matrice inclut l'etat de fin de session (END, ~52% des probabilites). Pour comparer avec les donnees reelles (qui ne stockent pas END), les probabilites sont normalisees :

text
P_norm(A -> B) = P(A -> B) / (1 - P(A -> END))
Seuils d'Alerte Comportementale :

Deviation	Statut
< 5%	Normal
5% - 15%	Deviation moderee
> 25%	Alerte critique
Flush -> Sink < 40%	Alerte hygiene (absence de lavage post-WC)




4. Agent IA Agentique (Cerveau)
L'intelligence est pilotee par un agent LangGraph utilisant le modele de raisonnement ReAct (Reason + Act).

Fallback Resilient :
Rotation automatique entre modeles en cas de saturation de quota :

text
1. OpenRouter - meta-llama/llama-3.3-70b-instruct (priorite)
2. OpenRouter - meta-llama/llama-3.1-8b-instruct:free
3. Groq - llama-3.3-70b-versatile
4. Groq - llama-3.1-8b-instant
Memoire Persistante :
Utilisation de SQLite via SqliteSaver pour conserver le contexte des conversations sur plusieurs sessions utilisateur.

Les 10 Outils de l'Agent :

#	Outil	Fonction	Parametres
1	query_water_stats	Statistiques de consommation	customer, date_from, date_to, tag, cabin
2	get_anomaly_report	Anomalies Z-Score et debits impossibles	customer, top_n
3	analyze_behavior_shift	Analyse Markov ou patterns generaux	customer, time_slot
4	generate_consumption_chart	Graphiques Plotly JSON	customer, chart_type, tag, date_from, date_to
5	check_hardware_health	Sante capteurs IoT	customer
6	get_hydraulic_stats	Durees, debits, intensite, gaps	customer, time_slot, tag, cabin, sub_cat
7	get_trend_analysis	Tendance semaine/mois/trimestre	customer, period, tag, cabin
8	compare_customers	Comparaison multi-clients	customers, metric, time_slot, date_from, date_to
9	get_session_analysis	Sessions (nombre, duree, nocturnes)	customer, time_slot, tag, cabin
10	generate_alert_report	Rapport d'alertes consolide	customers, severity, date_from
System Prompt :
L'agent est conditionne avec une voix metier experte :

Identification obligatoire du client avant tout appel d'outil

Extraction systematique de tous les parametres explicites

Reponses structurees en 4 sections : Verdict, Donnees, Analyse, Recommandation

Graphiques encapsules dans <CHART_JSON>...</CHART_JSON>

Vocabulaire domaine eau exclusivement (debits, sessions, routines, fuites)



Modelisation Mathematique
Concept	Formule	Utilite
Debit instantane	
Q
=
V
/
t
Q=V/t	Identification des fuites franches et debits impossibles
Intensite hydraulique	
I
=
V
/
t
I=V/ 
t
​
 	Signature unique du type d'appareil (WC vs Robinet)
Anomalie Z-Score	
z
=
(
x
−
μ
)
/
σ
z=(x−μ)/σ	Scoring statistique des evenements par capteur
Anomaly Score	$a = \min(	z	/ 3, 1)$	Score normalise entre 0 et 1
Markov (Ordre 1)	
P
(
A
→
B
)
=
c
o
u
n
t
(
A
∩
B
)
c
o
u
n
t
(
A
)
P(A→B)= 
count(A)
count(A∩B)
​
 	Probabilite de transition d'un etat a un autre
Normalisation sans END	
P
n
o
r
m
(
A
→
B
)
=
P
(
A
→
B
)
∑
X
≠
E
N
D
P
(
A
→
X
)
P 
norm
​
 (A→B)= 
∑ 
X

=END
​
 P(A→X)
P(A→B)
​
 	Comparaison correcte avec le dataset reel
Detection de session	
n
e
w
_
s
e
s
s
i
o
n
=
(
g
a
p
>
120
s
)
∨
(
g
a
p
=
−
1
)
new_session=(gap>120s)∨(gap=−1)	Regroupement des evenements en sequences coherentes


Structure du Projet
aquamind/
│
├── data/                                   # Data Lake (stockage local)
│   ├── gym_consumption_data.csv            # Donnees brutes Gym
│   ├── customerA_consumption.csv           # Donnees brutes CustomerA
│   ├── customerB_consumption.csv           # Donnees brutes CustomerB
│   ├── customerC_consumption.csv           # Donnees brutes CustomerC
│   ├── gym_base.parquet                    # Gym nettoye
│   ├── gym_errors.parquet                  # Erreurs materiel Gym
│   ├── customerA_base.parquet              # CustomerA nettoye
│   ├── customerA_errors.parquet            # Erreurs materiel CustomerA
│   ├── customerB_base.parquet              # CustomerB nettoye
│   ├── customerB_errors.parquet            # Erreurs materiel CustomerB
│   ├── customerC_base.parquet              # CustomerC nettoye
│   ├── customerC_errors.parquet            # Erreurs materiel CustomerC
│   ├── master_dataset.parquet              # Dataset final (150k+ lignes, 15 features)
│   ├── master_health.parquet               # Journal consolide des erreurs
│   ├── behavioral_matrix.json              # Matrice de probabilites
│   └── aquamind_memory.db                  # Memoire SQLite de l'agent
│
├── data_preprocessed/                      # Couche ETL et Feature Engineering
│   ├── gym_loader.py                       # Loader et cleaner Gym
│   ├── customerA_loader.py                 # Loader et cleaner CustomerA
│   ├── customerB_loader.py                 # Loader et cleaner CustomerB
│   ├── customerC_loader.py                 # Loader et cleaner CustomerC
│   ├── master_fusion.py                    # Fusion et enrichissement
│   ├── verify_master.py                    # Script d'audit
│   └── behavior_graph.py                   # Graphe comportemental
│
├── agent/                                  # Couche Intelligence Artificielle
│   ├── __init__.py                         # Package marker
│   ├── agent.py                            # Orchestrateur LLM (LangChain ReAct)
│   └── tools.py                            # 10 outils specialises
│
├── app.py                                  # Interface Streamlit (Chatbot + Plotly)
├── requirements.txt                        # Dependances Python
└── README.md                               # Documentation (ce fichier)


Installation et Deploiement
1. Environnement
bash
# Cloner le projet depuis le depot officiel
git clone https://github.com/letdatatalk-ao/-AquaMind-Intelligence-Hydrique-par-IA-Explicable.git
cd -AquaMind-Intelligence-Hydrique-par-IA-Explicable

# Creer un environnement virtuel (recommande)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dependances
pip install -r requirements.txt
2. Variables d'Environnement
Creer un fichier .env a la racine du projet :

bash
# OpenRouter API Key (prioritaire)
OPENAI_API_KEY=sk-or-v1-votre-cle-openrouter

# Groq API Key (fallback)
GROQ_API_KEY=gsk_votre-cle-groq
Ou les exporter directement :

bash
export OPENAI_API_KEY="sk-or-v1-votre-cle-openrouter"
export GROQ_API_KEY="gsk_votre-cle-groq"
3. Initialisation du Pipeline de Donnees
bash
# Etape 1 : Nettoyage individuel des 4 clients
python data_preprocessed/gym_loader.py
python data_preprocessed/customerA_loader.py
python data_preprocessed/customerB_loader.py
python data_preprocessed/customerC_loader.py

# Etape 2 : Fusion et Feature Engineering (15 features expertes)
python data_preprocessed/master_fusion.py

# Etape 3 : Construction du Graphe Comportemental (Markov)
python data_preprocessed/behavior_graph.py

# Etape 4 (optionnel) : Verification du master dataset
python data_preprocessed/verify_master.py
4. Lancement de l'Interface
bash
streamlit run app.py
L'interface sera accessible a l'adresse : http://localhost:8501

5. Test de l'Agent en Ligne de Commande
bash
python -m agent.agent
Pipeline de Donnees
Diagramme de Sequence
text
CSV Bruts (4 clients)
    │
    ▼
Loaders Individuels (gym, A, B, C)
    │
    ├── *_base.parquet (donnees saines, schema commun)
    └── *_errors.parquet (journal des erreurs qualifiees)
    │
    ▼
master_fusion.py
    │
    ├── Fusion des 4 bases
    ├── Calcul des 15 features expertes
    ├── Detection des sessions (seuil 120s)
    ├── Construction des paires de transition
    │
    ▼
master_dataset.parquet + master_health.parquet
    │
    ▼
behavior_graph.py
    │
    ├── Isolation de CustomerC (Gold Standard)
    ├── Calcul Niveau 1 (Ordre 1)
    ├── Calcul Niveau 2 (Temporel)
    ├── Calcul Niveau 3 (Ordre 2)
    │
    ▼
behavioral_matrix.json


Volume de Donnees
Fichier	Lignes	Colonnes
Gym brut	49,773	4
CustomerA brut	57,117	5
CustomerB brut	54,403	5
CustomerC brut	43,583	6
Master Dataset	150,000+	15
Master Health	481	10
Behavioral Matrix	JSON	3 niveaux




