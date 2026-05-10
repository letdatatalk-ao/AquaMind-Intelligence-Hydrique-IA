"""
app.py — AquaMind · Interface Streamlit
Version : 3.1 (Corrigée)

CORRECTIONS:
- use_container_width supprimé (remplacé par use_container_width=True, paramètre natif Streamlit)
- Extraction chart_json robuste (balises CHART_JSON ou JSON brut)
- Historique de conversation passé au backend pour contexte
- Gestion d'état propre sans collisions de clés
"""

import os
import sys
import json
import streamlit as st
import plotly.io as pio
from datetime import datetime

# ─── Import du backend ───────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from agent.agent import run_agent as backend_run_agent

# =============================================================================
# WRAPPER BACKEND
# =============================================================================
def run_agent(prompt: str, chat_history: list = None) -> dict:
    """Appelle le backend réel avec gestion d'erreur."""
    try:
        result = backend_run_agent(
            prompt,
            chat_history=chat_history,
            thread_id=st.session_state.get("thread_id", "aquamind_demo_user"),
        )
        return {
            "response":     result.get("response", "Pas de réponse."),
            "chart_json":   result.get("chart_json"),
            "active_model": result.get("active_model", "aquamind-backend"),
        }
    except Exception as e:
        return {
            "response":     f"## ⚠️ Erreur backend\n\n```\n{e}\n```",
            "chart_json":   None,
            "active_model": "error",
        }


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="AquaMind Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# DESIGN SYSTEM — Abyssal Water · Dark Luxury · Neon Cyan
# =============================================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;600;700;900&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">

<style>
:root {
    --void:        #01070f;
    --deep:        #020c18;
    --surface:     #041220;
    --glass:       rgba(4, 18, 32, 0.78);
    --cyan:        #00f0ff;
    --cyan-dim:    rgba(0, 240, 255, 0.10);
    --cyan-glow:   rgba(0, 240, 255, 0.28);
    --cyan-border: rgba(0, 240, 255, 0.18);
    --emerald:     #00ffc8;
    --amber:       #ffb800;
    --danger:      #ff4b6e;
    --txt-1: #dff4fa;
    --txt-2: #7ab8cc;
    --txt-3: #355f70;
    --r-md:   18px;
    --r-lg:   26px;
    --r-full: 9999px;
    --f-display: 'Outfit', sans-serif;
    --f-body:    'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse 70% 55% at -5% -5%, rgba(0,180,210,0.20) 0%, rgba(0,0,0,0) 65%),
        radial-gradient(ellipse 55% 45% at 105% 105%, rgba(0,240,255,0.10) 0%, rgba(0,0,0,0) 60%),
        var(--void) !important;
    font-family: var(--f-body);
    color: var(--txt-1);
    min-height: 100vh;
    overflow-x: hidden;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem !important; max-width: 900px !important; }

/* ─── MOVING GRID ────────────────────────────────────────────────────────── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,240,255,0.020) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,240,255,0.020) 1px, transparent 1px);
    background-size: 72px 72px;
    pointer-events: none;
    z-index: 0;
    animation: gridMove 40s linear infinite;
}
@keyframes gridMove {
    0%   { background-position: 0 0; }
    100% { background-position: 72px 72px; }
}

/* ─── TOP BAR ────────────────────────────────────────────────────────────── */
.aq-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.85rem 0; margin-bottom: 0.5rem;
    border-bottom: 1px solid var(--cyan-border);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    position: sticky; top: 0; z-index: 100;
    background: rgba(4,18,32,0.65);
}
.aq-brand { display: flex; align-items: center; gap: 10px; }
.aq-brand-icon {
    width: 32px; height: 32px; border-radius: 10px;
    background: linear-gradient(135deg, #0096c7, #00f0ff);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; box-shadow: 0 0 14px var(--cyan-glow);
}
.aq-brand-text {
    font-family: var(--f-display); font-size: 0.9rem; font-weight: 700;
    background: linear-gradient(135deg, #dff4fa, var(--cyan));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.aq-status {
    display: flex; align-items: center; gap: 7px;
    padding: 4px 12px; border-radius: var(--r-full);
    background: rgba(0,255,200,0.07);
    border: 1px solid rgba(0,255,200,0.18);
    font-family: var(--f-display); font-size: 0.6rem; font-weight: 600;
    letter-spacing: 0.1em; color: var(--emerald);
}
.aq-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--emerald); box-shadow: 0 0 7px var(--emerald);
    animation: dotBlink 2.2s ease-in-out infinite;
}
@keyframes dotBlink { 0%,100%{opacity:1;} 50%{opacity:0.2;} }

/* ─── HERO ───────────────────────────────────────────────────────────────── */
.aq-hero { text-align: center; padding: 2rem 1rem 1rem; }
.aq-hero h1 {
    font-family: var(--f-display); font-size: 1.8rem; font-weight: 700;
    background: linear-gradient(135deg, #dff4fa, var(--cyan), #dff4fa);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; animation: textShine 6s linear infinite;
    margin-bottom: 0.5rem;
}
@keyframes textShine { 0%{background-position:0% center;} 100%{background-position:200% center;} }
.aq-hero p { font-size: 0.85rem; color: var(--txt-2); font-weight: 300; }

/* ─── CHAT INPUT ─────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: rgba(4,18,32,0.88) !important;
    border: 1px solid rgba(0,240,255,0.22) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(20px) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(0,240,255,0.50) !important;
    box-shadow: 0 0 0 3px rgba(0,240,255,0.08), 0 0 28px rgba(0,240,255,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important; color: var(--txt-1) !important;
    font-family: var(--f-body) !important; font-size: 0.95rem !important;
    font-weight: 300 !important; caret-color: var(--cyan) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--txt-3) !important; font-style: italic !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #0096c7, #00f0ff) !important;
    border-radius: 12px !important; border: none !important; color: var(--void) !important;
}
[data-testid="stChatInput"] button:hover { box-shadow: 0 0 16px var(--cyan-glow) !important; }

/* ─── CHAT MESSAGES ──────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--glass) !important;
    border: 1px solid var(--cyan-border) !important;
    border-radius: var(--r-lg) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    margin-bottom: 0.7rem !important; padding: 1rem 1.4rem !important;
    animation: fadeUp 0.35s ease both;
    position: relative; overflow: hidden;
}
[data-testid="stChatMessage"]::before {
    content: '';
    position: absolute;
    top: 0; left: 20%; right: 20%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,240,255,0.28), transparent);
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: var(--txt-1) !important; font-size: 0.93rem !important;
    font-weight: 300 !important; line-height: 1.75 !important;
}
[data-testid="stChatMessage"] strong { color: var(--cyan) !important; font-weight: 600 !important; }
[data-testid="stChatMessage"] code {
    background: rgba(0,240,255,0.08) !important; border: 1px solid var(--cyan-border) !important;
    border-radius: 5px !important; color: var(--cyan) !important;
    font-size: 0.82rem !important; padding: 1px 6px !important;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #0096c7, #00f0ff) !important;
    border-radius: 50% !important; box-shadow: 0 0 14px var(--cyan-glow) !important;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
    background: rgba(0,240,255,0.10) !important;
    border: 1px solid var(--cyan-border) !important;
    border-radius: 50% !important;
}

/* ─── BUTTONS ────────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: var(--f-display) !important;
    font-size: 0.72rem !important; font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    border-radius: var(--r-full) !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.22s ease !important;
    border: 1px solid var(--cyan-border) !important;
    background: rgba(0,240,255,0.05) !important;
    color: var(--txt-2) !important;
}
.stButton > button:hover {
    background: rgba(0,240,255,0.12) !important;
    border-color: rgba(0,240,255,0.38) !important;
    color: var(--cyan) !important;
    transform: translateY(-1px) !important;
}

/* ─── SPINNER ────────────────────────────────────────────────────────────── */
.stSpinner > div { border-color: var(--cyan) transparent transparent transparent !important; }

/* ─── SCROLLBAR ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: rgba(0,240,255,0.18); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

/* ─── FOOTER ─────────────────────────────────────────────────────────────── */
.aq-footer {
    text-align: center; margin-top: 2rem; padding-top: 1rem;
    border-top: 1px solid rgba(0,240,255,0.06);
    font-family: var(--f-display); font-size: 0.55rem;
    letter-spacing: 0.22em; text-transform: uppercase;
    color: #1e3d4f;
}

/* ─── TABLES ─────────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] table {
    border-collapse: collapse; width: 100%; margin: 0.5rem 0;
}
[data-testid="stChatMessage"] th {
    background: rgba(0,240,255,0.08) !important; color: var(--cyan) !important;
    font-family: var(--f-display) !important; font-size: 0.7rem !important;
    font-weight: 700 !important; letter-spacing: 0.08em !important;
    padding: 6px 10px !important; border: 1px solid var(--cyan-border) !important;
}
[data-testid="stChatMessage"] td {
    color: var(--txt-1) !important; font-size: 0.8rem !important;
    padding: 5px 10px !important; border: 1px solid rgba(0,240,255,0.07) !important;
    background: rgba(4,18,32,0.5) !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================
WELCOME_MSG = (
    "Bonjour. Je suis **AquaMind**, votre agent IA spécialisé en gestion intelligente de l'eau. "
    "Je suis connecté en temps réel à vos capteurs IoT, aux journaux de maintenance et à la matrice comportementale.\n\n"
    "**Ce que je peux analyser :**\n"
    "- 📊 Stats de consommation (volume, pics, tendances)\n"
    "- 🚨 Détection d'anomalies (fuites, surconsommations)\n"
    "- 🧠 Analyse comportementale (routines hygiène Flush/Sink)\n"
    "- 📈 Graphiques (horaire, journalier, anomalies)\n"
    "- 🔧 Santé matérielle (pannes capteurs, batteries)\n\n"
    "*Posez votre première question pour commencer l'analyse.*"
)

if "messages" not in st.session_state:
    st.session_state.messages     = [{"role": "assistant", "content": WELCOME_MSG, "chart": None}]
if "active_model" not in st.session_state:
    st.session_state.active_model = "aquamind-backend"
if "query_count" not in st.session_state:
    st.session_state.query_count  = 0
if "thread_id" not in st.session_state:
    # Thread unique par session pour la mémoire persistante
    st.session_state.thread_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if "_chip_query" not in st.session_state:
    st.session_state._chip_query = None


# =============================================================================
# TOP BAR
# =============================================================================
st.markdown(f"""
<div class="aq-topbar">
    <div class="aq-brand">
        <div class="aq-brand-icon">💧</div>
        <span class="aq-brand-text">AquaMind</span>
    </div>
    <div class="aq-status">
        <div class="aq-dot"></div>
        {st.session_state.active_model}
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# HERO (seulement au démarrage)
# =============================================================================
if len(st.session_state.messages) <= 1:
    st.markdown("""
    <div class="aq-hero">
        <h1>L'eau, comprise en un instant</h1>
        <p>Intelligence artificielle · Capteurs IoT · Analyse temps réel</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# CHIPS CLIQUABLES + RESET
# =============================================================================
col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 0.8])

with col1:
    if st.button("📊 Stats Gym",    use_container_width=True, key="chip_stats"):
        st.session_state._chip_query = "Stats de consommation Gym"
with col2:
    if st.button("🚨 Anomalies",    use_container_width=True, key="chip_anomalies"):
        st.session_state._chip_query = "Détecte les anomalies de CustomerA"
with col3:
    if st.button("🧠 Hygiène C",   use_container_width=True, key="chip_hygiene"):
        st.session_state._chip_query = "Analyse les routines hygiène de CustomerC"
with col4:
    if st.button("📈 Graphique",    use_container_width=True, key="chip_chart"):
        st.session_state._chip_query = "Graphique journalier Gym"
with col5:
    if st.button("🔧 Santé",        use_container_width=True, key="chip_health"):
        st.session_state._chip_query = "Santé capteurs CustomerB"
with col6:
    if st.button("🔄",             use_container_width=True, key="reset_btn", help="Nouvelle session"):
        st.session_state.messages     = [{"role": "assistant", "content": WELCOME_MSG, "chart": None}]
        st.session_state.query_count  = 0
        st.session_state.thread_id    = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state._chip_query  = None
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# RENDU GRAPHIQUE PLOTLY
# =============================================================================
def _render_chart(chart_json: str):
    """Applique le thème AquaMind et affiche un graphique Plotly."""
    if not chart_json:
        return
    try:
        parsed = json.loads(chart_json)
        if "error" in parsed:
            st.warning(f"Erreur graphique : {parsed['error']}")
            return

        fig = pio.from_json(chart_json)
        
        # ✅ Appliquer le thème via update_xaxes/update_yaxes (pas update_layout)
        fig.update_xaxes(
            gridcolor="rgba(0,240,255,0.07)",
            zerolinecolor="rgba(0,240,255,0.12)",
        )
        fig.update_yaxes(
            gridcolor="rgba(0,240,255,0.07)",
            zerolinecolor="rgba(0,240,255,0.12)",
        )
        
        # ✅ Layout global (sans xaxis/yaxis)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#7ab8cc", size=11),
            margin=dict(l=16, r=16, t=32, b=16),
        )
        
        st.plotly_chart(fig, use_container_width=True)

    except json.JSONDecodeError:
        st.error("Le JSON du graphique est invalide.")
    except Exception as e:
        st.error(f"Erreur rendu graphique : {e}")

# =============================================================================
# HISTORIQUE DES MESSAGES
# =============================================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("chart"):
            _render_chart(message["chart"])


# =============================================================================
# GESTION CHIPS + CHAT INPUT
# =============================================================================
SPINNER_MESSAGES = [
    "Interrogation des capteurs IoT…",
    "Analyse comportementale en cours…",
    "Consultation de la matrice Markov…",
    "Détection des anomalies Z-score…",
    "Chargement des données de maintenance…",
    "Calcul des probabilités de transition…",
]

# Récupérer la requête chip (si cliqué)
prompt = st.session_state._chip_query
if prompt:
    st.session_state._chip_query = None  # Consommer le chip

# Sinon, lire l'input texte
if prompt is None:
    prompt = st.chat_input("Posez votre question… (ex: Analyse la consommation du Gym)")

if prompt:
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt, "chart": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        spinner_idx = st.session_state.query_count % len(SPINNER_MESSAGES)
        with st.spinner(SPINNER_MESSAGES[spinner_idx]):
            n_history = 2
            history_to_send = st.session_state.messages[-(n_history+1):-1] if len(st.session_state.messages) > n_history else st.session_state.messages[:-1]
            # Construire l'historique pour le backend (sans les graphiques)
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in history_to_send
                if m["role"] in ("user", "assistant")
            ]

            result = run_agent(prompt, chat_history)

        response_text  = result.get("response", "Erreur de communication avec l'agent.")
        chart_data     = result.get("chart_json")
        active_model   = result.get("active_model", st.session_state.active_model)

        # Mettre à jour le modèle actif dans la topbar
        st.session_state.active_model = active_model

        st.markdown(response_text)
        if chart_data:
            _render_chart(chart_data)

        # Sauvegarder la réponse
        st.session_state.messages.append({
            "role":    "assistant",
            "content": response_text,
            "chart":   chart_data,
        })
        st.session_state.query_count += 1

    st.rerun()


# =============================================================================
# FOOTER
# =============================================================================
st.markdown(f"""
<div class="aq-footer">
    AquaMind · Intelligence de l'eau · v3.1 · Session {st.session_state.thread_id[-8:]} · {st.session_state.query_count} requêtes
</div>
""", unsafe_allow_html=True)
