"""
agent/agent.py — AquaMind Agent LLM (FINAL PRODUCTION-READY v5.3)
=====================================
CORRECTIONS APPLIQUÉES:
  ✅ Imports réduits aux outils qui existent réellement
  ✅ API Keys sécurisées (variables d'env)
  ✅ SystemMessage correctement injecté
  ✅ Fallback OpenRouter → Groq robuste

PROVIDERS :
  1. OpenRouter  (priorité)  — meta-llama/llama-3.3-70b-instruct
  2. Groq        (fallback)  — llama-3.3-70b-versatile → llama-3.1-8b-instant

LANCEMENT:
    python -m agent.agent
    OU
    streamlit run app.py
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import sqlite3
import warnings
import logging
from typing import Dict, Any, Optional

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── PATH FIX ─────────────────────────────────────────────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_this_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI          # OpenRouter (compatible OpenAI)
from langchain_groq import ChatGroq              # Groq fallback
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

# ✅ IMPORTS CORRIGÉS — Uniquement les tools qui existent dans tools.py
from agent.tools import (
    query_water_stats,
    get_anomaly_report,
    analyze_behavior_shift,
    generate_consumption_chart,
    check_hardware_health,
    get_hydraulic_stats,
    get_trend_analysis,
    compare_customers,
    get_session_analysis,
    generate_alert_report,
)
# ─── CLÉS API — SÉCURISÉES ────────────────────────────────────────────────────
# ✅ Utiliser variables d'env, pas des valeurs en dur
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

logger.info(f"OPENAI_API_KEY configured: {bool(os.environ.get('OPENAI_API_KEY'))}")
logger.info(f"GROQ_API_KEY configured: {bool(os.environ.get('GROQ_API_KEY'))}")

# ─── MÉMOIRE SQLite ───────────────────────────────────────────────────────────
DB_PATH = os.path.join(_root_dir, "aquamind_memory.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)

# ─── MODEL POOL ───────────────────────────────────────────────────────────────
# Format : (provider, model_name)
# OpenRouter en priorité, Groq en fallback automatique
MODEL_POOL = [
    ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    ("openrouter", "meta-llama/llama-3.1-8b-instruct:free"),
    ("groq", "llama-3.3-70b-versatile"),
    ("groq", "llama-3.1-8b-instant"),
]

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are AquaMind, SUEZ's AI agent specialized in intelligent water management.

Your role is to analyze water consumption, anomalies, and IoT sensor health for customers equipped with WaterSec sensors.

==================================================
SECTION 1 - CUSTOMER IDENTIFICATION (MANDATORY)
==================================================

You monitor 4 customer profiles.

1. GYM
Aliases: gym, shower block, sports facility, aquatic center, shower, douche, cabin,OSMOSIS,osmosis
unit
- Shower block with 4 cabins
- "osmosis unit" / "osmosis" = Gym
- Shower block with 4 cabins
- Each cabin has hot and cold water measured independently
- If user mentions "cabine 1", "cabin 2", ALWAYS pass cabin parameter (values: "1", "2", "3", "4")

2. CUSTOMERA
Aliases: customera, office, bureau, commercial, office building
- Aggregated toilet block (commercial/office usage)
- Sub-categories: Flush, Sink, Tap

3. CUSTOMERB
Aliases: customerb, residential standard, kitchen, residential, wc
- Single device covering 2 WC, central sink, 2 flushes
- Sub-categories: Flush, Sink, Tap

4. CUSTOMERC
Aliases: customerc, bathroom, toilet, granular, detailed residential
- Most detailed profile with separate sensors
- Supports detailed behavioral analysis
- Sub-categories: Flush, Sink, Tap

If customer is unclear, ask for clarification BEFORE calling tools.

==================================================
SECTION 2 - AVAILABLE TOOLS
==================================================

1. query_water_stats
Use for: total consumption, daily average, hourly peak, detailed statistics
Parameters: customer (required), date_from (opt), date_to (opt), tag ('hot'/'cold', opt), cabin ('1'-'4', opt)

IMPORTANT: If cabin is mentioned, ALWAYS pass it.

DATES: You MUST pass dates in ISO format (YYYY-MM-DD):
- "last 30 days" → date_from should be 30 days before today
- "March" → date_from="2025-03-01"
- "March 10 to April 11" → date_from="2025-03-10", date_to="2025-04-11"

Example:
"How much cold water in cabin 2 last March?"
→ query_water_stats(customer="Gym", tag="cold", cabin="2", date_from="2025-03-01", date_to="2025-03-31")

---

2. get_anomaly_report
Use for: z-score anomalies > 2.5, impossible flow > 500 ml/s, suspicious activity
Parameters: customer (required), top_n (int, default=5)

IMPORTANT: top_n MUST be an integer. If user says "top 5", pass top_n=5 (not "5").

---

3. analyze_behavior_shift
Use for: CustomerC Markov behavior analysis, usage patterns, routine changes
Parameters: customer (required), time_slot (opt: morning/afternoon/evening/night)

---

4. generate_consumption_chart
Use for: charts, plots, visualizations
Chart types: hourly, daily, cabin_comparison, anomaly
Parameters: customer (required), chart_type (required), tag (opt), date_from (opt, ex: "14 days ago", "2025-03-01"), date_to (opt)

CRITICAL: If user mentions a time period ("past 2 weeks", "last month", "March"), 
ALWAYS pass date_from parameter.
Example: "Plot daily consumption trend for gym over past 2 weeks"
→ generate_consumption_chart(customer="Gym", chart_type="daily", date_from="14 days ago")

When returning chart, encapsulate JSON in:
<CHART_JSON>
plotly_json_here
</CHART_JSON>

"plot", "show me", "graph", "visualize", "display chart" 
→ ALWAYS use generate_consumption_chart, NEVER query_water_stats.

---

5. check_hardware_health
Use for: sensor health, IoT status, faulty devices
Parameters: customer (required)

---

6. get_hydraulic_stats
Use for: shower duration, flow rate analysis, intensity ratio, inter-event gaps
Parameters: customer (required), time_slot (opt), tag (opt), cabin (opt), sub_cat (opt: 'Flush'/'Sink'/'Tap'/'Shower')

7. get_trend_analysis
Use for: consumption trends (week/month/quarter), period-over-period comparison
Parameters: customer (required), period ('week'/'month'/'quarter'), tag (opt), cabin (opt)

8. compare_customers
Use for: multi-customer ranking by a metric (consumption, anomalies, flow rate, sessions)
Parameters: customers ('all' or 'Gym,CustomerA'), metric (opt), time_slot (opt), date_from (opt), date_to (opt)

9. get_session_analysis
Use for: session counts, duration, long sessions, night sessions
Parameters: customer (required), time_slot (opt), tag (opt), cabin (opt)

10. generate_alert_report
Use for: consolidated prioritized alerts (CRITICAL/WARNING) across customers
Parameters: customers ('all' or list), severity ('critical'/'warning'/'all'), date_from (opt)
==================================================
SECTION 3 - RESPONSE FORMAT (MANDATORY)
==================================================

All responses MUST follow:

## Clear Title

**Verdict:** one sentence summary.

**Data:** metrics with units (L, ml/s, hours, etc.)

**Analysis:** 2-3 interpretations (flow rate, leakage, routines, anomalies).

**Recommendation:** one concrete action.

**Filters applied:** (if applicable)

==================================================
SECTION 4 - LANGUAGE
==================================================

Always speak like a water management expert.

Preferred: flow rate, volume, session, routine, leakage, pressure.
NEVER: DataFrame, rows, columns, dataset, object.

Never invent numbers. Never expose internal tools.

If insufficient information, ask clarification.
"""

# ══════════════════════════════════════════════════════════════════════════════
# FACTORY LLM
# ══════════════════════════════════════════════════════════════════════════════
def _build_llm(provider: str, model_name: str):
    """Instancie le LLM selon le provider."""
    if provider == "openrouter":
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            max_retries=1,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://aquamind.suez.com",
                "X-Title": "AquaMind",
            },
        )
    elif provider == "groq":
        return ChatGroq(model=model_name, temperature=0, max_retries=1)
    else:
        raise ValueError(f"Provider inconnu : {provider}")


# ══════════════════════════════════════════════════════════════════════════════
# OUTILS — UNIQUEMENT les 5 qui existent
# ══════════════════════════════════════════════════════════════════════════════
def _make_tools():
    return [
        StructuredTool.from_function(
            func=query_water_stats,
            name="query_water_stats",
            description=(
                "Statistiques consommation eau (total, moyenne, pic, breakdown). "
                "Args: customer (str), date_from (str ISO YYYY-MM-DD, opt), date_to (str ISO, opt), "
                "tag ('hot'/'cold', opt Gym), cabin ('1'-'4', opt Gym). "
                "IMPORTANT: Si 'cabine X' ou 'cabin X' est mentionné, TOUJOURS passer cabin='X'."
            ),
        ),
        StructuredTool.from_function(
            func=get_anomaly_report,
            name="get_anomaly_report",
            description=(
                "Anomalies Z-score > 2.5 et debits > 500 ml/s. "
                "Args: customer (str), top_n (int, default 5)."
            ),
        ),
        StructuredTool.from_function(
            func=analyze_behavior_shift,
            name="analyze_behavior_shift",
            description=(
                "Analyse comportementale Markov (CustomerC) ou patterns (autres). "
                "Args: customer (str), time_slot ('morning'/'afternoon'/'evening'/'night', opt)."
            ),
        ),
        StructuredTool.from_function(
            func=generate_consumption_chart,
            name="generate_consumption_chart",
            description=(
                "Graphique Plotly JSON avec filtre temporel. "
                "Args: customer (str), chart_type ('hourly'/'daily'/'cabin_comparison'/'anomaly'), "
                "tag ('hot'/'cold', opt), date_from (str, opt, ex: '14 days ago'), date_to (str, opt)."
            ),
        ),
        StructuredTool.from_function(
            func=check_hardware_health,
            name="check_hardware_health",
            description=(
                "Sante materielle capteurs IoT. "
                "Args: customer (str)."
            ),
        ),
        StructuredTool.from_function(
            func=get_hydraulic_stats,
            name="get_hydraulic_stats",
            description=(
                "Statistiques hydrauliques avancées: durée, débit, intensité, délai entre événements. "
                "Args: customer (str), time_slot (opt), tag (opt), cabin (opt), sub_cat ('Flush'/'Sink'/'Tap'/'Shower', opt)."
            ),
        ),
        StructuredTool.from_function(
            func=get_trend_analysis,
            name="get_trend_analysis",
            description=(
                "Tendance temporelle période courante vs précédente (semaine/mois/trimestre). "
                "Args: customer (str), period ('week'/'month'/'quarter'), tag (opt), cabin (opt)."
            ),
        ),
        StructuredTool.from_function(
            func=compare_customers,
            name="compare_customers",
            description=(
                "Comparaison multi-clients sur une métrique. "
                "Args: customers ('all' ou 'Gym,CustomerA'), metric ('consumption'/'anomaly_rate'/'flow_rate'/'session_count'), "
                "time_slot (opt), date_from (opt), date_to (opt)."
            ),
        ),
        StructuredTool.from_function(
            func=get_session_analysis,
            name="get_session_analysis",
            description=(
                "Analyse des sessions d'usage (session_id). "
                "Args: customer (str), time_slot (opt), tag (opt), cabin (opt)."
            ),
        ),
        StructuredTool.from_function(
            func=generate_alert_report,
            name="generate_alert_report",
            description=(
                "Rapport d'alertes consolidé priorisé (CRITICAL/WARNING). "
                "Args: customers ('all' ou liste), severity ('critical'/'warning'/'all'), date_from (opt)."
            ),
        ),
    ]
# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION CHART JSON
# ══════════════════════════════════════════════════════════════════════════════
def _extract_chart_json(text: str) -> Optional[str]:
    """Extrait le JSON Plotly de la réponse."""
    import re
    # Balises explicites
    m = re.search(r"<CHART_JSON>(.*?)</CHART_JSON>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # JSON brut Plotly
    idx = text.find('{"data":')
    if idx != -1:
        depth, end = 0, idx
        for i, ch in enumerate(text[idx:], start=idx):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        candidate = text[idx:end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    return None


def _clean_chart_tags(text: str) -> str:
    """Retire les balises CHART_JSON."""
    import re
    return re.sub(r"<CHART_JSON>.*?</CHART_JSON>", "", text, flags=re.DOTALL).strip()


# ══════════════════════════════════════════════════════════════════════════════
# AGENT PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
def run_agent(
    user_input: str,
    chat_history: list = None,
    thread_id: str = "aquamind_session",
) -> Dict[str, Any]:
    """
    Point d'entrée principal.

    Fallback strategy:
      1. OpenRouter llama-3.3-70b-instruct
      2. OpenRouter llama-3.1-8b-instruct:free
      3. Groq llama-3.3-70b-versatile
      4. Groq llama-3.1-8b-instant
    """
    config = {"configurable": {"thread_id": thread_id}}

    for provider, model_name in MODEL_POOL:
        try:
            logger.info(f"Trying [{provider}] {model_name}")

            llm = _build_llm(provider, model_name)
            tools = _make_tools()
            
            # ✅ Créer agent avec SystemMessage dans les messages initiaux
            agent = create_react_agent(model=llm, tools=tools, checkpointer=memory)

            # Préparer messages avec SYSTEM_PROMPT
            messages = [SystemMessage(content=SYSTEM_PROMPT)]

            # Historique conversationnel
            if chat_history:
                for msg in chat_history[-6:]:
                    cls = HumanMessage if msg["role"] == "user" else AIMessage
                    messages.append(cls(content=msg["content"]))

            messages.append(HumanMessage(content=user_input))

            # Invocation
            result = agent.invoke({"messages": messages}, config=config)

            # Log tools appelés
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    names = [
                        tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                        for tc in msg.tool_calls
                    ]
                    logger.info(f"Tools: {', '.join(names)}")

            raw_text = result["messages"][-1].content
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)

            chart_json = _extract_chart_json(raw_text)
            clean_text = _clean_chart_tags(raw_text) if chart_json else raw_text

            active_label = f"{provider}/{model_name.split('/')[-1]}"
            logger.info(f"✅ [{active_label}] {len(clean_text)} chars")

            return {
                "response": clean_text,
                "chart_json": chart_json,
                "active_model": active_label,
            }

        except Exception as e:
            err = str(e).lower()
            logger.warning(f"[{provider}] {model_name} failed: {str(e)[:100]}")
            # Continue to next provider
            continue

    # All providers failed
    return {
        "response": "⚠️ Service temporairement indisponible. Réessayez dans quelques secondes.",
        "chart_json": None,
        "active_model": "error",
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  AQUAMIND AGENT — TEST (OpenRouter + Groq fallback)")
    print("=" * 65)

    queries = [
        ("STATS", "What is the average daily cold water consumption in the gym for the last 30 days?"),
        ("ANOMALY", "Is there any unusual behaviour on the residential toilet flush this month?"),
        ("BEHAVIOR", "Analyze behavior patterns for CustomerC"),
        ("CHART", "Plot the daily consumption trend for the gym"),
        ("HARDWARE", "Check sensor health for CustomerA"),
    ]

    for category, query in queries:
        print(f"\n[{category}] {query[:60]}...")
        result = run_agent(query, thread_id="test")
        preview = result["response"][:200].replace("\n", " ")
        print(f"Response: {preview}...")
        print(f"Model: {result.get('active_model')}")

    print("\n" + "=" * 65)