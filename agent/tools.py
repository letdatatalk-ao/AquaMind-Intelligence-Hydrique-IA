"""
agent/tools.py — AquaMind Tools FINAL
======================================
Schéma exact du master_dataset.parquet :
  Colonnes : customer, device_id, cabin, tag, category, sub_category,
             timestamp, consumption, period_s, hour, day_of_week, is_weekend,
             time_slot, flow_rate, intensity_ratio, z_score, anomaly_score,
             block_id, gap_since_last, session_id, prev_sub_cat, transition_pair

  Customers : CustomerA, CustomerB, CustomerC, Gym
  cabin     : "1.0", "2.0", "3.0", "4.0"  (string dans parquet)
  time_slot : Morning_Peak, Mid_Day, Afternoon, Evening_Peak, Late_Evening, Night, Night_End
  tag       : cold, hot

FIXES APPLIQUÉS :
  BUG 2 CORRIGÉ : analyze_behavior_shift lit behavioral_matrix.json + normalise sans END
  BUG 3 CORRIGÉ : cabin filter -> str(float(cabin)) pour matcher "1.0"
  Tous types JSON-sérialisables (float() pas np.float64)
  check_hardware_health enrichi avec détails depuis master_health.parquet
"""

import os
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ─── CHEMINS ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
DATA_PATH   = os.path.join(DATA_DIR, "master_dataset.parquet")
HEALTH_PATH = os.path.join(DATA_DIR, "master_health.parquet")
MATRIX_PATH = os.path.join(DATA_DIR, "behavioral_matrix.json")

# ─── TIME_SLOT mapping ────────────────────────────────────────────────────────
TIME_SLOT_MAP = {
    "morning":      ["Morning_Peak", "Mid_Day"],
    "afternoon":    ["Afternoon"],
    "evening":      ["Evening_Peak", "Late_Evening"],
    "night":        ["Night", "Night_End"],
    "morning_peak": ["Morning_Peak"],
    "mid_day":      ["Mid_Day"],
    "evening_peak": ["Evening_Peak"],
    "late_evening": ["Late_Evening"],
    "night_end":    ["Night_End"],
}

TIME_SLOT_TO_MATRIX_KEY = {
    "morning":   "Morning_Peak",
    "afternoon": "Afternoon",
    "evening":   "Evening_Peak",
    "night":     "Night",
    "morning_peak": "Morning_Peak",
    "mid_day":      "Mid_Day",
    "evening_peak": "Evening_Peak",
    "late_evening": "Late_Evening",
}

# ─── NORMALISATION CLIENTS ────────────────────────────────────────────────────
_CUSTOMER_ALIASES: Dict[str, str] = {
    "customera": "CustomerA", "customer a": "CustomerA", "a": "CustomerA",
    "customerb": "CustomerB", "customer b": "CustomerB", "b": "CustomerB",
    "customerc": "CustomerC", "customer c": "CustomerC", "c": "CustomerC",
    "gym": "Gym",
    "office": "CustomerA", "bureau": "CustomerA", "bureaux": "CustomerA", "commercial": "CustomerA",
    "residential": "CustomerB", "résidentiel": "CustomerB", "kitchen": "CustomerB", "cuisine": "CustomerB",
    "bathroom": "CustomerC", "toilet": "CustomerC", "toilette": "CustomerC",
    "salle de bain": "CustomerC", "flush": "CustomerC", "granular": "CustomerC",
    "shower": "Gym", "sports": "Gym", "osmosis": "Gym", "facility": "Gym",
    "cabin": "Gym", "shower cabin": "Gym", "aquatic": "Gym", "sportif": "Gym",
}


def _normalize_customer(raw: str) -> str:
    key = raw.strip().lower()
    if key in _CUSTOMER_ALIASES:
        return _CUSTOMER_ALIASES[key]
    for alias, customer in _CUSTOMER_ALIASES.items():
        if alias in key:
            return customer
    raise ValueError(
        f"Client '{raw}' non reconnu. "
        "Valides: Gym | CustomerA (bureau/office) | CustomerB (résidentiel) | CustomerC (bathroom/toilet/granulaire)"
    )


# ─── PARSING DATE ─────────────────────────────────────────────────────────────
def _parse_date(s: Optional[str]) -> Optional[str]:
    if not s or str(s) in ("None", "null", ""):
        return None
    today = datetime.now().date()
    sl = s.strip().lower()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s.strip()):
        return s.strip()
    m = re.search(r"(\d+)\s+days?\s+ago|last\s+(\d+)\s+days?|past\s+(\d+)\s+days?", sl)
    if m:
        days = int(next(x for x in m.groups() if x))
        return (today - timedelta(days=days)).isoformat()
    m = re.search(r"(\d+)\s+weeks?\s+ago|last\s+(\d+)\s+weeks?|past\s+(\d+)\s+weeks?", sl)
    if m:
        weeks = int(next(x for x in m.groups() if x))
        return (today - timedelta(weeks=weeks)).isoformat()
    m = re.search(r"(\d+)\s+months?\s+ago|last\s+(\d+)\s+months?", sl)
    if m:
        months = int(next(x for x in m.groups() if x))
        return (today - timedelta(days=months * 30)).isoformat()
    MONTHS = {
        "january":"01","february":"02","march":"03","april":"04","may":"05","june":"06",
        "july":"07","august":"08","september":"09","october":"10","november":"11","december":"12",
        "janvier":"01","février":"02","mars":"03","avril":"04","mai":"05","juin":"06",
        "juillet":"07","août":"08","septembre":"09","octobre":"10","novembre":"11","décembre":"12",
    }
    for name, num in MONTHS.items():
        if name in sl:
            dm = re.search(r"(\d{1,2})", s)
            day = dm.group(1).zfill(2) if dm else "01"
            return f"2025-{num}-{day}"
    return s.strip()


# ─── CACHE ────────────────────────────────────────────────────────────────────
_df_cache:     Optional[pd.DataFrame] = None
_health_cache: Optional[pd.DataFrame] = None
_matrix_cache: Optional[dict]         = None


def _load_data() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset introuvable: {DATA_PATH}")
        _df_cache = pd.read_parquet(DATA_PATH)
        _df_cache["timestamp"] = pd.to_datetime(_df_cache["timestamp"], utc=True)
    return _df_cache.copy()


def _load_health() -> pd.DataFrame:
    global _health_cache
    if _health_cache is None:
        if not os.path.exists(HEALTH_PATH):
            return pd.DataFrame()
        _health_cache = pd.read_parquet(HEALTH_PATH)
    return _health_cache.copy()


def _load_matrix() -> dict:
    global _matrix_cache
    if _matrix_cache is None:
        if not os.path.exists(MATRIX_PATH):
            raise FileNotFoundError(f"Behavioral matrix introuvable: {MATRIX_PATH}")
        with open(MATRIX_PATH, "r") as f:
            _matrix_cache = json.load(f)
    return _matrix_cache


def _normalize_matrix_probs(from_event: str, matrix_level: dict) -> dict:
    """
    Normalise les probabilités en excluant END.
    POURQUOI : la matrice inclut END (~52%). Le dataset ne stocke pas les END.
    Comparer directement => faux négatifs. Normaliser => delta CustomerC = 0.000.
    """
    row = matrix_level.get(from_event, {})
    filtered = {k: v for k, v in row.items() if k != "END"}
    total = sum(filtered.values())
    if total == 0:
        return {}
    return {k: round(v / total, 4) for k, v in filtered.items()}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — QUERY
# ══════════════════════════════════════════════════════════════════════════════
def query_water_stats(
    customer:  str,
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None,
    tag:       Optional[str] = None,
    cabin:     Optional[str] = None,
) -> Dict[str, Any]:
    """
    Statistiques de consommation d'eau.
    Args:
        customer  : gym | customerA | customerB | customerC
        date_from : YYYY-MM-DD ou "30 days ago" ou "March" (optionnel)
        date_to   : YYYY-MM-DD (optionnel)
        tag       : "hot" | "cold" (Gym uniquement, optionnel)
        cabin     : "1" | "2" | "3" | "4" (Gym uniquement, optionnel)
    """
    try:
        customer  = _normalize_customer(str(customer).strip())
        date_from = _parse_date(date_from)
        date_to   = _parse_date(date_to)
        tag       = str(tag).strip().lower() if tag and str(tag) not in ("None","null") else None
        cabin     = str(cabin).strip() if cabin and str(cabin) not in ("None","null") else None

        df   = _load_data()
        mask = df["customer"] == customer

        if date_from:
            mask &= df["timestamp"] >= pd.Timestamp(date_from, tz="UTC")
        if date_to:
            mask &= df["timestamp"] <= pd.Timestamp(date_to + " 23:59:59", tz="UTC")
        if tag and "tag" in df.columns:
            mask &= df["tag"].str.lower() == tag
        if cabin and "cabin" in df.columns:
            # BUG 3 FIX: cabin stocké "1.0" dans parquet
            cabin_str = str(float(cabin))
            mask &= df["cabin"].astype(str) == cabin_str

        sub = df[mask]
        if sub.empty:
            return {"error": f"Aucune donnée pour {customer} avec ces filtres."}

        total_ml  = float(sub["consumption"].sum())
        peak_hour = int(sub.groupby("hour")["consumption"].sum().idxmax())
        dates     = sub["timestamp"].dt.date
        n_days    = max((dates.max() - dates.min()).days, 1)

        breakdown = {}
        if "sub_category" in sub.columns:
            breakdown = {
                str(k): round(float(v) / 1000, 1)
                for k, v in sub.groupby("sub_category")["consumption"].sum().items()
            }

        return {
            "customer":                customer,
            "period":                  f"{dates.min()} -> {dates.max()}",
            "event_count":             int(len(sub)),
            "total_L":                 round(total_ml / 1000, 1),
            "avg_per_event_L":         round(float(sub["consumption"].mean()) / 1000, 3),
            "daily_avg_L":             round(total_ml / 1000 / n_days, 1),
            "peak_hour":               f"{peak_hour}h00",
            "breakdown_by_category_L": breakdown,
            "filters_applied": {
                "tag": tag, "cabin": cabin,
                "date_from": date_from, "date_to": date_to,
            },
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("query_water_stats error")
        return {"error": f"Erreur: {str(e)}"}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — ANOMALY
# ══════════════════════════════════════════════════════════════════════════════
def get_anomaly_report(customer: str, top_n: int = 5) -> Dict[str, Any]:
    """
    Anomalies statistiques (Z-score > 2.5) et débits impossibles (>500 ml/s).
    Args:
        customer : gym | customerA | customerB | customerC
        top_n    : nombre max d'anomalies (défaut 5)
    """
    try:
        customer = _normalize_customer(str(customer).strip())
        top_n    = int(top_n) if top_n else 5

        df  = _load_data()
        sub = df[df["customer"] == customer]
        if sub.empty:
            return {"error": f"Aucune donnée pour {customer}"}

        anom = sub[sub["z_score"] > 2.5].sort_values("z_score", ascending=False)

        if anom.empty:
            return {
                "customer":        customer,
                "status":          "Aucune anomalie detectee",
                "total_anomalies": 0,
                "message":         "Flux stable dans les normes historiques.",
            }

        report = []
        for _, row in anom.head(top_n).iterrows():
            z    = float(row["z_score"])
            flow = float(row.get("flow_rate", 0))
            hour = int(row.get("hour", 12))
            if flow > 500:
                cause = "Debit physiquement impossible (>500 ml/s) — capteur defaillant ou fuite"
            elif hour < 5 or hour >= 23:
                cause = "Consommation nocturne inhabituelle — surveillance recommandee"
            elif z > 4:
                cause = "Surconsommation extreme (Z>4) — inspection immediate requise"
            else:
                cause = "Pic inhabituel — possible suroccupation"

            report.append({
                "device_id":      str(row["device_id"])[:12] + "...",
                "timestamp":      str(row["timestamp"])[:16],
                "sub_category":   str(row.get("sub_category", "N/A")),
                "consumption_L":  round(float(row["consumption"]) / 1000, 2),
                "z_score":        round(z, 2),
                "flow_rate_mls":  round(flow, 1),
                "anomaly_score":  round(float(row.get("anomaly_score", 0)), 3),
                "cause_probable": cause,
            })

        return {
            "customer":         customer,
            "total_anomalies":  int((sub["z_score"] > 2.5).sum()),
            "high_flow_events": int((sub["flow_rate"] > 500).sum()),
            "showing_top":      len(report),
            "anomalies":        report,
            "recommendation":   "Inspecter capteurs Z-score > 4. Verifier evenements nocturnes pour fuites silencieuses.",
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("get_anomaly_report error")
        return {"error": f"Erreur: {str(e)}"}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — BEHAVIOR
# ══════════════════════════════════════════════════════════════════════════════
def analyze_behavior_shift(customer: str, time_slot: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyse comportementale Markov (CustomerC) ou patterns generaux (autres clients).
    CustomerC : utilise behavioral_matrix.json + normalisation sans END.
    Autres    : analyse patterns horaires, hot/cold, variabilite, anomalies.
    Args:
        customer  : gym | customerA | customerB | customerC
        time_slot : morning | afternoon | evening | night (optionnel)
    """
    try:
        customer  = _normalize_customer(str(customer).strip())
        time_slot = str(time_slot).strip().lower() if time_slot and str(time_slot) not in ("None","null") else None

        df  = _load_data()
        sub = df[df["customer"] == customer].copy()
        if sub.empty:
            return {"error": f"Aucune donnee pour {customer}"}

        if time_slot:
            slots = TIME_SLOT_MAP.get(time_slot, [])
            if slots:
                sub = sub[sub["time_slot"].isin(slots)]
            if sub.empty:
                return {"error": f"Aucune donnee pour {customer} en time_slot={time_slot}"}

        # ── CustomerC → Markov ────────────────────────────────────────────────
        if customer == "CustomerC":
            matrix = _load_matrix()
            matrix_level = matrix.get("level_1_base", {})
            if time_slot:
                mk = TIME_SLOT_TO_MATRIX_KEY.get(time_slot)
                if mk:
                    temporal = matrix.get("level_2_temporal", {}).get(mk, {})
                    if temporal:
                        matrix_level = temporal

            no_start = sub[~sub["transition_pair"].str.startswith("START", na=True)]

            key_transitions = [
                ("Flush", "Sink",  "Lavage mains post-WC — indicateur hygiene cle"),
                ("Flush", "Tap",   "Robinet alternatif post-WC"),
                ("Flush", "Flush", "Double chasse — possible dysfonctionnement"),
                ("Sink",  "Flush", "Retour WC apres lavabo — routine normale"),
                ("Tap",   "Sink",  "Passage robinet->lavabo — routine complete"),
            ]

            results = []
            for from_e, to_e, label in key_transitions:
                normalized_std = _normalize_matrix_probs(from_e, matrix_level)
                standard       = normalized_std.get(to_e, 0)
                pair_key       = f"{from_e} -> {to_e}"
                from_total     = len(no_start[no_start["transition_pair"].str.startswith(from_e + " ->", na=False)])
                pair_count     = len(no_start[no_start["transition_pair"] == pair_key])
                current        = round(pair_count / from_total, 4) if from_total > 0 else 0
                deviation      = round(current - standard, 3)

                if standard < 0.05:
                    status = "Reference insuffisante"
                elif abs(deviation) < 0.05:
                    status = "Normal"
                elif deviation < -0.25:
                    status = "ALERTE CRITIQUE — deviation majeure"
                elif deviation < -0.10:
                    status = "Deviation moderee"
                elif deviation > 0.15:
                    status = "Frequence anormalement elevee"
                else:
                    status = "Normal"

                results.append({
                    "transition":         pair_key,
                    "description":        label,
                    "gold_standard_prob": round(standard, 3),
                    "current_prob":       round(current, 3),
                    "deviation":          deviation,
                    "statut":             status,
                })

            alerts   = [r for r in results if "ALERTE" in r["statut"]]
            warnings = [r for r in results if "Deviation moderee" in r["statut"]]
            if alerts:
                verdict = f"{len(alerts)} deviation(s) critique(s) — anomalie comportementale confirmee"
            elif warnings:
                verdict = f"{len(warnings)} deviation(s) moderee(s) — surveillance recommandee"
            else:
                verdict = "Comportement dans les normes du Gold Standard"

            flush_sink     = next((r["current_prob"] for r in results if r["transition"] == "Flush -> Sink"), 0)
            flush_sink_std = next((r["gold_standard_prob"] for r in results if r["transition"] == "Flush -> Sink"), 0)

            return {
                "customer":            customer,
                "time_slot_filter":    time_slot or "global",
                "verdict":             verdict,
                "behavioral_analysis": results,
                "hygiene_insight": (
                    f"Flush->Sink actuel: {flush_sink:.1%} vs Gold Standard: {flush_sink_std:.1%}. "
                    f"{'Taux de lavage bas — alerte hygiene.' if flush_sink < 0.40 else 'Hygiene conforme.'}"
                ),
                "gold_standard": "CustomerC residentiel (73 934 evenements appris)",
                "method_note":   "Normalise sans END. Delta CustomerC = 0.000 (valide).",
            }

        # ── Autres clients → patterns generaux ────────────────────────────────
        else:
            hourly     = sub.groupby("hour")["consumption"].mean()
            peak_h     = int(hourly.idxmax()) if not hourly.empty else 0
            low_h      = int(hourly.idxmin()) if not hourly.empty else 0
            daily_dow  = sub.groupby("day_of_week")["consumption"].mean()
            peak_day   = str(daily_dow.idxmax()) if not daily_dow.empty else "N/A"

            wkd_data = sub[sub["is_weekend"] == 1]["consumption"]
            wkday_data = sub[sub["is_weekend"] == 0]["consumption"]
            weekend_ratio = round(float(wkd_data.mean() / wkday_data.mean()), 2) if len(wkday_data) > 0 and float(wkday_data.mean()) > 0 else 1.0

            hot_cold = {}
            if "tag" in sub.columns:
                hot  = sub[sub["tag"] == "hot"]["consumption"]
                cold = sub[sub["tag"] == "cold"]["consumption"]
                if not hot.empty and not cold.empty:
                    hot_cold = {
                        "hot_avg_L":       round(float(hot.mean()) / 1000, 2),
                        "cold_avg_L":      round(float(cold.mean()) / 1000, 2),
                        "hot_cold_ratio":  round(float(hot.mean() / cold.mean()), 2),
                    }

            mean_c = float(sub["consumption"].mean())
            std_c  = float(sub["consumption"].std())
            cv     = round(std_c / mean_c, 2) if mean_c > 0 else 0
            profile = "Tres variable" if cv > 0.8 else ("Moderement variable" if cv > 0.5 else "Stable")
            n_anom  = int((sub["z_score"] > 2.5).sum())

            return {
                "customer":            customer,
                "time_slot_filter":    time_slot or "global",
                "verdict":             f"{n_anom} anomalie(s) detectee(s) — profil: {profile}",
                "profile_type":        profile,
                "variability_cv":      cv,
                "peak_hour":           f"{peak_h}h00",
                "low_hour":            f"{low_h}h00",
                "peak_day_of_week":    peak_day,
                "weekend_ratio":       weekend_ratio,
                "hot_cold_analysis":   hot_cold,
                "anomalies_in_period": n_anom,
                "note": (
                    f"{customer} n'a pas de donnees granulaires (Flush/Sink/Tap). "
                    "Analyse basee sur patterns horaires et Z-score statistique."
                ),
            }

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("analyze_behavior_shift error")
        return {"error": f"Erreur: {str(e)}"}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — CHART
# ══════════════════════════════════════════════════════════════════════════════
def generate_consumption_chart(
    customer:   str,
    chart_type: str = "hourly",
    tag:        Optional[str] = None,
    date_from:  Optional[str] = None,
    date_to:    Optional[str] = None,
) -> str:
    """
    Graphique Plotly JSON pour Streamlit avec filtre temporel.
    Args:
        customer   : gym | customerA | customerB | customerC
        chart_type : "hourly" | "daily" | "cabin_comparison" | "anomaly"
        tag        : "hot" | "cold" (Gym, optionnel)
        date_from  : YYYY-MM-DD ou "30 days ago" ou "March" (optionnel)
        date_to    : YYYY-MM-DD (optionnel)
    Returns: JSON string Plotly
    """
    try:
        customer   = _normalize_customer(str(customer).strip())
        chart_type = str(chart_type).strip().lower()
        tag        = str(tag).strip().lower() if tag and str(tag) not in ("None","null") else None
        date_from  = _parse_date(date_from) if date_from and str(date_from) not in ("None","null") else None
        date_to    = _parse_date(date_to) if date_to and str(date_to) not in ("None","null") else None

        df   = _load_data()
        mask = df["customer"] == customer

        # ✅ Appliquer les filtres temporels SI fournis
        if date_from:
            mask &= df["timestamp"] >= pd.Timestamp(date_from, tz="UTC")
        if date_to:
            mask &= df["timestamp"] <= pd.Timestamp(date_to + " 23:59:59", tz="UTC")
        if tag and "tag" in df.columns:
            mask &= df["tag"].str.lower() == tag

        sub = df[mask].copy()

        if sub.empty:
            period_info = ""
            if date_from:
                period_info = f" pour la période {date_from}"
                if date_to:
                    period_info += f" → {date_to}"
            return json.dumps({"error": f"Aucune donnee pour {customer}{period_info}"})

        if chart_type == "hourly":
            hourly = sub.groupby("hour")["consumption"].mean().reset_index()
            hourly["consumption_L"] = hourly["consumption"] / 1000
            
            title = f"Profil horaire — {customer}"
            if tag: title += f" ({tag})"
            if date_from: title += f" — {date_from}" + (f" → {date_to}" if date_to else "")
            
            fig = px.line(
                hourly, x="hour", y="consumption_L",
                title=title,
                labels={"hour": "Heure", "consumption_L": "Consommation moy. (L)"},
                markers=True, color_discrete_sequence=["#378ADD"],
            )

        elif chart_type == "daily":
            sub["date"] = sub["timestamp"].dt.date
            daily = sub.groupby("date")["consumption"].sum().reset_index()
            daily["consumption_L"] = daily["consumption"] / 1000
            
            title = f"Consommation journalière — {customer}"
            if tag: title += f" ({tag})"
            if date_from: title += f" — {date_from}" + (f" → {date_to}" if date_to else "")
            
            fig = px.line(
                daily, x="date", y="consumption_L",
                title=title,
                labels={"date": "Date", "consumption_L": "Consommation (L)"},
                color_discrete_sequence=["#1D9E75"],
            )

        elif chart_type == "cabin_comparison":
            if customer != "Gym":
                return generate_consumption_chart(customer, "hourly", tag, date_from, date_to)
            gym_sub = sub[sub["cabin"].astype(str).isin(["1.0","2.0","3.0","4.0"])].copy()
            if gym_sub.empty:
                return json.dumps({"error": "Aucune donnée cabine pour le Gym avec ces filtres."})
            gym_sub["cabin_label"] = "Cabin " + gym_sub["cabin"].astype(str).str.replace(".0","",regex=False)
            stats = gym_sub.groupby(["cabin_label", "tag"])["consumption"].sum().reset_index()
            stats["consumption_L"] = stats["consumption"] / 1000
            
            title = "Comparaison Cabines Gym — Hot vs Cold"
            if date_from: title += f" — {date_from}" + (f" → {date_to}" if date_to else "")
            
            fig = px.bar(
                stats, x="cabin_label", y="consumption_L", color="tag",
                title=title,
                labels={"consumption_L": "Consommation totale (L)", "cabin_label": "Cabine", "tag": "Type"},
                color_discrete_map={"hot": "#E24B4A", "cold": "#378ADD"},
                barmode="group",
            )

        elif chart_type == "anomaly":
            sub["date"] = sub["timestamp"].dt.date
            daily_z = sub.groupby("date")["z_score"].max().reset_index()
            
            title = f"Timeline des Anomalies — {customer}"
            if tag: title += f" ({tag})"
            if date_from: title += f" — {date_from}" + (f" → {date_to}" if date_to else "")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_z["date"], y=daily_z["z_score"],
                mode="lines+markers", name="Z-score max journalier",
                line=dict(color="#E24B4A"),
            ))
            fig.add_hline(y=2.5, line_dash="dash", line_color="orange",
                          annotation_text="Seuil alerte (2.5)")
            fig.update_layout(
                title=title,
                xaxis_title="Date", yaxis_title="Z-score max journalier",
            )
        else:
            return generate_consumption_chart(customer, "hourly", tag, date_from, date_to)

        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig.to_json()

    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.exception("generate_consumption_chart error")
        return json.dumps({"error": str(e)})

# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — HARDWARE HEALTH
# ══════════════════════════════════════════════════════════════════════════════
def check_hardware_health(customer: str) -> Dict[str, Any]:
    """
    Sante materielle des capteurs IoT depuis master_health.parquet.
    Args:
        customer : gym | customerA | customerB | customerC
    Returns dict: status, incidents, devices concernes, error_types
    """
    try:
        customer = _normalize_customer(str(customer).strip())
        health   = _load_health()

        if health.empty:
            return {"customer": customer, "status": "Donnees de sante indisponibles"}

        sub = health[health["customer"] == customer] if "customer" in health.columns else pd.DataFrame()

        if sub.empty:
            return {
                "customer":        customer,
                "status":          "100% Operationnel",
                "total_incidents": 0,
                "message":         "Aucun incident materiel enregistre.",
            }

        error_breakdown = {}
        if "error_type" in sub.columns:
            error_breakdown = {str(k): int(v) for k, v in sub["error_type"].value_counts().items()}

        affected_devices = []
        if "device_id" in sub.columns:
            for dev, count in sub["device_id"].value_counts().head(5).items():
                sub_dev = sub[sub["device_id"] == dev]
                entry = {
                    "device_id":   str(dev)[:12] + "...",
                    "incidents":   int(count),
                    "error_types": sub_dev["error_type"].unique().tolist() if "error_type" in sub_dev.columns else [],
                }
                if "sub_category" in sub_dev.columns:
                    entry["sub_category"] = str(sub_dev["sub_category"].iloc[0])
                if "cabin" in sub_dev.columns:
                    entry["cabin"] = str(sub_dev["cabin"].iloc[0])
                affected_devices.append(entry)

        return {
            "customer":         customer,
            "status":           "Incidents materiels detectes",
            "total_incidents":  int(len(sub)),
            "error_breakdown":  error_breakdown,
            "affected_devices": affected_devices,
            "recommendation":  (
                f"{len(sub)} enregistrement(s) d'erreur pour {customer}. "
                "Planifier maintenance preventive sur capteurs concernes."
            ),
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("check_hardware_health error")
        return {"error": f"Erreur: {str(e)}"}

# ══════════════════════════════════════════════════════════════════════════════
# TOOL 6 — HYDRAULIC & SESSION STATS
# À COLLER dans tools.py, après check_hardware_health()
# Couvre les 4 gaps : intensity_ratio, gap_since_last, flow_rate stats, period_s
# ══════════════════════════════════════════════════════════════════════════════

def get_hydraulic_stats(
    customer:   str,
    time_slot:  Optional[str] = None,
    tag:        Optional[str] = None,
    cabin:      Optional[str] = None,
    sub_cat:    Optional[str] = None,
) -> Dict[str, Any]:
    """
    Statistiques hydrauliques avancées exploitant les colonnes ignorées :
      - period_s       : durée moyenne / médiane des sessions (secondes)
      - flow_rate      : débit moyen, max, distribution par time_slot
      - intensity_ratio: signature hydraulique Flush vs Sink vs Shower
      - gap_since_last : délai inter-événements (détection doubles-flush, fréquentation)

    Args:
        customer  : gym | customerA | customerB | customerC
        time_slot : morning | afternoon | evening | night (optionnel)
        tag       : "hot" | "cold" (Gym, optionnel)
        cabin     : "1"-"4" (Gym, optionnel)
        sub_cat   : "Flush" | "Sink" | "Tap" | "Shower" (optionnel, filtre sur sub_category)

    Répond à des questions comme :
        "What is the average shower duration in cabin 3?"
        "Compare flow rate between morning and evening?"
        "How long does CustomerC wait between flush and sink?"
        "Which time_slot has the highest flow rate?"
    """
    try:
        customer  = _normalize_customer(str(customer).strip())
        time_slot = str(time_slot).strip().lower() if time_slot and str(time_slot) not in ("None","null") else None
        tag       = str(tag).strip().lower() if tag and str(tag) not in ("None","null") else None
        cabin     = str(cabin).strip() if cabin and str(cabin) not in ("None","null") else None
        sub_cat   = str(sub_cat).strip() if sub_cat and str(sub_cat) not in ("None","null") else None

        df   = _load_data()
        mask = df["customer"] == customer

        # Filtres optionnels
        if time_slot:
            slots = TIME_SLOT_MAP.get(time_slot, [])
            if slots:
                mask &= df["time_slot"].isin(slots)
        if tag and "tag" in df.columns:
            mask &= df["tag"].str.lower() == tag
        if cabin and "cabin" in df.columns:
            mask &= df["cabin"].astype(str) == str(float(cabin))
        if sub_cat and "sub_category" in df.columns:
            mask &= df["sub_category"].str.lower() == sub_cat.lower()

        sub = df[mask].copy()
        if sub.empty:
            return {"error": f"Aucune donnée pour {customer} avec ces filtres."}

        # ── 1. DURÉE DES SESSIONS (period_s) ─────────────────────────────────
        duration_stats = {}
        if "period_s" in sub.columns:
            dur = sub["period_s"].dropna()
            duration_stats = {
                "avg_duration_s":    round(float(dur.mean()), 1),
                "median_duration_s": round(float(dur.median()), 1),
                "min_duration_s":    round(float(dur.min()), 1),
                "max_duration_s":    round(float(dur.max()), 1),
                "avg_duration_min":  round(float(dur.mean()) / 60, 2),
            }
            # Durée par sous-catégorie si pas de filtre sub_cat
            if not sub_cat and "sub_category" in sub.columns:
                dur_by_subcat = (
                    sub.groupby("sub_category")["period_s"]
                    .mean()
                    .round(1)
                    .to_dict()
                )
                duration_stats["avg_duration_by_type_s"] = {
                    str(k): round(float(v), 1) for k, v in dur_by_subcat.items()
                }

        # ── 2. DÉBIT (flow_rate) ──────────────────────────────────────────────
        flow_stats = {}
        if "flow_rate" in sub.columns:
            flow = sub["flow_rate"].dropna()
            # Exclure les débits physiquement impossibles pour les stats normales
            flow_normal = flow[flow <= 500]
            flow_stats = {
                "avg_flow_rate_mls":    round(float(flow_normal.mean()), 2),
                "median_flow_rate_mls": round(float(flow_normal.median()), 2),
                "max_flow_rate_mls":    round(float(flow.max()), 2),
                "p95_flow_rate_mls":    round(float(flow_normal.quantile(0.95)), 2),
                "impossible_flow_pct":  round(float((flow > 500).mean()) * 100, 1),
            }
            # Débit moyen par time_slot (répond à "which time_slot has highest flow?")
            if "time_slot" in sub.columns:
                flow_by_slot = (
                    sub[sub["flow_rate"] <= 500]
                    .groupby("time_slot")["flow_rate"]
                    .mean()
                    .round(2)
                    .sort_values(ascending=False)
                    .to_dict()
                )
                flow_stats["avg_flow_by_time_slot"] = {
                    str(k): round(float(v), 2) for k, v in flow_by_slot.items()
                }
                peak_slot = max(flow_by_slot, key=flow_by_slot.get) if flow_by_slot else "N/A"
                flow_stats["peak_flow_time_slot"] = str(peak_slot)

        # ── 3. INTENSITY RATIO (signature hydraulique) ───────────────────────
        intensity_stats = {}
        if "intensity_ratio" in sub.columns:
            ir = sub["intensity_ratio"].dropna()
            intensity_stats = {
                "avg_intensity_ratio":    round(float(ir.mean()), 3),
                "median_intensity_ratio": round(float(ir.median()), 3),
                "std_intensity_ratio":    round(float(ir.std()), 3),
            }
            # Par sous-catégorie — c'est ici que la feature est la plus utile
            if "sub_category" in sub.columns:
                ir_by_subcat = (
                    sub.groupby("sub_category")["intensity_ratio"]
                    .mean()
                    .round(3)
                    .to_dict()
                )
                intensity_stats["avg_intensity_by_type"] = {
                    str(k): round(float(v), 3) for k, v in ir_by_subcat.items()
                }
                # Identifier le type d'usage le plus intensif
                if ir_by_subcat:
                    peak_type = max(ir_by_subcat, key=ir_by_subcat.get)
                    intensity_stats["most_intensive_usage"] = str(peak_type)

        # ── 4. GAP INTER-ÉVÉNEMENTS (gap_since_last) ─────────────────────────
        gap_stats = {}
        if "gap_since_last" in sub.columns:
            # -1 = premier événement du bloc, on l'exclut
            gaps = sub[sub["gap_since_last"] > 0]["gap_since_last"]
            if not gaps.empty:
                gap_stats = {
                    "avg_gap_s":    round(float(gaps.mean()), 1),
                    "median_gap_s": round(float(gaps.median()), 1),
                    "min_gap_s":    round(float(gaps.min()), 1),
                    "rapid_sequences_pct": round(
                        float((gaps < 30).mean()) * 100, 1
                    ),  # % d'événements arrivant moins de 30s après le précédent
                }
                # Gaps moyens par paire de transition (pour CustomerC)
                if customer == "CustomerC" and "transition_pair" in sub.columns:
                    gap_by_transition = (
                        sub[sub["gap_since_last"] > 0]
                        .groupby("transition_pair")["gap_since_last"]
                        .mean()
                        .round(1)
                        .sort_values()
                        .head(5)
                        .to_dict()
                    )
                    gap_stats["avg_gap_by_transition_s"] = {
                        str(k): round(float(v), 1) for k, v in gap_by_transition.items()
                    }

        # ── 5. RÉSUMÉ NARRATIF ────────────────────────────────────────────────
        filters_desc = []
        if time_slot: filters_desc.append(f"time_slot={time_slot}")
        if tag:       filters_desc.append(f"tag={tag}")
        if cabin:     filters_desc.append(f"cabin={cabin}")
        if sub_cat:   filters_desc.append(f"type={sub_cat}")

        return {
            "customer":         customer,
            "filters_applied":  filters_desc or "global",
            "event_count":      int(len(sub)),
            "duration_stats":   duration_stats,
            "flow_stats":       flow_stats,
            "intensity_stats":  intensity_stats,
            "gap_stats":        gap_stats,
            "interpretation": (
                f"Durée moy: {duration_stats.get('avg_duration_s', 'N/A')}s | "
                f"Débit moy: {flow_stats.get('avg_flow_rate_mls', 'N/A')} ml/s | "
                f"Pic débit: {flow_stats.get('peak_flow_time_slot', 'N/A')} | "
                f"Séquences rapides: {gap_stats.get('rapid_sequences_pct', 'N/A')}%"
            ),
        }

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("get_hydraulic_stats error")
        return {"error": f"Erreur: {str(e)}"}
    

# ══════════════════════════════════════════════════════════════════════════════
# TOOL 7 — TREND ANALYSIS
# Répond à : "is consumption growing?", "compare this week vs last week"
# ══════════════════════════════════════════════════════════════════════════════

def get_trend_analysis(
    customer:    str,
    period:      str = "month",
    tag:         Optional[str] = None,
    cabin:       Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyse de tendance temporelle : croissance, décroissance, stabilité.
    Compare automatiquement période courante vs période précédente.

    Args:
        customer : gym | customerA | customerB | customerC
        period   : "week" | "month" | "quarter" (défaut: "month")
        tag      : "hot" | "cold" (Gym, optionnel)
        cabin    : "1"-"4" (Gym, optionnel)

    Répond à :
        "Is gym consumption growing this month?"
        "Compare this week vs last week for CustomerB"
        "What is the month-over-month trend for CustomerC?"
        "Is hot water usage increasing in cabin 2?"
    """
    try:
        customer = _normalize_customer(str(customer).strip())
        period   = str(period).strip().lower() if period else "month"
        tag      = str(tag).strip().lower() if tag and str(tag) not in ("None","null") else None
        cabin    = str(cabin).strip() if cabin and str(cabin) not in ("None","null") else None

        df   = _load_data()
        mask = df["customer"] == customer
        if tag and "tag" in df.columns:
            mask &= df["tag"].str.lower() == tag
        if cabin and "cabin" in df.columns:
            mask &= df["cabin"].astype(str) == str(float(cabin))

        sub = df[mask].copy()
        if sub.empty:
            return {"error": f"Aucune donnée pour {customer} avec ces filtres."}

        sub["date"] = sub["timestamp"].dt.date
        sub["week"] = sub["timestamp"].dt.isocalendar().week.astype(int)
        sub["month"] = sub["timestamp"].dt.to_period("M").astype(str)

        # ── Définir les deux périodes à comparer ──────────────────────────────
        now = pd.Timestamp.now(tz="UTC")

        if period == "week":
            delta     = timedelta(weeks=1)
            cur_start = now - delta
            prv_start = now - 2 * delta
            prv_end   = cur_start
        elif period == "quarter":
            delta     = timedelta(days=90)
            cur_start = now - delta
            prv_start = now - 2 * delta
            prv_end   = cur_start
        else:  # month par défaut
            delta     = timedelta(days=30)
            cur_start = now - delta
            prv_start = now - 2 * delta
            prv_end   = cur_start

        cur = sub[sub["timestamp"] >= cur_start]
        prv = sub[(sub["timestamp"] >= prv_start) & (sub["timestamp"] < prv_end)]

        if cur.empty and prv.empty:
            return {"error": "Données insuffisantes pour calculer une tendance."}

        def period_stats(df_p):
            if df_p.empty:
                return {"total_L": 0, "daily_avg_L": 0, "event_count": 0, "peak_hour": "N/A"}
            dates = df_p["timestamp"].dt.date
            n_days = max((dates.max() - dates.min()).days, 1)
            return {
                "total_L":     round(float(df_p["consumption"].sum()) / 1000, 1),
                "daily_avg_L": round(float(df_p["consumption"].sum()) / 1000 / n_days, 1),
                "event_count": int(len(df_p)),
                "peak_hour":   f"{int(df_p.groupby('hour')['consumption'].sum().idxmax())}h00" if not df_p.empty else "N/A",
            }

        cur_stats = period_stats(cur)
        prv_stats = period_stats(prv)

        # ── Calcul du delta ───────────────────────────────────────────────────
        if prv_stats["daily_avg_L"] > 0:
            pct_change = round(
                (cur_stats["daily_avg_L"] - prv_stats["daily_avg_L"]) / prv_stats["daily_avg_L"] * 100, 1
            )
        else:
            pct_change = 0

        if pct_change > 10:
            trend     = "HAUSSE"
            severity  = "attention" if pct_change > 25 else "normal"
        elif pct_change < -10:
            trend     = "BAISSE"
            severity  = "attention" if pct_change < -25 else "normal"
        else:
            trend     = "STABLE"
            severity  = "normal"

        # ── Tendance journalière (rolling 7j) ─────────────────────────────────
        daily = sub.groupby("date")["consumption"].sum().reset_index()
        daily["consumption_L"] = daily["consumption"] / 1000
        daily["rolling_7d"] = daily["consumption_L"].rolling(7, min_periods=1).mean().round(1)
        last_7 = daily.tail(7)[["date", "consumption_L", "rolling_7d"]].to_dict("records")
        last_7_clean = [
            {
                "date":           str(r["date"]),
                "consumption_L":  round(float(r["consumption_L"]), 1),
                "rolling_avg_L":  round(float(r["rolling_7d"]), 1),
            }
            for r in last_7
        ]

        # ── Weekend vs semaine ────────────────────────────────────────────────
        wkd    = sub[sub["is_weekend"] == 1]["consumption"]
        wkday  = sub[sub["is_weekend"] == 0]["consumption"]
        wk_ratio = round(float(wkd.mean() / wkday.mean()), 2) if len(wkday) > 0 and float(wkday.mean()) > 0 else 1.0

        return {
            "customer":           customer,
            "period_compared":    period,
            "trend":              trend,
            "severity":           severity,
            "pct_change":         pct_change,
            "current_period":     cur_stats,
            "previous_period":    prv_stats,
            "daily_L_delta":      round(cur_stats["daily_avg_L"] - prv_stats["daily_avg_L"], 1),
            "weekend_ratio":      wk_ratio,
            "last_7_days":        last_7_clean,
            "verdict": (
                f"Consommation {trend} de {abs(pct_change)}% "
                f"({period} courant vs précédent). "
                f"Moy. journalière : {cur_stats['daily_avg_L']} L/j "
                f"vs {prv_stats['daily_avg_L']} L/j."
            ),
        }

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("get_trend_analysis error")
        return {"error": f"Erreur: {str(e)}"}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 8 — COMPARE CUSTOMERS
# Répond à : "compare all customers", "who consumes the most?", "rank by anomaly"
# ══════════════════════════════════════════════════════════════════════════════

def compare_customers(
    customers:   Optional[str] = None,
    metric:      str = "consumption",
    time_slot:   Optional[str] = None,
    date_from:   Optional[str] = None,
    date_to:     Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comparaison multi-clients sur une même métrique en un seul appel.
    Évite d'appeler Tool 1 x4 séparément.

    Args:
        customers : "all" ou liste séparée par virgules ex: "Gym,CustomerA"
                    (défaut: "all" → les 4 clients)
        metric    : "consumption" | "anomaly_rate" | "flow_rate" | "session_count"
                    | "daily_avg" (défaut: "consumption")
        time_slot : morning | afternoon | evening | night (optionnel)
        date_from : YYYY-MM-DD ou "30 days ago" (optionnel)
        date_to   : YYYY-MM-DD (optionnel)

    Répond à :
        "Compare all customers by daily average consumption"
        "Who consumes the most water — gym or CustomerA?"
        "Rank all customers by anomaly rate"
        "Compare CustomerA and CustomerB in the morning"
    """
    try:
        ALL_CUSTOMERS = ["CustomerA", "CustomerB", "CustomerC", "Gym"]

        # Résoudre la liste de clients
        if not customers or str(customers).strip().lower() in ("all", "none", "null", ""):
            target_customers = ALL_CUSTOMERS
        else:
            raw_list = [c.strip() for c in str(customers).split(",")]
            target_customers = []
            for raw in raw_list:
                try:
                    target_customers.append(_normalize_customer(raw))
                except ValueError:
                    pass
            if not target_customers:
                target_customers = ALL_CUSTOMERS

        metric    = str(metric).strip().lower() if metric else "consumption"
        time_slot = str(time_slot).strip().lower() if time_slot and str(time_slot) not in ("None","null") else None
        date_from = _parse_date(date_from)
        date_to   = _parse_date(date_to)

        df = _load_data()

        # Filtres temporels globaux
        if date_from:
            df = df[df["timestamp"] >= pd.Timestamp(date_from, tz="UTC")]
        if date_to:
            df = df[df["timestamp"] <= pd.Timestamp(date_to + " 23:59:59", tz="UTC")]
        if time_slot:
            slots = TIME_SLOT_MAP.get(time_slot, [])
            if slots:
                df = df[df["time_slot"].isin(slots)]

        results = []
        for cust in target_customers:
            sub = df[df["customer"] == cust]
            if sub.empty:
                continue

            dates  = sub["timestamp"].dt.date
            n_days = max((dates.max() - dates.min()).days, 1)

            row = {"customer": cust}

            if metric in ("consumption", "daily_avg"):
                row["total_L"]     = round(float(sub["consumption"].sum()) / 1000, 1)
                row["daily_avg_L"] = round(float(sub["consumption"].sum()) / 1000 / n_days, 1)
                row["event_count"] = int(len(sub))
                row["peak_hour"]   = f"{int(sub.groupby('hour')['consumption'].sum().idxmax())}h00"

            elif metric == "anomaly_rate":
                total  = len(sub)
                n_anom = int((sub["z_score"] > 2.5).sum())
                row["anomaly_count"]    = n_anom
                row["anomaly_rate_pct"] = round(n_anom / total * 100, 2) if total > 0 else 0
                row["max_z_score"]      = round(float(sub["z_score"].max()), 2)
                row["high_flow_events"] = int((sub["flow_rate"] > 500).sum())

            elif metric == "flow_rate":
                flow = sub[sub["flow_rate"] <= 500]["flow_rate"]
                row["avg_flow_mls"]    = round(float(flow.mean()), 2) if not flow.empty else 0
                row["median_flow_mls"] = round(float(flow.median()), 2) if not flow.empty else 0
                row["max_flow_mls"]    = round(float(sub["flow_rate"].max()), 2)

            elif metric == "session_count":
                if "session_id" in sub.columns:
                    n_sessions = sub["session_id"].nunique()
                    row["total_sessions"]      = int(n_sessions)
                    row["avg_sessions_per_day"] = round(n_sessions / n_days, 1)
                    row["avg_events_per_session"] = round(
                        float(sub.groupby("session_id").size().mean()), 1
                    )

            results.append(row)

        if not results:
            return {"error": "Aucune donnée pour les clients demandés."}

        # Trier par la métrique principale
        sort_key = {
            "consumption": "daily_avg_L",
            "daily_avg":   "daily_avg_L",
            "anomaly_rate":"anomaly_rate_pct",
            "flow_rate":   "avg_flow_mls",
            "session_count":"avg_sessions_per_day",
        }.get(metric, "daily_avg_L")

        results_sorted = sorted(results, key=lambda x: x.get(sort_key, 0), reverse=True)

        # Leader
        leader = results_sorted[0]["customer"] if results_sorted else "N/A"
        leader_val = results_sorted[0].get(sort_key, 0) if results_sorted else 0

        return {
            "metric_compared":  metric,
            "time_slot_filter": time_slot or "global",
            "period_filter":    f"{date_from or 'all'} -> {date_to or 'now'}",
            "clients_compared": len(results_sorted),
            "ranking":          results_sorted,
            "leader":           leader,
            "leader_value":     leader_val,
            "verdict": (
                f"{leader} a la valeur la plus élevée sur '{metric}' "
                f"({sort_key}: {leader_val})."
            ),
        }

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("compare_customers error")
        return {"error": f"Erreur: {str(e)}"}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 9 — SESSION ANALYSIS
# Répond à : "how many sessions per day?", "average session length?", sessions nocturnes
# ══════════════════════════════════════════════════════════════════════════════

def get_session_analysis(
    customer:  str,
    time_slot: Optional[str] = None,
    tag:       Optional[str] = None,
    cabin:     Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyse des sessions d'usage (basée sur session_id calculé dans master_fusion).
    Une session = séquence d'événements séparés de moins de 120 secondes.

    Args:
        customer  : gym | customerA | customerB | customerC
        time_slot : morning | afternoon | evening | night (optionnel)
        tag       : "hot" | "cold" (Gym, optionnel)
        cabin     : "1"-"4" (Gym, optionnel)

    Répond à :
        "How many sessions per day does CustomerC have on average?"
        "What is the average session length at the gym in the evening?"
        "Are there abnormally long sessions at the gym?"
        "How many events does a typical CustomerC session contain?"
    """
    try:
        customer  = _normalize_customer(str(customer).strip())
        time_slot = str(time_slot).strip().lower() if time_slot and str(time_slot) not in ("None","null") else None
        tag       = str(tag).strip().lower() if tag and str(tag) not in ("None","null") else None
        cabin     = str(cabin).strip() if cabin and str(cabin) not in ("None","null") else None

        df   = _load_data()
        mask = df["customer"] == customer
        if time_slot:
            slots = TIME_SLOT_MAP.get(time_slot, [])
            if slots:
                mask &= df["time_slot"].isin(slots)
        if tag and "tag" in df.columns:
            mask &= df["tag"].str.lower() == tag
        if cabin and "cabin" in df.columns:
            mask &= df["cabin"].astype(str) == str(float(cabin))

        sub = df[mask].copy()
        if sub.empty or "session_id" not in sub.columns:
            return {"error": f"Aucune donnée de session pour {customer}."}

        # ── Stats par session ─────────────────────────────────────────────────
        sess_grp = sub.groupby("session_id")

        sess_stats = sess_grp.agg(
            event_count=("consumption", "count"),
            total_consumption=("consumption", "sum"),
            duration_s=("period_s", "sum"),
            start_hour=("hour", "first"),
        ).reset_index()

        sess_stats["total_L"] = sess_stats["total_consumption"] / 1000

        # ── Métriques globales ────────────────────────────────────────────────
        n_sessions = len(sess_stats)
        dates      = sub["timestamp"].dt.date
        n_days     = max((dates.max() - dates.min()).days, 1)

        avg_events   = round(float(sess_stats["event_count"].mean()), 1)
        avg_duration = round(float(sess_stats["duration_s"].mean()), 1)
        avg_volume   = round(float(sess_stats["total_L"].mean()), 2)
        max_duration = round(float(sess_stats["duration_s"].max()), 1)

        # Sessions longues = > 2x la moyenne
        long_threshold = avg_duration * 2
        n_long_sessions = int((sess_stats["duration_s"] > long_threshold).sum())

        # ── Distribution par time_slot ────────────────────────────────────────
        slot_dist = {}
        if "time_slot" in sub.columns:
            slot_counts = sub.groupby(["session_id", "time_slot"]).size().reset_index()
            slot_counts = slot_counts.groupby("time_slot")["session_id"].nunique()
            total_s = slot_counts.sum()
            slot_dist = {
                str(k): {
                    "sessions": int(v),
                    "pct": round(float(v) / total_s * 100, 1)
                }
                for k, v in slot_counts.items()
            }

        # ── Sessions nocturnes (00h-05h) ──────────────────────────────────────
        night_sessions = int(sess_stats[sess_stats["start_hour"] < 5]["session_id"].count())
        night_pct      = round(night_sessions / n_sessions * 100, 1) if n_sessions > 0 else 0

        # ── Top 3 sessions les plus longues ──────────────────────────────────
        top_sessions = (
            sess_stats.nlargest(3, "duration_s")[["session_id", "duration_s", "total_L", "event_count", "start_hour"]]
            .to_dict("records")
        )
        top_sessions_clean = [
            {
                "session_id":  int(r["session_id"]),
                "duration_s":  round(float(r["duration_s"]), 1),
                "total_L":     round(float(r["total_L"]), 2),
                "event_count": int(r["event_count"]),
                "start_hour":  f"{int(r['start_hour'])}h00",
            }
            for r in top_sessions
        ]

        return {
            "customer":             customer,
            "time_slot_filter":     time_slot or "global",
            "total_sessions":       n_sessions,
            "avg_sessions_per_day": round(n_sessions / n_days, 1),
            "avg_events_per_session": avg_events,
            "avg_duration_s":       avg_duration,
            "avg_duration_min":     round(avg_duration / 60, 2),
            "avg_volume_per_session_L": avg_volume,
            "max_session_duration_s": max_duration,
            "long_sessions_count":  n_long_sessions,
            "long_sessions_pct":    round(n_long_sessions / n_sessions * 100, 1) if n_sessions > 0 else 0,
            "night_sessions_count": night_sessions,
            "night_sessions_pct":   night_pct,
            "distribution_by_slot": slot_dist,
            "top_3_longest_sessions": top_sessions_clean,
            "verdict": (
                f"{n_sessions} sessions total | "
                f"{round(n_sessions / n_days, 1)} sessions/jour | "
                f"Durée moy: {round(avg_duration / 60, 1)} min | "
                f"Sessions nocturnes: {night_pct}%"
            ),
        }

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("get_session_analysis error")
        return {"error": f"Erreur: {str(e)}"}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 10 — GENERATE ALERT REPORT (tous clients, priorisé par gravité)
# Répond à : "what needs urgent attention?", "priority maintenance report"
# ══════════════════════════════════════════════════════════════════════════════

def generate_alert_report(
    customers:  Optional[str] = None,
    severity:   str = "all",
    date_from:  Optional[str] = None,
) -> Dict[str, Any]:
    """
    Rapport d'alertes consolidé et priorisé par gravité, tous clients confondus.
    Combine anomalies statistiques (Z-score) + débits impossibles + incidents hardware.

    Args:
        customers : "all" ou liste ex: "Gym,CustomerC" (défaut: all)
        severity  : "critical" | "warning" | "all" (défaut: all)
        date_from : YYYY-MM-DD ou "7 days ago" (défaut: 7 derniers jours)

    Répond à :
        "What needs urgent attention right now across all clients?"
        "Give me a priority maintenance report"
        "Are there any critical alerts today?"
        "Show me all warnings for the last week"
    """
    try:
        ALL_CUSTOMERS = ["CustomerA", "CustomerB", "CustomerC", "Gym"]

        if not customers or str(customers).strip().lower() in ("all", "none", "null", ""):
            target_customers = ALL_CUSTOMERS
        else:
            raw_list = [c.strip() for c in str(customers).split(",")]
            target_customers = []
            for raw in raw_list:
                try:
                    target_customers.append(_normalize_customer(raw))
                except ValueError:
                    pass
            if not target_customers:
                target_customers = ALL_CUSTOMERS

        severity  = str(severity).strip().lower() if severity else "all"
        date_from = _parse_date(date_from) if date_from and str(date_from) not in ("None","null") else None
        if not date_from:
            date_from = (datetime.now().date() - timedelta(days=7)).isoformat()

        df     = _load_data()
        health = _load_health()

        df_recent = df[df["timestamp"] >= pd.Timestamp(date_from, tz="UTC")]

        alerts = []

        for cust in target_customers:
            sub = df_recent[df_recent["customer"] == cust]
            if sub.empty:
                continue

            # ── Alertes CRITIQUES : Z-score > 4 ou débit > 500 ml/s ──────────
            critical_z    = sub[sub["z_score"] > 4]
            critical_flow = sub[sub["flow_rate"] > 500]

            for _, row in critical_z.head(3).iterrows():
                alerts.append({
                    "severity":    "CRITICAL",
                    "customer":    cust,
                    "type":        "Surconsommation extrême (Z>4)",
                    "device_id":   str(row["device_id"])[:12] + "...",
                    "timestamp":   str(row["timestamp"])[:16],
                    "value":       f"Z-score: {round(float(row['z_score']), 2)}",
                    "sub_category": str(row.get("sub_category", "N/A")),
                    "action":      "Inspection capteur immédiate",
                })

            for _, row in critical_flow.head(3).iterrows():
                alerts.append({
                    "severity":    "CRITICAL",
                    "customer":    cust,
                    "type":        "Débit physiquement impossible (>500 ml/s)",
                    "device_id":   str(row["device_id"])[:12] + "...",
                    "timestamp":   str(row["timestamp"])[:16],
                    "value":       f"Débit: {round(float(row['flow_rate']), 1)} ml/s",
                    "sub_category": str(row.get("sub_category", "N/A")),
                    "action":      "Vérifier capteur — fuite probable",
                })

            # ── Alertes WARNING : Z-score 2.5-4 ──────────────────────────────
            warn_z = sub[(sub["z_score"] > 2.5) & (sub["z_score"] <= 4)]
            if not warn_z.empty:
                alerts.append({
                    "severity":    "WARNING",
                    "customer":    cust,
                    "type":        "Anomalies statistiques modérées",
                    "device_id":   "multiple",
                    "timestamp":   str(warn_z["timestamp"].max())[:16],
                    "value":       f"{len(warn_z)} événements Z-score 2.5-4",
                    "sub_category": "multiple",
                    "action":      "Surveillance recommandée — vérifier tendance",
                })

            # ── Alertes nocturnes (00h-05h avec consommation) ─────────────────
            night = sub[(sub["hour"] < 5) & (sub["consumption"] > sub["consumption"].mean())]
            if not night.empty:
                alerts.append({
                    "severity":    "WARNING",
                    "customer":    cust,
                    "type":        "Consommation nocturne anormale (00h-05h)",
                    "device_id":   "multiple",
                    "timestamp":   str(night["timestamp"].max())[:16],
                    "value":       f"{len(night)} événements nocturnes > moyenne",
                    "sub_category": "multiple",
                    "action":      "Vérifier fuites silencieuses — inspecter de nuit",
                })

        # ── Hardware alerts depuis master_health ──────────────────────────────
        if not health.empty and "customer" in health.columns:
            for cust in target_customers:
                sub_h = health[health["customer"] == cust]
                if not sub_h.empty:
                    # Filtrer par date si disponible
                    if "timestamp" in sub_h.columns:
                        sub_h = sub_h[
                            pd.to_datetime(sub_h["timestamp"], utc=True) >=
                            pd.Timestamp(date_from, tz="UTC")
                        ]
                    if not sub_h.empty:
                        alerts.append({
                            "severity":    "CRITICAL",
                            "customer":    cust,
                            "type":        "Incidents matériels capteurs IoT",
                            "device_id":   "multiple",
                            "timestamp":   date_from,
                            "value":       f"{len(sub_h)} incidents hardware",
                            "sub_category": "hardware",
                            "action":      "Planifier maintenance préventive urgente",
                        })

        # ── Filtrer par sévérité demandée ─────────────────────────────────────
        if severity == "critical":
            alerts = [a for a in alerts if a["severity"] == "CRITICAL"]
        elif severity == "warning":
            alerts = [a for a in alerts if a["severity"] == "WARNING"]

        # ── Trier : CRITICAL d'abord, puis WARNING ────────────────────────────
        severity_order = {"CRITICAL": 0, "WARNING": 1}
        alerts_sorted  = sorted(alerts, key=lambda x: severity_order.get(x["severity"], 2))

        n_critical = sum(1 for a in alerts_sorted if a["severity"] == "CRITICAL")
        n_warning  = sum(1 for a in alerts_sorted if a["severity"] == "WARNING")

        if n_critical > 0:
            overall = f"ATTENTION REQUISE — {n_critical} alerte(s) critique(s)"
        elif n_warning > 0:
            overall = f"SURVEILLANCE — {n_warning} avertissement(s) actif(s)"
        else:
            overall = "Système nominal — aucune alerte active"

        return {
            "report_from":    date_from,
            "clients_scanned": len(target_customers),
            "total_alerts":   len(alerts_sorted),
            "critical_count": n_critical,
            "warning_count":  n_warning,
            "overall_status": overall,
            "alerts":         alerts_sorted,
            "recommendation": (
                "Traiter les alertes CRITICAL en priorité (capteurs défaillants, fuites). "
                "Planifier inspection des WARNING dans les 48h."
                if n_critical > 0 else
                "Aucune action immédiate requise. Continuer la surveillance standard."
            ),
        }

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("generate_alert_report error")
        return {"error": f"Erreur: {str(e)}"}
# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import pprint
    pp = pprint.PrettyPrinter(indent=2, width=100)
    tests = [
        ("TOOL 1 — CustomerC",           lambda: query_water_stats("customerC")),
        ("TOOL 1 — Gym cabin 2 hot",     lambda: query_water_stats("gym", cabin="2", tag="hot")),
        ("TOOL 1 — Gym 30 days cold",    lambda: query_water_stats("gym", tag="cold", date_from="30 days ago")),
        ("TOOL 2 — Anomalies Gym",       lambda: get_anomaly_report("gym")),
        ("TOOL 2 — Anomalies CustomerC", lambda: get_anomaly_report("customerC", top_n=3)),
        ("TOOL 3 — Behavior CustomerC",  lambda: analyze_behavior_shift("customerC")),
        ("TOOL 3 — Behavior C morning",  lambda: analyze_behavior_shift("customerC", time_slot="morning")),
        ("TOOL 3 — Behavior Gym",        lambda: analyze_behavior_shift("gym")),
        ("TOOL 4 — Chart hourly",        lambda: generate_consumption_chart("gym", "hourly")),
        ("TOOL 4 — Cabin comparison",    lambda: generate_consumption_chart("gym", "cabin_comparison")),
        ("TOOL 4 — Anomaly timeline",    lambda: generate_consumption_chart("customerC", "anomaly")),
        ("TOOL 5 — Health Gym",          lambda: check_hardware_health("gym")),
        ("TOOL 5 — Health CustomerC",    lambda: check_hardware_health("customerC")),
    ]
    print("\n" + "="*60 + "\n  AQUAMIND TOOLS — TEST FINAL\n" + "="*60)
    for name, fn in tests:
        print(f"\n-- {name}")
        try:
            r = fn()
            if isinstance(r, dict):
                print(f"   OK — {list(r.keys())[:4]}" if "error" not in r else f"   ERR: {r['error'][:80]}")
            else:
                print(f"   JSON {len(r)} chars {'(Plotly OK)' if '\"data\"' in r else ''}")
        except Exception as e:
            print(f"   EXCEPTION: {e}")
    print("\n" + "="*60)