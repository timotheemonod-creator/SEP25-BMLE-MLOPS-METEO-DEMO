from __future__ import annotations
import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="MLOps Meteo - Soutenance",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _apply_premium_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg: #f4f7fb;
  --surface: #ffffff;
  --surface-soft: #f9fbff;
  --ink: #132238;
  --muted: #5a6b82;
  --line: #d7e1ee;
  --brand: #0f4c81;
  --brand-soft: #e7f1fb;
  --accent: #0fa3b1;
  --ok: #2d9d78;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 360px at 95% -10%, #d9ecff 0%, rgba(217,236,255,0) 65%),
    radial-gradient(900px 280px at 0% -20%, #d8f6f2 0%, rgba(216,246,242,0) 60%),
    var(--bg);
}

header[data-testid="stHeader"] {
  background-color: #f4f7fb;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #eef4fb 0%, #f5f8fc 60%, #f8fbff 100%);
  border-right: 1px solid var(--line);
}





html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  color: var(--ink);
}

h1, h2, h3, h4 {
  font-family: "Space Grotesk", "IBM Plex Sans", "Segoe UI", sans-serif;
  color: #0f2741 !important;
  letter-spacing: 0.2px;
}
.hero {
  background: linear-gradient(135deg, #0f4c81 0%, #14639f 65%, #1e7fb8 100%);
  color: #f8fbff;
  border-radius: 18px;
  padding: 1.15rem 1.25rem 1.05rem 1.25rem;
  border: 1px solid rgba(255,255,255,0.18);
  box-shadow: 0 16px 35px rgba(16,40,69,0.18);
}

.hero h2 {
  color: #ffffff !important;
  font-weight: 800 !important;
}

.hero p {
  margin: 0.45rem 0 0 0;
  color: #eaf4ff;
}

.panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 0.95rem 1rem;
  box-shadow: 0 6px 14px rgba(26,47,74,0.06);
}

.panel h4 {
  margin: 0 0 0.45rem 0;
  color: #103a63;
}

.schema-box pre {
  background: #0e2136 !important;
  color: #d9e8f7 !important;
  border-radius: 12px !important;
  border: 1px solid #1f4367 !important;
}

div[data-testid="metric-container"] {
  background: var(--surface-soft);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 0.4rem 0.7rem;
  box-shadow: 0 3px 10px rgba(18,41,68,0.05);
}

div[data-testid="metric-container"] label {
  color: var(--muted) !important;
}

.badge-line {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.25rem;
}

.badge {
  display: inline-block;
  border: 1px solid #b8d4ea;
  border-radius: 999px;
  padding: 0.18rem 0.55rem;
  font-size: 0.78rem;
  color: #0f4c81;
  background: #eff7ff;
}
.slide-title { font-size: 1.4rem; margin-bottom: 0.3rem; color: #0f4c81; }
.slide-sub { color: var(--muted); font-size: 1rem; }
.card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.04);
}
.metric-big { font-size: 1.5rem; font-weight: 700; color: #0f4c81; }
.muted { color: var(--muted); }
.accent { color: #0f4c81; }
</style>
""",
        unsafe_allow_html=True,
    )

PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_PATH = PROJECT_ROOT / "metrics" / "eval.json"
RETRAIN_QUALITY_PATH = PROJECT_ROOT / "metrics" / "retrain_quality_eval.json"
MIN_NEW_ROWS_FOR_RETRAIN = int(os.getenv("MIN_NEW_ROWS_FOR_RETRAIN", "60"))
ASSETS_DIR = PROJECT_ROOT / "assets" / "slides"
IMG_AIRFLOW_MAIN = ASSETS_DIR / "airflow_weather_main_graph.png"
IMG_AIRFLOW_OPTUNA = ASSETS_DIR / "airflow_optuna_graph.png"
IMG_DVC_PIPELINE = ASSETS_DIR / "dvc_pipeline_graph.png"
IMG_MLFLOW_RUNS = ASSETS_DIR / "mlflow_runs.png"

STATIONS = [
    "Adelaide", "Albany", "Albury-Wodonga", "Alice Springs", "Ballarat", "Bendigo", "Brisbane",
    "Broome", "Cairns", "Canberra", "Casey", "Christmas Island", "Cocos Islands", "Darwin",
    "Davis", "Devonport", "Gold Coast", "Hobart", "Kalgoorlie-Boulder", "Launceston",
    "Lord Howe Island", "Macquarie Island", "Mawson", "Melbourne", "Mount Gambier",
    "Norfolk Island", "Penrith", "Perth", "Port Lincoln", "Renmark", "Sydney",
    "Tennant Creek", "Townsville", "Tuggeranong", "Wollongong", "Wynyard",
]

def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _show_metric_cards(metrics: dict[str, Any], title: str) -> None:
    st.subheader(title)
    if not metrics:
        st.info("Fichier de metriques indisponible pour l'instant.")
        return
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{metrics.get('accuracy_live', metrics.get('accuracy', 0)):.3f}")
    cols[1].metric("Precision", f"{metrics.get('precision_live', metrics.get('precision', 0)):.3f}")
    cols[2].metric("Recall", f"{metrics.get('recall_live', metrics.get('recall', 0)):.3f}")
    cols[3].metric("F1", f"{metrics.get('f1_live', metrics.get('f1', 0)):.3f}")
    roc = metrics.get("roc_auc_live", metrics.get("roc_auc"))
    cols[4].metric("ROC AUC", f"{roc:.3f}" if isinstance(roc, (int, float)) else "n/a")

def _show_combined_metric_cards(metrics: dict[str, Any], title: str) -> None:
    st.subheader(title)
    if not metrics:
        st.info("Fichier de métriques combinées indisponible.")
        return
    if metrics.get("status") != "ok":
        st.warning(f"Statut métriques combinées: {metrics.get('status', 'unknown')}")
        return
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{float(metrics.get('accuracy_combined_cv', 0.0)):.3f}")
    cols[1].metric("Precision", f"{float(metrics.get('precision_combined_cv', 0.0)):.3f}")
    cols[2].metric("Recall", f"{float(metrics.get('recall_combined_cv', 0.0)):.3f}")
    cols[3].metric("F1", f"{float(metrics.get('f1_combined_cv', 0.0)):.3f}")
    roc = metrics.get("roc_auc_combined_cv")
    cols[4].metric("ROC AUC", f"{float(roc):.3f}" if isinstance(roc, (int, float)) else "n/a")
    
def _api_fetch_latest(api_url: str, station: str) -> dict[str, Any]:
    url = f"{api_url.rstrip('/')}/latest_weather"
    resp = requests.get(url, params={"station_name": station}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def _api_request(api_url: str, api_key: str, station: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "accept": "application/json"}
    url = f"{api_url.rstrip('/')}/predict"
    params = {"use_latest": "true", "station_name": station}
    resp = requests.post(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    payload["station_name"] = station
    return payload



def _accueil_view() -> None:
    st.markdown(
        """
    <div class="hero">
         <h2 style="margin:0;">Prévision météo en Australie</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    st.markdown(
            """
    <div class="card">
    <div class="metric-big">Objectifs du projet</div>
    <div class="muted">
    Le projet transforme un besoin métier simple en système décisionnel: anticiper la pluie du lendemain pour aider la planification terrain.<br>

    1. <b>Prédire RainTomorrow</b> (Yes/No) avec une bonne précision<br>
    3. <b>Fournir une API</b> pour des prédictions en temps réel<br>
    4. <b>Mettre en place un pipeline MLOps</b> reproductible et scalable
    </div>
    </div>
    """,unsafe_allow_html=True,)
        
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
    <div class="card">
    <div class="metric-big">Résultats atteints</div>
    <div class="muted">
        <ul>
            <li><b>Cas d'usage clarifié:</b> prédire RainTomorrow sur des stations australiennes.</li>
            <li><b>Fournir des prédictions via API :</b> pour une utilisation en production.</li>
            <li><b>Attentes fonctionnelles formalisées:</b> prédire, exposer la prédiction, suivre la qualité dans le temps.</li>
            <li><b>Objectif MLOps posé dès le début:</b> un pipeline reproductible, observable et automatisable.</li>
            <li><b>Métrique clé choisie:</b> recall en priorité, complétée par precision, f1, roc_auc et accuracy.</li>
            <li><b>Stack outillée:</b> Python ML, FastAPI, MLflow, DVC/DagsHub, Airflow, Docker, Streamlit.</li>
        </ul>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    cols = st.columns(4)
    overview = [
        ("Données", "weatherAUS.csv", "~87k lignes, 23 colonnes"),
        ("Modèle", "XGBoost + SMOTE", "Pipeline sklearn/imblearn"),
        ("API", "FastAPI", "POST /predict"),
        ("Déploiement", "Docker", "MLflow, DVC, Airflow"),
    ]
    for i, (title, tech, desc) in enumerate(overview):
        with cols[i]:
            st.markdown(
                f"""
            <div class="card">
            <div class="accent"><b>{title}</b></div>
            <div class="muted" style="font-size: 0.9rem;"><b>{tech}</b><br>{desc}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

def _modelisation_view() -> None:
    st.markdown(
        """
    <div class="hero">
         <h2 style="margin:0;">Prévision météo en Australie</h2>
         <p>Bâtir un socle fiable et mesurable sur lequel reposera par la suite tout le cycle ML .</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    metric_style = """
        <div style='display: flex; flex-direction: column; align-items: flex-start;'>
            <div style='font-size: 0.95rem; color: var(--muted, #5a6b82); margin-bottom: 0.15em;'>{label}</div>
            <div style='font-size: 1.3rem; color: #2b79c4;font-weight: 700;'>{value}</div>
            <div style='font-size: 0.85rem; color: var(--muted, #132238);'>{delta}</div>
        </div>
    """
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    with col1:
        st.markdown(metric_style.format(label="Dataset", value="<span style='font-size:0.9em;'>weatherAUS.csv</span>", delta="Kaggle"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_style.format(label="Période", value="<span style='font-size:0.9em;'>~10 ans</span>", delta="Observations quotidiennes"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_style.format(label="Lignes", value="<span style='font-size:0.9em;'>~87k</span>", delta="23 colonnes"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_style.format(label="Cible", value="<span style='font-size:0.9em;'>RainTomorrow</span>", delta="Yes/No"), unsafe_allow_html=True)
    
    st.markdown("""
        <div class="card">
        <div class="metric-big">Pipeline de modélisation</div>
        <div class="muted">
        <ol>
        <li><b>Collecte et chargement</b> des données depuis le .csv</li>
        <li><b>Nettoyage et feature engineering</b> des données</li>
        <li><b>Imputation</b> des valeurs manquantes (fit sur train uniquement)</li>
        <li><b>One-hot encoding</b> des variables catégorielles</li>
        <li><b>Encodage cyclique</b> pour Dayofyear et Month</li>
        <li><b>Scaling</b> (StandardScaler, with_mean=False)</li>
        <li><b>SMOTE</b> pour rééquilibrer les classes (pluie minoritaire)</li>
        <li><b>XGBoost</b> pour la classification</li>
        <li><b>Évaluation</b> avec le set de métriques choisi.</li>
        </ol>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    st.markdown("""
                <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 2em;'>
            <div style='
                background: #f5f7fa;
                border-radius: 2em;
                padding: 0.85em 2em;
                font-size: 1.2rem;
                color: #223047;
                box-shadow: 0 1px 8px 0 rgba(32,48,77,.10);
                font-weight: 600;
                display: inline-block;
                border: 1.5px solid #e0e6ef;
                '>
                <span style='color:#2b79c4;'>Collecte</span>
                <span style='margin: 0 0.5em; font-size:1.3em;'>→</span>
                <span style='color:#2b79c4;'>Preparation des données</span>
                <span style='margin: 0 0.5em; font-size:1.3em;'>→</span>
                <span style='color:#2b79c4;'>Entraînement</span>
                <span style='margin: 0 0.5em; font-size:1.3em;'>→</span>
                <span style='color:#2b79c4;'>Évaluation</span>
                <span style='margin: 0 0.5em; font-size:1.3em;'>→</span>
                <span style='color:#2b79c4;'>Prédiction</span>
                <span style="margin-left:1.2em; font-size:0.99em;color:var(--muted,#708099);font-weight:500;">
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)   

def _api_view() -> None:
    
    st.markdown(
        """
    <div class="hero">
         <h2 style="margin:0;">Prévision météo en Australie</h2>
         <p>Transformer le modèle en service opérationnel, exploitable par des systèmes métiers.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="card">
        <div class="metric-big">API FastAPI</div>
        <div class="muted">
        <ul>
        <li>Endpoint <code>/health</code> pour vérifier rapidement la disponibilité du service.</li>
        <li>Création de l'endpoint <code>/predict</code> pour produire une prédiction sur demande.</li>
        <li>Chargement du modèle (joblib.load), sans ré-entraînement.</li>
        <li>Tests manuels via Swagger UI et requêtes HTTP (curl).</li>
        <li>Tests unitaires pour simuler des appels à l'API (pytest).</li>
        <li>Securisation de l'api par API key</li>
        <li>Journalisation des prédictions pour le monitoring et la traçabilité.</li>
        </ul>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    st.markdown("""
                <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 2em;'>
            <div style='
                background: #f5f7fa;
                border-radius: 2em;
                padding: 0.85em 2em;
                font-size: 1.2rem;
                color: #223047;
                box-shadow: 0 1px 8px 0 rgba(32,48,77,.10);
                font-weight: 600;
                display: inline-block;
                border: 1.5px solid #e0e6ef;
                '>
                <span style='color:#2b79c4;'>GET<code>/health</code></span>
                <span style='margin: 0 0.5em; font-size:1.3em;'>→</span>
                <span style='color:#2b79c4;'>Vérification de l'état du service</span><br>
                <span style='color:#2b79c4;'>GET<code>/latest_weather</code></span>
                <span style='margin: 0 0.5em; font-size:1.3em;'>→</span>
                <span style='color:#2b79c4;'>Récuperation des données du jour</span><br>
                <span style='color:#2b79c4;'>POST<code>/predict</code></span>
                <span style='margin: 0 0.5em; font-size:1.3em;'>→</span>
                <span style='color:#2b79c4;'>pipeline.joblib chargé en mémoire</span>
                <span style='margin: 0 0.5em; font-size:1.3em;'>→</span>
                <span style='color:#2b79c4;'>Prédiction</span><br>
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def _versionning_view() -> None:
    
    st.markdown(
        """
    <div class="hero">
         <h2 style="margin:0;">Prévision météo en Australie</h2>
         <p>Rendre le projet pilotable, traçable et maîtrisable dans le temps.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown("""
        <div class="card">
        <div class="metric-big">Ce qui a été fait</div>
        <div class="muted">
        <ul>
        <li>Tracking MLflow des runs: paramètres, métriques, artefacts.</li>
        <li>Comparaison des essais pour choisir les configurations utiles.</li>
        <li>Versioning data/artefacts avec DVC et stockage distant DagsHub.</li>
        </ul>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    with col_b:
        st.markdown("""
        <div class="card">
        <div class="metric-big">Pourquoi ce choix</div>
        <div class="muted">
        <ul>
        <li>MLflow donne une mémoire expérimentale exploitable par toute l'équipe.</li>
        <li>DVC complète Git pour les fichiers volumineux et la reproductibilité.</li>
        </ul>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    st.markdown("""
                <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 2em;'>
            <div style='
                background: #f5f7fa;
                border-radius: 2em;
                padding: 0.85em 2em;
                font-size: 1.2rem;
                color: #223047;
                box-shadow: 0 1px 8px 0 rgba(32,48,77,.10);
                font-weight: 600;
                display: inline-block;
                border: 1.5px solid #e0e6ef;
                '>
                <span style='color:#2b79c4;'>Entraînements</span>
                <span style='margin: 0 0.5em; font-size:1.3em;'>→</span>
                <span style='color:#2b79c4;'>MLflow (runs comparables)</span><br>
                <span style='color:#2b79c4;'>Prédictions/datasets</span>
                <span style='margin: 0 0.5em; font-size:1.3em;'>→</span>
                <span style='color:#2b79c4;'> DVC</span>
                <span style='margin: 0 0.5em; font-size:1.3em;'>→</span>
                <span style='color:#2b79c4;'>DagsHub</span><br>
                <span style='color:#2b79c4;'>Code/config</span>
                <span style='margin: 0 0.5em; font-size:1.3em;'>→</span>
                <span style='color:#2b79c4;'> Git</span>
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='panel'><h4>Pipeline de données (DVC)</h4></div>", unsafe_allow_html=True)
    if IMG_DVC_PIPELINE.exists():
            st.image(
                str(IMG_DVC_PIPELINE),
                caption=(
                    "Vue DVC: elle montre le flux complet données -> preprocessing -> entraînement -> évaluation "
                    "et les artefacts suivis."
                ),
                use_container_width=True,
            )
    else:
            st.info(
                "Image manquante: ajoutez `assets/slides/dvc_pipeline_graph.png` pour afficher la vue DVC dans cette slide."
            )
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='panel'><h4>Suivi des expériences MLflow</h4></div>", unsafe_allow_html=True)
    if IMG_MLFLOW_RUNS.exists():
            st.image(
                str(IMG_MLFLOW_RUNS),
                caption=(
                    "Vue MLflow des runs: comparaison des entraînements/évaluations et des runs Optuna, "
                    "avec leurs métriques et durées."
                ),
                use_container_width=True,
            )
    else:
            st.info(
                "Image manquante: ajoutez `assets/slides/mlflow_runs.png`."
            )

def _orchestration_view() -> None:
    
    st.markdown(
        """
    <div class="hero">
         <h2 style="margin:0;">Prévision météo en Australie</h2>
         <p>Orchestrer et automatiser le cycle de vie ML de bout en bout.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([2.5, 1])

    with col_a:
        st.markdown("""
        <div class="card">
        <div class="metric-big">Ce qui a été fait</div>
        <div class="muted">
        <ul>
        <li>Mises à jour automatisées du cycle prédiction → monitoring → décision.</li>
        <li>Récupération et traitement automatiques des données live via scripts orchestrables.</li>
        <li>Découpage en microservices: API, tracking, orchestration, jobs ML.</li>
        <li>Conteneurisation des services (Dockerfiles dédiés API/Airflow/MLflow).</li>
        <li>Connexion des services avec Docker Compose.</li>
    
        </ul>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    with col_b:
        st.markdown("""
        <div class="card">
        <div class="metric-big">Pourquoi ce choix</div>
        <div class="muted">
        <ul>
        <li>Le découpage par service isole les responsabilités et simplifie les dépannages.</li>
        <li>L'orchestration conditionnelle évite des retrains inutiles.</li>
        </ul>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    st.markdown("""
                <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 2em;'>
            <div style='
                background: #f5f7fa;
                border-radius: 2em;
                padding: 0.85em 2em;
                font-size: 1.2rem;
                color: #223047;
                box-shadow: 0 1px 8px 0 rgba(32,48,77,.10);
                font-weight: 600;
                display: inline-block;
                border: 1.5px solid #e0e6ef;
                '>
                <span style='color:#2b79c4;'>Docker Compose</span><br>
                <span style='margin: 0 0.5em; font-size:1.3em;'>|-</span>
                <span style='color:#2b79c4;'>FastAPI (inférence)</span><br>
                <span style='margin: 0 0.5em; font-size:1.3em;'>|-</span>
                <span style='color:#2b79c4;'>MLflow (tracking)</span><br>
                <span style='margin: 0 0.5em; font-size:1.3em;'>|-</span>
                <span style='color:#2b79c4;'>Airflow (orchestration)</span><br>
                <span style='margin: 0 0.5em; font-size:1.3em;'>|-</span>
                <span style='color:#2b79c4;'>Postgres Airflow (metadata)</span><br>
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def _dags_view() -> None:
    
    st.markdown(
        """
    <div class="hero">
         <h2 style="margin:0;">Prévision météo en Australie</h2>
         <p>Quand et pourquoi l’optimisation est déclenchée, et comment les outils s’articulent pour piloter la décision de bout en bout.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([2.45, 1.35])

    with col_a:
        st.markdown("""
        <div class="card" style="min-height: 290px;">
        <div class="metric-big">Ce qui a été fait</div>
        <div class="muted">
        <ul>
        <li>DAG principal <code>weather_main_dag</code>: prédiction, scoring live, branche conditionnelle.</li>
        <li>Condition Optuna: recall_combined_cv < seuil ET new_rows_for_retrain ≥ seuil minimal.</li>
        <li>DAG <code>optuna_tuning_dag</code>: build dataset retrain → optuna → export best params → retrain modèle → maj watermark.</li>
        <li>Watermark retrain: évite de relancer Optuna sur les mêmes nouvelles lignes déjà consommées.</li>
        <li>Architecture complète relie API, monitoring, orchestration, versioning et tracking.</li>
       
        </ul>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    with col_b:
        st.markdown("""
        <div class="card" style="min-height: 290px;">
        <div class="metric-big">Pourquoi ce choix</div>
        <div class="muted">
        <ul>
        <li>Le watermark stabilise la logique métier dans le temps.</li>
        <li>Les contraintes explicites rendent la décision audit-able et présentable au jury.</li>
        </ul>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div class="card" style="min-height: 185px;">
            <div class="metric-big">Architecture outils</div>
            <div class="muted" style="line-height: 1.75;">
                <b>Flux produit :</b> Streamlit -> FastAPI -> outputs/preds_api.csv -> live_monitoring -> metrics -> Airflow décision<br>
                <b>Boucle entraînement :</b> MLflow <-> training / optuna<br>
                <b>Versioning :</b> DVC + DagsHub <-> données et artefacts
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("""
        <div class="card" style="min-height: 430px;">
            <div class="metric-big">weather_main_dag</div>
            <div class="muted" style="line-height: 1.65;">
                predict_all_and_push -> predictions_ready -> compute_live_metrics -> compute_retrain_quality -> branch_on_recall<br><br>
                <b>Règle de décision :</b><br>
                if new_rows &lt; MIN_NEW_ROWS_FOR_RETRAIN: skip_optuna<br>
                elif recall_combined_cv &lt; RECALL_THRESHOLD: trigger_optuna<br>
                else: skip_optuna<br><br>
                Fin branche qualité: <b>join_after_branch</b>
            </div>
        </div>
    """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="card" style="min-height: 430px;">
            <div class="metric-big">optuna_tuning_dag</div>
            <div class="muted" style="line-height: 1.75;">
                run_optuna (dataset retrain + recherche)<br>
                -> save_best_params<br>
                -> retrain_with_best<br>
                -> mark_retrain_consumed (watermark)<br><br>
                <b>Effet :</b> évite de retraiter le même lot de lignes live déjà consommées.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='panel'><h4>Graphes Airflow</h4></div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
            if IMG_AIRFLOW_MAIN.exists():
                st.image(
                    str(IMG_AIRFLOW_MAIN),
                    caption=(
                        "DAG principal: prédiction -> métriques live -> branche de décision "
                        "(trigger_optuna ou skip_optuna)."
                    ),
                    use_container_width=True,
                )
            else:
                st.info(
                    "Image manquante: ajoutez `assets/slides/airflow_weather_main_graph.png`."
                )
    with c2:
            if IMG_AIRFLOW_OPTUNA.exists():
                st.image(
                    str(IMG_AIRFLOW_OPTUNA),
                    caption=(
                        "DAG Optuna: run_optuna -> save_best_params -> retrain_with_best -> mark_retrain_consumed "
                        "(watermark pour éviter de retraiter les mêmes lignes)."
                    ),
                    use_container_width=True,
                )
            else:
                st.info(
                    "Image manquante: ajoutez `assets/slides/airflow_optuna_graph.png`."
                )

def _outro_view() -> None:
    
    st.markdown(
        """
    <div class="hero">
         <h2 style="margin:0;">Prévision météo en Australie</h2>
         <p>Le socle est en place. La suite vise surtout la robustesse de production, l'observabilité et l'opération à plus grande échelle.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="card">
        <div class="metric-big">Evolutions possibles</div>
        <div class="muted">
        <ul>
        <li>Ajouter une couche Nginx (reverse proxy, TLS, routage).</li>
        <li>Mettre en place Prometheus + Grafana pour monitorer API, DAGs et métriques métier.</li>
        <li>Ajouter des alertes automatiques (latence API, baisse recall, échec DAG).</li>
        <li>Industrialiser davantage la CI/CD (tests, build images, déploiement contrôlé).</li>
        <li>Approfondir le monitoring data/model: drift, calibration, analyses par station.</li>
        </ul>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div class="card">
        <div class="metric-big">Pourquoi ce choix</div>
        <div class="muted">
        <ul>
        <li>Ces briques transforment un bon POC en plateforme exploitable en continu.</li>
        <li>La priorité est de garder une systémique simple, observable et explicable.</li>
        </ul>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col_a, col_b = st.columns([1, 1.3])

    with col_a:
        st.markdown("""
                <div class="card" style="margin-top:1em;">
            <div class="metric-big">Aujourd'hui</div>
            <div class="muted">
                <ul>
                    <li>Pipeline opérationnel et orchestré</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="card" style="margin-top:1em;">
            <div class="metric-big">Demain</div>
            <div class="muted">
                <ul>
                    <li>Nginx (reverse proxy)</li>
                    <li>Prometheus + Grafana</li>
                    <li>Alerting automatique</li>
                    <li>CI/CD renforcée</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

def _live_demo_view() -> None:
    st.markdown(
        """
<div class="hero">
  <h2 style="margin:0;">Démonstration</h2>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)

    api_url = st.sidebar.text_input("API URL", value=os.getenv("STREAMLIT_API_URL", "http://127.0.0.1:8000"))
    api_key = st.sidebar.text_input("API KEY", value=os.getenv("API_KEY", ""), type="password")
    

    tab1, tab2, tab3 = st.tabs(
        ["1) Santé & prédiction API", "2) Batch 36 stations", "3) Monitoring métriques"]
    )

    with tab1:
        st.subheader("Test santé API")
        if st.button("Appeler /health"):
            try:
                resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=15)
                st.json(resp.json())
            except Exception as e:
                st.error(f"Erreur API: {e}")

        st.subheader("Récupération automatique")


        st.subheader("Fetch meteo BOM (/latest_weather)")
        fetch_station = st.selectbox("Station pour fetch", STATIONS, index=0, key="fetch_station")
        if st.button("Appeler /latest_weather"):
            try:
                payload = _api_fetch_latest(api_url, fetch_station)
                st.success("Fetch météo réussi.")
                st.json(payload)
            except Exception as e:
                st.error(f"Erreur fetch: {e}")

        st.subheader("Prediction unitaire")
        station = st.selectbox("Station", STATIONS, index=0, key="predict_station")
        if st.button("Prédire pour la station sélectionnée"):
            if not api_key:
                st.warning("Renseigne API KEY dans la sidebar.")
            else:
                try:
                    result = _api_request(api_url, api_key, station)
                    st.success("Prédiction réalisée.")
                    st.json(result)
                except Exception as e:
                    st.error(f"Erreur prediction: {e}")

    with tab2:
        st.subheader("Batch 36 stations (via API)")
        st.caption("Ce bouton déclenche 36 appels API successifs.")
        if st.button("Lancer batch API (36 stations)"):
            if not api_key:
                st.warning("Renseigne API KEY dans la sidebar.")
            else:
                rows = []
                progress = st.progress(0)
                for idx, station in enumerate(STATIONS, start=1):
                    try:
                        rows.append(_api_request(api_url, api_key, station))
                    except Exception as e:
                        rows.append({"station_name": station, "error": str(e)})
                    progress.progress(idx / len(STATIONS))
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab3:
        cta1, cta2 = st.columns([1, 3])
        with cta1:
            if st.button("Rafraichir les dernieres metriques"):
                st.rerun()
        with cta2:
            if RETRAIN_QUALITY_PATH.exists():
                ts = datetime.fromtimestamp(RETRAIN_QUALITY_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                st.caption(f"Dernière mise à jour combinée: {ts}")
            else:
                st.caption("Fichier combiné non trouvé: metrics/retrain_quality_eval.json")

        combined_metrics = _load_json(RETRAIN_QUALITY_PATH)
        offline_metrics = _load_json(EVAL_PATH)

        _show_combined_metric_cards(combined_metrics, "Métriques combinées (décision Optuna)")
        st.divider()
        _show_metric_cards(offline_metrics, "Métriques offline")

def main() -> None:
    _apply_premium_theme()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Vision d'ensemble","Modélisation","API FastAPI","Suivi et Versioning","Orchestration et déploiement", "DAGs","Outro", "Demo live"])

    if page == "Vision d'ensemble":
        _accueil_view()
    elif page == "Modélisation":
        _modelisation_view()
    elif page == "API FastAPI":
        _api_view()
    elif page == "Suivi et Versioning":
        _versionning_view()
    elif page == "Orchestration et déploiement":
        _orchestration_view()
    elif page == "DAGs":
        _dags_view()
    elif page == "Outro":
        _outro_view()
    else:
        _live_demo_view()


if __name__ == "__main__":
    main()
