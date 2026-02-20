from __future__ import annotations

import json
import os
import textwrap
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="MLOps Meteo - Soutenance",
    page_icon="🌦️",
    layout="wide",
)


def _apply_premium_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&display=swap');

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

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #eef4fb 0%, #f5f8fc 60%, #f8fbff 100%);
  border-right: 1px solid var(--line);
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  color: var(--ink);
}

h1, h2, h3 {
  font-family: "IBM Plex Serif", Georgia, serif !important;
  color: #0f2741 !important;
  letter-spacing: 0.1px;
}

.hero {
  background: linear-gradient(135deg, #0f4c81 0%, #14639f 65%, #1e7fb8 100%);
  color: #f8fbff;
  border-radius: 18px;
  padding: 1.15rem 1.25rem 1.05rem 1.25rem;
  border: 1px solid rgba(255,255,255,0.18);
  box-shadow: 0 16px 35px rgba(16,40,69,0.18);
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
</style>
""",
        unsafe_allow_html=True,
    )


PROJECT_ROOT = Path(__file__).resolve().parent
LIVE_METRICS_PATH = PROJECT_ROOT / "metrics" / "live_eval.json"
PREDS_PATH = PROJECT_ROOT / "outputs" / "preds_api.csv"
SCORED_PREDS_PATH = PROJECT_ROOT / "outputs" / "preds_api_scored.csv"
EVAL_PATH = PROJECT_ROOT / "metrics" / "eval.json"
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


def _load_preds_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # header ignored on purpose (mixed historical formats)
        for row in reader:
            if not row:
                continue
            if len(row) == 7:
                rows.append(
                    {
                        "logged_at_utc": row[0],
                        "feature_date": None,
                        "target_date": row[1],
                        "location": row[2],
                        "use_latest": row[3],
                        "station_name": row[4],
                        "rain_probability": row[5],
                        "predicted_rain": row[6],
                    }
                )
            elif len(row) >= 8:
                rows.append(
                    {
                        "logged_at_utc": row[0],
                        "feature_date": row[1],
                        "target_date": row[2],
                        "location": row[3],
                        "use_latest": row[4],
                        "station_name": row[5],
                        "rain_probability": row[6],
                        "predicted_rain": row[7],
                    }
                )
    return pd.DataFrame(rows)


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


def _api_request(api_url: str, api_key: str, station: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "accept": "application/json"}
    url = f"{api_url.rstrip('/')}/predict"
    params = {"use_latest": "true", "station_name": station}
    resp = requests.post(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    payload["station_name"] = station
    return payload


def _render_schema(title: str, schema_text: str) -> None:
    st.markdown(f"<div class='panel'><h4>{title}</h4></div>", unsafe_allow_html=True)
    st.markdown("<div class='schema-box'>", unsafe_allow_html=True)
    st.code(textwrap.dedent(schema_text).strip("\n"), language="text")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_visual_evidence(slide_title: str) -> None:
    if slide_title.startswith("2. Phase 1"):
        st.markdown("<div class='panel'><h4>Preuve visuelle: pipeline de données (DVC)</h4></div>", unsafe_allow_html=True)
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

    if slide_title.startswith("4. Phase 2"):
        st.markdown("<div class='panel'><h4>Preuve visuelle: suivi des expériences MLflow</h4></div>", unsafe_allow_html=True)
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

    if slide_title.startswith("6. DAGs"):
        st.markdown("<div class='panel'><h4>Preuve visuelle: graphes Airflow</h4></div>", unsafe_allow_html=True)
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


def _slide_payload() -> list[dict[str, Any]]:
    return [
        {
            "title": "1. Le projet météo: vision d'ensemble",
            "context": "Le projet transforme un besoin métier simple en système décisionnel: anticiper la pluie du lendemain pour aider la planification terrain.",
            "done": [
                "Cas d'usage clarifié: prévoir RainTomorrow sur des stations australiennes.",
                "Attentes fonctionnelles formalisées: prédire, exposer la prédiction, suivre la qualité dans le temps.",
                "Objectif MLOps posé dès le début: un pipeline reproductible, observable et automatisable.",
                "Métrique clé choisie: recall en priorité, complétée par precision, f1, roc_auc et accuracy.",
                "Stack outillée: Python ML, FastAPI, MLflow, DVC/DagsHub, Airflow, Docker, Streamlit.",
            ],
            "choices": [
                "Le recall est prioritaire car rater un vrai épisode de pluie est plus coûteux qu'une fausse alerte.",
                "La combinaison de métriques évite de juger le modèle sur un seul angle.",
                "Le choix des outils suit une logique de complémentarité: développer, servir, monitorer, automatiser.",
            ],
            "schema": """
Besoin métier (anticiper la pluie J+1)
          |
          v
Objectifs techniques (prédire + suivre + automatiser)
          |
          v
Indicateurs de qualité (recall, precision, f1, roc_auc, accuracy)
""",
        },
        {
            "title": "2. Phase 1: Fondations et conteneurisation",
            "context": "Cette phase construit le socle data/model. Le principe: d'abord fiabiliser les données, puis établir une baseline measurable.",
            "done": [
                "Collecte des données depuis les sources du projet et chargement dans des scripts Python.",
                "Nettoyage: gestion des manquants, valeurs aberrantes, cohérence de type.",
                "Feature engineering: transformation temporelle et prétraitements utiles au modèle.",
                "Entraînement d'un modèle de base pour obtenir un point de comparaison.",
                "Évaluation offline avec un set de métriques interprétable par l'équipe.",
            ],
            "choices": [
                "Une baseline permet de mesurer l'apport réel des optimisations ultérieures.",
                "Un prétraitement propre en amont réduit les erreurs en production.",
            ],
            "schema": """
Collecte -> Nettoyage -> Feature engineering -> Entraînement baseline -> Évaluation
                        (pipeline data robuste)
""",
        },
        {
            "title": "3. API d'inférence (FastAPI)",
            "context": "Le modèle devient un service actionnable: il peut être appelé en temps réel, testé et supervisé.",
            "done": [
                "Création de l'endpoint `/predict` pour produire une prédiction sur demande.",
                "Chargement du modèle entraîné via artefact sérialisé (joblib), sans ré-entraînement.",
                "Endpoint `/health` pour vérifier rapidement la disponibilité du service.",
                "Tests manuels via Swagger UI et requêtes HTTP (curl).",
                "Journalisation des prédictions pour le monitoring et la traçabilité.",
            ],
            "choices": [
                "FastAPI apporte une doc interactive immédiate, utile pour la démo et le debug.",
                "La séparation API/modèle facilite les évolutions sans casser le service.",
            ],
            "schema": """
Client -> /predict -> pipeline.joblib chargé en mémoire -> prédiction
     -> /health  -> validation rapide du service
     -> Swagger  -> test et documentation
""",
        },
        {
            "title": "4. Phase 2: Microservices, suivi et versioning",
            "context": "Cette phase rend le projet gouvernable: on suit les expériences, on versionne les données et on fiabilise l'API.",
            "done": [
                "Tracking MLflow des runs: paramètres, métriques, artefacts.",
                "Comparaison des essais pour choisir les configurations utiles.",
                "Versioning data/artefacts avec DVC et stockage distant DagsHub.",
                "Renforcement API: tests unitaires et authentification par token.",
            ],
            "choices": [
                "MLflow donne une mémoire expérimentale exploitable par toute l'équipe.",
                "DVC complète Git pour les fichiers volumineux et la reproductibilité.",
            ],
            "schema": """
Entraînements -> MLflow (runs comparables)
Prédictions/datasets -> DVC -> DagsHub
Code/config -> Git
""",
        },
        {
            "title": "5. Phase 3: Orchestration et déploiement",
            "context": "Airflow automatise la chaîne métier: produire, mesurer, puis décider s'il faut relancer une optimisation.",
            "done": [
                "Mises à jour automatisées du cycle prédiction -> monitoring -> décision.",
                "Récupération et traitement automatiques des données live via scripts orchestrables.",
                "Découpage en microservices: API, tracking, orchestration, jobs ML.",
                "Conteneurisation des services (Dockerfiles dédiés API/Airflow/MLflow).",
                "Connexion des services avec Docker Compose.",
            ],
            "choices": [
                "Le découpage par service isole les responsabilités et simplifie les dépannages.",
                "L'orchestration conditionnelle évite des retrains inutiles.",
            ],
            "schema": """
Docker Compose
  |- FastAPI (inférence)
  |- MLflow (tracking)
  |- Airflow (orchestration)
  |- Postgres Airflow (metadata)
""",
        },
        {
            "title": "6. DAGs + architecture outillage (logique et contraintes)",
            "context": "Cette slide assemble la logique décisionnelle complète: quand Optuna part, quand il ne part pas, et comment les outils se complètent.",
            "done": [
                "DAG principal `weather_main_dag`: prédiction, scoring live, branche conditionnelle.",
                "Condition Optuna: recall_live < seuil ET new_rows_for_retrain >= seuil minimal.",
                "DAG `optuna_tuning_dag`: build dataset retrain -> optuna -> export best params -> retrain modèle -> maj watermark.",
                "Watermark retrain: évite de relancer Optuna sur les mêmes nouvelles lignes déjà consommées.",
                "Architecture complète relie API, monitoring, orchestration, versioning et tracking.",
            ],
            "choices": [
                "Le watermark stabilise la logique métier dans le temps.",
                "Les contraintes explicites rendent la décision audit-able et présentable au jury.",
            ],
            "schema": """
weather_main_dag
  predict_all_and_push
    -> compute_live_metrics
    -> read_quality_metrics
    -> branch_on_recall
         if new_rows < MIN_NEW_ROWS_FOR_RETRAIN: skip_optuna
         else if recall_live < RECALL_THRESHOLD: trigger_optuna
         else: skip_optuna

optuna_tuning_dag
  run_optuna (dataset retrain + recherche)
    -> save_best_params
    -> retrain_with_best
    -> mark_retrain_consumed (watermark)

Architecture outils:
  Streamlit -> FastAPI -> outputs/preds_api.csv -> live_monitoring
                         -> metrics -> Airflow décision
  MLflow <-> training/optuna
  DVC + DagsHub <-> versioning data/artefacts
""",
        },
        {
            "title": "7. Outro: prochaines evolutions possibles",
            "context": "Le socle est en place. La suite vise surtout la robustesse de production, l'observabilité et l'opération à plus grande échelle.",
            "done": [
                "Ajouter une couche Nginx (reverse proxy, TLS, routage).",
                "Mettre en place Prometheus + Grafana pour monitorer API, DAGs et métriques métier.",
                "Ajouter des alertes automatiques (latence API, baisse recall, échec DAG).",
                "Industrialiser davantage la CI/CD (tests, build images, déploiement contrôlé).",
                "Approfondir le monitoring data/model: drift, calibration, analyses par station.",
            ],
            "choices": [
                "Ces briques transforment un bon POC en plateforme exploitable en continu.",
                "La priorité est de garder une systémique simple, observable et explicable.",
            ],
            "schema": """
Aujourd'hui:
  pipeline opérationnel et orchestré

Demain:
  Nginx + Prometheus + Grafana + alerting + CI/CD renforcée
""",
        },
    ]


def _slides_view() -> None:
    st.markdown(
        """
<div class="hero">
  <h2 style="margin:0;">Soutenance MLOps Météo</h2>
  <p>Mode présentation: 1 slide par page, contenu pédagogique, schémas techniques.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    slides = _slide_payload()
    total = len(slides)

    if "slide_idx" not in st.session_state:
        st.session_state.slide_idx = 0

    col_prev, col_mid, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ Précédente", use_container_width=True):
            st.session_state.slide_idx = max(0, st.session_state.slide_idx - 1)
    with col_mid:
        labels = [f"{i+1:02d}. {s['title']}" for i, s in enumerate(slides)]
        chosen = st.selectbox("Sommaire (jump slide)", options=labels, index=st.session_state.slide_idx)
        st.session_state.slide_idx = int(chosen.split(".")[0]) - 1
    with col_next:
        if st.button("Suivante ➡️", use_container_width=True):
            st.session_state.slide_idx = min(total - 1, st.session_state.slide_idx + 1)

    idx = st.session_state.slide_idx
    slide = slides[idx]
    st.progress((idx + 1) / total, text=f"Slide {idx + 1}/{total}")

    st.markdown(f"### {slide['title']}")
    st.markdown(
        f"<div class='panel'><h4>Contexte</h4><div>{slide['context']}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='badge-line'><span class='badge'>MLOps</span><span class='badge'>Airflow</span><span class='badge'>FastAPI</span><span class='badge'>MLflow</span><span class='badge'>DVC</span></div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("<div class='panel'><h4>Ce qui a été fait</h4></div>", unsafe_allow_html=True)
        for item in slide["done"]:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("<div class='panel'><h4>Pourquoi ce choix</h4></div>", unsafe_allow_html=True)
        for item in slide["choices"]:
            st.markdown(f"- {item}")

    _render_schema("Schema explicatif", slide["schema"])
    _render_visual_evidence(slide["title"])

def _live_demo_view() -> None:
    st.markdown(
        """
<div class="hero">
  <h2 style="margin:0;">Demo live du projet</h2>
  <p>Validation en direct: API, batch 36 stations, monitoring des métriques.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    api_url = st.sidebar.text_input("API URL", value=os.getenv("STREAMLIT_API_URL", "http://127.0.0.1:8000"))
    api_key = st.sidebar.text_input("API KEY", value=os.getenv("API_KEY", ""), type="password")
    st.sidebar.caption(
        f"Decision actuelle: Optuna si recall < 0.59 ET nouvelles lignes retrain >= {MIN_NEW_ROWS_FOR_RETRAIN}"
    )

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

        st.subheader("Prediction unitaire")
        station = st.selectbox("Station", STATIONS, index=0)
        if st.button("Predire pour la station selectionnee"):
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
        live_metrics = _load_json(LIVE_METRICS_PATH)
        offline_metrics = _load_json(EVAL_PATH)

        _show_metric_cards(live_metrics, "Métriques live")
        st.divider()
        _show_metric_cards(offline_metrics, "Métriques offline")

        st.subheader("Apercu des fichiers de suivi")
        c1, c2 = st.columns(2)
        with c1:
            if PREDS_PATH.exists():
                df = _load_preds_flexible(PREDS_PATH)
                st.write(f"`preds_api.csv` - {len(df)} lignes")
                st.dataframe(df.tail(20), use_container_width=True)
            else:
                st.info("`outputs/preds_api.csv` introuvable.")
        with c2:
            if SCORED_PREDS_PATH.exists():
                sdf = pd.read_csv(SCORED_PREDS_PATH)
                st.write(f"`preds_api_scored.csv` - {len(sdf)} lignes")
                st.dataframe(sdf.tail(20), use_container_width=True)
            else:
                st.info("`outputs/preds_api_scored.csv` introuvable.")


def main() -> None:
    _apply_premium_theme()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choisir une vue", ["Slides projet", "Demo live"])
    st.sidebar.caption(f"Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if page == "Slides projet":
        _slides_view()
    else:
        _live_demo_view()


if __name__ == "__main__":
    main()
