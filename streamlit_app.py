from __future__ import annotations

import json
import os
import subprocess
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
COMBINED_METRICS_PATH = PROJECT_ROOT / "metrics" / "combined_eval.json"
PREDS_PATH = PROJECT_ROOT / "outputs" / "preds_api.csv"
SCORED_PREDS_PATH = PROJECT_ROOT / "outputs" / "preds_api_scored.csv"
EVAL_PATH = PROJECT_ROOT / "metrics" / "eval.json"
OPTUNA_SCRIPT = PROJECT_ROOT / "optimisations" / "optuna_search_recall_small.py"

INSPIRATION_ODP = Path(r"C:\Users\timot\Downloads\Présentation_MLOPS_Analyse_des_sentiments.odp")
INSPIRATION_PDF = Path(r"C:\Users\timot\Downloads\parivision_presentation.pdf")

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


def _run_optuna_demo(timeout_seconds: int) -> tuple[int, str]:
    cmd = ["python", str(OPTUNA_SCRIPT)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return proc.returncode, (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + ("\n" + e.stderr if e.stderr else "")
        return 124, f"Arret automatique apres {timeout_seconds}s (mode demo).\n\n{out}"


def _render_schema(title: str, schema_text: str) -> None:
    st.markdown(f"<div class='panel'><h4>{title}</h4></div>", unsafe_allow_html=True)
    st.markdown("<div class='schema-box'>", unsafe_allow_html=True)
    st.code(textwrap.dedent(schema_text).strip("\n"), language="text")
    st.markdown("</div>", unsafe_allow_html=True)


def _slide_payload() -> list[dict[str, Any]]:
    return [
        {
            "title": "Slide 1 - Cadrage: le probleme que nous resolvons",
            "context": "Notre objectif est simple: aider a anticiper la pluie du lendemain sur 36 stations pour mieux planifier les operations.",
            "done": [
                "Le besoin metier a ete formalise: reduire les surprises liees a la pluie.",
                "Nous avons defini des indicateurs lisibles pour suivre la qualite du modele.",
                "Une regle de decision a ete fixee: si la qualite baisse, on relance l'amelioration automatiquement.",
            ],
            "choices": [
                "Le recall est prioritaire: manquer un vrai jour de pluie coute plus cher qu'une fausse alerte.",
                "Plusieurs metriques sont combinees pour eviter les conclusions trompeuses.",
                "Une regle claire permet d'expliquer facilement les decisions au jury et aux operations.",
            ],
            "schema": """
Besoin metier -> Indicateurs de qualite -> Decision automatique
      |                 |                        |
      |                 |                        +--> Re-optimiser si performance insuffisante
      |                 +--> recall, precision, f1, roc_auc, accuracy
      +--> Predire la pluie du lendemain
""",
        },
        {
            "title": "Slide 2 - Phase 1: donner des bases solides au projet",
            "context": "Avant d'optimiser, nous avons construit un premier cycle complet: donnees -> modele -> evaluation.",
            "done": [
                "Les donnees ont ete recuperees, nettoyees et harmonisees.",
                "Les variables utiles ont ete preparees pour simplifier l'apprentissage.",
                "Un premier modele de reference a ete entraine.",
                "Les performances ont ete mesurees sur des donnees non vues.",
            ],
            "choices": [
                "Commencer par une baseline evite de perdre du temps sur de l'optimisation prematuree.",
                "Un pipeline standard rend les executions reproductibles et compréhensibles.",
            ],
            "schema": """
Donnees brutes -> Preparation -> Entrainement -> Modele initial
                               -> Evaluation -> Metriques de reference
""",
        },
        {
            "title": "Slide 3 - API: transformer le modele en service utilisable",
            "context": "Le modele n'est utile que s'il est accessible facilement: l'API permet cette mise en production.",
            "done": [
                "Un endpoint de prediction a ete cree pour interroger le modele en direct.",
                "Un endpoint de sante confirme rapidement que le service fonctionne.",
                "Une cle API protege les appels.",
                "Chaque prediction est enregistree pour assurer la tracabilite.",
            ],
            "choices": [
                "FastAPI facilite les tests, la documentation et la maintenance.",
                "Le mode 'use_latest' rend la demo fluide sans manipulations complexes.",
            ],
            "schema": """
Utilisateur -> API /predict -> Modele charge en memoire
           -> Reponse immediate (pluie oui/non + probabilite)
           -> Journalisation pour suivi et audit
""",
        },
        {
            "title": "Slide 4 - Phase 2: suivi des essais et versioning des donnees",
            "context": "Nous avons outille le projet pour savoir ce qui a ete teste, pourquoi, et avec quels resultats.",
            "done": [
                "Chaque entrainement est historise avec ses parametres et ses metriques.",
                "Les fichiers de predictions sont versions avec DVC/DagsHub.",
                "Des tests simples verifient que les fonctions critiques restent stables.",
            ],
            "choices": [
                "MLflow permet de comparer objectivement les essais.",
                "DVC complete Git pour suivre proprement les fichiers data.",
            ],
            "schema": """
Experiences modele -> Historique MLflow
Predictions produites -> DVC -> Stockage DagsHub
""",
        },
        {
            "title": "Slide 5 - Phase 3: orchestration quotidienne avec Airflow",
            "context": "Le pipeline tourne automatiquement deux fois par jour et verifie si le modele reste fiable.",
            "done": [
                "Un DAG principal gere les predictions et les controles qualite.",
                "Un DAG dedie a l'optimisation est lance seulement si necessaire.",
                "Les metriques sont recalculees a chaque execution.",
            ],
            "choices": [
                "Separer les DAGs clarifie la lecture et limite les risques.",
                "Le retrain est conditionne pour eviter des reactions trop rapides sur peu de donnees.",
            ],
            "schema": """
weather_main_dag (06:00, 18:00)
  1) Predire
  2) Evaluer la qualite recente
  3) Decider: optimiser ou continuer tel quel
""",
        },
        {
            "title": "Slide 6 - Architecture: une application decoupee par role",
            "context": "Chaque composant a une responsabilite claire pour simplifier l'exploitation et les evolutions.",
            "done": [
                "Les services ont ete conteneurises (API, orchestration, suivi des experiences).",
                "Airflow s'appuie sur une base dediee pour son fonctionnement.",
                "Docker Compose demarre l'ensemble en une commande.",
            ],
            "choices": [
                "Le decoupage limite les effets de bord quand on fait evoluer un composant.",
                "Compose rend la demo reproductible sur une machine propre.",
            ],
            "schema": """
[Interface demo] -> [API prediction] -> [Fichiers de suivi]
        |                   |                   |
        |                   +--> [Modele]       +--> [DVC / DagsHub]
        +--> [Airflow] ----> Jobs planifies ----> [Metriques + MLflow]
""",
        },
        {
            "title": "Slide 7 - CI: automatiser les taches repetitives",
            "context": "Une partie du travail de publication est automatisee pour limiter les oublis manuels.",
            "done": [
                "Un workflow GitHub Actions publie les nouvelles predictions.",
                "Le fichier est suivi avec DVC puis pousse vers DagsHub.",
                "Les identifiants sont stockes de maniere securisee via GitHub Secrets.",
            ],
            "choices": [
                "Automatiser reduit la charge operative et les erreurs humaines.",
                "Un workflow visible facilite l'audit et la reprise par un autre membre d'equipe.",
            ],
            "schema": """
GitHub Actions -> versionne les predictions -> publie sur DagsHub
""",
        },
        {
            "title": "Slide 8 - Monitoring: verifier la qualite en continu",
            "context": "Nous comparons la qualite initiale et la qualite en conditions reelles pour agir au bon moment.",
            "done": [
                "Les metriques live sont calculees sur les observations recentes.",
                "Une vue combinee melange historique + donnees recentes.",
                "La decision d'optimisation suit une regle explicite (qualite + volume minimal).",
            ],
            "choices": [
                "Le live reflète la realite terrain, donc c'est le signal principal.",
                "La vue combinee donne du recul et evite les decisions basees sur du bruit court terme.",
            ],
            "schema": """
Predictions + donnees reelles -> Metriques live
Historique + live -> Metriques combinees
Metriques -> Decision automatique de retrain
""",
        },
        {
            "title": "Slide 9 - Conclusion: preuve, limites et prochaines etapes",
            "context": "La soutenance montre un systeme fonctionnel, explicable, et pret a etre ameliore de maniere progressive.",
            "done": [
                "L'application Streamlit combine explication du projet et demonstration live.",
                "Un mode demo d'optimisation court garantit une presentation fluide.",
                f"Des inspirations visuelles ont ete prises dans {INSPIRATION_ODP.name} et {INSPIRATION_PDF.name}.",
            ],
            "choices": [
                "Le fil narratif suit une logique simple: besoin -> solution -> preuve -> suite.",
                "La roadmap vise surtout la robustesse: calibration, drift et analyses par station.",
            ],
            "schema": """
Slides claires -> Demo en direct -> Questions du jury
""",
        },
    ]


def _slides_view() -> None:
    st.markdown(
        """
<div class="hero">
  <h2 style="margin:0;">Soutenance MLOps Meteo</h2>
  <p>Mode presentation: 1 slide par page, contenu pedagogique, schemas techniques.</p>
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
        if st.button("⬅️ Precedente", use_container_width=True):
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
        st.markdown("<div class='panel'><h4>Ce qui a ete fait</h4></div>", unsafe_allow_html=True)
        for item in slide["done"]:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("<div class='panel'><h4>Pourquoi ce choix</h4></div>", unsafe_allow_html=True)
        for item in slide["choices"]:
            st.markdown(f"- {item}")

    _render_schema("Schema explicatif", slide["schema"])

    if idx == total - 1:
        st.divider()
        a, b = st.columns(2)
        with a:
            st.write(f"ODP inspiration: `{INSPIRATION_ODP}`")
            if INSPIRATION_ODP.exists():
                with open(INSPIRATION_ODP, "rb") as f:
                    st.download_button("Telecharger ODP inspiration", f, file_name=INSPIRATION_ODP.name)
            else:
                st.info("ODP non accessible depuis cet environnement.")
        with b:
            st.write(f"PDF inspiration: `{INSPIRATION_PDF}`")
            if INSPIRATION_PDF.exists():
                with open(INSPIRATION_PDF, "rb") as f:
                    st.download_button("Telecharger PDF inspiration", f, file_name=INSPIRATION_PDF.name)
            else:
                st.info("PDF non accessible depuis cet environnement.")


def _live_demo_view() -> None:
    st.markdown(
        """
<div class="hero">
  <h2 style="margin:0;">Demo live du projet</h2>
  <p>Validation en direct: API, batch 36 stations, monitoring des metriques, test Optuna en mode soutenance.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    api_url = st.sidebar.text_input("API URL", value=os.getenv("STREAMLIT_API_URL", "http://127.0.0.1:8000"))
    api_key = st.sidebar.text_input("API KEY", value=os.getenv("API_KEY", ""), type="password")
    min_labels = st.sidebar.number_input("Seuil labels min (monitoring)", min_value=1, value=72, step=1)
    st.sidebar.caption(f"Decision actuelle: Optuna si recall < 0.59 ET labels >= {min_labels}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["1) Sante & prediction API", "2) Batch 36 stations", "3) Monitoring metriques", "4) Optuna mode demo"]
    )

    with tab1:
        st.subheader("Test sante API")
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
                    st.success("Prediction realisee.")
                    st.json(result)
                except Exception as e:
                    st.error(f"Erreur prediction: {e}")

    with tab2:
        st.subheader("Batch 36 stations (via API)")
        st.caption("Ce bouton declenche 36 appels API successifs.")
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
        combined_metrics = _load_json(COMBINED_METRICS_PATH)
        offline_metrics = _load_json(EVAL_PATH)

        _show_metric_cards(live_metrics, "Metriques live")
        st.divider()
        _show_metric_cards(combined_metrics, "Metriques combinees (historique + live)")
        st.divider()
        _show_metric_cards(offline_metrics, "Metriques offline")

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

    with tab4:
        st.subheader("Optuna mode demonstration")
        st.caption("Execution limitee a 10 secondes pour soutenance.")
        timeout_seconds = st.slider("Timeout (secondes)", min_value=5, max_value=30, value=10, step=1)
        if st.button("Lancer Optuna en mode demo"):
            if not OPTUNA_SCRIPT.exists():
                st.error(f"Script introuvable: {OPTUNA_SCRIPT}")
            else:
                code, logs = _run_optuna_demo(timeout_seconds)
                if code == 0:
                    st.success("Optuna termine dans la fenetre de demo.")
                elif code == 124:
                    st.warning("Optuna arrete automatiquement (timeout demo).")
                else:
                    st.error(f"Optuna a echoue (code {code}).")
                st.code(logs[-8000:] if logs else "Aucune sortie.", language="text")


def main() -> None:
    _apply_premium_theme()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choisir une vue", ["Slides projet", "Demo live"])
    st.sidebar.caption(f"Derniere mise a jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if page == "Slides projet":
        _slides_view()
    else:
        _live_demo_view()


if __name__ == "__main__":
    main()
