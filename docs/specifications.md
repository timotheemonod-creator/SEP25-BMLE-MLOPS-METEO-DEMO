# Spécifications - Phase 1

## Contexte & cas d’usage
Le client final est l’utilisateur de l’application. L’objectif est de prédire si
`RainTomorrow` sera "Yes" ou "No" à partir d’observations météo journalières.
Le système doit permettre une utilisation simple via API et des runs batch.

## Objectifs métier
- Réduire l’incertitude sur la pluie du lendemain pour la planification.
- Offrir une prédiction rapide avec explication des scores de base.

## Type de problème
Classification binaire supervisée (RainTomorrow ∈ {0,1}).

## Métriques clés
- **ROC AUC** : mesure globale de ranking.
- **Recall (classe pluie)** : limiter les faux négatifs.
- **Precision (classe pluie)** : limiter les fausses alertes.
- **F1-score** : équilibre précision/rappel.
- **Accuracy** : indicateur global, non suffisant seul.

## Hypothèses
- Les données historiques sont représentatives du futur.
- Les valeurs manquantes peuvent être imputées sans biais majeur.
- Les caractéristiques temporelles (jour/mois) améliorent la prédiction.

## Données & collecte
- Source principale : Kaggle `weatherAUS.csv`.
- Format : CSV téléchargé manuellement (Phase 1).
- Chemin attendu : `data/raw/weatherAUS.csv`.

## Prétraitement (Phase 1)
- Suppression des lignes sans cible.
- Outliers forcés en NaN pour certaines variables.
- Création des variables temporelles `Dayofyear` et `Month`.
- Imputation des valeurs manquantes (train uniquement).
- Encodage one-hot des variables catégorielles.

## Modèle baseline
- Pipeline : preprocessing → scaling → SMOTE → XGBoost.
- Split train/test stratifié (80/20).

## API d’inférence
- Endpoint : `POST /predict`
- Entrée : dictionnaire des features météo (incluant `Date`).
- Sortie : probabilité + classe binaire (seuil 0.5).

## Tests manuels
- Swagger UI : `http://127.0.0.1:8000/docs`
- `curl`/Postman pour la validation des réponses.
