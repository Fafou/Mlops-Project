# ToxiGuard — détection de commentaires toxiques

Projet du module **MLOps & Industrialisation de l'IA** (2026) — groupe X.

Sur les réseaux sociaux, des millions de commentaires sont publiés chaque jour et la modération humaine ne suit pas.
ToxiGuard attribue à chaque commentaire un **score de toxicité** (6 catégories) servi par une API, pour que les
modérateurs ne vérifient que les cas douteux. L'objectif du projet est d'industrialiser ce modèle de bout en bout.

## Équipe

| Asma Jleili |
| Farah Charfeddine | 
| Ranim Khalifa | 
| Chaima Ben Radhia | 

## Données

- [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) :
  ~160 000 commentaires annotés, 6 étiquettes (toxic, severe_toxic, obscene, threat, insult, identity_hate).
- [Civil Comments](https://huggingface.co/datasets/google/civil_comments) (commentaires datés) pour étudier la dérive.
- Problème : classification **multi-label**. Métriques : ROC-AUC, F1, rappel (priorité au rappel).

Placer `train.csv` de Kaggle dans `data/raw/` (non versionné par Git, il sera suivi avec DVC).

## Architecture cible

Données (DVC + remote) → entraînement (TF-IDF + régression logistique, MLflow) → Model Registry →
API FastAPI (Docker Compose) → monitoring de la dérive (Evidently). Orchestration Prefect, CI/CD GitHub Actions + CML.

## Structure

```
src/toxiguard/
├── config.py          # chemins, étiquettes, graine
├── data/
│   ├── preprocess.py  # nettoyage du texte
│   └── load.py        # chargement du dataset Jigsaw
└── models/
    ├── pipeline.py    # TF-IDF + régression logistique
    └── train.py       # entraînement + suivi MLflow
tests/                 # tests unitaires (Pytest)
.github/workflows/     # CI : Ruff + Pytest
```

## Installation et utilisation

```bash
poetry install
poetry run pre-commit install
poetry run pytest
poetry run python -m toxiguard.models.train   # nécessite data/raw/train.csv
poetry run mlflow ui                           # consulter les expériences
```

## Avancement

| Pilier | Statut |
|---|---|
| Dépôt, structure src/, Poetry, pre-commit, Ruff | Fait |
| Tests unitaires + CI GitHub Actions | Fait |
| Code du modèle de base + suivi MLflow | Écrit, premier entraînement à lancer |
| DVC + remote | À faire |
| API FastAPI + Docker Compose | À faire |
| Prefect + Evidently (dérive) | À faire |
