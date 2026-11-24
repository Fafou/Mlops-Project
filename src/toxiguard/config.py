"""Chemins et paramètres partagés par tout le projet."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Les 6 étiquettes du dataset Jigsaw Toxic Comment
LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
TEXT_COLUMN = "comment_text"
RANDOM_STATE = 42
