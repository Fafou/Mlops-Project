"""Chargement du dataset Jigsaw Toxic Comment (Kaggle)."""

from pathlib import Path

import pandas as pd

from toxiguard.config import LABELS, RAW_DIR, TEXT_COLUMN
from toxiguard.data.preprocess import clean_text


def load_jigsaw(path: Path = RAW_DIR / "train.csv") -> pd.DataFrame:
    """Lit train.csv et ajoute la colonne clean_text (texte nettoyé)."""
    df = pd.read_csv(path)
    missing = [col for col in [TEXT_COLUMN, *LABELS] if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path} : {missing}")
    df["clean_text"] = df[TEXT_COLUMN].fillna("").astype(str).map(clean_text)
    return df
